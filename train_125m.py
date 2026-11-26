#!/usr/bin/env python3
"""125M-scale training for word/pair/trigram value-memory experiments on FineWeb GPT-2 tokens.

Clean, bug-fixed, single-file trainer supporting:
  - One GPU per process (e.g. 4 independent runs on 4x H20 / A100)
  - Optional torchrun DDP (`--distributed 1`)
  - Memory-safe chunked cross-entropy + proportional 10% prefix threshold scaling
  - Fast CPU/GPU routing (skips unnecessary binary search for plain/word/hash arms)
  - Automatic end-of-training evaluation at final token target + latency benchmark + final metrics JSON

Arms
----
plain              body only (Equal Active FLOPs baseline, default L=12, d=768, H=12)
plain_eq_weights   wider plain body (e.g. d=1152, H=16) matching total parameter budget (Equal Total Weights)
word               local full-d word table in the last 2 blocks (Bank of Values baseline)
hard_stack         local full-d word + hashed pair table in the last 2 blocks, decoupled LR
mp_ve              shared word + tiered pair bank injected into the last 2 blocks
u1                 shared word + tiered pair bank injected into all blocks (low-latency variant)
u123               u1 + context gates + residual injection + pre-head readout + exact 3-gram tier
u123_no3g          u123 without exact 3-gram tier (isolates 3-gram contribution)
ctrl_word          u123 dense architecture with ONLY word bank (controls for dense adapter FLOPs)
ctrl_hs            u123 dense architecture with hashed pair table (no maturity pruning, no exact 3g)
ctrl_hs_pruned     ctrl_hs with <mid_thresh low-count pairs masked (separates pruning vs exact de-collision)
uniform_mgram      value-side Over-Tokenized control: uniform unpruned, untiered 2g+3g hash pool at matched table budget
engram             multi-head hashed 2/3-gram table + K/V proj + context gate + causal depthwise conv
xgram              frequency-split word value table (top 80% token cover at d_mem, rest at d_mem/4 + proj)
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_main() -> bool:
    return get_rank() == 0


# ---------------------------------------------------------------------------
# Model pieces
# ---------------------------------------------------------------------------


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[-2]
        c, s = self.cos[:, :, :t], self.sin[:, :, :t]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon for 2-D block/adapter matrices."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, ns_steps: int = 5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, ns = group["lr"], group["momentum"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                m = self.state[p].get("m")
                if m is None:
                    m = torch.zeros_like(p, dtype=torch.float32)
                    self.state[p]["m"] = m
                m.mul_(mom).add_(grad)
                update = newton_schulz(grad + mom * m, ns)
                p.add_(update.to(p.dtype), alpha=-lr * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class ShardStream:
    """Memory-mapped uint16 token stream over FineWeb .bin shards."""

    def __init__(self, paths: list[str], header: int = 1024, dtype=np.uint16):
        if not paths:
            raise ValueError("No .bin shards found.")
        self.paths = paths
        self.header = header
        self.dtype = dtype
        self.mmaps = []
        self.lengths = []
        for p in paths:
            mm = np.memmap(p, dtype=dtype, mode="r", offset=header)
            self.mmaps.append(mm)
            self.lengths.append(len(mm))
        self.cum = np.cumsum([0] + self.lengths)
        self.total = int(self.cum[-1])

    def read(self, start: int, n: int) -> np.ndarray:
        if self.total <= n:
            raise ValueError(f"Token stream too small: {self.total} <= {n}")
        start = int(start) % (self.total - n)
        end = start + n
        chunks = []
        for i, mm in enumerate(self.mmaps):
            s, e = int(self.cum[i]), int(self.cum[i + 1])
            if start < e and end > s:
                lo = max(start, s) - s
                hi = min(end, e) - s
                chunks.append(np.asarray(mm[lo:hi]))
            if end <= e:
                break
        out = np.concatenate(chunks) if chunks else np.empty(0, dtype=self.dtype)
        if len(out) != n:
            raise RuntimeError(f"Short read: {len(out)} != {n}")
        return out

    def get_batch(self, start: int, n_seqs: int, seq_len: int) -> np.ndarray:
        n = n_seqs * (seq_len + 1)
        return self.read(start, n).astype(np.int64).reshape(n_seqs, seq_len + 1)


class RandomStream:
    """Smoke-test stream: random tokens, no disk."""

    def __init__(self, V: int, total: int = 10_000_000):
        self.V = V
        self.total = total

    def read(self, start: int, n: int) -> np.ndarray:
        start = int(start) % max(1, self.total - n)
        rng = np.random.default_rng(1234 + start)
        return rng.integers(0, self.V, size=n, dtype=np.int64)

    def get_batch(self, start: int, n_seqs: int, seq_len: int) -> np.ndarray:
        return self.read(start, n_seqs * (seq_len + 1)).reshape(n_seqs, seq_len + 1)


@dataclass
class CountData:
    V: int
    hi_keys: np.ndarray        # sorted int64 pair keys with c >= eff_hi (capped at n_hi)
    mid_keys: np.ndarray       # sorted int64 pair keys with eff_mid <= c < eff_hi (capped at n_mid)
    g3_keys: np.ndarray        # sorted int64 trigram keys with c >= eff_g3 (capped at n_3g)
    mature_keys: np.ndarray    # sorted int64 pair keys with c >= eff_mid (~10x smaller than all_keys!)
    all_keys: np.ndarray       # sorted int64 pair keys observed in prefix (for eval bucketing only)
    all_counts: np.ndarray     # counts aligned with all_keys (for eval bucketing only)
    eff_hi: int
    eff_mid: int
    eff_g3: int
    xgram_hi_words: np.ndarray | None = None  # sorted top token IDs covering ~80% of prefix tokens


def compute_xgram_hi_words(toks: np.ndarray, V: int, cover: float = 0.80) -> np.ndarray:
    """Select minimum prefix of token IDs sorted by count covering `cover` of `toks`."""
    w_counts = np.bincount(toks, minlength=V)
    order = np.argsort(-w_counts, kind="stable")
    cum = np.cumsum(w_counts[order])
    target = cover * max(1, int(cum[-1]))
    k = int(np.searchsorted(cum, target, side="left")) + 1
    k = max(1, min(k, max(1, V - 1)))
    return np.sort(order[:k].astype(np.int64))


def build_counts(stream: ShardStream | RandomStream, V: int, count_tokens: int,
                 eff_hi: int, eff_mid: int, eff_g3: int,
                 n_hi: int = 500_000, n_mid: int = 1_500_000, n_3g: int = 150_000) -> CountData:
    toks = stream.read(0, count_tokens).astype(np.int64)
    xgram_hi_words = compute_xgram_hi_words(toks, V, cover=0.80)
    pair_keys = toks[:-1] * V + toks[1:]
    all_keys, all_counts = np.unique(pair_keys, return_counts=True)

    hi_mask = all_counts >= eff_hi
    mid_mask = (all_counts >= eff_mid) & (all_counts < eff_hi)
    mature_keys = all_keys[all_counts >= eff_mid]  # already sorted!

    hi_keys, hi_counts = all_keys[hi_mask], all_counts[hi_mask]
    mid_keys, mid_counts = all_keys[mid_mask], all_counts[mid_mask]
    if len(hi_keys) > n_hi:
        order = np.argsort(-hi_counts)[:n_hi]
        hi_keys = np.sort(hi_keys[order])
    if len(mid_keys) > n_mid:
        order = np.argsort(-mid_counts)[:n_mid]
        mid_keys = np.sort(mid_keys[order])

    tri = (toks[:-2] * V + toks[1:-1]) * V + toks[2:]
    u3, c3 = np.unique(tri, return_counts=True)
    g3_keys, g3_counts = u3[c3 >= eff_g3], c3[c3 >= eff_g3]
    if len(g3_keys) > n_3g:
        order = np.argsort(-g3_counts)[:n_3g]
        g3_keys = np.sort(g3_keys[order])

    if is_main():
        print(
            f"[counts] prefix={count_tokens/1e6:.1f}M tok | unique={len(all_keys):,} | "
            f"hi(>={eff_hi})={len(hi_keys):,} | mid({eff_mid}..{eff_hi-1})={len(mid_keys):,} | "
            f"mature(>={eff_mid})={len(mature_keys):,} | 3g(>={eff_g3})={len(g3_keys):,} | "
            f"xgram_top80={len(xgram_hi_words):,}/{V:,}"
        )
    return CountData(V, hi_keys, mid_keys, g3_keys, mature_keys, all_keys, all_counts,
                     eff_hi, eff_mid, eff_g3, xgram_hi_words=xgram_hi_words)


def _lookup_pair_ids(tokens: np.ndarray, keys: np.ndarray, V: int) -> np.ndarray:
    """Strictly causal bigrams for input x=tokens[:, :-1] of shape (B, T): pos j gets (tokens[:, j-1], tokens[:, j])."""
    T = tokens.shape[1] - 1
    out = np.zeros((tokens.shape[0], T), dtype=np.int64)
    if len(keys) == 0 or T < 2:
        return out
    ids = tokens[:, :-1]
    p = ids[:, :-1] * V + ids[:, 1:]
    ix = np.clip(np.searchsorted(keys, p), 0, len(keys) - 1)
    out[:, 1:] = np.where(keys[ix] == p, ix + 1, 0)
    return out


def _lookup_pair_mask(tokens: np.ndarray, keys: np.ndarray, V: int) -> np.ndarray:
    T = tokens.shape[1] - 1
    out = np.zeros((tokens.shape[0], T), dtype=bool)
    if len(keys) == 0 or T < 2:
        return out
    ids = tokens[:, :-1]
    p = ids[:, :-1] * V + ids[:, 1:]
    ix = np.clip(np.searchsorted(keys, p), 0, len(keys) - 1)
    out[:, 1:] = (keys[ix] == p)
    return out


def _lookup_tri_ids(tokens: np.ndarray, keys: np.ndarray, V: int) -> np.ndarray:
    """Strictly causal trigrams for input x=tokens[:, :-1]: pos j gets (tokens[:, j-2], tokens[:, j-1], tokens[:, j])."""
    T = tokens.shape[1] - 1
    out = np.zeros((tokens.shape[0], T), dtype=np.int64)
    if len(keys) == 0 or T < 3:
        return out
    ids = tokens[:, :-1]
    t = (ids[:, :-2] * V + ids[:, 1:-1]) * V + ids[:, 2:]
    ix = np.clip(np.searchsorted(keys, t), 0, len(keys) - 1)
    out[:, 2:] = np.where(keys[ix] == t, ix + 1, 0)
    return out


def _lookup_pair_counts(tokens: np.ndarray, keys: np.ndarray, counts: np.ndarray, V: int) -> np.ndarray:
    T = tokens.shape[1] - 1
    out = np.zeros((tokens.shape[0], T), dtype=np.int64)
    if len(keys) == 0 or T < 2:
        return out
    ids = tokens[:, :-1]
    p = ids[:, :-1] * V + ids[:, 1:]
    ix = np.clip(np.searchsorted(keys, p), 0, len(keys) - 1)
    out[:, 1:] = np.where(keys[ix] == p, counts[ix], 0)
    return out


def make_train_ids(tokens: np.ndarray, cd: CountData, V: int, cfg: dict) -> dict:
    """Build ONLY the lookup IDs needed by the active arm (avoids multi-million-key searchsorted overhead)."""
    out = {}
    mmode = cfg["memory_mode"]
    if mmode == "tiered":
        out["hi_id"] = _lookup_pair_ids(tokens, cd.hi_keys, V)
        out["mid_id"] = _lookup_pair_ids(tokens, cd.mid_keys, V)
        if cfg["use_3g"]:
            out["g3_id"] = _lookup_tri_ids(tokens, cd.g3_keys, V)
    elif mmode == "hash" and cfg["gate_mode"] == "tier":
        out["mature_mask"] = _lookup_pair_mask(tokens, cd.mature_keys, V)
    return out


# ---------------------------------------------------------------------------
# Memory bank
# ---------------------------------------------------------------------------


class MemoryBank(nn.Module):
    """Word / xgram frequency-split / hashed-pair / tiered-pair memory bank."""

    def __init__(self, V: int, d_mem: int, mode: str = "tiered",
                 n_hi: int = 0, n_mid: int = 0, n_3g: int = 0,
                 pair_rows: int = 1_000_000, gate_mode: str = "tier",
                 mid_rank: int | None = None,
                 xgram_hi_words: np.ndarray | None = None,
                 n_w_hi: int | None = None,
                 use_hash_3g: bool = False):
        super().__init__()
        self.V = V
        self.d_mem = d_mem
        self.mode = mode
        self.gate_mode = gate_mode
        self.n_hi, self.n_mid, self.n_3g = n_hi, n_mid, n_3g
        self.pair_rows = pair_rows
        self.mid_rank = mid_rank or max(4, d_mem // 4)
        self.use_hash_3g = use_hash_3g

        self.ve_word = None
        self.ve_word_hi = self.ve_word_lo = self.ve_word_proj = None
        self.ve_pair = None
        self.ve_hi = self.ve_mid_a = self.ve_mid_p = self.ve_3g = None

        if mode == "xgram":
            if n_w_hi is not None:
                n_hi_w = max(1, min(int(n_w_hi), max(1, V - 1)))
                is_hi = np.zeros(V, dtype=bool)
                is_hi[:n_hi_w] = True
            else:
                hi_ids = np.asarray(
                    xgram_hi_words if xgram_hi_words is not None else np.arange(max(1, min(V // 20, max(1, V - 1)))),
                    dtype=np.int64,
                )
                is_hi = np.zeros(V, dtype=bool)
                is_hi[hi_ids[(hi_ids >= 0) & (hi_ids < V)]] = True
                n_hi_w = int(is_hi.sum())
            n_lo_w = max(1, V - n_hi_w)
            slots = np.zeros(V, dtype=np.int64)
            slots[is_hi] = np.arange(n_hi_w, dtype=np.int64)
            slots[~is_hi] = np.arange(V - n_hi_w, dtype=np.int64)
            self.register_buffer("xgram_is_hi", torch.from_numpy(is_hi), persistent=True)
            self.register_buffer("xgram_slot", torch.from_numpy(slots), persistent=True)

            w_rank = max(4, d_mem // 4)
            self.ve_word_hi = nn.Parameter(torch.randn(n_hi_w, d_mem) * d_mem ** -0.5)
            self.ve_word_lo = nn.Parameter(torch.randn(n_lo_w, w_rank) * w_rank ** -0.5)
            self.ve_word_proj = nn.Linear(w_rank, d_mem, bias=False)
            nn.init.normal_(self.ve_word_proj.weight, std=(w_rank * d_mem) ** -0.5)
        elif mode != "none":
            self.ve_word = nn.Parameter(torch.randn(V, d_mem) * d_mem ** -0.5)

        if mode == "hash":
            self.ve_pair = nn.Parameter(torch.randn(pair_rows, d_mem) * d_mem ** -0.5)
        elif mode == "tiered":
            self.ve_hi = nn.Parameter(torch.randn(max(1, n_hi + 1), d_mem) * d_mem ** -0.5)
            self.ve_mid_a = nn.Parameter(torch.randn(max(1, n_mid + 1), self.mid_rank) * self.mid_rank ** -0.5)
            self.ve_mid_p = nn.Parameter(torch.randn(self.mid_rank, d_mem) * (self.mid_rank * d_mem) ** -0.5)
            if n_3g > 0:
                self.ve_3g = nn.Parameter(torch.randn(n_3g + 1, d_mem) * d_mem ** -0.5)
        elif mode not in ("word", "xgram", "none"):
            raise ValueError(f"Unknown memory mode: {mode}")

        with torch.no_grad():
            for p in (self.ve_hi, self.ve_mid_a, self.ve_3g):
                if p is not None:
                    p[0].zero_()

        self.b5_logits = nn.Parameter(torch.zeros(5)) if gate_mode == "bucket5" else None

    def word_table_params(self):
        return [p for p in (self.ve_word, self.ve_word_hi, self.ve_word_lo) if p is not None]

    def table_params(self):
        ps = self.word_table_params()
        for p in (self.ve_pair, self.ve_hi, self.ve_mid_a, self.ve_mid_p, self.ve_3g):
            if p is not None:
                ps.append(p)
        return ps

    def lookup(self, ids, hi_id=None, mid_id=None, g3_id=None, mature_mask=None, bucket_id=None):
        if self.mode == "xgram":
            hi_mask = self.xgram_is_hi[ids]
            slot = self.xgram_slot[ids]
            w_hi = self.ve_word_hi[torch.where(hi_mask, slot, 0)]
            w_lo = self.ve_word_proj(self.ve_word_lo[torch.where(hi_mask, 0, slot)])
            return torch.where(hi_mask.unsqueeze(-1), w_hi.to(w_lo.dtype), w_lo), None

        w = self.ve_word[ids] if self.ve_word is not None else None
        if self.mode in ("word", "none"):
            return w, None
        if self.mode == "hash":
            # Compute fast hash_id directly on GPU
            p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
            hash_id = (p1 * 1000003 + ids) % self.pair_rows
            p = self.ve_pair[hash_id]
            if self.use_hash_3g:
                p2 = F.pad(ids, (2, 0), value=self.V)[:, :-2]
                h3 = (p2 * 998244353 + p1 * 1000003 + ids + 17) % self.pair_rows
                p = p + self.ve_pair[h3]
        else:
            p = self.ve_hi[hi_id] * (hi_id > 0).unsqueeze(-1).to(w.dtype)
            if self.n_mid > 0 and mid_id is not None:
                p = p + (self.ve_mid_a[mid_id] @ self.ve_mid_p) * (mid_id > 0).unsqueeze(-1).to(w.dtype)
            if self.ve_3g is not None and g3_id is not None:
                p = p + self.ve_3g[g3_id] * (g3_id > 0).unsqueeze(-1).to(w.dtype)

        if self.gate_mode == "tier" and self.mode == "hash":
            p = p * mature_mask.unsqueeze(-1).to(p.dtype)
        elif self.gate_mode == "bucket5" and bucket_id is not None:
            p = p * torch.sigmoid(self.b5_logits[bucket_id]).unsqueeze(-1).to(p.dtype)
        return w, p


class EngramModule(nn.Module):
    """Multi-head hashed 2/3-gram Engram conditional memory (key/value proj, context gate, causal conv)."""

    def __init__(self, V: int, d: int, d_mem: int, pair_rows: int = 1_000_000, n_heads: int = 4):
        super().__init__()
        self.V = V
        self.K = min(n_heads, max(1, d_mem // 4))
        self.ng_n = max(1, pair_rows // 2)
        self.ng_dim = max(4, d_mem // self.K)
        concat_dim = 2 * self.K * self.ng_dim

        self.ng2 = nn.Parameter(torch.randn(self.K, self.ng_n, self.ng_dim) * d ** -0.5)
        self.ng3 = nn.Parameter(torch.randn(self.K, self.ng_n, self.ng_dim) * d ** -0.5)
        self.ngk = nn.Linear(concat_dim, d, bias=False)
        self.ngv = nn.Linear(concat_dim, d, bias=False)
        nn.init.normal_(self.ngk.weight, std=(2 * d) ** -0.5)
        nn.init.normal_(self.ngv.weight, std=(2 * d) ** -0.5)
        self.ngc = nn.Parameter(torch.zeros(d, 1, 4))

        self.register_buffer("hk", torch.arange(self.K, dtype=torch.long), persistent=False)
        m1 = [1000003, 998244353, 754974721, 167772161][:self.K]
        m2 = [7919, 104729, 1299709, 15485863][:self.K]
        self.register_buffer("m1", torch.tensor(m1, dtype=torch.long), persistent=False)
        self.register_buffer("m2", torch.tensor(m2, dtype=torch.long), persistent=False)

    def table_params(self):
        return [self.ng2, self.ng3]

    def forward(self, x: torch.Tensor, ids: torch.Tensor, escale: float) -> torch.Tensor:
        p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
        p2 = F.pad(ids, (2, 0), value=self.V)[:, :-2]
        h2 = (p1.unsqueeze(-1) * self.m1 + ids.unsqueeze(-1) * self.m2 + self.hk) % self.ng_n
        h3 = (p2.unsqueeze(-1) * (self.m2 * 31) + p1.unsqueeze(-1) * self.m1 + ids.unsqueeze(-1) * self.m2 + self.hk * 7) % self.ng_n
        e = torch.cat([self.ng2[self.hk, h2].flatten(-2), self.ng3[self.hk, h3].flatten(-2)], dim=-1)
        kk, vv = self.ngk(e), self.ngv(e)
        gate = torch.sigmoid(
            (F.rms_norm(x.float(), (x.shape[-1],)) * F.rms_norm(kk.float(), (kk.shape[-1],))).sum(-1, keepdim=True)
            / (x.shape[-1] ** 0.5)
        )
        v = (gate * vv.float()).to(x.dtype)
        vn = F.rms_norm(v, (v.shape[-1],)).transpose(1, 2)
        c = F.conv1d(F.pad(vn, (9, 0)), self.ngc.to(vn.dtype), dilation=3, groups=v.shape[-1]).transpose(1, 2)
        return x + (F.silu(c) + v) * escale


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, d: int, H: int, rope: Rotary, *,
                 memory: MemoryBank | None = None, shared: bool = False,
                 shared_has_pair: bool = True,
                 d_mem: int = 0, use_ctx_gate: bool = False, inject_res: bool = False):
        super().__init__()
        self.d, self.H, self.dk = d, H, d // H
        self.memory = memory
        self.shared = shared
        self.use_ctx_gate = use_ctx_gate
        self.inject_res = inject_res
        self.rope = rope

        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qn, self.kn = nn.RMSNorm(d // H), nn.RMSNorm(d // H)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        f = max(8, int(8 * d / 3) // 8 * 8)
        self.gate, self.up, self.down = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

        has_mem = (memory is not None) or shared
        has_pair = ((memory is not None and memory.mode not in ("word", "xgram", "none"))
                    or (shared and shared_has_pair))
        self.lam_w = nn.Parameter(torch.tensor(1.0)) if has_mem else None
        self.lam_p = nn.Parameter(torch.tensor(1.0)) if has_pair else None

        self.proj_w_v = self.proj_p_v = None
        self.gate_v = None
        self.proj_res = self.gate_res = None

        need_proj = (memory is None and shared) or (memory is not None and memory.d_mem != d)
        mem_in_dim = d_mem if shared else (memory.d_mem if memory is not None else d)
        if has_mem and need_proj:
            self.proj_w_v = nn.Linear(mem_in_dim, d, bias=False)
            if has_pair:
                self.proj_p_v = nn.Linear(mem_in_dim, d, bias=False)
        if has_mem and use_ctx_gate:
            self.gate_v = nn.Linear(d, d, bias=True)
            nn.init.zeros_(self.gate_v.weight)
            nn.init.zeros_(self.gate_v.bias)
        if has_mem and inject_res:
            self.proj_res = nn.Linear(mem_in_dim, d, bias=False)
            self.gate_res = nn.Linear(d, d, bias=True)
            nn.init.zeros_(self.gate_res.weight)
            nn.init.constant_(self.gate_res.bias, -1.0)

    def forward(self, x, ids, mem_ids, shared_vecs=None):
        b, t, d = x.shape
        h = self.n1(x)
        sp = lambda z: z.view(b, t, self.H, self.dk).transpose(1, 2)
        v = self.v(h)
        mem_for_res = None

        if self.shared:
            w_raw, p_raw = shared_vecs
        elif self.memory is not None:
            w_raw, p_raw = self.memory.lookup(
                ids,
                hi_id=mem_ids.get("hi_id"),
                mid_id=mem_ids.get("mid_id"),
                g3_id=mem_ids.get("g3_id"),
                mature_mask=mem_ids.get("mature_mask"),
                bucket_id=mem_ids.get("bucket_id"),
            )
        else:
            w_raw, p_raw = None, None

        if w_raw is not None:
            wv = self.proj_w_v(w_raw) if self.proj_w_v is not None else w_raw
            pv = (self.proj_p_v(p_raw) if self.proj_p_v is not None else p_raw) if p_raw is not None else None
            ve = self.lam_w * wv + (self.lam_p * pv if pv is not None else 0.0)
            mem_for_res = w_raw if p_raw is None else (w_raw + p_raw)
            if self.gate_v is not None:
                v = v + (2.0 * torch.sigmoid(self.gate_v(h))) * ve
            else:
                v = v + ve

        q, k = self.rope(self.qn(sp(self.q(h)))), self.rope(self.kn(sp(self.k(h))))
        y = F.scaled_dot_product_attention(q, k, sp(v), is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(b, t, d))

        if mem_for_res is not None and self.inject_res:
            g = torch.sigmoid(self.gate_res(self.n2(x)))
            x = x + g * self.proj_res(mem_for_res)

        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Model(nn.Module):
    def __init__(self, V: int, d: int, layers: int, heads: int, seq_len: int,
                 arm: str = "plain", ve_layers: tuple[int, ...] = (),
                 d_mem: int = 256, memory_mode: str = "tiered",
                 n_hi: int = 0, n_mid: int = 0, n_3g: int = 0,
                 pair_rows: int = 1_000_000, gate_mode: str = "tier",
                 use_ctx_gate: bool = False, inject_res: bool = False,
                 pre_head_mem: bool = False, share_bank_across_layers: bool = False,
                 use_engram: bool = False,
                 xgram_hi_words: np.ndarray | None = None,
                 n_w_hi: int | None = None,
                 use_hash_3g: bool = False):
        super().__init__()
        self.V, self.d = V, d
        self.arm = arm
        self.escale = 4.0
        self.ve_layers = set(ve_layers)
        self.shared = share_bank_across_layers
        self.pre_head_mem = pre_head_mem

        rope = Rotary(d // heads, seq_len)
        self.emb = nn.Embedding(V, d)
        self.head = nn.Linear(d, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

        self.engram = EngramModule(V, d, d_mem, pair_rows) if use_engram else None
        self.bank = None
        if self.shared:
            self.bank = MemoryBank(V, d_mem, memory_mode, n_hi, n_mid, n_3g, pair_rows, gate_mode,
                                   xgram_hi_words=xgram_hi_words, n_w_hi=n_w_hi, use_hash_3g=use_hash_3g)
        self.blocks = nn.ModuleList()
        for i in range(layers):
            active = i in self.ve_layers
            local = None
            if active and not self.shared:
                local = MemoryBank(V, d_mem, memory_mode, n_hi, n_mid, n_3g, pair_rows, gate_mode,
                                   xgram_hi_words=xgram_hi_words, n_w_hi=n_w_hi, use_hash_3g=use_hash_3g)
            self.blocks.append(Block(
                d, heads, rope,
                memory=local,
                shared=self.shared and active,
                shared_has_pair=(self.bank is not None and self.bank.mode not in ("word", "xgram", "none")),
                d_mem=d_mem,
                use_ctx_gate=use_ctx_gate and active,
                inject_res=inject_res and active,
            ))
        self.norm = nn.RMSNorm(d)
        if self.pre_head_mem:
            assert self.shared
            self.head_proj = nn.Linear(d_mem, d, bias=False)
            self.head_gate = nn.Linear(d, d, bias=True)
            nn.init.zeros_(self.head_gate.weight)
            nn.init.constant_(self.head_gate.bias, -1.0)

    def forward(self, ids, targets=None, hi_id=None, mid_id=None, g3_id=None, mature_mask=None, bucket_id=None):
        x = self.emb(ids) * self.escale
        if self.engram is not None:
            x = self.engram(x, ids, self.escale)
        shared_vecs = None
        if self.shared:
            shared_vecs = self.bank.lookup(
                ids, hi_id=hi_id, mid_id=mid_id, g3_id=g3_id,
                mature_mask=mature_mask, bucket_id=bucket_id,
            )
        mem_ids = dict(hi_id=hi_id, mid_id=mid_id, g3_id=g3_id, mature_mask=mature_mask, bucket_id=bucket_id)
        for blk in self.blocks:
            x = blk(x, ids, mem_ids, shared_vecs=shared_vecs)
        h = self.norm(x)
        if self.pre_head_mem:
            w_raw, p_raw = shared_vecs
            mem = w_raw if p_raw is None else (w_raw + p_raw)
            h = h + torch.sigmoid(self.head_gate(h)) * self.head_proj(mem)

        if targets is None:
            return self.head(h)

        # Chunked cross entropy avoids multi-GB logits materialization on V=50,257
        h_flat = h.reshape(-1, h.shape[-1])
        t_flat = targets.reshape(-1)
        loss_sum = torch.zeros((), device=h.device, dtype=torch.float32)
        chunk = 2048
        for s0 in range(0, h_flat.shape[0], chunk):
            lg = self.head(h_flat[s0:s0 + chunk])
            loss_sum = loss_sum + F.cross_entropy(lg.float(), t_flat[s0:s0 + chunk], reduction="sum")
        return loss_sum / t_flat.numel()


# ---------------------------------------------------------------------------
# Arm presets
# ---------------------------------------------------------------------------


def arm_config(arm: str, layers: int, d: int, d_mem: int):
    last2 = tuple(range(max(0, layers - 2), layers))
    all_layers = tuple(range(layers))
    if arm in ("plain", "plain_eq_weights"):
        return dict(ve_layers=(), memory_mode="word", shared=False, d_mem=d,
                    gate_mode="none", ctx_gate=False, inject_res=False, prehead=False, use_3g=False, use_engram=False)
    if arm == "word":
        # Local full-d word value embedding (matches Bank of Values without projection bottleneck)
        return dict(ve_layers=last2, memory_mode="word", shared=False, d_mem=d,
                    gate_mode="none", ctx_gate=False, inject_res=False, prehead=False, use_3g=False, use_engram=False)
    if arm == "xgram":
        # Frequency-split word table: full-d rows for top 80% token cover, d/4 rows + shared projection for tail
        return dict(ve_layers=last2, memory_mode="xgram", shared=False, d_mem=d,
                    gate_mode="none", ctx_gate=False, inject_res=False, prehead=False, use_3g=False, use_engram=False)
    if arm == "engram":
        # Multi-head hashed 2/3-gram Engram conditional memory with K/V projections, context gate & causal conv
        return dict(ve_layers=(), memory_mode="none", shared=False, d_mem=d_mem,
                    gate_mode="none", ctx_gate=False, inject_res=False, prehead=False, use_3g=False, use_engram=True)
    if arm == "hard_stack":
        # Use d_mem with a projection: a full-d 1M-row pair table would be ~1.5B
        # parameters in two layers and would not fit the intended table budget.
        return dict(ve_layers=last2, memory_mode="hash", shared=False, d_mem=d_mem,
                    gate_mode="none", ctx_gate=False, inject_res=False, prehead=False, use_3g=False, use_engram=False)
    if arm == "mp_ve":
        return dict(ve_layers=last2, memory_mode="tiered", shared=True, d_mem=d_mem,
                    gate_mode="tier", ctx_gate=False, inject_res=False, prehead=False, use_3g=False, use_engram=False)
    if arm == "u1":
        return dict(ve_layers=all_layers, memory_mode="tiered", shared=True, d_mem=d_mem,
                    gate_mode="tier", ctx_gate=False, inject_res=False, prehead=False, use_3g=False, use_engram=False)
    if arm == "u123":
        return dict(ve_layers=all_layers, memory_mode="tiered", shared=True, d_mem=d_mem,
                    gate_mode="tier", ctx_gate=True, inject_res=True, prehead=True, use_3g=True, use_engram=False)
    if arm == "u123_no3g":
        return dict(ve_layers=all_layers, memory_mode="tiered", shared=True, d_mem=d_mem,
                    gate_mode="tier", ctx_gate=True, inject_res=True, prehead=True, use_3g=False, use_engram=False)
    if arm == "ctrl_word":
        return dict(ve_layers=all_layers, memory_mode="word", shared=True, d_mem=d_mem,
                    gate_mode="none", ctx_gate=True, inject_res=True, prehead=True, use_3g=False, use_engram=False)
    if arm == "ctrl_hs":
        return dict(ve_layers=all_layers, memory_mode="hash", shared=True, d_mem=d_mem,
                    gate_mode="none", ctx_gate=True, inject_res=True, prehead=True, use_3g=False, use_engram=False)
    if arm == "ctrl_hs_pruned":
        return dict(ve_layers=all_layers, memory_mode="hash", shared=True, d_mem=d_mem,
                    gate_mode="tier", ctx_gate=True, inject_res=True, prehead=True, use_3g=False, use_engram=False)
    if arm == "uniform_mgram":
        # Value-side Over-Tokenized multi-gram control: uniform unpruned, untiered 2g+3g hash pool
        return dict(ve_layers=all_layers, memory_mode="hash", shared=True, d_mem=d_mem,
                    gate_mode="none", ctx_gate=True, inject_res=True, prehead=True,
                    use_3g=False, use_hash_3g=True, use_engram=False)
    raise ValueError(f"Unknown arm: {arm}")


# ---------------------------------------------------------------------------
# Optimizer groups
# ---------------------------------------------------------------------------


def build_optimizers(model: Model, args):
    tables = []
    if model.bank is not None:
        tables.extend(model.bank.table_params())
    if model.engram is not None:
        tables.extend(model.engram.table_params())
    for blk in model.blocks:
        if blk.memory is not None:
            tables.extend(blk.memory.table_params())
    seen = set()
    tables = [p for p in tables if not (id(p) in seen or seen.add(id(p)))]
    table_ids = {id(p) for p in tables}

    mats = []
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in table_ids:
            continue
        if p.ndim == 2 and (n.startswith("blocks") or n.startswith("head_proj") or n.startswith("head_gate") or n.startswith("engram")):
            mats.append(p)
    mat_ids = {id(p) for p in mats}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in table_ids and id(p) not in mat_ids]

    opts = [
        Muon(mats, lr=args.lr_muon),
        torch.optim.AdamW([dict(params=rest, lr=args.lr_adam, weight_decay=0.0)], betas=(0.9, 0.95)),
    ]
    word_tabs = model.bank.word_table_params() if model.bank is not None else []
    for blk in model.blocks:
        if blk.memory is not None:
            word_tabs.extend(blk.memory.word_table_params())
    word_ids = {id(p) for p in word_tabs}
    pair_tabs = [p for p in tables if id(p) not in word_ids]
    if word_tabs:
        opts.append(torch.optim.Adam([dict(params=word_tabs, lr=args.lr_word)], betas=(0.9, 0.95)))
    if pair_tabs:
        opts.append(torch.optim.Adam([dict(params=pair_tabs, lr=args.lr_pair)], betas=(0.9, 0.95)))
    return opts, tables


def lr_scale(step: int, total_steps: int, warmup_frac: float, min_frac: float = 0.1) -> float:
    warmup = max(1, int(warmup_frac * total_steps))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Evaluation & Latency Benchmark
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: Model, val_tokens: np.ndarray, cd: CountData, cfg: dict, args, device) -> dict:
    model.eval()
    V = args.V
    losses, buckets = [], []
    bs = args.eval_batch
    for i in range(0, len(val_tokens), bs):
        toks = val_tokens[i:i + bs]
        t_md = make_train_ids(toks, cd, V, cfg)
        counts = _lookup_pair_counts(toks, cd.all_keys, cd.all_counts, V)
        # Counts come from a prefix; rescale to full-corpus estimates before bucketing.
        scale = (args.count_tokens / max(1.0, float(args.train_tokens))) if args.auto_scale_thresh else 1.0
        b_id = np.digitize(counts.astype(np.float64) / max(1e-9, scale),
                           [1, 10, 100, 1000]).astype(np.int64)
        ids = torch.from_numpy(toks[:, :-1]).to(device)
        tgt = torch.from_numpy(toks[:, 1:]).to(device)
        mem = {k: torch.from_numpy(v).to(device) for k, v in t_md.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(ids, **mem)
            loss = F.cross_entropy(logits.float().reshape(-1, V), tgt.reshape(-1), reduction="none")
        losses.append(loss.cpu())
        buckets.append(torch.from_numpy(b_id).reshape(-1))
    loss = torch.cat(losses)
    bucket = torch.cat(buckets)
    out = {"val": round(float(loss.mean()), 5)}
    for b in range(5):
        sel = bucket == b
        if int(sel.sum()) > 0:
            out[f"pc_{b}"] = round(float(loss[sel].mean()), 5)
    model.train()
    return out


@torch.no_grad()
def bench_latency(model: Model, cfg: dict, args, device) -> dict:
    model.eval()
    T = args.seq_len
    def mk_inputs(B, L):
        ids = torch.randint(0, args.V, (B, L), device=device, dtype=torch.long)
        mem = {}
        if cfg["memory_mode"] == "tiered":
            mem["hi_id"] = torch.zeros((B, L), device=device, dtype=torch.long)
            mem["mid_id"] = torch.zeros((B, L), device=device, dtype=torch.long)
            if cfg["use_3g"]:
                mem["g3_id"] = torch.zeros((B, L), device=device, dtype=torch.long)
        elif cfg["memory_mode"] == "hash" and cfg["gate_mode"] == "tier":
            mem["mature_mask"] = torch.ones((B, L), device=device, dtype=torch.bool)
        return ids, mem

    ids1, mem1 = mk_inputs(1, T)
    ids16, mem16 = mk_inputs(16, T)
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        for _ in range(10):
            _ = model(ids1, **mem1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_b1 = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(ids1, **mem1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_b1.append((time.perf_counter() - t0) * 1000.0)

        for _ in range(5):
            _ = model(ids16, **mem16)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            _ = model(ids16, **mem16)
        if device.type == "cuda":
            torch.cuda.synchronize()
        el16 = time.perf_counter() - t0

    model.train()
    return {
        "b1_seq_ms_mean": round(float(np.mean(t_b1)), 3),
        "b1_seq_ms_std": round(float(np.std(t_b1, ddof=1)), 3),
        "b16_infer_tok_s": int((20 * 16 * T) / max(1e-6, el16)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_stream(args):
    if args.smoke:
        return RandomStream(args.V, total=10_000_000)
    paths = sorted(glob.glob(os.path.join(args.data_dir, "fineweb_train_*.bin")))
    if not paths:
        raise FileNotFoundError(f"No fineweb_train_*.bin in {args.data_dir}; run download_data.py")
    stream = ShardStream(paths)
    if is_main():
        print(f"[data] {len(paths)} train shards, {stream.total/1e9:.2f}B tokens")
    return stream


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="u123",
                   choices=["plain", "plain_eq_weights", "word", "xgram", "engram",
                            "hard_stack", "mp_ve", "u1", "u123", "u123_no3g",
                            "ctrl_word", "ctrl_hs", "ctrl_hs_pruned", "uniform_mgram"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--val-shard", default=None)
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--V", type=int, default=50257)
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--eq-width", type=int, default=1152, help="Width when --arm plain_eq_weights")
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--eq-heads", type=int, default=16, help="Heads when --arm plain_eq_weights")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--train-tokens", type=float, default=2.5e9)
    p.add_argument("--steps", type=int, default=-1)
    p.add_argument("--count-tokens", type=int, default=-1, help="-1 = auto 10%% of train_tokens (capped at stream)")
    p.add_argument("--auto-scale-thresh", type=int, default=1, help="Scale thresholds proportionally by count_tokens/train_tokens")
    p.add_argument("--d-mem", type=int, default=256)
    p.add_argument("--pair-rows", type=int, default=1_000_000)
    p.add_argument("--n-hi", type=int, default=500_000)
    p.add_argument("--n-mid", type=int, default=1_500_000)
    p.add_argument("--n-3g", type=int, default=150_000)
    p.add_argument("--hi-thresh", type=int, default=100)
    p.add_argument("--mid-thresh", type=int, default=10)
    p.add_argument("--g3-thresh", type=int, default=30)
    p.add_argument("--lr-muon", type=float, default=0.02)
    p.add_argument("--lr-adam", type=float, default=6e-4)
    p.add_argument("--lr-word", type=float, default=0.03)
    p.add_argument("--lr-pair", type=float, default=0.02)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-tokens", default="50000000,100000000,250000000,500000000,1000000000,2500000000")
    p.add_argument("--val-seqs", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--bench-latency", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--distributed", type=int, default=0)
    p.add_argument("--compile", type=int, default=0)
    p.add_argument("--save-optimizer", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.arm == "plain_eq_weights":
        args.d = args.eq_width
        args.heads = args.eq_heads

    if args.smoke:
        args.V = 1024
        args.d = 64 if args.arm != "plain_eq_weights" else 96
        args.layers = 4
        args.heads = 4
        args.seq_len = 64
        args.micro_batch = 4
        args.grad_accum = 1
        args.train_tokens = 1024
        args.count_tokens = 200_000
        args.auto_scale_thresh = 0
        args.d_mem = 32
        args.n_hi = 200
        args.n_mid = 400
        args.n_3g = 100
        args.pair_rows = 1000
        args.eval_batch = 2
        args.val_seqs = 8
        args.hi_thresh = 2
        args.mid_thresh = 1
        args.g3_thresh = 2
        args.steps = 3
        args.log_interval = 1
        args.eval_tokens = "256,512,768"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    world = get_world()
    rank = get_rank()
    os.makedirs(args.out, exist_ok=True)
    if is_main():
        with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    stream = make_stream(args)
    if args.count_tokens <= 0:
        args.count_tokens = min(int(round(0.10 * args.train_tokens)), stream.total - 1024)
    scale_ratio = args.count_tokens / max(1.0, float(args.train_tokens))
    if args.auto_scale_thresh and scale_ratio < 0.99:
        eff_hi = max(2, int(round(args.hi_thresh * scale_ratio)))
        eff_mid = max(1, int(round(args.mid_thresh * scale_ratio)))
        eff_g3 = max(2, int(round(args.g3_thresh * scale_ratio)))
    else:
        eff_hi, eff_mid, eff_g3 = args.hi_thresh, args.mid_thresh, args.g3_thresh

    cd = build_counts(stream, args.V, args.count_tokens, eff_hi, eff_mid, eff_g3,
                      args.n_hi, args.n_mid, args.n_3g)
    args.eff_hi, args.eff_mid, args.eff_g3 = eff_hi, eff_mid, eff_g3
    if is_main():
        with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    cfg = arm_config(args.arm, args.layers, args.d, args.d_mem)
    actual_n_hi = len(cd.hi_keys)
    actual_n_mid = len(cd.mid_keys)
    actual_n_3g = len(cd.g3_keys) if cfg["use_3g"] else 0

    model = Model(
        args.V, args.d, args.layers, args.heads, args.seq_len,
        arm=args.arm, ve_layers=cfg["ve_layers"], d_mem=cfg["d_mem"],
        memory_mode=cfg["memory_mode"], n_hi=actual_n_hi, n_mid=actual_n_mid,
        n_3g=actual_n_3g, pair_rows=args.pair_rows,
        gate_mode=cfg["gate_mode"], use_ctx_gate=cfg["ctx_gate"],
        inject_res=cfg["inject_res"], pre_head_mem=cfg["prehead"],
        share_bank_across_layers=cfg["shared"],
        use_engram=cfg.get("use_engram", False),
        xgram_hi_words=cd.xgram_hi_words,
        use_hash_3g=cfg.get("use_hash_3g", False),
    ).to(device)

    nparam = sum(p.numel() for p in model.parameters())
    tables_n = 0
    if model.bank is not None:
        tables_n += sum(p.numel() for p in model.bank.table_params())
    if model.engram is not None:
        tables_n += sum(p.numel() for p in model.engram.table_params())
    for blk in model.blocks:
        if blk.memory is not None:
            tables_n += sum(p.numel() for p in blk.memory.table_params())
    body_n = nparam - tables_n
    if is_main():
        print(f"[model] arm={args.arm} total={nparam/1e6:.2f}M table={tables_n/1e6:.2f}M body={body_n/1e6:.2f}M")

    opts, _ = build_optimizers(model, args)
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]

    tokens_per_step = world * args.micro_batch * args.grad_accum * args.seq_len
    total_steps = args.steps if args.steps > 0 else math.ceil(args.train_tokens / tokens_per_step)
    milestones = sorted(float(x) for x in args.eval_tokens.split(",") if x)

    val_tokens = None
    if not args.smoke:
        val_path = args.val_shard or os.path.join(args.data_dir, "fineweb_val_000000.bin")
        val_stream = ShardStream([val_path])
        starts = np.linspace(0, val_stream.total - args.seq_len - 2, args.val_seqs).astype(np.int64)
        val_tokens = np.stack([val_stream.read(int(s), args.seq_len + 1) for s in starts]).astype(np.int64)

    if is_main():
        print(f"[train] steps={total_steps} tokens/step={tokens_per_step:,} total={total_steps*tokens_per_step/1e9:.3f}B world={world}")

    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
    if args.compile and device.type == "cuda":
        model = torch.compile(model)

    next_eval = 0
    t0 = time.time()
    tokens_seen = 0
    last_eval = {}
    log_path = os.path.join(args.out, "log.jsonl")

    for step in range(total_steps):
        scale = lr_scale(step, total_steps, args.warmup_frac)
        for o, bl in zip(opts, base_lrs):
            for g, lr0 in zip(o.param_groups, bl):
                g["lr"] = lr0 * scale

        step_start = step * tokens_per_step
        rank_start = step_start + rank * (args.micro_batch * args.grad_accum * args.seq_len)
        total_loss = 0.0
        for micro in range(args.grad_accum):
            start = rank_start + micro * args.micro_batch * args.seq_len
            toks = stream.get_batch(start, args.micro_batch, args.seq_len)
            t_md = make_train_ids(toks, cd, args.V, cfg)
            ids = torch.from_numpy(toks[:, :-1]).to(device)
            tgt = torch.from_numpy(toks[:, 1:]).to(device)
            mem = {k: torch.from_numpy(v).to(device) for k, v in t_md.items()}
            sync_ctx = (model.no_sync()
                        if (args.distributed and micro < args.grad_accum - 1)
                        else contextlib.nullcontext())
            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    loss = model(ids, targets=tgt, **mem)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}, micro {micro}")
                (loss / args.grad_accum).backward()
            total_loss += float(loss.detach()) / args.grad_accum

        if args.distributed:
            t_tensor = torch.tensor(total_loss, device=device)
            dist.all_reduce(t_tensor, op=dist.ReduceOp.AVG)
            total_loss = float(t_tensor.item())

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip)
        for o in opts:
            o.step()
        model.zero_grad(set_to_none=True)

        tokens_seen = (step + 1) * tokens_per_step
        if is_main() and (step % args.log_interval == 0 or step == total_steps - 1):
            dt = time.time() - t0
            tok_s = tokens_seen / max(1e-9, dt)
            mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            rec = dict(step=step + 1, tokens=tokens_seen, loss=round(total_loss, 5),
                       lr=base_lrs[0][0] * scale, tok_s=round(tok_s, 1), mem_gb=round(mem_gb, 2))
            print(f"[step {step+1}/{total_steps}] loss={total_loss:.4f} lr={rec['lr']:.2e} tok/s={tok_s:,.0f} mem={mem_gb:.1f}GB")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        trigger_eval = (next_eval < len(milestones) and tokens_seen >= milestones[next_eval])
        force_final = (step == total_steps - 1 and not last_eval)
        if trigger_eval or force_final:
            while next_eval < len(milestones) and tokens_seen >= milestones[next_eval]:
                next_eval += 1
            if is_main():
                raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
                raw_m = raw_m.module if hasattr(raw_m, "module") else raw_m
                rec = dict(step=step + 1, tokens=tokens_seen, stage="eval")
                if val_tokens is not None:
                    rec.update(evaluate(raw_m, val_tokens, cd, cfg, args, device))
                last_eval = rec
                print(f"[eval] tokens={tokens_seen/1e9:.3f}B {rec}")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                if not args.smoke:
                    ckpt = {"model": raw_m.state_dict(), "config": vars(args), "step": step + 1, "tokens": tokens_seen}
                    if args.save_optimizer:
                        ckpt["opts"] = [o.state_dict() for o in opts]
                    torch.save(ckpt, os.path.join(args.out, f"ckpt_{int(tokens_seen/1e6)}M.pt"))
            if args.distributed:
                dist.barrier()

    if is_main():
        raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_m = raw_m.module if hasattr(raw_m, "module") else raw_m
        lat = bench_latency(raw_m, cfg, args, device) if args.bench_latency else {}
        summary = dict(
            arm=args.arm, seed=args.seed, params=nparam, body_params=body_n, table_params=tables_n,
            tokens_seen=tokens_seen, elapsed_s=round(time.time() - t0, 1),
            train_tok_s=round(tokens_seen / max(1e-9, time.time() - t0), 1),
            peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0,
            final_eval=last_eval, latency=lat,
        )
        with open(os.path.join(args.out, "metrics_final.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if not args.smoke:
            ckpt = {"model": raw_m.state_dict(), "config": vars(args), "step": total_steps, "tokens": tokens_seen}
            if args.save_optimizer:
                ckpt["opts"] = [o.state_dict() for o in opts]
            torch.save(ckpt, os.path.join(args.out, "ckpt_final.pt"))
        print(f"[done] summary saved to {os.path.join(args.out, 'metrics_final.json')}")
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
