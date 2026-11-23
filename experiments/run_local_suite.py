"""Complete Local Multi-Seed Experimental Suite for Paper 06 / MP-VE (RTX 4070 GPU).

Runs 5 comprehensive 3-seed experimental blocks and saves incrementally to local_suite_results.json:
  Block A: Orthogonality (Engram Full x Maturity Gate: mul / add / bucket5 / word+engram_mat), 3 seeds.
  Block B: (M, T) Collision-Variance Phase Diagram (V=1500, 6 table sizes M x 4 token budgets T), 3 seeds.
  Block C: Width Scaling Check of c* ~ d (d in {8, 12, 16, 32, 64, 128}, V=6000), 3 seeds.
  Block D: Component Ablation + Equal-Weights Plain Baseline + Last-2 vs All-4 + Dense vs Sparse-Row Adam + Latency (L=4, w=32, V=6000), 3 seeds.
  Block E: 3-Seed Held-Out Vocabulary Crossover Verification (V=1500, V=3000, V=6000, 150 & 300 steps).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
EOT = 50256


def rope(x):
    t, d = x.shape[-2], x.shape[-1]
    f = 1.0 / 10000 ** (torch.arange(0, d, 2, device=x.device).float() / d)
    ang = torch.outer(torch.arange(t, device=x.device).float(), f)
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


def newton_schulz(G, steps=5):
    A, B, C = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tr_ = X.shape[0] > X.shape[1]
    if tr_:
        X = X.T
    for _ in range(steps):
        M = X @ X.T
        X = A * X + (B * M + C * M @ M) @ X
    return X.T if tr_ else X


class Muon(torch.optim.Optimizer):
    def __init__(self, ps, lr):
        super().__init__(ps, dict(lr=lr))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for w in g["params"]:
                if w.grad is None:
                    continue
                m = self.state[w].setdefault("m", torch.zeros_like(w))
                m.mul_(0.95).add_(w.grad)
                w.add_(newton_schulz(w.grad + 0.95 * m), alpha=-g["lr"] * max(1, w.shape[0] / w.shape[1]) ** 0.5)


class SparseRowAdam(torch.optim.Optimizer):
    """Exact per-row Sparse Adam: updates m, v, and per-row step count ONLY for active rows (grad != 0)."""
    def __init__(self, param_groups, betas=(0.9, 0.95), eps=1e-8):
        super().__init__(param_groups, dict(betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                    st["row_t"] = torch.zeros(p.shape[0], device=p.device, dtype=torch.int64) if p.ndim == 2 else torch.zeros((), device=p.device, dtype=torch.int64)
                if p.ndim == 2:
                    active = (g.abs().sum(dim=1) > 0)
                    if not active.any():
                        continue
                    idx = active.nonzero(as_tuple=False).squeeze(-1)
                    st["row_t"][idx] += 1
                    rt = st["row_t"][idx].float().unsqueeze(-1)
                    gi = g[idx]
                    mi = st["m"][idx].mul_(b1).add_(gi, alpha=1.0 - b1)
                    vi = st["v"][idx].mul_(b2).addcmul_(gi, gi, value=1.0 - b2)
                    st["m"][idx] = mi
                    st["v"][idx] = vi
                    m_hat = mi / (1.0 - b1 ** rt)
                    v_hat = vi / (1.0 - b2 ** rt)
                    p.index_add_(0, idx, (m_hat / (v_hat.sqrt() + eps)) * (-lr))
                else:
                    st["row_t"] += 1
                    rt = float(st["row_t"])
                    st["m"].mul_(b1).add_(g, alpha=1.0 - b1)
                    st["v"].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    p.add_((st["m"] / (1.0 - b1 ** rt)) / ((st["v"] / (1.0 - b2 ** rt)).sqrt() + eps), alpha=-lr)


_DATA_CACHE = {}


def load_data(V_arg, steps, batch, L=64, nval=1000):
    key = (V_arg, steps, batch, L, nval)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    counts = np.load(os.path.join(DATA, "counts.npy")).astype(np.float64)
    keep = np.unique(np.concatenate([np.argsort(counts)[-(V_arg - 2):], [EOT]]))
    V = len(keep) + 1
    UNK = V - 1
    look = np.full(65536, UNK, np.int64)
    look[keep] = np.arange(len(keep))
    freq = np.append(counts[keep], counts.sum() - counts[keep].sum())
    order = np.argsort(-freq[:-1], kind="stable")
    cover = np.cumsum(freq[order]) / freq[:-1].sum()
    cov_of = np.empty(V)
    cov_of[order] = cover - freq[order] / freq[:-1].sum()
    cov_of[UNK] = 2.0
    bucket = np.full(V, 3)
    for i, (_, lo, _) in enumerate([("common", 0.0, 0.5), ("middle", 0.5, 0.8), ("rare", 0.8, 1.01)]):
        bucket[cov_of >= lo] = i
    bucket[UNK] = 3

    need = steps * batch * L
    tr = np.memmap(os.path.join(DATA, "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
    rows = look[np.asarray(tr[:need]).astype(np.int64).reshape(-1, L)]
    va = np.memmap(os.path.join(DATA, "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)
    val = look[np.stack([np.asarray(va[x:x + L]) for x in np.linspace(0, len(va) - L - 1, nval).astype(np.int64)]).astype(np.int64)]

    tp = rows[:, :-1].reshape(-1).astype(np.int64) * (V + 1) + rows[:, 1:].reshape(-1)
    u_pairs, c_pairs = np.unique(tp, return_counts=True)

    def get_pair_counts(arr):
        p = arr[:, :-1].astype(np.int64) * (V + 1) + arr[:, 1:].astype(np.int64)
        ix = np.clip(np.searchsorted(u_pairs, p), 0, len(u_pairs) - 1)
        cnt = np.zeros_like(arr, dtype=np.int64)
        cnt[:, 1:] = np.where(u_pairs[ix] == p, c_pairs[ix], 0)
        return cnt

    tr_pcnt = get_pair_counts(rows)
    val_pcnt = get_pair_counts(val)

    high_pairs = u_pairs[c_pairs >= 100]
    mid_pairs = u_pairs[(c_pairs >= 10) & (c_pairs < 100)]

    def get_tier_ids(arr, tier_keys):
        if len(tier_keys) == 0:
            return np.zeros_like(arr, dtype=np.int64)
        p = arr[:, :-1].astype(np.int64) * (V + 1) + arr[:, 1:].astype(np.int64)
        ix = np.clip(np.searchsorted(tier_keys, p), 0, len(tier_keys) - 1)
        out = np.zeros_like(arr, dtype=np.int64)
        out[:, 1:] = np.where(tier_keys[ix] == p, ix + 1, 0)
        return out

    d = dict(
        V=V, UNK=UNK, bucket=bucket, rows=rows, val=val,
        tr_pcnt=tr_pcnt, val_pcnt=val_pcnt,
        tr_hi=get_tier_ids(rows, high_pairs), val_hi=get_tier_ids(val, high_pairs), n_hi=len(high_pairs),
        tr_mid=get_tier_ids(rows, mid_pairs), val_mid=get_tier_ids(val, mid_pairs), n_mid=len(mid_pairs),
        n_unique_pairs=len(u_pairs),
    )
    _DATA_CACHE[key] = d
    return d


def bucket_index(pcnt):
    b_idx = torch.zeros_like(pcnt)
    b_idx = torch.where(pcnt >= 1, torch.ones_like(b_idx), b_idx)
    b_idx = torch.where(pcnt >= 10, torch.full_like(b_idx, 2), b_idx)
    b_idx = torch.where(pcnt >= 100, torch.full_like(b_idx, 3), b_idx)
    b_idx = torch.where(pcnt >= 1000, torch.full_like(b_idx, 4), b_idx)
    return b_idx


class Block(nn.Module):
    def __init__(self, d, H, V, active_ve, ve_mode, pair_rows, n_hi, n_mid, tau=100.0):
        super().__init__()
        self.H, self.dk, self.V = H, d // H, V
        self.active_ve, self.ve_mode, self.tau = active_ve, ve_mode, tau
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qn, self.kn = nn.RMSNorm(d // H), nn.RMSNorm(d // H)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        f = max(8, int(8 * d / 3) // 8 * 8)
        self.gate, self.up, self.down = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

        self.ve_word = self.ve_pair = self.ve_hi = self.ve_mid_a = self.ve_mid_p = self.b5_logits = None
        self.lam_w = nn.Parameter(torch.tensor(1.0))
        self.lam_p = nn.Parameter(torch.tensor(1.0))

        if active_ve:
            if ve_mode in ("word", "hard_stack", "st_gate", "bucket5_gate", "mp_ve", "word_engram_mat"):
                self.ve_word = nn.Parameter(torch.randn(V, d) * d ** -0.5)
            if ve_mode in ("pair_ve", "hard_stack", "st_gate", "bucket5_gate"):
                self.ve_pair = nn.Parameter(torch.randn(pair_rows, d) * d ** -0.5)
                self.pair_rows = pair_rows
            if ve_mode == "bucket5_gate":
                # UNIFORM zero init (sigmoid=0.5 for all 5 buckets) — zero prior bias!
                self.b5_logits = nn.Parameter(torch.zeros(5))
            if ve_mode == "mp_ve":
                self.ve_hi = nn.Parameter(torch.randn(n_hi + 1, d) * d ** -0.5)
                with torch.no_grad():
                    self.ve_hi[0].zero_()
                r = max(2, d // 4)
                self.ve_mid_a = nn.Parameter(torch.randn(n_mid + 1, r) * r ** -0.5)
                with torch.no_grad():
                    self.ve_mid_a[0].zero_()
                self.ve_mid_p = nn.Parameter(torch.randn(r, d) * (r * d) ** -0.5)

    def forward(self, x, ids, pcnt=None, hi_id=None, mid_id=None):
        b, t, d = x.shape
        h = self.n1(x)
        sp = lambda z: z.view(b, t, self.H, self.dk).transpose(1, 2)
        v = self.v(h)

        if self.active_ve:
            if self.ve_word is not None:
                v = v + self.lam_w * self.ve_word[ids]
            if self.ve_pair is not None:
                p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
                h_idx = (p1 * 1000003 + ids) % self.pair_rows
                tp = self.ve_pair[h_idx]
                if self.ve_mode in ("pair_ve", "hard_stack"):
                    v = v + self.lam_p * tp
                elif self.ve_mode == "st_gate":
                    g = (pcnt.float() / (pcnt.float() + self.tau)).unsqueeze(-1)
                    v = v + self.lam_p * (tp + (g - 1.0) * tp.detach())
                elif self.ve_mode == "bucket5_gate":
                    g = torch.sigmoid(self.b5_logits[bucket_index(pcnt)]).unsqueeze(-1)
                    v = v + self.lam_p * (g * tp.detach() + (tp - tp.detach()))
            if self.ve_mode == "mp_ve":
                hi_vec = self.ve_hi[hi_id] * (hi_id > 0).unsqueeze(-1).float()
                mid_vec = (self.ve_mid_a[mid_id] @ self.ve_mid_p) * (mid_id > 0).unsqueeze(-1).float()
                v = v + self.lam_p * (hi_vec + mid_vec)

        q, k = rope(self.qn(sp(self.q(h)))), rope(self.kn(sp(self.k(h))))
        y = F.scaled_dot_product_attention(q, k, sp(v), is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(b, t, d))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Model(nn.Module):
    def __init__(self, V, d, layers, heads, ve_layers, ve_mode, engram_mode, pair_rows, n_hi, n_mid, tau=100.0):
        super().__init__()
        self.V, self.d, self.engram_mode, self.tau = V, d, engram_mode, tau
        self.escale = 4.0
        self.head = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        self.emb = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        self.blocks = nn.ModuleList(
            Block(d, heads, V, i in ve_layers, ve_mode, pair_rows, n_hi, n_mid, tau=tau) for i in range(layers)
        )
        self.norm = nn.RMSNorm(d)

        self.ng_n = pair_rows
        self.ng_head = min(4, d // 2) if d >= 8 else 2
        self.ng2 = self.ng3 = self.ngk = self.ngv = self.ngc = self.eng_b5 = None
        if engram_mode == "core":
            self.ng2 = nn.Parameter(torch.randn(pair_rows, d) * d ** -0.5)
            self.ng3 = nn.Parameter(torch.randn(pair_rows, d) * d ** -0.5)
        elif engram_mode != "none":
            K = self.ng_head
            self.ng2 = nn.Parameter(torch.randn(K, pair_rows, d // K) * d ** -0.5)
            self.ng3 = nn.Parameter(torch.randn(K, pair_rows, d // K) * d ** -0.5)
            self.ngk = nn.Parameter(torch.randn(d, 2 * (d // K) * K) * (2 * d) ** -0.5)
            self.ngv = nn.Parameter(torch.randn(d, 2 * (d // K) * K) * (2 * d) ** -0.5)
            self.ngc = nn.Parameter(torch.zeros(d, 1, 4))
            if engram_mode in ("full_b5_mul", "full_b5_add", "word_engram_b5"):
                # UNIFORM zero init (sigmoid=0.5 for all 5 buckets) — zero prior bias!
                self.eng_b5 = nn.Parameter(torch.zeros(5))
        self.last_engram_gates = None

    def forward(self, ids, pcnt=None, hi_id=None, mid_id=None, capture_gate=False):
        x = self.emb[ids] * self.escale
        for i, blk in enumerate(self.blocks):
            if i == 0 and self.engram_mode == "core":
                p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
                p2 = F.pad(ids, (2, 0), value=self.V)[:, :-2]
                g2 = (p1 * 1000003 + ids) % self.ng_n
                g3 = (p2 * 998244353 + p1 * 1000003 + ids) % self.ng_n
                gate2 = torch.sigmoid(
                    (F.rms_norm(x, (x.shape[-1],)) * F.rms_norm(self.ng2[g2], (x.shape[-1],))).sum(-1, keepdim=True)
                    / x.shape[-1] ** 0.5
                )
                if capture_gate:
                    self.last_engram_gates = gate2.squeeze(-1).detach()
                x = x + gate2 * self.ng2[g2] * self.escale
                gate3 = torch.sigmoid(
                    (F.rms_norm(x, (x.shape[-1],)) * F.rms_norm(self.ng3[g3], (x.shape[-1],))).sum(-1, keepdim=True)
                    / x.shape[-1] ** 0.5
                )
                x = x + gate3 * self.ng3[g3] * self.escale
            elif i == 0 and self.engram_mode != "none":
                p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
                p2 = F.pad(ids, (2, 0), value=self.V)[:, :-2]
                K = self.ng_head
                hk = torch.arange(K, device=ids.device)
                M1 = torch.tensor([1000003, 998244353, 754974721, 167772161][:K], device=ids.device)
                M2 = torch.tensor([7919, 104729, 1299709, 15485863][:K], device=ids.device)
                h2 = (p1.unsqueeze(-1) * M1 + ids.unsqueeze(-1) * M2 + hk) % self.ng_n
                h3 = (p2.unsqueeze(-1) * M2 * 31 + p1.unsqueeze(-1) * M1 + ids.unsqueeze(-1) * M2 + hk * 7) % self.ng_n
                e = torch.cat([self.ng2[hk, h2].flatten(-2), self.ng3[hk, h3].flatten(-2)], -1)
                kk, vv = F.linear(e, self.ngk), F.linear(e, self.ngv)
                ctx_logit = (F.rms_norm(x, (x.shape[-1],)) * F.rms_norm(kk, (kk.shape[-1],))).sum(-1, keepdim=True) / x.shape[-1] ** 0.5
                g_ctx = torch.sigmoid(ctx_logit)
                if capture_gate:
                    self.last_engram_gates = g_ctx.squeeze(-1).detach()

                if self.engram_mode == "full":
                    v = g_ctx * vv
                elif self.engram_mode == "full_mat_mul":
                    g_mat = (pcnt.float() / (pcnt.float() + self.tau)).unsqueeze(-1)
                    # Straight-through for vv so rare entries still train, multiplied by g_ctx
                    vv_st = vv + (g_mat - 1.0) * vv.detach()
                    v = g_ctx * vv_st
                elif self.engram_mode == "full_mat_add":
                    g_mat = (pcnt.float() / (pcnt.float() + self.tau)).unsqueeze(-1).clamp(1e-4, 1.0 - 1e-4)
                    mat_logit = torch.log(g_mat / (1.0 - g_mat))
                    g_tot = torch.sigmoid(ctx_logit + mat_logit)
                    v = g_tot * vv.detach() + g_ctx * (vv - vv.detach())
                elif self.engram_mode in ("full_b5_mul", "word_engram_b5"):
                    g_mat = torch.sigmoid(self.eng_b5[bucket_index(pcnt)]).unsqueeze(-1)
                    vv_st = g_mat * vv.detach() + (vv - vv.detach())
                    v = g_ctx * vv_st
                elif self.engram_mode == "full_b5_add":
                    mat_logit = self.eng_b5[bucket_index(pcnt)].unsqueeze(-1)
                    g_tot = torch.sigmoid(ctx_logit + mat_logit)
                    v = g_tot * vv.detach() + g_ctx * (vv - vv.detach())

                c = F.conv1d(F.pad(F.rms_norm(v, (v.shape[-1],)).transpose(1, 2), (9, 0)), self.ngc, dilation=3, groups=v.shape[-1]).transpose(1, 2)
                x = x + (F.silu(c) + v) * self.escale
            x = blk(x, ids, pcnt, hi_id, mid_id)
        return F.linear(self.norm(x), self.head)


def train_and_eval(data, ve_layers, ve_mode="none", engram_mode="none", seed=0, width=16, layers=2, heads=2,
                   steps=150, batch=128, elr=0.1, tlr_w=0.1, tlr_p=0.03, lr=0.02, pair_rows=6000,
                   table_beta2=0.95, sparse_adam=False, measure_latency=False, capture_gate=False, dev="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    DEV = torch.device(dev)
    V = data["V"]
    m = Model(V, width, layers, heads, set(ve_layers), ve_mode, engram_mode, pair_rows, data["n_hi"], data["n_mid"]).to(DEV)

    word_tabs = [b.ve_word for b in m.blocks if b.ve_word is not None]
    pair_tabs = [z for z in (m.ng2, m.ng3, m.ngk, m.ngv, m.ngc) if z is not None] + [
        z for b in m.blocks for z in (b.ve_pair, b.ve_hi, b.ve_mid_a, b.ve_mid_p) if z is not None
    ]
    all_tabs = set(id(w) for w in word_tabs + pair_tabs)
    mats = [w for n, w in m.named_parameters() if w.ndim == 2 and n.startswith("blocks") and id(w) not in all_tabs]
    mat_ids = set(id(w) for w in mats)
    rest = [w for w in m.parameters() if w.requires_grad and id(w) not in all_tabs and id(w) not in mat_ids]

    opts = [Muon(mats, lr), torch.optim.Adam([dict(params=rest, lr=elr)], betas=(0.9, 0.95))]
    tgroups = []
    if word_tabs:
        tgroups.append(dict(params=word_tabs, lr=tlr_w))
    if pair_tabs:
        tgroups.append(dict(params=pair_tabs, lr=tlr_p))
    if tgroups:
        if sparse_adam:
            opts.append(SparseRowAdam(tgroups, betas=(0.9, table_beta2)))
        else:
            opts.append(torch.optim.Adam(tgroups, betas=(0.9, table_beta2)))

    base = [[g["lr"] for g in o.param_groups] for o in opts]
    X = torch.tensor(data["rows"], device=DEV)
    X_pc = torch.tensor(data["tr_pcnt"], device=DEV)
    X_hi = torch.tensor(data["tr_hi"], device=DEV)
    X_mid = torch.tensor(data["tr_mid"], device=DEV)

    t0 = time.time()
    for step in range(steps):
        sc = min((step + 1) / min(50, max(1, steps // 10)), 1.0, (steps - step) / (0.3 * steps))
        for o, bl in zip(opts, base):
            for g, lr0 in zip(o.param_groups, bl):
                g["lr"] = lr0 * sc
        s0, e0 = step * batch, (step + 1) * batch
        xb = X[s0:e0]
        logits = m(xb[:, :-1], X_pc[s0:e0, :-1], X_hi[s0:e0, :-1], X_mid[s0:e0, :-1])
        F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1)).backward()
        for o in opts:
            o.step()
        m.zero_grad(set_to_none=True)

    if DEV.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.time() - t0

    m.eval()
    VX = torch.tensor(data["val"], device=DEV)
    V_pc = torch.tensor(data["val_pcnt"], device=DEV)
    V_hi = torch.tensor(data["val_hi"], device=DEV)
    V_mid = torch.tensor(data["val_mid"], device=DEV)
    per = []
    gates_list = []
    with torch.no_grad():
        for i in range(0, len(VX), 200):
            xb = VX[i:i + 200]
            logits = m(xb[:, :-1], V_pc[i:i + 200, :-1], V_hi[i:i + 200, :-1], V_mid[i:i + 200, :-1], capture_gate=capture_gate)
            per.append(F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1), reduction="none"))
            if capture_gate and m.last_engram_gates is not None:
                gates_list.append(m.last_engram_gates.cpu().numpy())
    per = torch.cat(per).float().cpu().numpy()

    vtg = data["bucket"][data["val"][:, 1:].reshape(-1)]
    pcnt_flat = data["val_pcnt"][:, :-1].reshape(-1)
    first_mask = np.zeros_like(data["val"][:, :-1], dtype=bool)
    first_mask[:, 0] = True
    first_flat = first_mask.reshape(-1)

    PB = [0, 1, 10, 100, 1000, 10 ** 12]
    scores = {"val": float(per.mean()), "val_known": float(per[vtg < 3].mean())}
    for lo, hi in zip(PB[:-1], PB[1:]):
        sel = (~first_flat) & (pcnt_flat >= lo) & (pcnt_flat < hi) & (vtg < 3)
        if sel.sum() >= 50:
            scores[f"pc_{lo}"] = float(per[sel].mean())

    gate_diag = None
    if gates_list:
        g_all = np.concatenate(gates_list, axis=0).reshape(-1)
        valid = (~first_flat) & (vtg < 3)
        gv, cv = g_all[valid], pcnt_flat[valid]
        pearson = float(np.corrcoef(gv, np.log1p(cv.astype(np.float64)))[0, 1])
        rg = np.argsort(np.argsort(gv)).astype(np.float64)
        rc = np.argsort(np.argsort(cv)).astype(np.float64)
        spearman = float(np.corrcoef(rg, rc)[0, 1])
        per_b = {}
        for lo, hi in zip(PB[:-1], PB[1:]):
            sel = (~first_flat) & (pcnt_flat >= lo) & (pcnt_flat < hi) & (vtg < 3)
            if sel.sum() >= 50:
                per_b[f"pc_{lo}"] = round(float(g_all[sel].mean()), 4)
        gate_diag = {"pearson_logc": round(pearson, 4), "spearman": round(spearman, 4), "buckets": per_b}

    lat_ms = None
    if measure_latency:
        xb1 = VX[:1, :-1]
        pc1, hi1, mid1 = V_pc[:1, :-1], V_hi[:1, :-1], V_mid[:1, :-1]
        with torch.no_grad():
            for _ in range(10):
                _ = m(xb1, pc1, hi1, mid1)
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            tl0 = time.time()
            for _ in range(50):
                _ = m(xb1, pc1, hi1, mid1)
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            lat_ms = round((time.time() - tl0) / 50.0 * 1000.0, 3)

    nparam = sum(w.numel() for w in m.parameters())
    tab_param = sum(w.numel() for w in word_tabs + pair_tabs)
    eng_b5_val = [round(float(x), 3) for x in torch.sigmoid(m.eng_b5).detach().cpu()] if m.eng_b5 is not None else None
    ve_b5_val = [[round(float(x), 3) for x in torch.sigmoid(b.b5_logits).detach().cpu()] for b in m.blocks if b.b5_logits is not None] or None
    return dict(params=nparam, tab_params=tab_param, train_s=round(train_s, 2), lat_ms=lat_ms,
                scores=scores, gate_diag=gate_diag, eng_b5=eng_b5_val, ve_b5=ve_b5_val)


def summarize(runs):
    vk = [r["scores"]["val_known"] for r in runs]
    bm = {}
    for b in ("pc_0", "pc_1", "pc_10", "pc_100", "pc_1000"):
        vals = [r["scores"][b] for r in runs if b in r["scores"]]
        bm[b] = round(float(np.mean(vals)), 4) if vals else None
    ve_b5_all = [row for r in runs if r["ve_b5"] for row in r["ve_b5"]]
    ve_b5_stat = None
    if ve_b5_all:
        arr = np.array(ve_b5_all)
        ve_b5_stat = {"mean": [round(float(x), 3) for x in arr.mean(0)], "std": [round(float(x), 3) for x in arr.std(0, ddof=1) if len(arr) > 1]} if len(arr) > 1 else {"mean": [round(float(x), 3) for x in arr.mean(0)]}
    eng_b5_all = [r["eng_b5"] for r in runs if r["eng_b5"]]
    eng_b5_stat = None
    if eng_b5_all:
        arr = np.array(eng_b5_all)
        eng_b5_stat = {"mean": [round(float(x), 3) for x in arr.mean(0)], "std": [round(float(x), 3) for x in arr.std(0, ddof=1)] if len(arr) > 1 else [0.0]*5}
    return dict(
        mean=round(float(np.mean(vk)), 4),
        std=round(float(np.std(vk, ddof=1)) if len(vk) > 1 else 0.0, 4),
        params=runs[0]["params"],
        tab_params=runs[0]["tab_params"],
        lat_ms=runs[0]["lat_ms"],
        buckets=bm,
        eng_b5=eng_b5_stat,
        ve_b5=ve_b5_stat,
        raw=runs,
    )


def paired_str(ref_s, arm_s):
    diffs = [r0["scores"]["val_known"] - r1["scores"]["val_known"] for r0, r1 in zip(ref_s["raw"], arm_s["raw"])]
    m = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    t = m / (sd / math.sqrt(len(diffs))) if sd > 1e-9 else float("inf")
    return f"Δ={m:+.4f}±{sd:.4f} (t={t:+.1f})"


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "local_suite_results.json")
    all_res = {}

    # ------------------------------------------------------------------
    # BLOCK A: Orthogonality (Engram Full x Maturity Gate: mul / add / b5)
    # ------------------------------------------------------------------
    print("\n================ BLOCK A: Engram Full x Maturity Gate Orthogonality (3 seeds) ================")
    all_res["block_A_orthogonality"] = {}
    for st in (150, 300):
        data = load_data(6000, st, 128)
        all_res["block_A_orthogonality"][str(st)] = {}
        print(f"\n--- V=6,000, L=2, steps={st} ---")
        cfgs = [
            ("engram_full",       dict(ve_layers=[],    ve_mode="none",           engram_mode="full")),
            ("engram_mat_mul",    dict(ve_layers=[],    ve_mode="none",           engram_mode="full_mat_mul")),
            ("engram_mat_add",    dict(ve_layers=[],    ve_mode="none",           engram_mode="full_mat_add")),
            ("engram_b5_mul",     dict(ve_layers=[],    ve_mode="none",           engram_mode="full_b5_mul")),
            ("engram_b5_add",     dict(ve_layers=[],    ve_mode="none",           engram_mode="full_b5_add")),
            ("word_plus_eng_b5",  dict(ve_layers=[0,1], ve_mode="word_engram_mat",engram_mode="word_engram_b5")),
        ]
        for name, cfg in cfgs:
            runs = [train_and_eval(data, seed=sd, steps=st, pair_rows=6000, dev=args.dev, **cfg) for sd in range(args.seeds)]
            s = summarize(runs)
            all_res["block_A_orthogonality"][str(st)][name] = s
            b_str = " ".join(f"{k}:{v:.3f}" for k, v in s["buckets"].items() if v is not None)
            extra = f" | b5={s['eng_b5']}" if s["eng_b5"] else ""
            print(f"{name:17s} | known={s['mean']:.4f}±{s['std']:.4f} | {b_str}{extra}")
    save_json(out_path, all_res)

    # ------------------------------------------------------------------
    # BLOCK B: (M, T) Collision-Variance Phase Diagram (V=1,500, 3 seeds)
    # ------------------------------------------------------------------
    print("\n================ BLOCK B: (M, T) Collision-Variance Phase Diagram (V=1,500, 3 seeds) ================")
    all_res["block_B_phase_diagram"] = {}
    M_list = [256, 512, 1024, 2048, 4096, 16384]
    T_steps = [75, 150, 300, 600]
    for st in T_steps:
        data = load_data(1500, st, 128)
        all_res["block_B_phase_diagram"][str(st)] = {}
        rw = summarize([train_and_eval(data, [0, 1], ve_mode="word", seed=sd, steps=st, elr=0.05, tlr_w=0.1, dev=args.dev) for sd in range(args.seeds)])
        all_res["block_B_phase_diagram"][str(st)]["word"] = rw
        line = [f"T={st*128*64/1e6:4.2f}M (word={rw['mean']:.4f}):"]
        for M in M_list:
            rp = summarize([train_and_eval(data, [0, 1], ve_mode="pair_ve", seed=sd, steps=st, elr=0.05, tlr_p=0.03, pair_rows=M, dev=args.dev) for sd in range(args.seeds)])
            all_res["block_B_phase_diagram"][str(st)][f"M_{M}"] = rp
            gap = rp["mean"] - rw["mean"]  # <0 means pair beats word!
            line.append(f"M={M:5d}:{gap:+.4f}")
        print("  ".join(line))
    save_json(out_path, all_res)

    # ------------------------------------------------------------------
    # BLOCK C: Width Scaling Check of c* ~ d (d in 8..128, V=6000, 3 seeds)
    # ------------------------------------------------------------------
    print("\n================ BLOCK C: Width Scaling Check c* ~ d (V=6,000, 150 steps, 3 seeds) ================")
    all_res["block_C_width_scaling"] = {}
    data_c = load_data(6000, 150, 128)
    for d in (8, 12, 16, 32, 64, 128):
        H = 1 if d == 8 else (2 if d <= 32 else 4)
        all_res["block_C_width_scaling"][str(d)] = {}
        rw = summarize([train_and_eval(data_c, [0, 1], ve_mode="word", seed=sd, width=d, heads=H, steps=150, dev=args.dev) for sd in range(args.seeds)])
        re = summarize([train_and_eval(data_c, [], ve_mode="none", engram_mode="full", seed=sd, width=d, heads=H, steps=150, pair_rows=6000, dev=args.dev) for sd in range(args.seeds)])
        rm = summarize([train_and_eval(data_c, [0, 1], ve_mode="mp_ve", seed=sd, width=d, heads=H, steps=150, dev=args.dev) for sd in range(args.seeds)])
        all_res["block_C_width_scaling"][str(d)] = dict(word=rw, engram_full=re, mp_ve=rm)
        d_b10 = re["buckets"]["pc_10"] - rw["buckets"]["pc_10"]
        d_b100 = re["buckets"]["pc_100"] - rw["buckets"]["pc_100"]
        print(f"d={d:3d} | word={rw['mean']:.4f}±{rw['std']:.4f} | eng={re['mean']:.4f}±{re['std']:.4f} | mp_ve={rm['mean']:.4f}±{rm['std']:.4f} | "
              f"(eng-word) pc_10={d_b10:+.3f} pc_100={d_b100:+.3f} | mp_ve gain vs word={rw['mean']-rm['mean']:+.4f}")
    save_json(out_path, all_res)

    # ------------------------------------------------------------------
    # BLOCK D: Ablation + Equal-Weights + Last-2 vs All-4 + Sparse vs Dense Adam + Latency (L=4, w=32)
    # ------------------------------------------------------------------
    print("\n================ BLOCK D: Ablations, Equal-Weights, Sparse Adam & Latency (L=4, w=32, V=6,000, 300s, 3 seeds) ================")
    all_res["block_D_ablations_L4"] = {}
    data_d = load_data(6000, 300, 128)
    # Exact param check: plain L=4, w=32 has ~433K params; mp_ve_last2 has ~1.41M params -> plain w=96 has ~1.58M params (approx equal weights!)
    d_cfgs = [
        ("plain_w32_eqFLOP",     dict(width=32, heads=2, ve_layers=[],         ve_mode="none",         engram_mode="none", sparse_adam=False)),
        ("plain_w88_eqWeights",  dict(width=88, heads=4, ve_layers=[],         ve_mode="none",         engram_mode="none", sparse_adam=False)),
        ("word_last2",           dict(width=32, heads=2, ve_layers=[2, 3],     ve_mode="word",         engram_mode="none", sparse_adam=False)),
        ("engram_full",          dict(width=32, heads=2, ve_layers=[],         ve_mode="none",         engram_mode="full", sparse_adam=False)),
        ("hard_stack_sameLR",    dict(width=32, heads=2, ve_layers=[2, 3],     ve_mode="hard_stack",   engram_mode="none", tlr_p=0.1,  pair_rows=24000, sparse_adam=False)),
        ("hard_stack_decoupled", dict(width=32, heads=2, ve_layers=[2, 3],     ve_mode="hard_stack",   engram_mode="none", tlr_p=0.03, pair_rows=24000, sparse_adam=False)),
        ("bucket5_last2",        dict(width=32, heads=2, ve_layers=[2, 3],     ve_mode="bucket5_gate", engram_mode="none", tlr_p=0.03, pair_rows=24000, sparse_adam=False)),
        ("mp_ve_last2_dense",    dict(width=32, heads=2, ve_layers=[2, 3],     ve_mode="mp_ve",        engram_mode="none", tlr_p=0.03, sparse_adam=False)),
        ("mp_ve_all4_dense",     dict(width=32, heads=2, ve_layers=[0,1,2,3],  ve_mode="mp_ve",        engram_mode="none", tlr_p=0.03, sparse_adam=False)),
        ("mp_ve_last2_sparse",   dict(width=32, heads=2, ve_layers=[2, 3],     ve_mode="mp_ve",        engram_mode="none", tlr_p=0.03, sparse_adam=True)),
    ]
    for name, cfg in d_cfgs:
        runs = [train_and_eval(data_d, seed=sd, layers=4, steps=300, measure_latency=(sd == 0), dev=args.dev, **cfg) for sd in range(args.seeds)]
        s = summarize(runs)
        all_res["block_D_ablations_L4"][name] = s
        extra_b5 = f" | b5_mean={s['ve_b5']['mean']}" if s["ve_b5"] else ""
        print(f"{name:21s} | known={s['mean']:.4f}±{s['std']:.4f} | params={s['params']:9,d} (tab={s['tab_params']:7,d}) | b1_lat={s['lat_ms']:5.2f}ms | "
              f"pc_0:{s['buckets']['pc_0']:.3f} pc_100:{s['buckets']['pc_100']:.3f}{extra_b5}")
    hs_d = all_res["block_D_ablations_L4"]["hard_stack_decoupled"]
    wl_d = all_res["block_D_ablations_L4"]["word_last2"]
    print(f"  [Paired vs word_last2] hard_stack_decoupled: {paired_str(wl_d, hs_d)}")
    for nm in ("bucket5_last2", "mp_ve_last2_dense", "mp_ve_all4_dense", "mp_ve_last2_sparse"):
        print(f"  [Paired vs hard_stack_decoupled] {nm:20s}: {paired_str(hs_d, all_res['block_D_ablations_L4'][nm])}")
    save_json(out_path, all_res)

    # ------------------------------------------------------------------
    # BLOCK E: 3-Seed Held-Out Vocabulary Verification (V=1500, V=3000, V=6000)
    # ------------------------------------------------------------------
    print("\n================ BLOCK E: 3-Seed Held-Out Vocabulary Crossover (V=1500, 3000, 6000, UNIFORM b5 init!) ================")
    all_res["block_E_heldout_3seed"] = {}
    for Vw in (1500, 3000, 6000):
        all_res["block_E_heldout_3seed"][str(Vw)] = {}
        elr = 0.05 if Vw <= 3000 else 0.1
        for st in (150, 300):
            data_e = load_data(Vw, st, 128)
            rw = summarize([train_and_eval(data_e, [0, 1], ve_mode="word", seed=sd, steps=st, elr=elr, tlr_w=0.1, dev=args.dev) for sd in range(args.seeds)])
            re = summarize([train_and_eval(data_e, [], ve_mode="none", engram_mode="full", seed=sd, steps=st, elr=elr, tlr_p=0.03, pair_rows=Vw, dev=args.dev) for sd in range(args.seeds)])
            rh = summarize([train_and_eval(data_e, [0, 1], ve_mode="hard_stack", seed=sd, steps=st, elr=elr, tlr_w=0.1, tlr_p=0.03, pair_rows=Vw * 4, dev=args.dev) for sd in range(args.seeds)])
            rm = summarize([train_and_eval(data_e, [0, 1], ve_mode="mp_ve", seed=sd, steps=st, elr=elr, tlr_w=0.1, tlr_p=0.03, dev=args.dev) for sd in range(args.seeds)])
            rb = summarize([train_and_eval(data_e, [0, 1], ve_mode="bucket5_gate", seed=sd, steps=st, elr=elr, tlr_w=0.1, tlr_p=0.03, pair_rows=Vw * 4, dev=args.dev) for sd in range(args.seeds)])
            all_res["block_E_heldout_3seed"][str(Vw)][str(st)] = dict(word=rw, engram_full=re, hard_stack=rh, mp_ve=rm, bucket5=rb)
            b5m = rb["ve_b5"]["mean"] if rb["ve_b5"] else None
            print(f"V={Vw:4d} s={st:3d} | word={rw['mean']:.4f} | hs={rh['mean']:.4f} | mp_ve={rm['mean']:.4f} | b5={rb['mean']:.4f} "
                  f"| b5_vs_hs:{paired_str(rh, rb)} | b5_uniform_learned={b5m}")
    save_json(out_path, all_res)

    # ------------------------------------------------------------------
    # BLOCK F: 3-Seed Engram Gate Correlation + 3-Seed Table-Only beta2 Sweep
    # ------------------------------------------------------------------
    print("\n================ BLOCK F: 3-Seed Engram Gate Diag & 3-Seed Table-Only beta2 Sweep ================")
    all_res["block_F_3seed_diags"] = {"diag_engram": {}, "beta2_table_only": {}}
    for st in (150, 300):
        data_f = load_data(6000, st, 128)
        all_res["block_F_3seed_diags"]["diag_engram"][str(st)] = {}
        for em in ("core", "full"):
            runs = [train_and_eval(data_f, [], ve_mode="none", engram_mode=em, seed=sd, steps=st, elr=0.1, tlr_p=0.03, pair_rows=6000, capture_gate=True, dev=args.dev) for sd in range(args.seeds)]
            rs = [r["gate_diag"]["pearson_logc"] for r in runs]
            sp = [r["gate_diag"]["spearman"] for r in runs]
            bmeans = {b: round(float(np.mean([r["gate_diag"]["buckets"][b] for r in runs])), 3) for b in ("pc_0", "pc_1", "pc_10", "pc_100", "pc_1000")}
            all_res["block_F_3seed_diags"]["diag_engram"][str(st)][em] = dict(r_list=rs, r_mean=round(float(np.mean(rs)), 4), r_std=round(float(np.std(rs, ddof=1)), 4), spearman_mean=round(float(np.mean(sp)), 4), bucket_means=bmeans)
            print(f"[engram_{em:4s} s={st}] 3-seed r={rs} -> {np.mean(rs):+.4f}±{np.std(rs, ddof=1):.4f} | buckets={bmeans}")

    data_b2 = load_data(6000, 150, 128)
    for b2 in (0.9, 0.95, 0.99, 0.999):
        rw = summarize([train_and_eval(data_b2, [0, 1], ve_mode="word", seed=sd, steps=150, elr=0.1, tlr_w=0.1, table_beta2=b2, dev=args.dev) for sd in range(args.seeds)])
        rc = summarize([train_and_eval(data_b2, [], ve_mode="none", engram_mode="core", seed=sd, steps=150, elr=0.1, tlr_p=0.03, pair_rows=6000, table_beta2=b2, dev=args.dev) for sd in range(args.seeds)])
        rf = summarize([train_and_eval(data_b2, [], ve_mode="none", engram_mode="full", seed=sd, steps=150, elr=0.1, tlr_p=0.03, pair_rows=6000, table_beta2=b2, dev=args.dev) for sd in range(args.seeds)])
        dc = {b: round(rc["buckets"][b] - rw["buckets"][b], 4) for b in ("pc_0", "pc_1", "pc_10", "pc_100", "pc_1000")}
        df = {b: round(rf["buckets"][b] - rw["buckets"][b], 4) for b in ("pc_0", "pc_1", "pc_10", "pc_100", "pc_1000")}
        all_res["block_F_3seed_diags"]["beta2_table_only"][str(b2)] = dict(word=rw, engram_core=rc, engram_full=rf, diff_core=dc, diff_full=df)
        print(f"table_beta2={b2:5.3f} | (core-word) pc10={dc['pc_10']:+.4f} pc100={dc['pc_100']:+.4f} | (full-word) pc10={df['pc_10']:+.4f} pc100={df['pc_100']:+.4f}")

    save_json(out_path, all_res)
    print(f"\nALL 6 LOCAL MULTI-SEED BLOCKS COMPLETE! Saved to {out_path}")


if __name__ == "__main__":
    main()
