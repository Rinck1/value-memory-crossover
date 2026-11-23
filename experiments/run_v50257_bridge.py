"""Medium-scale local bridge experiment on REAL V=50,257 GPT-2 FineWeb tokens (RTX 4070 8GB, bf16).
Features:
- Real V=50,257 vocabulary (NO <UNK> collapse)
- Realistic 10% prefix pair counting (ZERO future oracle leakage)
- Intermediate checkpoint evaluations along training tokens T (pc_0, pc_1, pc_10, pc_100, pc_1000)
- Compares Equal Active FLOPs Plain vs Word Last-2 vs MP-VE (exact_hi + hash_mid) Last-2 vs Equal-Weights Plain
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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "v50257_bridge_results.json")
V_GPT2 = 50257


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
                grad = w.grad.float()
                m = self.state[w].setdefault("m", torch.zeros_like(w, dtype=torch.float32))
                m.mul_(0.95).add_(grad)
                upd = newton_schulz(grad + 0.95 * m)
                w.add_(upd.to(w.dtype), alpha=-g["lr"] * max(1, w.shape[0] / w.shape[1]) ** 0.5)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


class BridgeBlock(nn.Module):
    def __init__(self, d, H, V, active_ve, mode, n_hi, mid_hash_rows):
        super().__init__()
        self.H, self.dk, self.V = H, d // H, V
        self.active_ve, self.mode = active_ve, mode
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qn, self.kn = RMSNorm(d // H), RMSNorm(d // H)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        f = max(8, int(8 * d / 3) // 8 * 8)
        self.gate, self.up, self.down = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

        self.ve_word = self.ve_hi = self.ve_mid_a = self.ve_mid_p = None
        self.lam_w = nn.Parameter(torch.tensor(1.0))
        self.lam_p = nn.Parameter(torch.tensor(1.0))
        self.mid_hash_rows = mid_hash_rows

        if active_ve:
            if mode in ("word", "mp_ve"):
                self.ve_word = nn.Parameter(torch.randn(V, d) * d ** -0.5)
            if mode == "mp_ve":
                self.ve_hi = nn.Parameter(torch.randn(n_hi + 1, d) * d ** -0.5)
                with torch.no_grad():
                    self.ve_hi[0].zero_()
                r = max(4, d // 4)
                self.ve_mid_a = nn.Parameter(torch.randn(mid_hash_rows, r) * r ** -0.5)
                self.ve_mid_p = nn.Parameter(torch.randn(r, d) * (r * d) ** -0.5)

    def forward(self, x, ids, hi_id=None, is_mid=None):
        b, t, d = x.shape
        h = self.n1(x)
        sp = lambda z: z.view(b, t, self.H, self.dk).transpose(1, 2)
        v = self.v(h)

        if self.active_ve:
            if self.ve_word is not None:
                v = v + self.lam_w * self.ve_word[ids].to(v.dtype)
            if self.mode == "mp_ve":
                hi_vec = self.ve_hi[hi_id] * (hi_id > 0).unsqueeze(-1).to(v.dtype)
                p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
                h_idx = (p1 * 1000003 + ids) % self.mid_hash_rows
                mid_vec = (self.ve_mid_a[h_idx].to(v.dtype) @ self.ve_mid_p.to(v.dtype)) * is_mid.unsqueeze(-1).to(v.dtype)
                v = v + self.lam_p * (hi_vec.to(v.dtype) + mid_vec)

        q, k = rope(self.qn(sp(self.q(h)))), rope(self.kn(sp(self.k(h))))
        y = F.scaled_dot_product_attention(q, k, sp(v), is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(b, t, d))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class BridgeModel(nn.Module):
    def __init__(self, V, d, layers, heads, ve_layers, mode, n_hi, mid_hash_rows):
        super().__init__()
        self.V, self.d = V, d
        self.escale = 4.0
        self.emb = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        self.head = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        self.blocks = nn.ModuleList(
            BridgeBlock(d, heads, V, i in ve_layers, mode, n_hi, mid_hash_rows) for i in range(layers)
        )
        self.norm = RMSNorm(d)

    def forward(self, ids, hi_id=None, is_mid=None, targets=None):
        x = self.emb[ids] * self.escale
        for blk in self.blocks:
            x = blk(x, ids, hi_id, is_mid)
        h = self.norm(x)
        if targets is None:
            return F.linear(h, self.head.to(h.dtype))
        # Chunked cross entropy along sequence tokens to keep peak VRAM tiny on 8GB GPU
        h_flat = h.reshape(-1, h.shape[-1])
        t_flat = targets.reshape(-1)
        head_w = self.head.to(h.dtype)
        loss_sum = torch.zeros((), device=h.device, dtype=torch.float32)
        chunk = 2048
        for s0 in range(0, h_flat.shape[0], chunk):
            lg = F.linear(h_flat[s0:s0 + chunk], head_w)
            loss_sum = loss_sum + F.cross_entropy(lg.float(), t_flat[s0:s0 + chunk], reduction="sum")
        return loss_sum / t_flat.numel()


def prepare_data(total_tokens, seq_len=256, prefix_frac=0.10, nval_seqs=512):
    t0 = time.time()
    tr_mm = np.memmap(os.path.join(DATA_DIR, "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
    va_mm = np.memmap(os.path.join(DATA_DIR, "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)

    n_seqs = total_tokens // seq_len
    need = n_seqs * seq_len
    assert need <= len(tr_mm), f"Requested {need} tokens > {len(tr_mm)} available in train shard"
    rows = np.asarray(tr_mm[:need], dtype=np.int64).reshape(n_seqs, seq_len)

    starts = np.linspace(0, len(va_mm) - seq_len - 1, nval_seqs, dtype=np.int64)
    val = np.stack([np.asarray(va_mm[s:s + seq_len], dtype=np.int64) for s in starts])

    # 1. Realistic 10% prefix estimation ONLY (for MP-VE routing!)
    n_pref = max(1, int(n_seqs * prefix_frac))
    pref = rows[:n_pref]
    tp_pref = pref[:, :-1].reshape(-1) * V_GPT2 + pref[:, 1:].reshape(-1)
    u_pref, c_pref = np.unique(tp_pref, return_counts=True)
    hi_th = max(2, int(round(100 * prefix_frac)))
    mid_th = max(1, int(round(10 * prefix_frac)))
    high_keys = u_pref[c_pref >= hi_th]
    mid_keys = u_pref[(c_pref >= mid_th) & (c_pref < hi_th)]

    # 2. Exact full-horizon counts ONLY for post-hoc evaluation bucketing
    tp_full = rows[:, :-1].reshape(-1) * V_GPT2 + rows[:, 1:].reshape(-1)
    u_full, c_full = np.unique(tp_full, return_counts=True)
    va_p = val[:, :-1] * V_GPT2 + val[:, 1:]
    ix_f = np.clip(np.searchsorted(u_full, va_p), 0, len(u_full) - 1)
    val_pcnt = np.zeros_like(val, dtype=np.int64)
    val_pcnt[:, 1:] = np.where(u_full[ix_f] == va_p, c_full[ix_f], 0)

    def lookup_hi(arr):
        p = arr[:, :-1] * V_GPT2 + arr[:, 1:]
        ix = np.clip(np.searchsorted(high_keys, p), 0, max(0, len(high_keys) - 1))
        out = np.zeros_like(arr, dtype=np.int32)
        if len(high_keys):
            out[:, 1:] = np.where(high_keys[ix] == p, ix + 1, 0)
        return out

    def lookup_mid_mask(arr):
        p = arr[:, :-1] * V_GPT2 + arr[:, 1:]
        ix = np.clip(np.searchsorted(mid_keys, p), 0, max(0, len(mid_keys) - 1))
        out = np.zeros_like(arr, dtype=bool)
        if len(mid_keys):
            out[:, 1:] = (mid_keys[ix] == p)
        return out

    # Note: to evaluate cleanly without RAM bloat, compute hi/mid for rows on CPU as uint8/int32
    tr_hi = lookup_hi(rows)
    tr_mid_mask = lookup_mid_mask(rows)
    val_hi = lookup_hi(val)
    val_mid_mask = lookup_mid_mask(val)
    print(f"Prepared {need/1e6:.2f}M tokens in {time.time()-t0:.1f}s | Prefix({prefix_frac*100:.0f}%): hi>={hi_th} ({len(high_keys):,} pairs), mid>={mid_th} ({len(mid_keys):,} pairs)")
    return dict(
        rows=rows, val=val, val_pcnt=val_pcnt,
        tr_hi=tr_hi, tr_mid_mask=tr_mid_mask,
        val_hi=val_hi, val_mid_mask=val_mid_mask,
        n_hi=len(high_keys), n_mid=len(mid_keys), seq_len=seq_len
    )


def evaluate(m, data, dev="cuda"):
    m.eval()
    val = data["val"]
    val_hi = data["val_hi"]
    val_mid = data["val_mid_mask"]
    per = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for s0 in range(0, len(val), 16):
            xb = torch.from_numpy(val[s0:s0 + 16]).to(dev, dtype=torch.long)
            hi = torch.from_numpy(val_hi[s0:s0 + 16, :-1]).to(dev, dtype=torch.long)
            md = torch.from_numpy(val_mid[s0:s0 + 16, :-1]).to(dev)
            lg = m(xb[:, :-1], hi, md)
            per.append(F.cross_entropy(lg.float().reshape(-1, V_GPT2), xb[:, 1:].reshape(-1), reduction="none").cpu())
    per = torch.cat(per).numpy()
    pc_flat = data["val_pcnt"][:, :-1].reshape(-1)
    fm = np.zeros_like(val[:, :-1], dtype=bool)
    fm[:, 0] = True
    ff = fm.reshape(-1)
    scores = {"val": float(per.mean())}
    for lo, hi_b in [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 10**12)]:
        sel = (~ff) & (pc_flat >= lo) & (pc_flat < hi_b)
        if sel.sum() >= 50:
            scores[f"pc_{lo}"] = round(float(per[sel].mean()), 4)
    m.train()
    return scores


def run_arm(name, data, d, L, H, ve_layers, mode, mid_hash_rows=200000, batch_seqs=32, accum=2,
            elr=0.05, tlr_w=0.05, tlr_p=0.03, lr=0.015, eval_fracs=(0.25, 0.5, 1.0), seed=0, dev="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.reset_peak_memory_stats()
    m = BridgeModel(V_GPT2, d, L, H, set(ve_layers), mode, data["n_hi"], mid_hash_rows).to(dev)

    word_tabs = [b.ve_word for b in m.blocks if b.ve_word is not None]
    pair_tabs = [z for b in m.blocks for z in (b.ve_hi, b.ve_mid_a, b.ve_mid_p) if z is not None]
    all_tabs = set(id(w) for w in word_tabs + pair_tabs)
    mats = [w for n, w in m.named_parameters() if w.ndim == 2 and n.startswith("blocks") and id(w) not in all_tabs]
    mat_ids = set(id(w) for w in mats)
    rest = [w for w in m.parameters() if w.requires_grad and id(w) not in all_tabs and id(w) not in mat_ids]

    opts = [Muon(mats, lr), torch.optim.AdamW([dict(params=rest, lr=elr, weight_decay=0.0)], betas=(0.9, 0.95))]
    tgroups = []
    if word_tabs:
        tgroups.append(dict(params=word_tabs, lr=tlr_w))
    if pair_tabs:
        tgroups.append(dict(params=pair_tabs, lr=tlr_p))
    if tgroups:
        opts.append(torch.optim.Adam(tgroups, betas=(0.9, 0.95)))
    base = [[g["lr"] for g in o.param_groups] for o in opts]

    n_seqs = len(data["rows"])
    steps = n_seqs // batch_seqs
    mb = batch_seqs // accum
    eval_steps = sorted(set(max(1, min(steps, int(round(steps * fr)))) for fr in eval_fracs))
    ckpts = {}

    t0 = time.time()
    m.train()
    for step in range(steps):
        sc = min((step + 1) / max(1, min(100, steps // 10)), 1.0, (steps - step) / max(1.0, 0.25 * steps))
        for o, bl in zip(opts, base):
            for g, lr0 in zip(o.param_groups, bl):
                g["lr"] = lr0 * sc
        for a in range(accum):
            s0 = step * batch_seqs + a * mb
            s1 = s0 + mb
            xb = torch.from_numpy(data["rows"][s0:s1]).to(dev, dtype=torch.long)
            hi = torch.from_numpy(data["tr_hi"][s0:s1, :-1]).to(dev, dtype=torch.long)
            md = torch.from_numpy(data["tr_mid_mask"][s0:s1, :-1]).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = m(xb[:, :-1], hi, md, targets=xb[:, 1:]) / accum
            loss.backward()
        for o in opts:
            o.step()
        m.zero_grad(set_to_none=True)

        st_1 = step + 1
        if st_1 in eval_steps:
            tok_m = st_1 * batch_seqs * data["seq_len"] / 1e6
            sc_dict = evaluate(m, data, dev)
            sc_dict["tokens_M"] = round(tok_m, 2)
            sc_dict["elapsed_s"] = round(time.time() - t0, 1)
            ckpts[f"{tok_m:.1f}M"] = sc_dict
            print(f"    [{name} @ {tok_m:5.1f}M tok, {sc_dict['elapsed_s']:6.1f}s] val={sc_dict['val']:.4f} | "
                  f"pc0={sc_dict.get('pc_0')} pc1={sc_dict.get('pc_1')} pc10={sc_dict.get('pc_10')} pc100={sc_dict.get('pc_100')} pc1000={sc_dict.get('pc_1000')}")

    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    nparam = sum(w.numel() for w in m.parameters())
    tab_param = sum(w.numel() for w in word_tabs + pair_tabs)
    tot_s = round(time.time() - t0, 1)
    tok_s = int((steps * batch_seqs * data["seq_len"]) / max(1e-3, tot_s))
    return dict(name=name, params=nparam, tab_params=tab_param, peak_vram_gb=peak_gb, total_s=tot_s, tok_per_s=tok_s, checkpoints=ckpts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--tokens_m", type=float, default=30.0)
    args = p.parse_args()

    tot_tok = int(args.tokens_m * 1e6) if not args.probe else int(1.0 * 1e6)
    data = prepare_data(tot_tok, seq_len=256, prefix_frac=0.10)
    eval_fr = (1.0,) if args.probe else (0.25, 0.5, 0.75, 1.0)

    # Active FLOPs equal baseline: L=6, d=256, H=8
    # Equal-Weights check: L=6, d=384, H=8 (approx ~2x params of d=256 plain)
    arms = [
        ("plain_d256_eqFLOP",   dict(d=256, L=6, H=8, ve_layers=[],     mode="none")),
        ("word_d256_last2",     dict(d=256, L=6, H=8, ve_layers=[4, 5], mode="word")),
        ("mp_ve_d256_prefix10", dict(d=256, L=6, H=8, ve_layers=[4, 5], mode="mp_ve", mid_hash_rows=200000)),
    ]
    if not args.probe:
        arms.append(("plain_d384_eqWeights", dict(d=384, L=6, H=8, ve_layers=[], mode="none")))

    results = {"tokens_M": tot_tok / 1e6, "n_hi_prefix10": data["n_hi"], "n_mid_prefix10": data["n_mid"], "arms": {}}
    for nm, cfg in arms:
        print(f"\n--- Running {nm} (V=50,257, tokens={tot_tok/1e6:.2f}M) ---")
        res_arm = run_arm(nm, data, eval_fracs=eval_fr, **cfg)
        results["arms"][nm] = res_arm
        print(f"  DONE {nm}: params={res_arm['params']:,} (tab={res_arm['tab_params']:,}), peak_VRAM={res_arm['peak_vram_gb']} GB, "
              f"time={res_arm['total_s']}s, throughput={res_arm['tok_per_s']:,} tok/s")
        if not args.probe:
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
