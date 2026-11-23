"""Sequential P0 Validation Suite for Paper 06 / MP-VE (RTX 4070 GPU).
Blocks:
  step2: Preregistered held-out test (V=4500, d=48, L=4, H=4, 250 steps, 3 seeds)
  step3: Equal-compute plain shape search (V=6000, 300 steps, 3 seeds)
  step4: Disentangled L=4 ablations + LR grid sensitivity (V=6000, L=4, w=32, 300s, 3 seeds)
  step5: 5-seed headline upgrade + SparseRowAdam global-step mechanism + 1000+ pair lexical check
  step6: Prefix counting (10%/20%) vs Oracle + Standard AdamW backbone check (3 seeds)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_local_suite as rls

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "p0_results.json")


def load_p0():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_p0(obj):
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


class SparseRowAdamGlobalT(torch.optim.Optimizer):
    """Ablates why SparseRowAdam differs from Dense Adam: mode in {'both_frozen', 'decay_m_only', 'decay_v_only'}."""
    def __init__(self, param_groups, betas=(0.9, 0.95), eps=1e-8, mode="both_frozen"):
        super().__init__(param_groups, dict(betas=betas, eps=eps, mode=mode))
        self.global_t = 0

    @torch.no_grad()
    def step(self):
        self.global_t += 1
        t = self.global_t
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            mode = group["mode"]
            bc1 = 1.0 - b1 ** t
            bc2 = 1.0 - b2 ** t
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                if p.ndim == 2:
                    if mode == "decay_m_only":
                        st["m"].mul_(b1)
                    elif mode == "decay_v_only":
                        st["v"].mul_(b2)
                    active = (g.abs().sum(dim=1) > 0)
                    if not active.any():
                        continue
                    idx = active.nonzero(as_tuple=False).squeeze(-1)
                    gi = g[idx]
                    mi = st["m"][idx] * (1.0 if mode == "decay_m_only" else b1) + (1.0 - b1) * gi
                    vi = st["v"][idx] * (1.0 if mode == "decay_v_only" else b2) + (1.0 - b2) * (gi * gi)
                    st["m"][idx] = mi
                    st["v"][idx] = vi
                    p.index_add_(0, idx, ((mi / bc1) / ((vi / bc2).sqrt() + eps)) * (-lr))
                else:
                    st["m"].mul_(b1).add_(g, alpha=1.0 - b1)
                    st["v"].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    p.add_((st["m"] / bc1) / ((st["v"] / bc2).sqrt() + eps), alpha=-lr)


class FlexBlock(nn.Module):
    """Supports flexible ablation variants cleanly alongside exact run_local_suite semantics."""
    def __init__(self, d, H, V, active_ve, variant, pair_rows, n_hi, n_mid, b5_init=None):
        super().__init__()
        self.H, self.dk, self.V = H, d // H, V
        self.active_ve, self.variant = active_ve, variant
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qn, self.kn = nn.RMSNorm(d // H), nn.RMSNorm(d // H)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        f = max(8, int(8 * d / 3) // 8 * 8)
        self.gate, self.up, self.down = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

        self.ve_word = self.ve_pair = self.ve_hi = self.ve_mid_a = self.ve_mid_p = self.b5_logits = None
        self.lam_w = nn.Parameter(torch.tensor(1.0))
        self.lam_p = nn.Parameter(torch.tensor(1.0))
        self.pair_rows = pair_rows

        if active_ve:
            use_word = variant in ("word", "hard_stack", "hash_pruned_c10", "bucket5", "mp_ve",
                                   "exact_hi_only", "exact_hi_mid_full", "mp_ve_exact_hi_hash_mid")
            if use_word:
                self.ve_word = nn.Parameter(torch.randn(V, d) * d ** -0.5)
            if variant in ("hard_stack", "hash_pruned_c10", "bucket5"):
                self.ve_pair = nn.Parameter(torch.randn(pair_rows, d) * d ** -0.5)
            if variant == "bucket5":
                init_v = torch.zeros(5) if b5_init is None else torch.tensor(b5_init, dtype=torch.float32)
                self.b5_logits = nn.Parameter(init_v)
            if variant in ("mp_ve", "mp_ve_no_word", "exact_hi_only"):
                self.ve_hi = nn.Parameter(torch.randn(n_hi + 1, d) * d ** -0.5)
                with torch.no_grad():
                    self.ve_hi[0].zero_()
            if variant in ("mp_ve", "mp_ve_no_word"):
                r = max(2, d // 4)
                self.ve_mid_a = nn.Parameter(torch.randn(n_mid + 1, r) * r ** -0.5)
                with torch.no_grad():
                    self.ve_mid_a[0].zero_()
                self.ve_mid_p = nn.Parameter(torch.randn(r, d) * (r * d) ** -0.5)
            if variant == "exact_hi_mid_full":
                # Full-rank exact rows for both >=100 (n_hi) and 10-99 (n_mid)
                self.ve_hi = nn.Parameter(torch.randn(n_hi + 1, d) * d ** -0.5)
                self.ve_mid_a = nn.Parameter(torch.randn(n_mid + 1, d) * d ** -0.5)
                with torch.no_grad():
                    self.ve_hi[0].zero_()
                    self.ve_mid_a[0].zero_()
            if variant == "mp_ve_exact_hi_hash_mid":
                # Exact full-rank for >=100 (small set!), hashed low-rank d/4 for 10-99!
                self.ve_hi = nn.Parameter(torch.randn(n_hi + 1, d) * d ** -0.5)
                with torch.no_grad():
                    self.ve_hi[0].zero_()
                r = max(2, d // 4)
                self.ve_mid_a = nn.Parameter(torch.randn(pair_rows, r) * r ** -0.5)
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
                if self.variant == "hard_stack":
                    v = v + self.lam_p * tp
                elif self.variant == "hash_pruned_c10":
                    mask = (pcnt >= 10).unsqueeze(-1).float()
                    v = v + self.lam_p * (tp * mask)
                elif self.variant == "bucket5":
                    g = torch.sigmoid(self.b5_logits[rls.bucket_index(pcnt)]).unsqueeze(-1)
                    v = v + self.lam_p * (g * tp.detach() + (tp - tp.detach()))
            elif self.variant in ("mp_ve", "mp_ve_no_word"):
                hi_vec = self.ve_hi[hi_id] * (hi_id > 0).unsqueeze(-1).float()
                mid_vec = (self.ve_mid_a[mid_id] @ self.ve_mid_p) * (mid_id > 0).unsqueeze(-1).float()
                v = v + self.lam_p * (hi_vec + mid_vec)
            elif self.variant == "exact_hi_only":
                hi_vec = self.ve_hi[hi_id] * (hi_id > 0).unsqueeze(-1).float()
                v = v + self.lam_p * hi_vec
            elif self.variant == "exact_hi_mid_full":
                hi_vec = self.ve_hi[hi_id] * (hi_id > 0).unsqueeze(-1).float()
                mid_vec = self.ve_mid_a[mid_id] * (mid_id > 0).unsqueeze(-1).float()
                v = v + self.lam_p * (hi_vec + mid_vec)
            elif self.variant == "mp_ve_exact_hi_hash_mid":
                hi_vec = self.ve_hi[hi_id] * (hi_id > 0).unsqueeze(-1).float()
                p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
                h_idx = (p1 * 1000003 + ids) % self.pair_rows
                is_mid = ((pcnt >= 10) & (pcnt < 100)).unsqueeze(-1).float()
                mid_vec = (self.ve_mid_a[h_idx] @ self.ve_mid_p) * is_mid
                v = v + self.lam_p * (hi_vec + mid_vec)

        q, k = rls.rope(self.qn(sp(self.q(h)))), rls.rope(self.kn(sp(self.k(h))))
        y = F.scaled_dot_product_attention(q, k, sp(v), is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(b, t, d))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class FlexModel(nn.Module):
    def __init__(self, V, d, layers, heads, ve_layers, variant, pair_rows, n_hi, n_mid, b5_init=None):
        super().__init__()
        self.V, self.d = V, d
        self.escale = 4.0
        self.head = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        self.emb = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        self.blocks = nn.ModuleList(
            FlexBlock(d, heads, V, i in ve_layers, variant, pair_rows, n_hi, n_mid, b5_init=b5_init)
            for i in range(layers)
        )
        self.norm = nn.RMSNorm(d)

    def forward(self, ids, pcnt=None, hi_id=None, mid_id=None):
        x = self.emb[ids] * self.escale
        for blk in self.blocks:
            x = blk(x, ids, pcnt, hi_id, mid_id)
        return F.linear(self.norm(x), self.head)


def train_eval_flex(data, ve_layers, variant="none", seed=0, width=32, layers=4, heads=2,
                    steps=300, batch=128, elr=0.1, tlr_w=0.1, tlr_p=0.03, lr=0.02,
                    pair_rows=24000, b5_init=None, sparse_mode="none", backbone_opt="muon",
                    adamw_lr=3e-3, measure_latency=False, dev="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    DEV = torch.device(dev)
    V = data["V"]
    m = FlexModel(V, width, layers, heads, set(ve_layers), variant, pair_rows, data["n_hi"], data["n_mid"], b5_init=b5_init).to(DEV)

    word_tabs = [b.ve_word for b in m.blocks if b.ve_word is not None]
    pair_tabs = [z for b in m.blocks for z in (b.ve_pair, b.ve_hi, b.ve_mid_a, b.ve_mid_p) if z is not None]
    all_tabs = set(id(w) for w in word_tabs + pair_tabs)
    mats = [w for n, w in m.named_parameters() if w.ndim == 2 and n.startswith("blocks") and id(w) not in all_tabs]
    mat_ids = set(id(w) for w in mats)
    rest = [w for w in m.parameters() if w.requires_grad and id(w) not in all_tabs and id(w) not in mat_ids]

    if backbone_opt == "muon":
        opts = [rls.Muon(mats, lr), torch.optim.Adam([dict(params=rest, lr=elr)], betas=(0.9, 0.95))]
    elif backbone_opt == "adamw":
        opts = [torch.optim.AdamW([dict(params=mats, lr=adamw_lr, weight_decay=0.01)], betas=(0.9, 0.95)),
                torch.optim.Adam([dict(params=rest, lr=elr)], betas=(0.9, 0.95))]
    else:
        raise ValueError(backbone_opt)

    tgroups = []
    if word_tabs:
        tgroups.append(dict(params=word_tabs, lr=tlr_w))
    if pair_tabs:
        tgroups.append(dict(params=pair_tabs, lr=tlr_p))
    if tgroups:
        if sparse_mode == "row_t":
            opts.append(rls.SparseRowAdam(tgroups, betas=(0.9, 0.95)))
        elif sparse_mode in ("global_t", "both_frozen"):
            opts.append(SparseRowAdamGlobalT(tgroups, betas=(0.9, 0.95), mode="both_frozen"))
        elif sparse_mode in ("decay_m_only", "decay_v_only"):
            opts.append(SparseRowAdamGlobalT(tgroups, betas=(0.9, 0.95), mode=sparse_mode))
        else:
            opts.append(torch.optim.Adam(tgroups, betas=(0.9, 0.95)))

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
    with torch.no_grad():
        for i in range(0, len(VX), 200):
            xb = VX[i:i + 200]
            logits = m(xb[:, :-1], V_pc[i:i + 200, :-1], V_hi[i:i + 200, :-1], V_mid[i:i + 200, :-1])
            per.append(F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1), reduction="none"))
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
    ve_b5_val = [[round(float(x), 3) for x in torch.sigmoid(b.b5_logits).detach().cpu()] for b in m.blocks if b.b5_logits is not None] or None
    return dict(params=nparam, tab_params=tab_param, train_s=round(train_s, 2), lat_ms=lat_ms,
                scores=scores, gate_diag=None, eng_b5=None, ve_b5=ve_b5_val)


def flops_per_tok(d, L, V, T_ctx=64):
    f = max(8, int(8 * d / 3) // 8 * 8)
    layer = 2 * (4 * d * d + 3 * d * f) + 2 * T_ctx * d
    bb = L * layer
    tot = bb + 2 * V * d
    return bb, tot


def run_step2(res):
    print("\n================ STEP 2: Preregistered Held-Out (V=4,500, d=48, L=4, H=4, 250 steps, 3 seeds) ================")
    data = rls.load_data(4500, 250, 128)
    cfgs = [
        ("plain_w48",            "rls",  dict(ve_layers=[],    ve_mode="none",       engram_mode="none")),
        ("word_last2",           "rls",  dict(ve_layers=[2,3], ve_mode="word",       engram_mode="none")),
        ("engram_full",          "rls",  dict(ve_layers=[],    ve_mode="none",       engram_mode="full", pair_rows=4500)),
        ("hard_stack_decoupled", "rls",  dict(ve_layers=[2,3], ve_mode="hard_stack", engram_mode="none", tlr_p=0.03, pair_rows=18000)),
        ("mp_ve_last2_dense",    "rls",  dict(ve_layers=[2,3], ve_mode="mp_ve",      engram_mode="none", tlr_p=0.03)),
    ]
    out = {}
    for name, _, cfg in cfgs:
        runs = [rls.train_and_eval(data, seed=sd, width=48, layers=4, heads=4, steps=250, elr=0.1, dev="cuda", **cfg) for sd in range(3)]
        s = rls.summarize(runs)
        out[name] = s
        print(f"  {name:22s} | known={s['mean']:.4f}±{s['std']:.4f} | params={s['params']:,} | tab={s['tab_params']:,}")
    wl, ef, hs, mv = out["word_last2"], out["engram_full"], out["hard_stack_decoupled"], out["mp_ve_last2_dense"]
    print(f"  Measured gaps vs preregistered:")
    print(f"    engram_full - word_last2  = {ef['mean'] - wl['mean']:+.4f} (pred +0.0099)")
    print(f"    word_last2 - hard_stack   = {wl['mean'] - hs['mean']:+.4f} (pred +0.0209) | {rls.paired_str(wl, hs)}")
    print(f"    hard_stack - mp_ve        = {hs['mean'] - mv['mean']:+.4f} (pred +0.0203) | {rls.paired_str(hs, mv)}")
    print(f"    word_last2 - mp_ve        = {wl['mean'] - mv['mean']:+.4f} (pred +0.0412) | {rls.paired_str(wl, mv)}")
    for b in ("pc_0", "pc_1", "pc_10", "pc_100", "pc_1000"):
        print(f"    bucket {b:7s}: eng-word = {ef['buckets'][b] - wl['buckets'][b]:+.4f} | hs-mpve = {hs['buckets'][b] - mv['buckets'][b]:+.4f}")
    res["step2_prereg_heldout"] = out
    save_p0(res)


def run_step3(res):
    print("\n================ STEP 3: Equal-Compute Plain Shape Search (V=6,000, 300 steps, 3 seeds) ================")
    data = rls.load_data(6000, 300, 128)
    shapes = [
        ("L2_w48", 2, 48, 4),
        ("L3_w40", 3, 40, 2),
        ("L4_w32", 4, 32, 2),
        ("L6_w24", 6, 24, 2),
        ("L4_w40", 4, 40, 2),
        ("L4_w48", 4, 48, 4),
    ]
    out = {}
    for name, L, w, H in shapes:
        runs = [train_eval_flex(data, [], "none", seed=sd, width=w, layers=L, heads=H, steps=300, elr=0.1,
                                measure_latency=(sd == 0), dev="cuda") for sd in range(3)]
        s = rls.summarize(runs)
        bb, tot = flops_per_tok(w, L, data["V"])
        s["bb_flops_tok"] = bb
        s["tot_flops_tok"] = tot
        s["mean_train_s"] = round(float(np.mean([r["train_s"] for r in runs])), 2)
        out[name] = s
        print(f"  {name:8s} (L={L},w={w:2d},H={H}) | known={s['mean']:.4f}±{s['std']:.4f} | params={s['params']:9,d} | bb_FLOP={bb:7,d} | tot_FLOP={tot:8,d} | b1={s['lat_ms']:5.2f}ms | tr={s['mean_train_s']:5.2f}s")
    res["step3_equal_compute_shape"] = out
    save_p0(res)


def run_step4(res):
    print("\n================ STEP 4: L=4 Disentangled Ablation + LR Grid (V=6,000, L=4, w=32, 300s, 3 seeds) ================")
    data = rls.load_data(6000, 300, 128)
    arms = [
        ("hash_hs_sameLR",          dict(ve_layers=[2,3], variant="hard_stack",              tlr_w=0.1, tlr_p=0.1,  elr=0.1, pair_rows=24000)),
        ("hash_hs_decoupled",       dict(ve_layers=[2,3], variant="hard_stack",              tlr_w=0.1, tlr_p=0.03, elr=0.1, pair_rows=24000)),
        ("hash_pruned_c10",         dict(ve_layers=[2,3], variant="hash_pruned_c10",         tlr_w=0.1, tlr_p=0.03, elr=0.1, pair_rows=24000)),
        ("exact_hi_only",           dict(ve_layers=[2,3], variant="exact_hi_only",           tlr_w=0.1, tlr_p=0.03, elr=0.1)),
        ("exact_hi_mid_full",       dict(ve_layers=[2,3], variant="exact_hi_mid_full",       tlr_w=0.1, tlr_p=0.03, elr=0.1)),
        ("mp_ve_no_word",           dict(ve_layers=[2,3], variant="mp_ve_no_word",           tlr_w=0.1, tlr_p=0.03, elr=0.1)),
        ("mp_ve_sameLR",            dict(ve_layers=[2,3], variant="mp_ve",                   tlr_w=0.1, tlr_p=0.1,  elr=0.1)),
        ("mp_ve_dense_ref",         dict(ve_layers=[2,3], variant="mp_ve",                   tlr_w=0.1, tlr_p=0.03, elr=0.1)),
        ("mp_ve_exact_hi_hash_mid", dict(ve_layers=[2,3], variant="mp_ve_exact_hi_hash_mid", tlr_w=0.1, tlr_p=0.03, elr=0.1, pair_rows=24000)),
        ("bucket5_prior_init",      dict(ve_layers=[2,3], variant="bucket5",                 tlr_w=0.1, tlr_p=0.03, elr=0.1, pair_rows=24000, b5_init=[-4.0,-2.5,-1.0,1.5,2.5])),
        ("bucket5_zero_init",       dict(ve_layers=[2,3], variant="bucket5",                 tlr_w=0.1, tlr_p=0.03, elr=0.1, pair_rows=24000, b5_init=None)),
    ]
    part_a = {}
    for name, cfg in arms:
        runs = [train_eval_flex(data, seed=sd, width=32, layers=4, heads=2, steps=300, measure_latency=(sd == 0), dev="cuda", **cfg) for sd in range(3)]
        s = rls.summarize(runs)
        part_a[name] = s
        extra = f" | b5={s['ve_b5']['mean']}" if s["ve_b5"] else ""
        print(f"  {name:24s} | known={s['mean']:.4f}±{s['std']:.4f} | tab={s['tab_params']:9,d} | pc0={s['buckets']['pc_0']:.3f} pc100={s['buckets']['pc_100']:.3f}{extra}")

    hs = part_a["hash_hs_decoupled"]
    print("  Paired diffs vs hash_hs_decoupled:")
    for nm in part_a:
        if nm != "hash_hs_decoupled":
            print(f"    {nm:24s}: {rls.paired_str(hs, part_a[nm])}")

    print("\n--- Step 4b: LR Sensitivity Sweep (tlr_p x elr) ---")
    lr_grid = {}
    for elr in (0.05, 0.1):
        for tlr_p in (0.01, 0.03, 0.06, 0.1):
            if elr == 0.05 and tlr_p in (0.01, 0.1):
                continue
            tag = f"elr{elr}_tlrp{tlr_p}"
            lr_grid[tag] = {}
            for arm_nm, var_nm in [("word", "word"), ("hs", "hard_stack"), ("mpve", "mp_ve"), ("mpve_ehm", "mp_ve_exact_hi_hash_mid")]:
                runs = [train_eval_flex(data, [2, 3], var_nm, seed=sd, width=32, layers=4, heads=2, steps=300,
                                        elr=elr, tlr_w=0.1, tlr_p=tlr_p, pair_rows=24000, dev="cuda") for sd in range(3)]
                lr_grid[tag][arm_nm] = rls.summarize(runs)
            w_s, h_s, m_s, me_s = lr_grid[tag]["word"], lr_grid[tag]["hs"], lr_grid[tag]["mpve"], lr_grid[tag]["mpve_ehm"]
            print(f"  {tag:16s} | word={w_s['mean']:.4f} hs={h_s['mean']:.4f} mpve={m_s['mean']:.4f} ({rls.paired_str(h_s, m_s)}) | ehm={me_s['mean']:.4f}")

    res["step4_ablation_and_lr"] = {"disentangle": part_a, "lr_grid": lr_grid}
    save_p0(res)


def run_step5(res):
    print("\n================ STEP 5: 5-Seed Headline + SparseRowAdam Global-t + 1000+ Lexical Diag ================")
    data = rls.load_data(6000, 300, 128)
    # Load existing 3 seeds from local_suite_results.json for seamless 5-seed extension!
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "local_suite_results.json")) as f:
        suite = json.load(f)
    bd = suite["block_D_ablations_L4"]

    extend_cfgs = [
        ("plain_w32_eqFLOP",     dict(width=32, heads=2, ve_layers=[],         ve_mode="none",         engram_mode="none")),
        ("word_last2",           dict(width=32, heads=2, ve_layers=[2,3],      ve_mode="word",         engram_mode="none")),
        ("engram_full",          dict(width=32, heads=2, ve_layers=[],         ve_mode="none",         engram_mode="full")),
        ("hard_stack_decoupled", dict(width=32, heads=2, ve_layers=[2,3],      ve_mode="hard_stack",   engram_mode="none", tlr_p=0.03, pair_rows=24000)),
        ("bucket5_last2",        dict(width=32, heads=2, ve_layers=[2,3],      ve_mode="bucket5_gate", engram_mode="none", tlr_p=0.03, pair_rows=24000)),
        ("mp_ve_last2_dense",    dict(width=32, heads=2, ve_layers=[2,3],      ve_mode="mp_ve",        engram_mode="none", tlr_p=0.03)),
        ("mp_ve_all4_dense",     dict(width=32, heads=2, ve_layers=[0,1,2,3],  ve_mode="mp_ve",        engram_mode="none", tlr_p=0.03)),
    ]
    five_seed = {}
    for name, cfg in extend_cfgs:
        raw_3 = bd[name]["raw"]
        extra_2 = [rls.train_and_eval(data, seed=sd, layers=4, steps=300, dev="cuda", **cfg) for sd in (3, 4)]
        five_seed[name] = rls.summarize(raw_3 + extra_2)
        s = five_seed[name]
        print(f"  [5-seed] {name:21s} | known={s['mean']:.4f}±{s['std']:.4f}")

    # Also 5-seed mp_ve_exact_hi_hash_mid
    ehm_3 = res.get("step4_ablation_and_lr", {}).get("disentangle", {}).get("mp_ve_exact_hi_hash_mid", {}).get("raw", [])
    if len(ehm_3) == 3:
        ehm_2 = [train_eval_flex(data, [2, 3], "mp_ve_exact_hi_hash_mid", seed=sd, width=32, layers=4, heads=2, steps=300, tlr_p=0.03, pair_rows=24000, dev="cuda") for sd in (3, 4)]
        five_seed["mp_ve_exact_hi_hash_mid"] = rls.summarize(ehm_3 + ehm_2)
        s = five_seed["mp_ve_exact_hi_hash_mid"]
        print(f"  [5-seed] {'mp_ve_exact_hi_hash_mid':21s} | known={s['mean']:.4f}±{s['std']:.4f}")

    hs5, wl5 = five_seed["hard_stack_decoupled"], five_seed["word_last2"]
    print(f"  [5-seed Paired vs word_last2] hard_stack: {rls.paired_str(wl5, hs5)}")
    for nm in ("bucket5_last2", "mp_ve_last2_dense", "mp_ve_all4_dense", "mp_ve_exact_hi_hash_mid"):
        if nm in five_seed:
            print(f"  [5-seed Paired vs hard_stack] {nm:24s}: {rls.paired_str(hs5, five_seed[nm])}")

    print("\n--- SparseRowAdam Mechanism Disentanglement (3 seeds) ---")
    sp_global = rls.summarize([
        train_eval_flex(data, [2, 3], "mp_ve", seed=sd, width=32, layers=4, heads=2, steps=300, tlr_p=0.03, sparse_mode="both_frozen", dev="cuda")
        for sd in range(3)
    ])
    sp_dm = rls.summarize([
        train_eval_flex(data, [2, 3], "mp_ve", seed=sd, width=32, layers=4, heads=2, steps=300, tlr_p=0.03, sparse_mode="decay_m_only", dev="cuda")
        for sd in range(3)
    ])
    sp_dv = rls.summarize([
        train_eval_flex(data, [2, 3], "mp_ve", seed=sd, width=32, layers=4, heads=2, steps=300, tlr_p=0.03, sparse_mode="decay_v_only", dev="cuda")
        for sd in range(3)
    ])
    print(f"  SparseRowAdam (per-row t, both frozen): {bd['mp_ve_last2_sparse']['mean']:.4f}±{bd['mp_ve_last2_sparse']['std']:.4f}")
    print(f"  SparseRowAdam (global  t, both frozen): {sp_global['mean']:.4f}±{sp_global['std']:.4f}")
    print(f"  SparseRowAdam (decay m only, freeze v): {sp_dm['mean']:.4f}±{sp_dm['std']:.4f}")
    print(f"  SparseRowAdam (freeze m, decay v only): {sp_dv['mean']:.4f}±{sp_dv['std']:.4f}")
    print(f"  Dense Adam    (both decay, Block D ref): {bd['mp_ve_last2_dense']['mean']:.4f}±{bd['mp_ve_last2_dense']['std']:.4f}")

    # Zero-GPU lexical breakdown of 100-999 vs 1000+ bucket (why b5 gate dips slightly at 1000+)
    counts = np.load(os.path.join(rls.DATA, "counts.npy")).astype(np.float64)
    keep = np.unique(np.concatenate([np.argsort(counts)[-(6000 - 2):], [rls.EOT]]))
    V = len(keep) + 1
    dec = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        dec = lambda tid: tok.decode([int(tid)])
    except Exception:
        dec = lambda tid: f"id_{int(tid)}"
    tp = data["rows"][:, :-1].reshape(-1).astype(np.int64) * (V + 1) + data["rows"][:, 1:].reshape(-1).astype(np.int64)
    u_pairs, c_pairs = np.unique(tp, return_counts=True)
    order = np.argsort(-c_pairs)
    top_1000 = []
    for idx in order[:20]:
        key = int(u_pairs[idx])
        cnt = int(c_pairs[idx])
        w1, w2 = key // (V + 1), key % (V + 1)
        t1 = "<UNK>" if w1 >= len(keep) else ("<EOT>" if int(keep[w1]) == rls.EOT else dec(keep[w1]))
        t2 = "<UNK>" if w2 >= len(keep) else ("<EOT>" if int(keep[w2]) == rls.EOT else dec(keep[w2]))
        top_1000.append({"pair": [t1, t2], "count": cnt})
    ge1000_unk_frac = float(np.mean([(int(u_pairs[idx]) // (V + 1) >= len(keep)) or (int(u_pairs[idx]) % (V + 1) >= len(keep))
                                     for idx in np.where(c_pairs >= 1000)[0]])) if (c_pairs >= 1000).any() else 0.0
    ge100_mid_unk_frac = float(np.mean([(int(u_pairs[idx]) // (V + 1) >= len(keep)) or (int(u_pairs[idx]) % (V + 1) >= len(keep))
                                        for idx in np.where((c_pairs >= 100) & (c_pairs < 1000))[0]]))
    print(f"  1000+ bucket count={int((c_pairs>=1000).sum())}, UNK-involving pair frac={ge1000_unk_frac:.3f} vs (100-999 UNK frac={ge100_mid_unk_frac:.3f})")
    print("  Top 10 pairs in 1000+:", [(x["pair"], x["count"]) for x in top_1000[:10]])

    res["step5_5seed_and_sparse"] = {
        "five_seed": five_seed,
        "sparse_global_t": sp_global,
        "sparse_decay_m_only": sp_dm,
        "sparse_decay_v_only": sp_dv,
        "top_1000_pairs": top_1000,
        "unk_frac_1000p": ge1000_unk_frac,
        "unk_frac_100_999": ge100_mid_unk_frac,
    }
    save_p0(res)


def load_data_prefix(V_arg, steps, batch, prefix_frac, L=64, nval=1000):
    base = rls.load_data(V_arg, steps, batch, L=L, nval=nval)
    rows = base["rows"]
    val = base["val"]
    V = base["V"]
    n_pref = max(1, int(len(rows) * prefix_frac))
    pref_rows = rows[:n_pref]
    tp = pref_rows[:, :-1].reshape(-1).astype(np.int64) * (V + 1) + pref_rows[:, 1:].reshape(-1).astype(np.int64)
    u_pairs, c_pairs = np.unique(tp, return_counts=True)

    hi_thresh = max(1, int(round(100 * prefix_frac)))
    mid_thresh = max(1, int(round(10 * prefix_frac)))
    high_pairs = u_pairs[c_pairs >= hi_thresh]
    mid_pairs = u_pairs[(c_pairs >= mid_thresh) & (c_pairs < hi_thresh)]

    def get_tier_ids(arr, tier_keys):
        if len(tier_keys) == 0:
            return np.zeros_like(arr, dtype=np.int64)
        p = arr[:, :-1].astype(np.int64) * (V + 1) + arr[:, 1:].astype(np.int64)
        ix = np.clip(np.searchsorted(tier_keys, p), 0, len(tier_keys) - 1)
        out = np.zeros_like(arr, dtype=np.int64)
        out[:, 1:] = np.where(tier_keys[ix] == p, ix + 1, 0)
        return out

    d = dict(base)
    d["tr_hi"] = get_tier_ids(rows, high_pairs)
    d["val_hi"] = get_tier_ids(val, high_pairs)
    d["n_hi"] = len(high_pairs)
    d["tr_mid"] = get_tier_ids(rows, mid_pairs)
    d["val_mid"] = get_tier_ids(val, mid_pairs)
    d["n_mid"] = len(mid_pairs)
    return d, hi_thresh, mid_thresh


def run_step6(res):
    print("\n================ STEP 6: Prefix Counting vs Oracle + Standard AdamW Backbone (3 seeds) ================")
    data_oracle = rls.load_data(6000, 300, 128)
    pref_res = {}
    oracle_runs = [train_eval_flex(data_oracle, [2, 3], "mp_ve", seed=sd, width=32, layers=4, heads=2, steps=300, tlr_p=0.03, dev="cuda") for sd in range(3)]
    pref_res["oracle_100pct"] = rls.summarize(oracle_runs)
    for frac, tag in [(0.2, "prefix_20pct"), (0.1, "prefix_10pct")]:
        d_pref, h_th, m_th = load_data_prefix(6000, 300, 128, frac)
        runs = [train_eval_flex(d_pref, [2, 3], "mp_ve", seed=sd, width=32, layers=4, heads=2, steps=300, tlr_p=0.03, dev="cuda") for sd in range(3)]
        s = rls.summarize(runs)
        s["n_hi"] = d_pref["n_hi"]
        s["n_mid"] = d_pref["n_mid"]
        s["thresholds"] = [m_th, h_th]
        pref_res[tag] = s
        print(f"  {tag:13s} (hi>={h_th:2d} [{d_pref['n_hi']:,d}], mid>={m_th:2d} [{d_pref['n_mid']:,d}]) | known={s['mean']:.4f}±{s['std']:.4f} | vs oracle: {rls.paired_str(pref_res['oracle_100pct'], s)}")

    print("\n--- Standard AdamW Backbone (no Muon, L=4, w=32, V=6,000, 300s, 3 seeds) ---")
    adamw_res = {}
    for arm_nm, ve_l, var_nm, pr in [
        ("plain_w32",            [],     "none",                    24000),
        ("word_last2",           [2, 3], "word",                    24000),
        ("hard_stack_decoupled", [2, 3], "hard_stack",              24000),
        ("mp_ve_last2_dense",    [2, 3], "mp_ve",                   24000),
        ("mp_ve_ehm",            [2, 3], "mp_ve_exact_hi_hash_mid", 24000),
    ]:
        runs = [train_eval_flex(data_oracle, ve_l, var_nm, seed=sd, width=32, layers=4, heads=2, steps=300,
                                elr=0.1, tlr_w=0.1, tlr_p=0.03, pair_rows=pr, backbone_opt="adamw", adamw_lr=5e-3, dev="cuda") for sd in range(3)]
        s = rls.summarize(runs)
        adamw_res[arm_nm] = s
        print(f"  [AdamW] {arm_nm:22s} | known={s['mean']:.4f}±{s['std']:.4f} | pc0={s['buckets']['pc_0']:.3f} pc100={s['buckets']['pc_100']:.3f}")
    print(f"  [AdamW Paired] hs vs word:   {rls.paired_str(adamw_res['word_last2'], adamw_res['hard_stack_decoupled'])}")
    print(f"  [AdamW Paired] mpve vs hs:   {rls.paired_str(adamw_res['hard_stack_decoupled'], adamw_res['mp_ve_last2_dense'])}")
    print(f"  [AdamW Paired] mpve vs word: {rls.paired_str(adamw_res['word_last2'], adamw_res['mp_ve_last2_dense'])}")

    res["step6_prefix_and_adamw"] = {"prefix": pref_res, "adamw": adamw_res}
    save_p0(res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", default="all", choices=["all", "step2", "step3", "step4", "step5", "step6"])
    args = p.parse_args()
    res = load_p0()
    if args.step in ("all", "step2"):
        run_step2(res)
    if args.step in ("all", "step3"):
        run_step3(res)
    if args.step in ("all", "step4"):
        run_step4(res)
    if args.step in ("all", "step5"):
        run_step5(res)
    if args.step in ("all", "step6"):
        run_step6(res)
    print("\nRequested P0 blocks complete! Saved to", OUT_PATH)


if __name__ == "__main__":
    main()
