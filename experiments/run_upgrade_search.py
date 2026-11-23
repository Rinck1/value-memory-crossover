"""Controlled test of 4 principled upgrades to amplify MP-VE's gain (Block D: V=6000, L=4, w=32, 300s, 3 seeds):
  U1: Shared-Bank All-Layer Injection (1 shared table bank -> all 4 layers via adapters)
  U2: Context-Modulated Gated Dual Injection (V + Residual x) + Pre-Head Memory Readout
  U3: Hierarchical 1+2+3-Gram Maturity Routing (add exact mature 3-gram tier c3 >= 30)
  U4: Information-Weighted Maturity Routing (rank pairs by c * KL(P(next|prev,cur) || P(next|cur)))
Plus strict matched-adapter controls: CTRL_word_d64_dual_prehead & CTRL_hs_d64_dual_prehead.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)
import run_local_suite as rls


def prepare_rich_data(V_arg=6000, steps=300, batch=128, L=64, c3_thresh=30):
    base = rls.load_data(V_arg, steps, batch, L=L)
    rows, val, V = base["rows"], base["val"], base["V"]

    # Encode 3-grams into int64 with base B = V + 1 (V <= 6000 -> B^3 < 2.2e11 << 2^63)
    B = V + 1
    tr3 = (rows[:, :-2].astype(np.int64) * B + rows[:, 1:-1].astype(np.int64)) * B + rows[:, 2:].astype(np.int64)
    u3, c3 = np.unique(tr3.reshape(-1), return_counts=True)
    hi3_keys = u3[c3 >= c3_thresh]

    def get_3g_ids(arr):
        if len(hi3_keys) == 0:
            return np.zeros_like(arr, dtype=np.int64)
        k3 = (arr[:, :-2].astype(np.int64) * B + arr[:, 1:-1].astype(np.int64)) * B + arr[:, 2:].astype(np.int64)
        ix = np.clip(np.searchsorted(hi3_keys, k3), 0, len(hi3_keys) - 1)
        out = np.zeros_like(arr, dtype=np.int64)
        out[:, 2:] = np.where(hi3_keys[ix] == k3, ix + 1, 0)
        return out

    # U4: Information-weighted pair priority S(w1,w2) = sum_{w3: c3>=2} c3 * max(0, log P(w3|w1,w2)/P(w3|w2))
    w1 = rows[:, :-2].reshape(-1).astype(np.int64)
    w2 = rows[:, 1:-1].reshape(-1).astype(np.int64)
    w3 = rows[:, 2:].reshape(-1).astype(np.int64)
    p12 = w1 * B + w2
    p23 = w2 * B + w3

    c_w2 = np.bincount(w2, minlength=B).astype(np.float64)
    u23, c23 = np.unique(p23, return_counts=True)
    u12, c12 = np.unique(p12, return_counts=True)

    uw3 = u3 % B
    rem = u3 // B
    uw2 = rem % B
    uw1 = rem // B
    up12 = uw1 * B + uw2
    up23 = uw2 * B + uw3

    ix12 = np.searchsorted(u12, up12)
    ix23 = np.searchsorted(u23, up23)
    alpha = 1.0
    p_w3_given_12 = c3 / (c12[ix12] + alpha)
    p_w3_given_2 = (c23[ix23] + 1e-3) / (c_w2[uw2] + 1.0)
    contrib = np.where(
        c3 >= 2,
        c3 * np.maximum(0.0, np.log(np.maximum(p_w3_given_12, 1e-9) / np.maximum(p_w3_given_2, 1e-9))),
        0.0,
    )
    score_12 = np.zeros(len(u12), dtype=np.float64)
    np.add.at(score_12, ix12, contrib)

    # Same budget (n_hi, n_mid) as standard count-based MP-VE, among pairs with c >= 5
    n_hi_ref = base["n_hi"]
    n_mid_ref = base["n_mid"]
    elig = np.where(c12 >= 5)[0]
    order_elig = elig[np.argsort(-score_12[elig])]
    hi_kl_keys = np.sort(u12[order_elig[:n_hi_ref]])
    mid_kl_keys = np.sort(u12[order_elig[n_hi_ref:n_hi_ref + n_mid_ref]])

    def get_pair_ids(arr, keys):
        if len(keys) == 0:
            return np.zeros_like(arr, dtype=np.int64)
        p = arr[:, :-1].astype(np.int64) * B + arr[:, 1:].astype(np.int64)
        ix = np.clip(np.searchsorted(keys, p), 0, len(keys) - 1)
        out = np.zeros_like(arr, dtype=np.int64)
        out[:, 1:] = np.where(keys[ix] == p, ix + 1, 0)
        return out

    d = dict(base)
    d["tr_3g"] = get_3g_ids(rows)
    d["val_3g"] = get_3g_ids(val)
    d["n_3g"] = len(hi3_keys)
    d["tr_hi_kl"] = get_pair_ids(rows, hi_kl_keys)
    d["val_hi_kl"] = get_pair_ids(val, hi_kl_keys)
    d["tr_mid_kl"] = get_pair_ids(rows, mid_kl_keys)
    d["val_mid_kl"] = get_pair_ids(val, mid_kl_keys)
    print(f"Rich data ready: n_hi={n_hi_ref:,}, n_mid={n_mid_ref:,}, n_3g(c>={c3_thresh})={len(hi3_keys):,}")
    return d


class SharedMemoryBank(nn.Module):
    """Memory bank supporting 'mp_ve', 'word_only', and 'hard_stack' modes."""
    def __init__(self, V, d_mem, n_hi, n_mid, n_3g=0, use_3g=False, bank_mode="mp_ve", pair_rows=24000):
        super().__init__()
        self.V = V
        self.d_mem = d_mem
        self.bank_mode = bank_mode
        self.pair_rows = pair_rows

        self.ve_word = nn.Parameter(torch.randn(V, d_mem) * d_mem ** -0.5)
        self.ve_hi = self.ve_mid_a = self.ve_mid_p = self.ve_3g = self.ve_pair = None

        if bank_mode == "mp_ve":
            self.ve_hi = nn.Parameter(torch.randn(n_hi + 1, d_mem) * d_mem ** -0.5)
            with torch.no_grad():
                self.ve_hi[0].zero_()
            r = max(4, d_mem // 4)
            self.ve_mid_a = nn.Parameter(torch.randn(n_mid + 1, r) * r ** -0.5)
            with torch.no_grad():
                self.ve_mid_a[0].zero_()
            self.ve_mid_p = nn.Parameter(torch.randn(r, d_mem) * (r * d_mem) ** -0.5)
            if use_3g and n_3g > 0:
                self.ve_3g = nn.Parameter(torch.randn(n_3g + 1, d_mem) * d_mem ** -0.5)
                with torch.no_grad():
                    self.ve_3g[0].zero_()
        elif bank_mode == "hard_stack":
            self.ve_pair = nn.Parameter(torch.randn(pair_rows, d_mem) * d_mem ** -0.5)
        elif bank_mode != "word_only":
            raise ValueError(f"Unknown bank_mode: {bank_mode}")

    def lookup(self, ids, hi_id, mid_id, g3_id=None):
        w_vec = self.ve_word[ids]
        if self.bank_mode == "word_only":
            return w_vec, None
        if self.bank_mode == "hard_stack":
            p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
            h_idx = (p1 * 1000003 + ids) % self.pair_rows
            return w_vec, self.ve_pair[h_idx]
        p_vec = self.ve_hi[hi_id] * (hi_id > 0).unsqueeze(-1).float()
        p_vec = p_vec + (self.ve_mid_a[mid_id] @ self.ve_mid_p) * (mid_id > 0).unsqueeze(-1).float()
        if self.ve_3g is not None and g3_id is not None:
            p_vec = p_vec + self.ve_3g[g3_id] * (g3_id > 0).unsqueeze(-1).float()
        return w_vec, p_vec


class UpgradedBlock(nn.Module):
    def __init__(self, d, H, V, active_ve, d_mem, shared_bank=False, ctx_gate=False, inject_res=False,
                 n_hi=0, n_mid=0, n_3g=0, use_3g=False, bank_mode="mp_ve", pair_rows=24000):
        super().__init__()
        self.H, self.dk, self.V = H, d // H, V
        self.active_ve = active_ve
        self.shared_bank = shared_bank
        self.ctx_gate = ctx_gate
        self.inject_res = inject_res
        self.bank_mode = bank_mode

        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qn, self.kn = nn.RMSNorm(d // H), nn.RMSNorm(d // H)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        f = max(8, int(8 * d / 3) // 8 * 8)
        self.gate, self.up, self.down = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

        self.local_bank = None
        self.proj_w_v = self.proj_p_v = self.gate_v = None
        self.proj_res = self.gate_res = None
        self.lam_w = nn.Parameter(torch.tensor(1.0))
        self.lam_p = nn.Parameter(torch.tensor(1.0)) if bank_mode != "word_only" else None

        if active_ve:
            if not shared_bank:
                self.local_bank = SharedMemoryBank(V, d, n_hi, n_mid, n_3g=n_3g, use_3g=use_3g,
                                                   bank_mode=bank_mode, pair_rows=pair_rows)
            else:
                self.proj_w_v = nn.Linear(d_mem, d, bias=False)
                if bank_mode != "word_only":
                    self.proj_p_v = nn.Linear(d_mem, d, bias=False)
            if ctx_gate:
                self.gate_v = nn.Linear(d, d, bias=True)
                nn.init.zeros_(self.gate_v.weight)
                nn.init.zeros_(self.gate_v.bias)  # 2 * sigmoid(0) = 1.0 at init
            if inject_res:
                self.proj_res = nn.Linear(d_mem if shared_bank else d, d, bias=False)
                self.gate_res = nn.Linear(d, d, bias=True)
                nn.init.zeros_(self.gate_res.weight)
                nn.init.constant_(self.gate_res.bias, -1.0)

    def forward(self, x, ids, hi_id, mid_id, g3_id, shared_vecs=None):
        b, t, d = x.shape
        h = self.n1(x)
        sp = lambda z: z.view(b, t, self.H, self.dk).transpose(1, 2)
        v = self.v(h)

        mem_for_res = None
        if self.active_ve:
            if self.shared_bank:
                w_raw, p_raw = shared_vecs
                wv = self.proj_w_v(w_raw)
                if p_raw is not None:
                    pv = self.proj_p_v(p_raw)
                    ve_sum = self.lam_w * wv + self.lam_p * pv
                    mem_for_res = w_raw + p_raw
                else:
                    ve_sum = self.lam_w * wv
                    mem_for_res = w_raw
            else:
                wv, pv = self.local_bank.lookup(ids, hi_id, mid_id, g3_id)
                if pv is not None:
                    ve_sum = self.lam_w * wv + self.lam_p * pv
                    mem_for_res = wv + pv
                else:
                    ve_sum = self.lam_w * wv
                    mem_for_res = wv

            if self.ctx_gate:
                g_v = 2.0 * torch.sigmoid(self.gate_v(h))
                v = v + g_v * ve_sum
            else:
                v = v + ve_sum

        q, k = rls.rope(self.qn(sp(self.q(h)))), rls.rope(self.kn(sp(self.k(h))))
        y = F.scaled_dot_product_attention(q, k, sp(v), is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(b, t, d))

        if self.active_ve and self.inject_res:
            h_mid = self.n2(x)
            g_r = torch.sigmoid(self.gate_res(h_mid))
            x = x + g_r * self.proj_res(mem_for_res)

        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class UpgradedModel(nn.Module):
    def __init__(self, V, d, layers, heads, ve_layers, d_mem, shared_bank, ctx_gate, inject_res,
                 n_hi, n_mid, n_3g, use_3g, bank_mode="mp_ve", pair_rows=24000, pre_head_mem=False):
        super().__init__()
        self.V, self.d = V, d
        self.escale = 4.0
        self.shared_bank_flag = shared_bank
        self.pre_head_mem = pre_head_mem
        self.emb = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        self.head = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        self.bank = (SharedMemoryBank(V, d_mem, n_hi, n_mid, n_3g=n_3g, use_3g=use_3g,
                                      bank_mode=bank_mode, pair_rows=pair_rows)
                     if shared_bank else None)
        self.blocks = nn.ModuleList(
            UpgradedBlock(d, heads, V, i in ve_layers, d_mem, shared_bank, ctx_gate, inject_res,
                          n_hi, n_mid, n_3g, use_3g, bank_mode=bank_mode, pair_rows=pair_rows)
            for i in range(layers)
        )
        self.norm = nn.RMSNorm(d)
        if pre_head_mem:
            assert shared_bank
            self.head_proj = nn.Linear(d_mem, d, bias=False)
            self.head_gate = nn.Linear(d, d, bias=True)
            nn.init.zeros_(self.head_gate.weight)
            nn.init.constant_(self.head_gate.bias, -1.0)

    def forward(self, ids, hi_id, mid_id, g3_id):
        x = self.emb[ids] * self.escale
        shared_vecs = self.bank.lookup(ids, hi_id, mid_id, g3_id) if self.shared_bank_flag else None
        for blk in self.blocks:
            x = blk(x, ids, hi_id, mid_id, g3_id, shared_vecs=shared_vecs)
        h = self.norm(x)
        if self.pre_head_mem:
            w_raw, p_raw = shared_vecs
            mem = w_raw if p_raw is None else (w_raw + p_raw)
            g_h = torch.sigmoid(self.head_gate(h))
            h = h + g_h * self.head_proj(mem)
        return F.linear(h, self.head)


def train_eval_upg(data, ve_layers, d_mem=32, shared_bank=False, ctx_gate=False, inject_res=False,
                   use_3g=False, use_kl=False, bank_mode="mp_ve", pair_rows=24000, pre_head_mem=False,
                   seed=0, width=32, layers=4, heads=2, steps=300, batch=128,
                   elr=0.1, tlr_w=0.1, tlr_p=0.03, lr=0.02, measure_latency=False, dev="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    DEV = torch.device(dev)
    V = data["V"]
    m = UpgradedModel(
        V, width, layers, heads, set(ve_layers), d_mem, shared_bank, ctx_gate, inject_res,
        data["n_hi"], data["n_mid"], data["n_3g"], use_3g,
        bank_mode=bank_mode, pair_rows=pair_rows, pre_head_mem=pre_head_mem,
    ).to(DEV)

    word_tabs, pair_tabs = [], []
    banks = ([m.bank] if m.bank is not None else []) + [b.local_bank for b in m.blocks if b.local_bank is not None]
    for bk in banks:
        word_tabs.append(bk.ve_word)
        pair_tabs.extend([z for z in (bk.ve_hi, bk.ve_mid_a, bk.ve_mid_p, bk.ve_3g, bk.ve_pair) if z is not None])
    all_tabs = set(id(w) for w in word_tabs + pair_tabs)
    mats = [
        w for n, w in m.named_parameters()
        if w.ndim == 2 and (n.startswith("blocks") or n.startswith("head_proj") or n.startswith("head_gate"))
        and id(w) not in all_tabs
    ]
    mat_ids = set(id(w) for w in mats)
    rest = [w for w in m.parameters() if w.requires_grad and id(w) not in all_tabs and id(w) not in mat_ids]

    opts = [rls.Muon(mats, lr), torch.optim.Adam([dict(params=rest, lr=elr)], betas=(0.9, 0.95))]
    tgroups = []
    if word_tabs:
        tgroups.append(dict(params=word_tabs, lr=tlr_w))
    if pair_tabs:
        tgroups.append(dict(params=pair_tabs, lr=tlr_p))
    if tgroups:
        opts.append(torch.optim.Adam(tgroups, betas=(0.9, 0.95)))
    base = [[g["lr"] for g in o.param_groups] for o in opts]

    X = torch.tensor(data["rows"], device=DEV)
    X_hi = torch.tensor(data["tr_hi_kl" if use_kl else "tr_hi"], device=DEV)
    X_mid = torch.tensor(data["tr_mid_kl" if use_kl else "tr_mid"], device=DEV)
    X_3g = torch.tensor(data["tr_3g"], device=DEV)

    t0 = time.time()
    for step in range(steps):
        sc = min((step + 1) / min(50, max(1, steps // 10)), 1.0, (steps - step) / (0.3 * steps))
        for o, bl in zip(opts, base):
            for g, lr0 in zip(o.param_groups, bl):
                g["lr"] = lr0 * sc
        s0, e0 = step * batch, (step + 1) * batch
        xb = X[s0:e0]
        logits = m(xb[:, :-1], X_hi[s0:e0, :-1], X_mid[s0:e0, :-1], X_3g[s0:e0, :-1])
        loss = F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1))
        if torch.isnan(loss):
            raise RuntimeError(f"NaN loss encountered at step {step}!")
        loss.backward()
        for o in opts:
            o.step()
        m.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    train_s = time.time() - t0

    m.eval()
    VX = torch.tensor(data["val"], device=DEV)
    V_hi = torch.tensor(data["val_hi_kl" if use_kl else "val_hi"], device=DEV)
    V_mid = torch.tensor(data["val_mid_kl" if use_kl else "val_mid"], device=DEV)
    V_3g = torch.tensor(data["val_3g"], device=DEV)
    per = []
    with torch.no_grad():
        for i in range(0, len(VX), 200):
            xb = VX[i:i + 200]
            logits = m(xb[:, :-1], V_hi[i:i + 200, :-1], V_mid[i:i + 200, :-1], V_3g[i:i + 200, :-1])
            per.append(F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1), reduction="none"))
    per = torch.cat(per).float().cpu().numpy()

    vtg = data["bucket"][data["val"][:, 1:].reshape(-1)]
    pcnt_flat = data["val_pcnt"][:, :-1].reshape(-1)
    fm = np.zeros_like(data["val"][:, :-1], dtype=bool)
    fm[:, 0] = True
    ff = fm.reshape(-1)
    scores = {"val": float(per.mean()), "val_known": float(per[vtg < 3].mean())}
    for lo, hi_b in [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 10**12)]:
        sel = (~ff) & (pcnt_flat >= lo) & (pcnt_flat < hi_b) & (vtg < 3)
        if sel.sum() >= 50:
            scores[f"pc_{lo}"] = float(per[sel].mean())

    lat_ms = None
    if measure_latency:
        xb1, hi1, mid1, g31 = VX[:1, :-1], V_hi[:1, :-1], V_mid[:1, :-1], V_3g[:1, :-1]
        with torch.no_grad():
            for _ in range(10):
                _ = m(xb1, hi1, mid1, g31)
            torch.cuda.synchronize()
            tl0 = time.time()
            for _ in range(50):
                _ = m(xb1, hi1, mid1, g31)
            torch.cuda.synchronize()
            lat_ms = round((time.time() - tl0) / 50.0 * 1000.0, 3)

    nparam = sum(w.numel() for w in m.parameters())
    tab_param = sum(w.numel() for w in word_tabs + pair_tabs)
    return dict(
        params=nparam, tab_params=tab_param, dense_params=nparam - tab_param,
        train_s=round(train_s, 2), lat_ms=lat_ms,
        scores=scores, gate_diag=None, eng_b5=None, ve_b5=None,
    )


def main():
    data = prepare_rich_data(6000, 300, 128, c3_thresh=30)
    with open(os.path.join(RESULTS, "local_suite_results.json"), "r", encoding="utf-8") as f:
        suite = json.load(f)
    ref_plain = suite["block_D_ablations_L4"]["plain_w32_eqFLOP"]
    ref_word = suite["block_D_ablations_L4"]["word_last2"]
    ref_hs = suite["block_D_ablations_L4"]["hard_stack_decoupled"]
    ref_mpve = suite["block_D_ablations_L4"]["mp_ve_last2_dense"]

    print(
        f"Baselines (L=4, w=32, V=6000, 300s, 3 seeds): "
        f"plain={ref_plain['mean']:.4f} | word={ref_word['mean']:.4f} | "
        f"hs={ref_hs['mean']:.4f} | mp_ve={ref_mpve['mean']:.4f} (tab={ref_mpve['tab_params']:,})\n"
    )

    cfgs = [
        ("U1_shared_d32_all4",         dict(ve_layers=[0,1,2,3], d_mem=32, shared_bank=True,  ctx_gate=False, inject_res=False, use_3g=False, use_kl=False, bank_mode="mp_ve",      pre_head_mem=False)),
        ("U1_shared_d64_all4",         dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True,  ctx_gate=False, inject_res=False, use_3g=False, use_kl=False, bank_mode="mp_ve",      pre_head_mem=False)),
        ("U2_ctxgate_V_last2",         dict(ve_layers=[2,3],     d_mem=32, shared_bank=False, ctx_gate=True,  inject_res=False, use_3g=False, use_kl=False, bank_mode="mp_ve",      pre_head_mem=False)),
        ("U2_dual_V_Res_last2",        dict(ve_layers=[2,3],     d_mem=32, shared_bank=False, ctx_gate=True,  inject_res=True,  use_3g=False, use_kl=False, bank_mode="mp_ve",      pre_head_mem=False)),
        ("U3_3gram_last2",             dict(ve_layers=[2,3],     d_mem=32, shared_bank=False, ctx_gate=False, inject_res=False, use_3g=True,  use_kl=False, bank_mode="mp_ve",      pre_head_mem=False)),
        ("U4_info_KL_last2",           dict(ve_layers=[2,3],     d_mem=32, shared_bank=False, ctx_gate=False, inject_res=False, use_3g=False, use_kl=True,  bank_mode="mp_ve",      pre_head_mem=False)),
        ("U12_shared_d64_dual_all4",   dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True,  ctx_gate=True,  inject_res=True,  use_3g=False, use_kl=False, bank_mode="mp_ve",      pre_head_mem=False)),
        ("U123_shared_d64_dual_3g",    dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True,  ctx_gate=True,  inject_res=True,  use_3g=True,  use_kl=False, bank_mode="mp_ve",      pre_head_mem=False)),
        ("U123_plus_prehead_d64",      dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True,  ctx_gate=True,  inject_res=True,  use_3g=True,  use_kl=False, bank_mode="mp_ve",      pre_head_mem=True)),
        ("U1234_full_KL_prehead_d64",  dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True,  ctx_gate=True,  inject_res=True,  use_3g=True,  use_kl=True,  bank_mode="mp_ve",      pre_head_mem=True)),
        ("CTRL_word_d64_dual_prehead", dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True,  ctx_gate=True,  inject_res=True,  use_3g=False, use_kl=False, bank_mode="word_only",  pre_head_mem=True)),
        ("CTRL_hs_d64_dual_prehead",   dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True,  ctx_gate=True,  inject_res=True,  use_3g=False, use_kl=False, bank_mode="hard_stack", pre_head_mem=True, pair_rows=24000)),
    ]

    results = {}
    out_path = os.path.join(RESULTS, "upgrade_search_results.json")
    for name, cfg in cfgs:
        runs = [train_eval_upg(data, seed=sd, measure_latency=(sd == 0), **cfg) for sd in range(3)]
        s = rls.summarize(runs)
        s["dense_params"] = runs[0]["dense_params"]
        results[name] = s
        print(
            f"{name:27s} | known={s['mean']:.4f}±{s['std']:.4f} | dense={s['dense_params']:7,d} | "
            f"tab={s['tab_params']:9,d} | b1={s['lat_ms']:5.2f}ms | "
            f"vs mpve:{rls.paired_str(ref_mpve, s)} | vs hs:{rls.paired_str(ref_hs, s)}"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    ctrl_w = results["CTRL_word_d64_dual_prehead"]
    ctrl_h = results["CTRL_hs_d64_dual_prehead"]
    print("\n--- Paired vs Matched-Adapter Controls (CTRL_word / CTRL_hs) ---")
    for name in ("U12_shared_d64_dual_all4", "U123_shared_d64_dual_3g", "U123_plus_prehead_d64", "U1234_full_KL_prehead_d64"):
        s = results[name]
        print(f"  {name:26s} | vs CTRL_word: {rls.paired_str(ctrl_w, s)} | vs CTRL_hs: {rls.paired_str(ctrl_h, s)} | vs ref_word: {rls.paired_str(ref_word, s)}")
    print("\nSaved upgrade search results to", out_path)


if __name__ == "__main__":
    main()
