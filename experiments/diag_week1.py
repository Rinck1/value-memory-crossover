"""Week-1 Decisive Diagnostics for Paper 06 / MP-VE:
1. --mode diag_engram : Measure correlation between Engram's learned context gate and pair training frequency.
2. --mode gonogo      : 3-seed comparison of Plain, Word, Engram-full, Pair-VE, Hard-Stack (decoupled LR),
                        Straight-Through Maturity Gate, 5-Bucket Gate, and MP-VE (3-tier capacity allocation).
3. --mode beta2       : Test beta2 in {0.9, 0.95, 0.99, 0.999} to measure the true direction of c*(beta2).
"""
import argparse
import json
import math
import os
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


def load_data(V_arg, steps, batch, L=64, nval=1000):
    counts_path = os.path.join(DATA, "counts.npy")
    train_path = os.path.join(DATA, "fineweb_train_000001.bin")
    val_path = os.path.join(DATA, "fineweb_val_000000.bin")
    if not os.path.exists(counts_path):
        tr_raw = np.memmap(train_path, dtype=np.uint16, mode="r", offset=1024)
        cnts = np.bincount(np.asarray(tr_raw, dtype=np.int64), minlength=65536).astype(np.int64)
        np.save(counts_path, cnts)
    counts = np.load(counts_path).astype(np.float64)
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
    tr = np.memmap(train_path, dtype=np.uint16, mode="r", offset=1024)
    rows = look[np.asarray(tr[:need]).astype(np.int64).reshape(-1, L)]
    va = np.memmap(val_path, dtype=np.uint16, mode="r", offset=1024)
    val = look[np.stack([np.asarray(va[x:x + L]) for x in np.linspace(0, len(va) - L - 1, nval).astype(np.int64)]).astype(np.int64)]

    # Exact pair counts on the training slice
    tp = rows[:, :-1].reshape(-1).astype(np.int64) * (V + 1) + rows[:, 1:].reshape(-1)
    u_pairs, c_pairs = np.unique(tp, return_counts=True)

    # Pre-compute pair counts for every position in rows and val
    def get_pair_counts(arr):
        p = arr[:, :-1].astype(np.int64) * (V + 1) + arr[:, 1:].astype(np.int64)
        ix = np.clip(np.searchsorted(u_pairs, p), 0, len(u_pairs) - 1)
        cnt = np.zeros_like(arr, dtype=np.int64)
        cnt[:, 1:] = np.where(u_pairs[ix] == p, c_pairs[ix], 0)
        return cnt

    tr_pcnt = get_pair_counts(rows)
    val_pcnt = get_pair_counts(val)

    # For MP-VE exact tier tables: build exact ID maps for high (c>=100) and mid (10<=c<100) pairs
    high_pairs = u_pairs[c_pairs >= 100]
    mid_pairs = u_pairs[(c_pairs >= 10) & (c_pairs < 100)]

    def get_tier_ids(arr, tier_keys):
        if len(tier_keys) == 0:
            return np.zeros_like(arr, dtype=np.int64)
        p = arr[:, :-1].astype(np.int64) * (V + 1) + arr[:, 1:].astype(np.int64)
        ix = np.clip(np.searchsorted(tier_keys, p), 0, len(tier_keys) - 1)
        out = np.zeros_like(arr, dtype=np.int64)  # 0 = absent / null slot
        out[:, 1:] = np.where(tier_keys[ix] == p, ix + 1, 0)
        return out

    tr_hi = get_tier_ids(rows, high_pairs)
    val_hi = get_tier_ids(val, high_pairs)
    tr_mid = get_tier_ids(rows, mid_pairs)
    val_mid = get_tier_ids(val, mid_pairs)

    return dict(
        V=V, UNK=UNK, bucket=bucket, rows=rows, val=val,
        tr_pcnt=tr_pcnt, val_pcnt=val_pcnt,
        tr_hi=tr_hi, val_hi=val_hi, n_hi=len(high_pairs),
        tr_mid=tr_mid, val_mid=val_mid, n_mid=len(mid_pairs),
        n_unique_pairs=len(u_pairs),
    )


class Block(nn.Module):
    def __init__(self, d, H, V, arm, pair_rows, n_hi, n_mid, tau=100.0):
        super().__init__()
        self.H, self.dk, self.V, self.arm = H, d // H, V, arm
        self.tau = tau
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qn, self.kn = nn.RMSNorm(d // H), nn.RMSNorm(d // H)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        f = int(8 * d / 3) // 8 * 8
        self.gate, self.up, self.down = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

        self.ve_word = None
        self.ve_pair = None
        self.ve_hi = None
        self.ve_mid_a = None
        self.ve_mid_p = None
        self.b5_logits = None
        self.lam_w = nn.Parameter(torch.tensor(1.0))
        self.lam_p = nn.Parameter(torch.tensor(1.0))

        if arm in ("word", "hard_stack", "st_gate", "bucket5_gate", "mp_ve", "mp_ve_hash"):
            self.ve_word = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        if arm in ("pair_ve", "hard_stack", "st_gate", "bucket5_gate"):
            self.ve_pair = nn.Parameter(torch.randn(pair_rows, d) * d ** -0.5)
            self.pair_rows = pair_rows
        if arm == "bucket5_gate":
            # 5 interpretable bucket gates for 0, 1-9, 10-99, 100-999, 1000+
            self.b5_logits = nn.Parameter(torch.tensor([-4.0, -2.5, -1.0, 1.5, 2.5]))
        if arm == "mp_ve":
            # Exact 3-tier capacity allocation: high -> full d, mid -> d/4 + shared P, rare -> 0
            # Row 0 is zero-padded null slot
            self.ve_hi = nn.Parameter(torch.randn(n_hi + 1, d) * d ** -0.5)
            with torch.no_grad():
                self.ve_hi[0].zero_()
            r = max(2, d // 4)
            self.ve_mid_a = nn.Parameter(torch.randn(n_mid + 1, r) * r ** -0.5)
            with torch.no_grad():
                self.ve_mid_a[0].zero_()
            self.ve_mid_p = nn.Parameter(torch.randn(r, d) * (r * d) ** -0.5)
        if arm == "mp_ve_hash":
            # Hashed 3-tier capacity allocation (fixed memory budget, no exact dict at inference)
            self.hi_rows = max(256, n_hi)
            self.mid_rows = max(256, n_mid)
            self.ve_hi = nn.Parameter(torch.randn(self.hi_rows, d) * d ** -0.5)
            r = max(2, d // 4)
            self.ve_mid_a = nn.Parameter(torch.randn(self.mid_rows, r) * r ** -0.5)
            self.ve_mid_p = nn.Parameter(torch.randn(r, d) * (r * d) ** -0.5)

    def forward(self, x, ids, pcnt=None, hi_id=None, mid_id=None):
        b, t, d = x.shape
        h = self.n1(x)
        sp = lambda z: z.view(b, t, self.H, self.dk).transpose(1, 2)
        v = self.v(h)

        if self.ve_word is not None:
            v = v + self.lam_w * self.ve_word[ids]

        if self.ve_pair is not None:
            p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
            h_idx = (p1 * 1000003 + ids) % self.pair_rows
            tp = self.ve_pair[h_idx]
            if self.arm in ("pair_ve", "hard_stack"):
                v = v + self.lam_p * tp
            elif self.arm == "st_gate":
                # Straight-Through maturity gate: forward uses g(c), backward preserves full grad=1 on tp
                g = (pcnt.float() / (pcnt.float() + self.tau)).unsqueeze(-1)
                tp_gated = tp + (g - 1.0) * tp.detach()
                v = v + self.lam_p * tp_gated
            elif self.arm == "bucket5_gate":
                # 5-bucket gate: 0 / 1-9 / 10-99 / 100-999 / 1000+
                b_idx = torch.zeros_like(pcnt)
                b_idx = torch.where(pcnt >= 1, torch.ones_like(b_idx), b_idx)
                b_idx = torch.where(pcnt >= 10, torch.full_like(b_idx, 2), b_idx)
                b_idx = torch.where(pcnt >= 100, torch.full_like(b_idx, 3), b_idx)
                b_idx = torch.where(pcnt >= 1000, torch.full_like(b_idx, 4), b_idx)
                g = torch.sigmoid(self.b5_logits[b_idx]).unsqueeze(-1)
                # Straight-through for tp, normal grad for b5_logits
                tp_gated = g * tp.detach() + (tp - tp.detach())
                v = v + self.lam_p * tp_gated

        if self.arm == "mp_ve":
            hi_vec = self.ve_hi[hi_id] * (hi_id > 0).unsqueeze(-1).float()
            mid_vec = (self.ve_mid_a[mid_id] @ self.ve_mid_p) * (mid_id > 0).unsqueeze(-1).float()
            v = v + self.lam_p * (hi_vec + mid_vec)

        if self.arm == "mp_ve_hash":
            p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
            hi_mask = (pcnt >= 100).unsqueeze(-1).float()
            mid_mask = ((pcnt >= 10) & (pcnt < 100)).unsqueeze(-1).float()
            h1 = (p1 * 1000003 + ids) % self.hi_rows
            h2 = (p1 * 998244353 + ids * 7919) % self.mid_rows
            hi_vec = self.ve_hi[h1] * hi_mask
            mid_vec = (self.ve_mid_a[h2] @ self.ve_mid_p) * mid_mask
            v = v + self.lam_p * (hi_vec + mid_vec)

        q, k = rope(self.qn(sp(self.q(h)))), rope(self.kn(sp(self.k(h))))
        y = F.scaled_dot_product_attention(q, k, sp(v), is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(b, t, d))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Model(nn.Module):
    def __init__(self, V, d, layers, heads, arm, pair_rows, n_hi, n_mid, tau=100.0):
        super().__init__()
        self.V, self.d, self.arm = V, d, arm
        self.escale = 4.0
        self.head = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        self.emb = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        self.blocks = nn.ModuleList(
            Block(d, heads, V, arm, pair_rows, n_hi, n_mid, tau=tau) for _ in range(layers)
        )
        self.norm = nn.RMSNorm(d)

        # Engram core / full design (exact match to pw.py lines 168-210)
        self.ng_n = pair_rows
        self.ng_head = 4
        self.ng2 = self.ng3 = self.ngk = self.ngv = self.ngc = None
        if arm == "engram_core":
            self.ng2 = nn.Parameter(torch.randn(pair_rows, d) * d ** -0.5)
            self.ng3 = nn.Parameter(torch.randn(pair_rows, d) * d ** -0.5)
        elif arm == "engram_full":
            K = self.ng_head
            self.ng2 = nn.Parameter(torch.randn(K, pair_rows, d // K) * d ** -0.5)
            self.ng3 = nn.Parameter(torch.randn(K, pair_rows, d // K) * d ** -0.5)
            self.ngk = nn.Parameter(torch.randn(d, 2 * (d // K) * K) * (2 * d) ** -0.5)
            self.ngv = nn.Parameter(torch.randn(d, 2 * (d // K) * K) * (2 * d) ** -0.5)
            self.ngc = nn.Parameter(torch.zeros(d, 1, 4))
        self.last_engram_gates = None

    def forward(self, ids, pcnt=None, hi_id=None, mid_id=None, capture_gate=False):
        x = self.emb[ids] * self.escale
        for i, blk in enumerate(self.blocks):
            if i == 0 and self.arm == "engram_full":
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
                gate = torch.sigmoid(
                    (F.rms_norm(x, (x.shape[-1],)) * F.rms_norm(kk, (kk.shape[-1],))).sum(-1, keepdim=True)
                    / x.shape[-1] ** 0.5
                )
                if capture_gate:
                    self.last_engram_gates = gate.squeeze(-1).detach()
                v = gate * vv
                c = F.conv1d(F.pad(F.rms_norm(v, (v.shape[-1],)).transpose(1, 2), (9, 0)), self.ngc, dilation=3, groups=v.shape[-1]).transpose(1, 2)
                x = x + (F.silu(c) + v) * self.escale
            elif i == 0 and self.arm == "engram_core":
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
            x = blk(x, ids, pcnt, hi_id, mid_id)
        return F.linear(self.norm(x), self.head)


def train_one(data, arm, seed=0, width=16, layers=2, heads=2, steps=150, batch=128,
              elr=0.1, tlr_w=0.1, tlr_p=0.03, lr=0.02, beta2=0.95, pair_rows=6000, tau=100.0, dev="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    DEV = torch.device(dev)
    V = data["V"]
    m = Model(V, width, layers, heads, arm, pair_rows, data["n_hi"], data["n_mid"], tau=tau).to(DEV)

    word_tabs = [b.ve_word for b in m.blocks if b.ve_word is not None]
    pair_tabs = [z for z in (m.ng2, m.ng3, m.ngk, m.ngv, m.ngc) if z is not None] + [
        z for b in m.blocks for z in (b.ve_pair, b.ve_hi, b.ve_mid_a, b.ve_mid_p) if z is not None
    ]
    all_tabs = set(id(w) for w in word_tabs + pair_tabs)
    mats = [w for n, w in m.named_parameters() if w.ndim == 2 and n.startswith("blocks") and id(w) not in all_tabs]
    mat_ids = set(id(w) for w in mats)
    rest = [w for w in m.parameters() if w.requires_grad and id(w) not in all_tabs and id(w) not in mat_ids]

    groups = [dict(params=rest, lr=elr)]
    if word_tabs:
        groups.append(dict(params=word_tabs, lr=tlr_w))
    if pair_tabs:
        groups.append(dict(params=pair_tabs, lr=tlr_p))

    opts = [Muon(mats, lr), torch.optim.Adam(groups, betas=(0.9, beta2))]
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

    # Evaluate
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
            logits = m(xb[:, :-1], V_pc[i:i + 200, :-1], V_hi[i:i + 200, :-1], V_mid[i:i + 200, :-1], capture_gate=True)
            per.append(F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1), reduction="none"))
            if m.last_engram_gates is not None:
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
        logc = np.log1p(cv.astype(np.float64))
        pearson = float(np.corrcoef(gv, logc)[0, 1])
        # Spearman via rank
        rg = np.argsort(np.argsort(gv)).astype(np.float64)
        rc = np.argsort(np.argsort(cv)).astype(np.float64)
        spearman = float(np.corrcoef(rg, rc)[0, 1])
        per_b = {}
        for lo, hi in zip(PB[:-1], PB[1:]):
            sel = (~first_flat) & (pcnt_flat >= lo) & (pcnt_flat < hi) & (vtg < 3)
            if sel.sum() >= 50:
                per_b[f"pc_{lo}"] = {"mean": round(float(g_all[sel].mean()), 4), "std": round(float(g_all[sel].std()), 4), "n": int(sel.sum())}
        gate_diag = {"pearson_logc": round(pearson, 4), "spearman": round(spearman, 4), "buckets": per_b}

    b5_learned = None
    if arm == "bucket5_gate":
        b5_learned = [[round(float(x), 3) for x in torch.sigmoid(b.b5_logits).detach().cpu()] for b in m.blocks]

    nparam = sum(w.numel() for w in m.parameters())
    tab_param = sum(w.numel() for w in word_tabs + pair_tabs)
    return dict(
        arm=arm, seed=seed, beta2=beta2, steps=steps, params=nparam, tab_params=tab_param,
        train_s=round(train_s, 2), scores=scores, gate_diag=gate_diag, b5_learned=b5_learned
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["diag_engram", "gonogo", "beta2", "all"], default="all")
    p.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--V", type=int, default=6000)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    out = {}

    if args.mode in ("diag_engram", "all"):
        print("\n=== DIAGNOSTIC 1: Engram Learned Context Gate vs Pair Training Sightings ===")
        out["diag_engram"] = []
        for st in (150, 300):
            data = load_data(args.V, st, 128)
            for arm in ("engram_core", "engram_full"):
                r = train_one(data, arm, seed=0, steps=st, elr=0.1, tlr_p=0.03, pair_rows=args.V, dev=args.dev)
                out["diag_engram"].append(r)
                gd = r["gate_diag"]
                b_str = " | ".join(f"{k}:{v['mean']:.3f}±{v['std']:.3f}" for k, v in gd["buckets"].items())
                print(f"[{arm} s={st}] known={r['scores']['val_known']:.4f} | r(gate,log1p(c))={gd['pearson_logc']:+.4f} spearman={gd['spearman']:+.4f} | {b_str}")

    if args.mode in ("gonogo", "all"):
        print("\n=== DIAGNOSTIC 2: 3-Seed Go/No-Go Method Comparison (V=6,000) ===")
        out["gonogo"] = {}
        for st in (150, 300):
            data = load_data(args.V, st, 128)
            print(f"\n--- Steps={st} (tokens={st*128*64:,}, unique_pairs={data['n_unique_pairs']:,}, n_hi={data['n_hi']:,}, n_mid={data['n_mid']:,}) ---")
            arms_cfg = [
                ("plain",        dict(elr=0.1, pair_rows=args.V)),
                ("word",         dict(elr=0.1, tlr_w=0.1, pair_rows=args.V)),
                ("engram_full",  dict(elr=0.1, tlr_p=0.03, pair_rows=args.V)),
                ("pair_ve_24k",  dict(arm_name="pair_ve", elr=0.1, tlr_p=0.03, pair_rows=24000)),
                ("hard_stack",   dict(elr=0.1, tlr_w=0.1, tlr_p=0.03, pair_rows=24000)),
                ("st_gate",      dict(elr=0.1, tlr_w=0.1, tlr_p=0.03, pair_rows=24000, tau=100.0)),
                ("bucket5_gate", dict(elr=0.1, tlr_w=0.1, tlr_p=0.03, pair_rows=24000)),
                ("mp_ve",        dict(elr=0.1, tlr_w=0.1, tlr_p=0.03, pair_rows=24000)),
                ("mp_ve_hash",   dict(elr=0.1, tlr_w=0.1, tlr_p=0.03, pair_rows=24000)),
            ]
            out["gonogo"][str(st)] = {}
            for label, cfg in arms_cfg:
                arm = cfg.pop("arm_name", label)
                runs = [train_one(data, arm, seed=sd, steps=st, dev=args.dev, **cfg) for sd in range(args.seeds)]
                out["gonogo"][str(st)][label] = runs
                vk = [r["scores"]["val_known"] for r in runs]
                b_means = {}
                for b in ("pc_0", "pc_1", "pc_10", "pc_100", "pc_1000"):
                    vals = [r["scores"][b] for r in runs if b in r["scores"]]
                    b_means[b] = float(np.mean(vals)) if vals else float("nan")
                b_str = " ".join(f"{b}:{b_means[b]:.3f}" for b in ("pc_0", "pc_1", "pc_10", "pc_100", "pc_1000"))
                extra = f" | b5={runs[0]['b5_learned'][0]}" if runs[0]["b5_learned"] else ""
                print(f"{label:13s} | known={np.mean(vk):.4f}±{np.std(vk):.4f} | tab_p={runs[0]['tab_params']:7d} | {b_str}{extra}")

    if args.mode in ("beta2", "all"):
        print("\n=== DIAGNOSTIC 3: beta2 Scaling Check (0.9 / 0.95 / 0.99 / 0.999) ===")
        data = load_data(args.V, 150, 128)
        out["beta2"] = {}
        for b2 in (0.9, 0.95, 0.99, 0.999):
            rw = train_one(data, "word", seed=0, steps=150, elr=0.1, tlr_w=0.1, beta2=b2, dev=args.dev)
            re = train_one(data, "engram_core", seed=0, steps=150, elr=0.1, tlr_p=0.03, beta2=b2, pair_rows=args.V, dev=args.dev)
            out["beta2"][str(b2)] = dict(word=rw, engram=re)
            diffs = {b: round(re["scores"][b] - rw["scores"][b], 4) for b in ("pc_0", "pc_1", "pc_10", "pc_100", "pc_1000")}
            print(f"beta2={b2:5.3f} | word={rw['scores']['val_known']:.4f} eng={re['scores']['val_known']:.4f} | (eng - word): {diffs}")

    save_path = os.path.join(RESULTS, "week1_diagnostics_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full diagnostic JSON to {save_path}")


if __name__ == "__main__":
    main()
