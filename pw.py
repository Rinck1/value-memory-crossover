"""Test bed: per-word weights (weights looked up by token ID) along long training, split by word frequency.

Model: pre-norm RMSNorm, RoPE, QK-norm, SwiGLU, no biases. Muon on the 2D block matrices, Adam on tables, norms, scalars.
Per-word weights tested:
  --tie 1            one shared table for input and output (the fewest per-word weights)
  --untie K / --ucover c   in a tied model, the top-K words (or the top words covering share c of the text) get their own
                     input row; --urand 1 picks K random words instead (control)
  --ve 0,1           value embeddings in those layers: v = W_V h + lam * T_l[word]
  --vek K / --vecover c    only the top-K words (or covering share c) get value-embedding rows; others get none;
                     --verand 1 picks K random words (control); --vefrozen 1 keeps the rows at their random init
Data: FineWeb (GPT-2 tokens) mapped to the top-V words plus one unknown token; each token read once.
Scores: validation loss, split into word buckets by share of text (common = the top words covering 50% of the kept text,
middle = 50-80%, rare = 80-100%, plus the unknown token), by the current input word and by the target word.
--log N records the scores every N steps (one run gives the whole curve). The final scores also include the
loss with the per-word part switched off at test time (val_off).

usage: nice -n 20 python3 pw.py --tie 1 --ucover 0.5 --steps 3000 --log 300 --threads 1 --tag name
Appends one JSON line to results.jsonl.
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from flops import flops_tok

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); EOT = 50256

p = argparse.ArgumentParser()
p.add_argument("--width", type=int, default=16); p.add_argument("--layers", type=int, default=2); p.add_argument("--heads", type=int, default=2)
p.add_argument("--V", type=int, default=1500); p.add_argument("--L", type=int, default=64); p.add_argument("--batch", type=int, default=32)
p.add_argument("--steps", type=int, default=3000); p.add_argument("--lr", type=float, default=0.02); p.add_argument("--elr", type=float, default=0.05)
p.add_argument("--tlr", type=float, default=-1, help="Adam rate of value-embedding tables and untied input rows (default: --elr)")
p.add_argument("--udelta", type=int, default=0, help="1: own rows are offsets added to the shared row (start at zero) instead of replacing it")
p.add_argument("--urank", type=int, default=0, help="tied model: input = shared row + A[w] @ B, rank r, A starts at zero (cheap full untie)")
p.add_argument("--umat", type=int, default=0, help="tied model: input = shared row @ M, one shared d x d matrix (starts at identity) for every word")
p.add_argument("--vehash", type=int, default=0, help="value embeddings: words without a full row share one of N hashed rows")
p.add_argument("--verank", type=int, default=0, help="value embeddings: words without a full row get a rank-r row A[w] @ P")
p.add_argument("--veunk", type=int, default=0, help="1: the unknown token always gets a full value row")
p.add_argument("--ngram", type=int, default=0, help="Engram-style baseline: hashed 2-gram and 3-gram tables of N rows each, gated by the hidden state, added to the residual before layer --nglayer")
p.add_argument("--nglayer", type=int, default=1)
p.add_argument("--ngfull", type=int, default=0, help="with --ngram N: the published Engram design: K hash heads per order (each N rows of width d/K, own hash), concatenated, key/value projections, gate sigmoid(rmsnorm(h).rmsnorm(k)/sqrt d), then RMSNorm + causal depthwise conv (kernel 4, dilation 3, starts at zero) + SiLU with a residual")
p.add_argument("--nghead", type=int, default=4)
p.add_argument("--vebig", type=int, default=0, help="value tables in the --ve layers indexed by a hashed bigram (previous, current word) into N rows instead of by the word")
p.add_argument("--veprev", type=int, default=0, help="1: value tables in the --ve layers indexed by the previous word instead of the current one")
p.add_argument("--probe", type=int, default=0, help="K>0: after training, linear readout accuracy of the current and the previous word (K most common known words) from the residual stream entering each layer")
p.add_argument("--vebigl", type=str, default="", help="with --vebig: only these layers (comma list) use bigram rows; the other --ve layers keep word rows")
p.add_argument("--inbig", type=int, default=0, help="Scone-style baseline: hashed bigram table of N rows added to the input vector")
p.add_argument("--veshare", type=int, default=0, help="1: the --ve layers share one value table (each layer keeps its own scale lam)")
p.add_argument("--bank", type=int, default=0, help="MoVE-style baseline: in the --ve layers, a shared bank of K value vectors mixed by a softmax router on the hidden state (no per-word rows)")
p.add_argument("--escale", type=float, default=4.0, help="multiply input vectors by this (tuned in trial-1)")
p.add_argument("--tie", type=int, default=0)
p.add_argument("--offeach", type=int, default=0, help="1: after training, switch off the value rows of one layer at a time at test time; loss rise on known next words per layer")
p.add_argument("--pcshare", type=int, default=0, help="1: print the share of scored positions in each pair-sighting bucket (data only, no training) and exit")
p.add_argument("--geom", type=int, default=0, help="1: after training, compare input rows with output rows per band: cosine of a word's two rows, R2 of one linear map input ~ output @ M, and top-10 neighbour overlap")
p.add_argument("--untie", type=int, default=0); p.add_argument("--ucover", type=float, default=0); p.add_argument("--urand", type=int, default=0)
p.add_argument("--uband", default="", help="lo,hi: untie the words whose text coverage starts in [lo, hi), e.g. 0.5,0.8 = middle words")
p.add_argument("--ve", default=""); p.add_argument("--vek", type=int, default=0); p.add_argument("--vecover", type=float, default=0)
p.add_argument("--veband", default="", help="lo,hi: value-embedding rows only for words in this coverage band");
p.add_argument("--verand", type=int, default=0); p.add_argument("--vefrozen", type=int, default=0); p.add_argument("--lam", type=float, default=1.0)
p.add_argument("--cool", type=float, default=0.3, help="share of steps in the final linear cooldown")
p.add_argument("--skip", type=int, default=0, help="start reading training tokens at this row (fresh data for a second run)")
p.add_argument("--log", type=int, default=0); p.add_argument("--nval", type=int, default=1000)
p.add_argument("--dev", default="cpu"); p.add_argument("--threads", type=int, default=1); p.add_argument("--tag", default=""); p.add_argument("--out", default="results.jsonl")
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0); DEV = torch.device(a.dev)
TLR = a.tlr if a.tlr > 0 else a.elr
VE = [int(x) for x in a.ve.split(",") if x != ""]

# ---------------- data ----------------
counts = np.load(os.path.join(DATA, "counts.npy")).astype(np.float64)
keep = np.unique(np.concatenate([np.argsort(counts)[-(a.V - 2):], [EOT]])); V = len(keep) + 1; UNK = V - 1
look = np.full(65536, UNK, np.int64); look[keep] = np.arange(len(keep))
freq = np.append(counts[keep], counts.sum() - counts[keep].sum())
order = np.argsort(-freq[:-1], kind="stable")                         # word ids by frequency, unknown left out
rank = np.empty(V, np.int64); rank[order] = np.arange(V - 1); rank[UNK] = V - 1
cover = np.cumsum(freq[order]) / freq[:-1].sum()                     # share of kept text covered by the top r+1 words
def topk_for(c): return int(np.searchsorted(cover, c)) + 1
cov_of = np.empty(V); cov_of[order] = cover - freq[order] / freq[:-1].sum(); cov_of[UNK] = 2.0   # share covered before this word
BUCK = [("common", 0.0, 0.5), ("middle", 0.5, 0.8), ("rare", 0.8, 1.01)]
bucket = np.full(V, 3); [bucket.__setitem__(cov_of >= lo, i) for i, (n, lo, hi) in enumerate(BUCK)]; bucket[UNK] = 3
NB = ["common", "middle", "rare", "unk"]
RB = [0, 10, 30, 100, 300, 1000, 3000, 10000, 10 ** 9]                 # finer bins by frequency rank of known words
rbin = np.searchsorted(RB, rank, side="right") - 1; rbin[UNK] = -1

need = a.steps * a.batch * a.L
tr = np.memmap(os.path.join(DATA, "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
off = a.skip * a.L
assert off + need <= len(tr), "not enough training tokens"
rows = look[np.asarray(tr[off:off + need]).astype(np.int64).reshape(-1, a.L)]
va = np.memmap(os.path.join(DATA, "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)
val = look[np.stack([np.asarray(va[x:x + a.L]) for x in np.linspace(0, len(va) - a.L - 1, a.nval).astype(np.int64)]).astype(np.int64)]


def pick(k, c, rnd, salt, band=""):
    if band:
        lo, hi = map(float, band.split(",")); ids = np.flatnonzero((cov_of >= lo) & (cov_of < hi)); k = len(ids)
    else:
        k = k or (topk_for(c) if c else 0); ids = order[:k]
    if not k: return None, 0
    if rnd: ids = np.random.default_rng(salt).choice(V - 1, k, replace=False)
    m = np.full(V, -1, np.int64); m[ids] = np.arange(k); return torch.tensor(m, device=DEV), k


UMAP, UK = pick(a.untie, a.ucover, a.urand, 2, a.uband) if a.tie else (None, 0)
VMAP, VK = pick(a.vek, a.vecover, a.verand, 3, a.veband) if VE else (None, 0)
CHEAP = bool(a.vehash or a.verank)
if VE and (CHEAP or (a.veunk and VMAP is not None)):            # full rows for a chosen set, cheap rows for the rest
    _m = VMAP.cpu().numpy().copy() if VMAP is not None else np.full(V, -1, np.int64)
    if a.veunk and _m[UNK] < 0: _m[UNK] = VK; VK += 1
    VMAP = torch.tensor(_m, device=DEV)
HID = torch.tensor(np.random.default_rng(4).integers(0, max(a.vehash, 1), V), device=DEV)
OFF = [False]


# ---------------- model ----------------
def rope(x):
    t, d = x.shape[-2], x.shape[-1]
    f = 1.0 / 10000 ** (torch.arange(0, d, 2, device=x.device).float() / d); ang = torch.outer(torch.arange(t, device=x.device).float(), f); c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]; return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Block(nn.Module):
    def __init__(s, d, H, ve, big=False):
        super().__init__(); s.H, s.dk = H, d // H; s.big = big
        s.n1, s.n2, s.qn, s.kn = nn.RMSNorm(d), nn.RMSNorm(d), nn.RMSNorm(d // H), nn.RMSNorm(d // H)
        s.q, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        f = int(8 * d / 3) // 8 * 8; s.gate, s.up, s.down = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)
        s.ve = s.vh = s.va = s.vp = None; s.on = False
        if ve:
            n = 0 if a.bank else (a.vebig if big else (VK if VMAP is not None else V))
            s.ve = nn.Parameter(torch.randn(n, d) * d ** -0.5, requires_grad=not a.vefrozen) if n else None; s.lam = nn.Parameter(torch.tensor(a.lam))
            s.vh = nn.Parameter(torch.randn(a.vehash, d) * d ** -0.5) if a.vehash else None
            s.va = nn.Parameter(torch.randn(V, a.verank)) if a.verank else None
            s.vp = nn.Parameter(torch.randn(a.verank, d) * (a.verank * d) ** -0.5) if a.verank else None
            s.bk = nn.Parameter(torch.randn(a.bank, d) * d ** -0.5) if a.bank else None
            s.br = nn.Parameter(torch.randn(d, a.bank) * d ** -0.5) if a.bank else None
        if not ve: s.bk = s.br = None
        s.on = bool(ve)

    def vt(s, ids, h=None):
        if s.bk is not None: return torch.softmax(h @ s.br, -1) @ s.bk
        if a.veprev: return s.ve[F.pad(ids, (1, 0), value=0)[:, :-1]]
        if s.big: return s.ve[(F.pad(ids, (1, 0), value=V)[:, :-1] * 1000003 + ids) % a.vebig]
        if VMAP is None: return s.ve[ids]
        j = VMAP[ids]; full = (j >= 0).unsqueeze(-1).float(); out = 0
        if s.ve is not None: out = s.ve[j.clamp(min=0)] * full
        if s.vh is not None: out = out + s.vh[HID[ids]] * (1 - full)
        if s.va is not None: out = out + (s.va[ids] @ s.vp) * (1 - full)
        return out

    def forward(s, x, ids):
        b, t, d = x.shape; h = s.n1(x); sp = lambda z: z.view(b, t, s.H, s.dk).transpose(1, 2)
        v = s.v(h)
        if s.on and not OFF[0]: v = v + s.lam * s.vt(ids, h)
        q, k = rope(s.qn(sp(s.q(h)))), rope(s.kn(sp(s.k(h))))
        y = F.scaled_dot_product_attention(q, k, sp(v), is_causal=True)
        x = x + s.o(y.transpose(1, 2).reshape(b, t, d)); h = s.n2(x)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))


class Model(nn.Module):
    def __init__(s):
        super().__init__(); d = a.width
        s.head = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        s.emb = None if a.tie else nn.Parameter(torch.randn(V, d) * d ** -0.5)
        s.urows = nn.Parameter(torch.randn(UK, d) * d ** -0.5 * (0 if a.udelta else 1)) if UMAP is not None else None
        s.ua = nn.Parameter(torch.zeros(V, a.urank)) if a.urank else None
        s.ub = nn.Parameter(torch.randn(a.urank, d) * d ** -0.5) if a.urank else None
        s.um = nn.Parameter(torch.eye(d)) if a.umat else None
        NGF = a.ngram and a.ngfull; K = a.nghead
        s.ng2 = nn.Parameter(torch.randn(*((K, a.ngram, d // K) if NGF else (a.ngram, d))) * d ** -0.5) if a.ngram else None
        s.ng3 = nn.Parameter(torch.randn(*((K, a.ngram, d // K) if NGF else (a.ngram, d))) * d ** -0.5) if a.ngram else None
        s.ngk = nn.Parameter(torch.randn(d, 2 * (d // K) * K) * (2 * d) ** -0.5) if NGF else None
        s.ngv = nn.Parameter(torch.randn(d, 2 * (d // K) * K) * (2 * d) ** -0.5) if NGF else None
        s.ngc = nn.Parameter(torch.zeros(d, 1, 4)) if NGF else None
        s.big = nn.Parameter(torch.randn(a.inbig, d) * d ** -0.5) if a.inbig else None
        s.blocks = nn.ModuleList(Block(d, a.heads, i in VE, bool(a.vebig) and (not a.vebigl or i in [int(z) for z in a.vebigl.split(',')])) for i in range(a.layers)); s.norm = nn.RMSNorm(d)

    def embed(s, ids):
        if s.emb is not None: return s.emb[ids]
        x = s.head[ids]
        if s.ua is not None and not OFF[0]: x = x + s.ua[ids] @ s.ub
        if s.um is not None and not OFF[0]: x = x @ s.um
        if s.urows is not None and not OFF[0]:
            j = UMAP[ids]
            x = x + s.urows[j.clamp(min=0)] * (j >= 0).unsqueeze(-1) if a.udelta else torch.where((j >= 0).unsqueeze(-1), s.urows[j.clamp(min=0)], x)
        return x

    def grams(s, ids):
        p1 = F.pad(ids, (1, 0), value=V)[:, :-1]; p2 = F.pad(ids, (2, 0), value=V)[:, :-2]
        return (p1 * 1000003 + ids) , (p2 * 998244353 + p1 * 1000003 + ids)

    def forward(s, ids):
        x = s.embed(ids)
        if s.big is not None and not OFF[0]: x = x + s.big[s.grams(ids)[0] % a.inbig]
        x = x * a.escale
        for i, blk in enumerate(s.blocks):
            if s.ngc is not None and i == a.nglayer and not OFF[0]:
                p1 = F.pad(ids, (1, 0), value=V)[:, :-1]; p2 = F.pad(ids, (2, 0), value=V)[:, :-2]; K = a.nghead; hk = torch.arange(K, device=ids.device)
                M1 = torch.tensor([1000003, 998244353, 754974721, 167772161, 469762049, 1004535809, 2013265921, 1224736769][:K], device=ids.device)
                M2 = torch.tensor([7919, 104729, 1299709, 15485863, 32452843, 49979687, 67867967, 86028121][:K], device=ids.device)
                h2 = (p1.unsqueeze(-1) * M1 + ids.unsqueeze(-1) * M2 + hk) % a.ngram
                h3 = (p2.unsqueeze(-1) * M2 * 31 + p1.unsqueeze(-1) * M1 + ids.unsqueeze(-1) * M2 + hk * 7) % a.ngram
                e = torch.cat([s.ng2[hk, h2].flatten(-2), s.ng3[hk, h3].flatten(-2)], -1)
                kk, vv = F.linear(e, s.ngk), F.linear(e, s.ngv)
                v = torch.sigmoid((F.rms_norm(x, (x.shape[-1],)) * F.rms_norm(kk, (kk.shape[-1],))).sum(-1, keepdim=True) / x.shape[-1] ** 0.5) * vv
                c = F.conv1d(F.pad(F.rms_norm(v, (v.shape[-1],)).transpose(1, 2), (9, 0)), s.ngc, dilation=3, groups=v.shape[-1]).transpose(1, 2)
                x = x + (F.silu(c) + v) * a.escale
            elif s.ng2 is not None and i == a.nglayer and not OFF[0]:
                g2, g3 = s.grams(ids)
                for e in (s.ng2[g2 % a.ngram], s.ng3[g3 % a.ngram]):
                    x = x + torch.sigmoid((F.rms_norm(x, (x.shape[-1],)) * F.rms_norm(e, (e.shape[-1],))).sum(-1, keepdim=True) / x.shape[-1] ** 0.5) * e * a.escale
            if CAP is not None: CAP.append(x)
            x = blk(x, ids)
        if CAP is not None: CAP.append(x)
        return F.linear(s.norm(x), s.head)


def newton_schulz(G, steps=5):
    A, B, C = 3.4445, -4.7750, 2.0315; X = G / (G.norm() + 1e-7); tr_ = X.shape[0] > X.shape[1]
    if tr_: X = X.T
    for _ in range(steps): M = X @ X.T; X = A * X + (B * M + C * M @ M) @ X
    return X.T if tr_ else X


class Muon(torch.optim.Optimizer):
    def __init__(s, ps, lr): super().__init__(ps, dict(lr=lr))

    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for w in g["params"]:
                if w.grad is None: continue
                m = s.state[w].setdefault("m", torch.zeros_like(w)); m.mul_(0.95).add_(w.grad)
                w.add_(newton_schulz(w.grad + 0.95 * m), alpha=-g["lr"] * max(1, w.shape[0] / w.shape[1]) ** 0.5)


VX = torch.tensor(val, device=DEV); VIN, VTG = bucket[val[:, :-1].reshape(-1)], bucket[val[:, 1:].reshape(-1)]
RIN, RTG = rbin[val[:, :-1].reshape(-1)], rbin[val[:, 1:].reshape(-1)]
OCC = {f"occ_{RB[i]}": float(np.bincount(rows.reshape(-1), minlength=V)[(rbin == i)].mean()) for i in range(len(RB) - 1) if (rbin == i).any()}


CAP = None
_tp = rows[:, :-1].reshape(-1).astype(np.int64) * (V + 1) + rows[:, 1:].reshape(-1); _u, _c = np.unique(_tp, return_counts=True)
_vp = val[:, :-2].reshape(-1).astype(np.int64) * (V + 1) + val[:, 1:-1].reshape(-1); _ix = np.clip(np.searchsorted(_u, _vp), 0, len(_u) - 1)
_vin = val[:, :-1]; _first = np.zeros_like(_vin, dtype=bool); _first[:, 0] = True
PCNT = np.zeros(_vin.shape, np.int64); PCNT[:, 1:] = np.where(_u[_ix] == _vp, _c[_ix], 0).reshape(len(val), -1); PCNT = PCNT.reshape(-1)
PFIRST = _first.reshape(-1); NPAIR = len(_u)
PB = [0, 1, 10, 100, 1000, 10**12]
if a.pcshare:
    _sel = (~PFIRST) & (VTG < 3); _n = _sel.sum()
    print(json.dumps({"V": a.V, "tokens": int(rows.size), "npair": int(NPAIR), "share": {f"pc_{lo}": round(float((_sel & (PCNT >= lo) & (PCNT < hi)).sum() / _n), 4) for lo, hi in zip(PB[:-1], PB[1:])}})); raise SystemExit


@torch.no_grad()
def probe(m, K):
    """Ridge readout of the current and previous word (K most common known words) from each layer's input."""
    global CAP
    m.eval(); x = VX[:400, :-1]; CAP = []; m(x); hs = [h.reshape(-1, h.shape[-1]).float().cpu().numpy().astype(np.float64) for h in CAP]; CAP = None; m.train()
    cur = x.reshape(-1).cpu().numpy(); prev = F.pad(x, (1, 0), value=-1)[:, :-1].reshape(-1).cpu().numpy()
    cnt = np.bincount(rows.reshape(-1), minlength=V); cnt[bucket >= 3] = 0; top = np.argsort(-cnt)[:K]; cls = -np.ones(V + 1, int); cls[top] = np.arange(K)
    out = {}
    for nm, lab in (("cur", cur), ("prev", prev)):
        y = cls[lab]; keep = y >= 0; acc = []
        for h in hs:
            X = np.c_[h[keep], np.ones(keep.sum())]; Y = np.eye(K)[y[keep]]; n = len(X) // 2
            W = np.linalg.solve(X[:n].T @ X[:n] + 1e-2 * n * np.eye(X.shape[1]), X[:n].T @ Y[:n])
            acc.append(round(float(((X[n:] @ W).argmax(1) == y[keep][n:]).mean()), 3))
        out[nm] = acc; out[nm + "_base"] = round(float(np.bincount(y[keep][len(y[keep]) // 2:]).max() / (len(y[keep]) - len(y[keep]) // 2)), 3)
    return out


@torch.no_grad()
def geom(m):
    """Is the input table a copy, a linear transform, or its own geometry relative to the output table?"""
    ids = torch.tensor(order, device=DEV); E = m.embed(ids).float().cpu().numpy().astype(np.float64); H = m.head[ids].float().cpu().numpy().astype(np.float64)
    bk = bucket[order]; out = {}
    cs = (E * H).sum(1) / (np.linalg.norm(E, axis=1) * np.linalg.norm(H, axis=1) + 1e-9)
    Hx = np.c_[H, np.ones(len(H))]; M = np.linalg.solve(Hx.T @ Hx + 1e-3 * len(H) * np.eye(Hx.shape[1]), Hx.T @ E); R = E - Hx @ M
    def nb(X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); S = Xn @ Xn.T; np.fill_diagonal(S, -9); return np.argsort(-S, 1)[:, :10]
    ne, nh = nb(E), nb(H); ov = np.array([len(set(ne[i]) & set(nh[i])) / 10 for i in range(len(E))])
    for i, n in enumerate(NB[:3]):
        sel = bk == i
        out[n] = dict(cos=round(float(cs[sel].mean()), 3), r2=round(float(1 - (R[sel] ** 2).sum() / ((E[sel] - E.mean(0)) ** 2).sum()), 3), nbr=round(float(ov[sel].mean()), 3))
    out["all"] = dict(cos=round(float(cs.mean()), 3), r2=round(float(1 - (R ** 2).sum() / ((E - E.mean(0)) ** 2).sum()), 3), nbr=round(float(ov.mean()), 3))
    sv = np.linalg.svd(R - R.mean(0), compute_uv=False) ** 2; out["resid_rank_share"] = [round(float(sv[:r].sum() / sv.sum()), 3) for r in (1, 2, 4, 8)]
    return out


@torch.no_grad()
def evaluate(m):
    m.eval(); per = []
    for i in range(0, len(VX), 200):
        x = VX[i:i + 200]; per.append(F.cross_entropy(m(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1), reduction="none"))
    per = torch.cat(per).float().cpu().numpy(); m.train()
    r = {"val": float(per.mean()), "val_known": float(per[VTG < 3].mean())}
    for side, bk in (("in", VIN), ("tg", VTG)):
        for i, n in enumerate(NB): r[f"{side}_{n}"] = float(per[bk == i].mean())
    for side, rb in (("rin", RIN), ("rtg", RTG)):
        for i in range(len(RB) - 1):
            if (rb == i).sum() >= 50: r[f"{side}_{RB[i]}"] = float(per[rb == i].mean())
    for lo, hi in zip(PB[:-1], PB[1:]):
        sel = (~PFIRST) & (PCNT >= lo) & (PCNT < hi) & (VTG < 3)
        if sel.sum() >= 50: r[f"pc_{lo}"] = float(per[sel].mean())
    return {k: round(v, 4) for k, v in r.items()}


def main():
    m = Model().to(DEV)
    if a.veshare:
        vb = [b for b in m.blocks if b.on]
        for b in vb[1:]:
            for nm in ("ve", "vh", "va", "vp"):
                if getattr(vb[0], nm) is not None: setattr(b, nm, getattr(vb[0], nm))
    TAB = lambda: [m.urows, m.ua, m.ub, m.um, m.ng2, m.ng3, m.big, m.ngk, m.ngv, m.ngc] + [z for b in m.blocks for z in (b.ve, b.vh, b.va, b.vp, b.bk, b.br)]
    mats = [w for n, w in m.named_parameters() if w.ndim == 2 and n.startswith("blocks") and all(w is not z for z in TAB())]
    tabs = list({id(w): w for w in TAB() if w is not None and w.requires_grad}.values())
    rest = [w for w in m.parameters() if w.requires_grad and all(w is not z for z in mats + tabs)]
    groups = [dict(params=rest, lr=a.elr)] + ([dict(params=tabs, lr=TLR)] if tabs else [])
    opts = [Muon(mats, a.lr), torch.optim.Adam(groups, betas=(0.9, 0.95))]
    base = [[g["lr"] for g in o.param_groups] for o in opts]
    curve = []; t0 = time.time(); X = torch.tensor(rows, device=DEV)
    for step in range(a.steps):
        sc = min((step + 1) / min(50, max(1, a.steps // 10)), 1.0, (a.steps - step) / (a.cool * a.steps))
        for o, bl in zip(opts, base):
            for g, lr in zip(o.param_groups, bl): g["lr"] = lr * sc
        x = X[step * a.batch:(step + 1) * a.batch]
        F.cross_entropy(m(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1)).backward()
        for o in opts: o.step()
        m.zero_grad(set_to_none=True)
        if a.log and (step + 1) % a.log == 0 and step + 1 < a.steps: curve.append(dict(step=step + 1, **evaluate(m)))
    if DEV.type == "mps": torch.mps.synchronize()
    r = evaluate(m)
    if UMAP is not None or VE or a.urank or a.umat or a.ngram or a.inbig:
        OFF[0] = True; o = evaluate(m); OFF[0] = False; r.update({f"off_{k}": v for k, v in o.items()})
    if a.offeach:
        oe = {}
        for i, blk in enumerate(m.blocks):
            if blk.on: blk.on = False; oe[i] = round(evaluate(m)["val_known"] - r["val_known"], 4); blk.on = True
        r["offeach"] = oe; print("offeach", json.dumps(oe), flush=True)
    pr = probe(m, a.probe) if a.probe else None
    gm = geom(m) if a.geom else None
    if gm: print("geom", json.dumps(gm), flush=True)
    nparam = sum(w.numel() for w in m.parameters())
    ntab = m.head.numel() + (m.emb.numel() if m.emb is not None else 0) + sum(w.numel() for w in {id(z): z for z in TAB() if z is not None}.values())
    res = dict(tag=a.tag, args=vars(a), params=nparam, body=nparam - ntab, tokens=need, tpw=round(need / nparam, 2),
               tpw_body=round(need / (nparam - ntab), 1), flops_tok=flops_tok(vars(a)), untie_k=UK, ve_k=VK, occ=OCC, npair=NPAIR, train_s=round(time.time() - t0, 1),
               lams=[round(float(b.lam.detach()), 3) for b in m.blocks if b.ve is not None], scores=r, probe=pr, geom=gm,
               curve=curve + [dict(step=a.steps, **{k: v for k, v in r.items() if not k.startswith("off_")})])
    with open(os.path.join(HERE, a.out), "a") as f: f.write(json.dumps(res) + "\n")
    print(f"{a.tag}: val {r['val']:.4f} known {r['val_known']:.4f} in c/m/r {r['in_common']:.3f}/{r['in_middle']:.3f}/{r['in_rare']:.3f}"
          + (f" off {r['off_val']:.4f}" if 'off_val' in r else "") + f" | params {nparam} tpw {res['tpw']} | {res['train_s']}s" + (f" | probe cur {pr['cur']} prev {pr['prev']} base {pr['cur_base']}/{pr['prev_base']}" if pr else ""), flush=True)


if __name__ == "__main__":
    main()
