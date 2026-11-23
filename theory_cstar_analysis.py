"""Zero-GPU analysis supporting the c*(d) theory write-up.

A. Fine-bucket (pure hashed pair_ve, M=24000 vs word, V=6000, L=2, 150 steps, 3 seeds) identifiability:
   - per-d mean gap curve, SE, sign changes, z of the positive side
   - interpolated c*, interval-censored feasible alpha range (representative and edge-conservative)
   - seed-level OLS alpha and seed-bootstrap alpha over identified widths
B. Coarse Block C (engram_full vs word) censoring bounds.
C. Data-level previous-token information Delta_hat per count bucket:
   smoothed trigram vs bigram log-likelihood gain on HELD-OUT validation positions, bucketed exactly
   like the model evaluation (key = (w_{t-1}, w_t) train count, known targets only, first position dropped).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import math
import numpy as np

sys.path.insert(0, r"C:\Users\ROG\Desktop\ICML")
import run_local_suite as rls

OUT = r"C:\Users\ROG\Desktop\ICML\theory_cstar_analysis.json"
rng = np.random.default_rng(0)


def feasible_alpha(xs, lo, hi, grid=np.linspace(-4, 4, 16001)):
    """alpha such that exists a with lo_d <= a + alpha*x_d <= hi_d for all d (log scale)."""
    ok = []
    for al in grid:
        if np.max(lo - al * xs) <= np.min(hi - al * xs) + 1e-12:
            ok.append(al)
    return (float(min(ok)), float(max(ok))) if ok else (None, None)


def first_crossing(g, logc):
    for j in range(len(g) - 1):
        if g[j] <= 0 < g[j + 1]:
            fr = -g[j] / (g[j + 1] - g[j])
            return j, float(logc[j] + fr * (logc[j + 1] - logc[j]))
    return None, None


def sign_changes(g):
    s = np.sign(g)
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1]))


# ---------------------------------------------------------------- A
p0 = json.load(open(r"C:\Users\ROG\Desktop\ICML\p0_results.json"))
s1 = p0["step1_fine_cstar"]
FB = s1["fine_bucket_c"]
keys_all = list(FB.keys())
edges = [(int(k.split("_")[1]), int(k.split("_")[2])) for k in keys_all]
inner = [i for i, (lo, hi) in enumerate(edges) if lo >= 1 and hi <= 1000]
inner_keys = [keys_all[i] for i in inner]
inner_edges = [edges[i] for i in inner]
logc = np.log(np.array([FB[k][1] for k in inner_keys]))

resA = {"buckets": inner_keys, "c_rep": [FB[k][1] for k in inner_keys], "per_d": {}}
mean_curves = {}
seed_G = {}
for d in ("16", "32", "64", "128"):
    W = s1["widths"][d]
    G = np.array([[W["word_runs"][s][k] - W["pair_runs"][s][k] for k in inner_keys] for s in range(3)])
    seed_G[d] = G
    gm = G.mean(0)
    gse = G.std(0, ddof=1) / math.sqrt(3)
    mean_curves[d] = gm
    j, lcs = first_crossing(gm, logc)
    pos = gm > 0
    z_pos = float(np.max(gm[pos] / gse[pos])) if pos.any() else 0.0
    seed_br = []
    for s in range(3):
        js, _ = first_crossing(G[s], logc)
        seed_br.append(js)
    n_same = sum(1 for x in seed_br if x is not None and x == j)
    nsc = sign_changes(gm)
    ident = (j is not None) and (nsc == 1) and (z_pos >= 2.0) and (n_same >= 2)
    entry = dict(
        gap_mean=[round(float(x), 4) for x in gm],
        gap_se=[round(float(x), 4) for x in gse],
        sign_changes=nsc, z_max_positive=round(z_pos, 2),
        seed_brackets=seed_br, n_seeds_same_bracket=n_same,
        identifiable=bool(ident),
        cstar_interp=(round(math.exp(lcs), 1) if lcs is not None else None),
        bracket_rep=([round(FB[inner_keys[j]][1], 1), round(FB[inner_keys[j + 1]][1], 1)] if j is not None else None),
        bracket_edge=([inner_edges[j][0], inner_edges[j + 1][1]] if j is not None else None),
        max_positive_gap=round(float(gm.max()), 4),
        amplitude=round(float(gm.max() - gm.min()), 4),
    )
    resA["per_d"][d] = entry
    print(f"d={d:>3} | signchg={nsc} z+={z_pos:5.2f} seedbr={seed_br} ident={ident} c*={entry['cstar_interp']} "
          f"rep={entry['bracket_rep']} edge={entry['bracket_edge']} max+={entry['max_positive_gap']} amp={entry['amplitude']}")

strict = [d for d in ("16", "32", "64", "128") if resA["per_d"][d]["identifiable"]]
weak = [d for d in ("16", "32", "64", "128") if resA["per_d"][d]["sign_changes"] == 1 and resA["per_d"][d]["cstar_interp"]]
resA["strict_identified"] = strict
resA["weak_identified"] = weak
print(f"strict-identified widths: {strict} | weak (single mean-curve sign change): {weak}")
idd = weak
xs = np.log(np.array([float(d) for d in idd]))
lo_rep = np.log(np.array([resA["per_d"][d]["bracket_rep"][0] for d in idd]))
hi_rep = np.log(np.array([resA["per_d"][d]["bracket_rep"][1] for d in idd]))
lo_edge = np.log(np.array([resA["per_d"][d]["bracket_edge"][0] for d in idd]))
hi_edge = np.log(np.array([resA["per_d"][d]["bracket_edge"][1] for d in idd]))
fa_rep = feasible_alpha(xs, lo_rep, hi_rep)
fa_edge = feasible_alpha(xs, lo_edge, hi_edge)
lc_id = np.log(np.array([resA["per_d"][d]["cstar_interp"] for d in idd]))
al_mean = float(np.polyfit(xs, lc_id, 1)[0])
print(f"identified widths={idd} | mean-curve alpha={al_mean:+.3f} | feasible alpha (rep)={fa_rep} | (edge)={fa_edge}")

# seed-level OLS
pts = []
for d in idd:
    for s in range(3):
        js, lcs = first_crossing(seed_G[d][s], logc)
        if lcs is not None:
            pts.append((math.log(float(d)), lcs))
px = np.array([p[0] for p in pts]); py = np.array([p[1] for p in pts])
sl, ic = np.polyfit(px, py, 1)
res = py - (sl * px + ic)
se = math.sqrt(np.sum(res ** 2) / (len(px) - 2) / np.sum((px - px.mean()) ** 2))
print(f"seed-level alpha={sl:+.3f}±{se:.3f} (n={len(px)}) t0={sl/se:+.2f} t1={(sl-1)/se:+.2f}")

# seed bootstrap (resample seeds within each width, mean curve -> crossing -> slope)
boots, fails = [], 0
for _ in range(4000):
    lcs_b = []
    good = True
    for d in idd:
        idx = rng.integers(0, 3, 3)
        gb = seed_G[d][idx].mean(0)
        _, lcb = first_crossing(gb, logc)
        if lcb is None:
            good = False
            break
        lcs_b.append(lcb)
    if not good:
        fails += 1
        continue
    boots.append(np.polyfit(xs, np.array(lcs_b), 1)[0])
boots = np.array(boots)
ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
p_ge1 = float(np.mean(boots >= 1.0))
print(f"bootstrap alpha median={np.median(boots):+.3f} 95%CI=[{ci[0]:+.3f},{ci[1]:+.3f}] P(alpha>=1)={p_ge1:.4f} fails={fails}")

resA["identified_widths"] = idd
resA["alpha_mean_curve"] = al_mean
resA["alpha_feasible_rep"] = fa_rep
resA["alpha_feasible_edge"] = fa_edge
resA["alpha_seed_ols"] = dict(slope=float(sl), se=float(se), n=len(px), t0=float(sl / se), t1=float((sl - 1) / se))
resA["alpha_bootstrap"] = dict(median=float(np.median(boots)), ci95=ci, p_alpha_ge_1=p_ge1, n=len(boots), fails=fails)

# full fit including d=128 (for the record)
all_lc = [(math.log(float(d)), math.log(resA["per_d"][d]["cstar_interp"])) for d in ("16", "32", "64", "128") if resA["per_d"][d]["cstar_interp"]]
ax = np.array([a for a, _ in all_lc]); ay = np.array([b for _, b in all_lc])
fs, fi = np.polyfit(ax, ay, 1)
r2 = 1 - np.sum((ay - (fs * ax + fi)) ** 2) / np.sum((ay - ay.mean()) ** 2)
resA["alpha_full_including_128"] = dict(slope=float(fs), r2=float(r2))
print(f"full fit incl. d=128: alpha={fs:+.3f} R2={r2:.3f}")

# ---------------------------------------------------------------- B
suite = json.load(open(r"C:\Users\ROG\Desktop\ICML\local_suite_results.json"))
resB = {}
xsC, loC_rep, hiC_rep, loC_edge, hiC_edge = [], [], [], [], []
for d in ("12", "16", "32", "64"):
    xsC.append(math.log(float(d)))
    loC_rep.append(math.log(29.42)); hiC_rep.append(math.log(290.21))
    loC_edge.append(math.log(10.0)); hiC_edge.append(math.log(1000.0))
resB["feasible_rep_d12_64"] = feasible_alpha(np.array(xsC), np.array(loC_rep), np.array(hiC_rep))
resB["feasible_edge_d12_64"] = feasible_alpha(np.array(xsC), np.array(loC_edge), np.array(hiC_edge))
print(f"Block C coarse: feasible alpha rep={resB['feasible_rep_d12_64']} edge={resB['feasible_edge_d12_64']}")


# ---------------------------------------------------------------- C
def lookup(u, c, k):
    ix = np.clip(np.searchsorted(u, k), 0, len(u) - 1)
    return np.where(u[ix] == k, c[ix], 0).astype(np.float64)


def ngram_gain(data, beta=5.0, gamma=1.0, unk=None):
    rows, val, V = data["rows"], data["val"], data["V"]
    Bs = V + 1
    uni = np.bincount(rows.ravel(), minlength=Bs).astype(np.float64)
    pu = (uni + 1.0) / (uni.sum() + Bs)
    ctx2 = np.bincount(rows[:, :-1].ravel(), minlength=Bs).astype(np.float64)
    k23 = rows[:, :-1].ravel().astype(np.int64) * Bs + rows[:, 1:].ravel()
    u23, c23 = np.unique(k23, return_counts=True)
    k12 = rows[:, :-2].ravel().astype(np.int64) * Bs + rows[:, 1:-1].ravel()
    u12, c12 = np.unique(k12, return_counts=True)
    k123 = k12 * Bs + rows[:, 2:].ravel()
    u123, c123 = np.unique(k123, return_counts=True)

    a, b, t = val[:, :-2].astype(np.int64), val[:, 1:-1].astype(np.int64), val[:, 2:].astype(np.int64)
    p2 = (lookup(u23, c23, b * Bs + t) + gamma * pu[t]) / (ctx2[b] + gamma)
    p12 = (lookup(u123, c123, (a * Bs + b) * Bs + t) + beta * p2) / (lookup(u12, c12, a * Bs + b) + beta)
    gain = np.log(p12) - np.log(p2)
    pc = data["val_pcnt"][:, 1:-1]
    known = data["bucket"][t] < 3
    unk_key = (a == unk) | (b == unk) if unk is not None else np.zeros_like(known)
    return gain, pc, known, unk_key


def by_bucket(gain, pc, known, unk_key, edges_list):
    out = {}
    for lo, hi in edges_list:
        sel = known & (pc >= lo) & (pc < hi)
        if sel.sum() < 30:
            continue
        sel_nu = sel & ~unk_key
        out[f"{lo}_{hi}"] = dict(
            n=int(sel.sum()), gain=round(float(gain[sel].mean()), 4),
            gain_nonUNK=(round(float(gain[sel_nu].mean()), 4) if sel_nu.sum() >= 30 else None),
            unk_frac=round(float(unk_key[sel].mean()), 4),
        )
    return out


FINE = [(0, 1), (1, 10), (10, 20), (20, 40), (40, 70), (70, 100), (100, 150), (150, 250), (250, 500), (500, 1000), (1000, 10 ** 12)]
COARSE = [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 10 ** 12)]
resC = {}
for steps in (150, 300):
    data = rls.load_data(6000, steps, 128)
    g, pc, kn, uk = ngram_gain(data, unk=data["UNK"])
    resC[f"V6000_s{steps}"] = dict(fine=by_bucket(g, pc, kn, uk, FINE), coarse=by_bucket(g, pc, kn, uk, COARSE))
    print(f"\n[V=6000, {steps} steps] held-out trigram-vs-bigram gain (nats) per pair-count bucket:")
    for k, v in resC[f"V6000_s{steps}"]["coarse"].items():
        print(f"   {k:>16s}: gain={v['gain']:+.4f} nonUNK={v['gain_nonUNK']} unk_frac={v['unk_frac']:.3f} n={v['n']}")

# correlation of model fine gap curves with data-level gain composition (150-step data = same as fine-bucket runs)
fine150 = resC["V6000_s150"]["fine"]
gain_inner = np.array([fine150[f"{lo}_{hi}"]["gain"] for (lo, hi) in inner_edges])
corr = {}
for d in ("16", "32", "64", "128"):
    gm = mean_curves[d]
    rg = np.argsort(np.argsort(gm)); rd = np.argsort(np.argsort(gain_inner))
    corr[d] = round(float(np.corrcoef(rg, rd)[0, 1]), 3)
print("\nSpearman(model pair-word gap, data-level gain) across fine buckets [1,1000):", corr)
resC["spearman_model_gap_vs_data_gain"] = corr

json.dump({"A_fine_bucket": resA, "B_blockC": resB, "C_data_delta": resC}, open(OUT, "w"), indent=2)
print("\nSaved", OUT)
