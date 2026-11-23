"""Step 1b: Fine-bucket c*(d) measurement on collision-free pair_ve vs word (V=6000, M=24000, L=2, 150s, 3 seeds)."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, json, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_local_suite as rls

FINE_EDGES = [0, 1, 10, 20, 40, 70, 100, 150, 250, 500, 1000, 10**12]


def train_and_eval_fine(data, ve_layers, ve_mode, engram_mode="none", seed=0, width=16, heads=2,
                        steps=150, batch=128, elr=0.1, tlr_w=0.1, tlr_p=0.03, pair_rows=24000, dev="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    DEV = torch.device(dev)
    V = data["V"]
    m = rls.Model(V, width, 2, heads, set(ve_layers), ve_mode, engram_mode, pair_rows, data["n_hi"], data["n_mid"]).to(DEV)

    word_tabs = [b.ve_word for b in m.blocks if b.ve_word is not None]
    pair_tabs = [z for z in (m.ng2, m.ng3, m.ngk, m.ngv, m.ngc) if z is not None] + [
        z for b in m.blocks for z in (b.ve_pair, b.ve_hi, b.ve_mid_a, b.ve_mid_p) if z is not None
    ]
    all_tabs = set(id(w) for w in word_tabs + pair_tabs)
    mats = [w for n, w in m.named_parameters() if w.ndim == 2 and n.startswith("blocks") and id(w) not in all_tabs]
    mat_ids = set(id(w) for w in mats)
    rest = [w for w in m.parameters() if w.requires_grad and id(w) not in all_tabs and id(w) not in mat_ids]

    opts = [rls.Muon(mats, 0.02), torch.optim.Adam([dict(params=rest, lr=elr)], betas=(0.9, 0.95))]
    tgroups = []
    if word_tabs:
        tgroups.append(dict(params=word_tabs, lr=tlr_w))
    if pair_tabs:
        tgroups.append(dict(params=pair_tabs, lr=tlr_p))
    if tgroups:
        opts.append(torch.optim.Adam(tgroups, betas=(0.9, 0.95)))

    base = [[g["lr"] for g in o.param_groups] for o in opts]
    X = torch.tensor(data["rows"], device=DEV)
    X_pc = torch.tensor(data["tr_pcnt"], device=DEV)
    X_hi = torch.tensor(data["tr_hi"], device=DEV)
    X_mid = torch.tensor(data["tr_mid"], device=DEV)

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

    scores = {"val_known": float(per[vtg < 3].mean())}
    for lo, hi in zip(FINE_EDGES[:-1], FINE_EDGES[1:]):
        sel = (~first_flat) & (pcnt_flat >= lo) & (pcnt_flat < hi) & (vtg < 3)
        if sel.sum() >= 30:
            scores[f"b_{lo}_{hi}"] = float(per[sel].mean())
    return scores


def main():
    data = rls.load_data(6000, 150, 128)
    vtg = data["bucket"][data["val"][:, 1:].reshape(-1)]
    pcnt_flat = data["val_pcnt"][:, :-1].reshape(-1)
    first_mask = np.zeros_like(data["val"][:, :-1], dtype=bool)
    first_mask[:, 0] = True
    first_flat = first_mask.reshape(-1)

    b_c = {}
    for lo, hi in zip(FINE_EDGES[:-1], FINE_EDGES[1:]):
        sel = (~first_flat) & (pcnt_flat >= lo) & (pcnt_flat < hi) & (vtg < 3)
        vals = pcnt_flat[sel]
        b_c[f"b_{lo}_{hi}"] = (int(sel.sum()), float(np.exp(np.log1p(vals).mean()) - 1) if len(vals) else 0.0)
    print("Fine bucket (count, geom_mean_c):", b_c)

    out = {"fine_bucket_c": b_c, "widths": {}}
    cross_pts = []
    for d in (16, 32, 64, 128):
        H = 2 if d <= 32 else 4
        w_runs = [train_and_eval_fine(data, [0, 1], "word", seed=sd, width=d, heads=H, pair_rows=24000) for sd in range(3)]
        p_runs = [train_and_eval_fine(data, [0, 1], "pair_ve", seed=sd, width=d, heads=H, pair_rows=24000) for sd in range(3)]
        keys = [f"b_{lo}_{hi}" for lo, hi in zip(FINE_EDGES[:-1], FINE_EDGES[1:]) if f"b_{lo}_{hi}" in w_runs[0]]
        gap_curve = {}
        for k in keys:
            diffs = [w[k] - p[k] for w, p in zip(w_runs, p_runs)]  # >0 means pair beats word
            gap_curve[k] = {"c_geom": b_c[k][1], "gap_mean": round(float(np.mean(diffs)), 5), "gap_sd": round(float(np.std(diffs, ddof=1)), 5)}
        # Find zero crossing on gap_mean(log c) for c >= 1
        pts_c = [(gap_curve[k]["c_geom"], gap_curve[k]["gap_mean"]) for k in keys if not k.startswith("b_0_")]
        cstar = None
        for i in range(len(pts_c) - 1):
            c0, g0 = pts_c[i]
            c1, g1 = pts_c[i + 1]
            if g0 <= 0 and g1 > 0:
                frac = -g0 / (g1 - g0)
                cstar = float(math.exp(math.log(c0) + frac * (math.log(c1) - math.log(c0))))
                cross_pts.append((d, cstar))
                break
        out["widths"][str(d)] = {"word_runs": w_runs, "pair_runs": p_runs, "gap_curve": gap_curve, "cstar_mean_curve": cstar}
        g_line = " | ".join(f"{k.split('_')[1]}:{gap_curve[k]['gap_mean']:+.3f}" for k in keys)
        print(f"d={d:3d} | c*={cstar if cstar else 'none':>6} | {g_line}")

    if len(cross_pts) >= 2:
        lx = np.log([x[0] for x in cross_pts])
        ly = np.log([x[1] for x in cross_pts])
        slope, intercept = np.polyfit(lx, ly, 1)
        pred = slope * lx + intercept
        r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
        print(f"\nFine-bucket pure pair_ve vs word fit (d=16..128): log(c*) ~ {slope:+.3f}*log(d) + {intercept:.3f}, R2={r2:.3f}")
        out["fit_mean_curve"] = {"points": cross_pts, "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}

    res_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "p0_results.json")
    all_p0 = {}
    if os.path.exists(res_path):
        with open(res_path) as f:
            all_p0 = json.load(f)
    all_p0["step1_fine_cstar"] = out
    with open(res_path, "w") as f:
        json.dump(all_p0, f, indent=2)
    print("Saved Step 1b to", res_path)


if __name__ == "__main__":
    main()
