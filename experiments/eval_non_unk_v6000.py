r"""Re-evaluate V=1500, 3000, 6000 (3 seeds, 300 steps) with and without <UNK> in the input pair (w_{t-1}, w_t).
Separates how much of the 1000+ bucket reversal in truncated-vocab experiments comes from:
  (1) <UNK> input pair collapse (51.9% of 1000+ pairs at V=6000 involve <UNK>), vs
  (2) genuine high-entropy function/punctuation pairs.
Saves to results/unk_filtered_eval.json.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_local_suite as rls

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "unk_filtered_eval.json")


def train_and_eval_dual_unk(data, ve_layers, ve_mode="none", engram_mode="none", seed=0,
                            elr=0.1, tlr_w=0.1, tlr_p=0.03, pair_rows=6000, dev="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    DEV = torch.device(dev)
    V = data["V"]
    UNK = data["UNK"]
    m = rls.Model(V, 16, 2, 2, set(ve_layers), ve_mode, engram_mode, pair_rows, data["n_hi"], data["n_mid"]).to(DEV)

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

    steps, batch = 300, 128
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

    val_np = data["val"]
    vtg = data["bucket"][val_np[:, 1:].reshape(-1)]
    pcnt_flat = data["val_pcnt"][:, :-1].reshape(-1)
    fm = np.zeros_like(val_np[:, :-1], dtype=bool)
    fm[:, 0] = True
    ff = fm.reshape(-1)

    # Check if w_{t-1} == UNK or w_t == UNK
    w_cur = val_np[:, :-1]
    w_prev = np.pad(val_np[:, :-1], ((0, 0), (1, 0)), constant_values=UNK)[:, :-1]
    has_unk_in_pair = ((w_cur == UNK) | (w_prev == UNK)).reshape(-1)

    PB = [0, 1, 10, 100, 1000, 10**12]
    all_b, clean_b, unk_only_b = {}, {}, {}
    for lo, hi in zip(PB[:-1], PB[1:]):
        base_sel = (~ff) & (pcnt_flat >= lo) & (pcnt_flat < hi) & (vtg < 3)
        clean_sel = base_sel & (~has_unk_in_pair)
        unk_sel = base_sel & has_unk_in_pair
        if base_sel.sum() >= 30:
            all_b[f"pc_{lo}"] = float(per[base_sel].mean())
        if clean_sel.sum() >= 30:
            clean_b[f"pc_{lo}"] = float(per[clean_sel].mean())
        if unk_sel.sum() >= 30:
            unk_only_b[f"pc_{lo}"] = float(per[unk_sel].mean())
    return dict(all_buckets=all_b, non_unk_pair_buckets=clean_b, unk_pair_buckets=unk_only_b)


def main():
    out = {}
    for Vw in (1500, 3000, 6000):
        elr = 0.05 if Vw <= 3000 else 0.1
        data = rls.load_data(Vw, 300, 128)
        w_runs = [train_and_eval_dual_unk(data, [0, 1], ve_mode="word", seed=sd, elr=elr, tlr_w=0.1) for sd in range(3)]
        e_runs = [train_and_eval_dual_unk(data, [], ve_mode="none", engram_mode="full", seed=sd, elr=elr, tlr_p=0.03, pair_rows=Vw) for sd in range(3)]
        m_runs = [train_and_eval_dual_unk(data, [0, 1], ve_mode="mp_ve", seed=sd, elr=elr, tlr_w=0.1, tlr_p=0.03) for sd in range(3)]

        def diff_stat(runs_a, runs_b, key):
            res = {}
            for b in ("pc_0", "pc_1", "pc_10", "pc_100", "pc_1000"):
                vals = [ra[key][b] - rb[key][b] for ra, rb in zip(runs_a, runs_b) if b in ra[key] and b in rb[key]]
                if vals:
                    res[b] = round(float(np.mean(vals)), 4)
            return res

        out[str(Vw)] = {
            "eng_minus_word_ALL": diff_stat(e_runs, w_runs, "all_buckets"),
            "eng_minus_word_NON_UNK_PAIR": diff_stat(e_runs, w_runs, "non_unk_pair_buckets"),
            "eng_minus_word_UNK_PAIR_ONLY": diff_stat(e_runs, w_runs, "unk_pair_buckets"),
            "mpve_minus_word_ALL": diff_stat(m_runs, w_runs, "all_buckets"),
            "mpve_minus_word_NON_UNK_PAIR": diff_stat(m_runs, w_runs, "non_unk_pair_buckets"),
        }
        print(f"\n=== V={Vw} (300 steps, 3 seeds): Engram - Word (<0 means Engram wins) ===")
        print("  ALL known targets     :", out[str(Vw)]["eng_minus_word_ALL"])
        print("  NON-UNK input pairs   :", out[str(Vw)]["eng_minus_word_NON_UNK_PAIR"])
        print("  UNK-involving pairs   :", out[str(Vw)]["eng_minus_word_UNK_PAIR_ONLY"])
        print("  MP-VE - Word (NON-UNK):", out[str(Vw)]["mpve_minus_word_NON_UNK_PAIR"])

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved", OUT_PATH)


if __name__ == "__main__":
    main()
