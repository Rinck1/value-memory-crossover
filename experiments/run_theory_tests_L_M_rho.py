r"""Empirical tests for the 3 phenomenological theory components (V=6000, w=32, 300 steps, 3 seeds):
  Test 1 (KKT sensitivity table): Tabulates c_hi/c_lo = ((1-\lambda)*r^{1/\nu} + \lambda*r) across variance-vs-rent weight \lambda,
          concavity ratio r = \phi'(0+)/\phi'(1) in {2.5, 3.0, 3.5, 4.0}, and \nu in {0.4, 0.5, 0.6}.
  Test 2 (Preregistered T_cross(M) shift):
          Preregistered prediction: In the 3-term crossing decomposition, increasing collision load (smaller M=6,000 vs M=24,000 vs M=96,000)
          accelerates and enlarges the crossover advantage of U123 over CTRL_hs(M) across steps T in {75, 150, 225, 300}.
  Test 3 (Shared-Bank Depth Scaling L in {2, 4, 6} + Direct Layer-Gradient Correlation \rho_{l,l'} measurement):
          Tests whether shared bank (U1_shared_d64_allL and U123_allL, fixed 1x shared bank) gains more over independent
          last-2 tables (mp_ve_last2, 2x table copies) as depth L grows from 2 -> 4 -> 6, and directly measures the
          inter-layer gradient correlation \rho_{l,l'} of backpropagated table gradients (W^{(l)T} grad_v^{(l)}).
Saves all results to results/theory_cstar_analysis.json['F_three_component_tests'].
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json, math, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_local_suite as rls
import run_upgrade_search as rus

THEORY_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "theory_cstar_analysis.json")


def kkt_sensitivity_table():
    """In KKT condition: phi'(u) = (B * c^{-nu} + mu * T * c^{-1}) / Delta_ctx.
    Let r = phi'(0+) / phi'(1) > 1 (ratio of marginal gain of first vs last rank dimension).
    - Pure variance regime (mu = 0): c_hi / c_lo = r^{1 / nu}.
    - Pure memory-rent regime (B = 0): c_hi / c_lo = r.
    - Mixed regime with rent share eta_rent in [0, 1] at c_hi:
       solving (1 - eta_rent)*(c_hi/c_lo)^nu + eta_rent*(c_hi/c_lo) = r.
    """
    grid = {}
    for r in (2.5, 3.0, 3.5, 4.0):
        grid[f"r={r}"] = {
            "pure_rent_dominated_ratio": round(r, 2),
            "pure_variance_nu0.5_ratio": round(r ** (1.0 / 0.5), 2),
            "pure_variance_nu0.6_ratio": round(r ** (1.0 / 0.6), 2),
            "mixed_50pct_rent_nu0.5_ratio": round(
                float(next(x for x in np.linspace(1.0, 30.0, 29001) if 0.5 * (x ** 0.5) + 0.5 * x >= r)), 2
            ),
        }
    return grid


def train_with_checkpoints(data, ve_layers, d_mem, shared_bank, ctx_gate, inject_res, use_3g,
                           bank_mode, pair_rows=24000, pre_head_mem=False, seed=0, width=32,
                           layers=4, heads=2, steps=300, eval_steps=(75, 150, 225, 300),
                           measure_grad_corr=False, dev="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    DEV = torch.device(dev)
    V = data["V"]
    m = rus.UpgradedModel(
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

    opts = [rls.Muon(mats, 0.02), torch.optim.Adam([dict(params=rest, lr=0.1)], betas=(0.9, 0.95))]
    tgroups = []
    if word_tabs:
        tgroups.append(dict(params=word_tabs, lr=0.1))
    if pair_tabs:
        tgroups.append(dict(params=pair_tabs, lr=0.03))
    if tgroups:
        opts.append(torch.optim.Adam(tgroups, betas=(0.9, 0.95)))
    base = [[g["lr"] for g in o.param_groups] for o in opts]

    X = torch.tensor(data["rows"], device=DEV)
    X_hi = torch.tensor(data["tr_hi"], device=DEV)
    X_mid = torch.tensor(data["tr_mid"], device=DEV)
    X_3g = torch.tensor(data["tr_3g"], device=DEV)

    VX = torch.tensor(data["val"], device=DEV)
    V_hi = torch.tensor(data["val_hi"], device=DEV)
    V_mid = torch.tensor(data["val_mid"], device=DEV)
    V_3g = torch.tensor(data["val_3g"], device=DEV)
    vtg = data["bucket"][data["val"][:, 1:].reshape(-1)]
    pcnt_flat = data["val_pcnt"][:, :-1].reshape(-1)
    fm = np.zeros_like(data["val"][:, :-1], dtype=bool); fm[:, 0] = True
    ff = fm.reshape(-1)

    def eval_now():
        m.eval()
        per = []
        with torch.no_grad():
            for i in range(0, len(VX), 200):
                xb = VX[i:i + 200]
                lg = m(xb[:, :-1], V_hi[i:i + 200, :-1], V_mid[i:i + 200, :-1], V_3g[i:i + 200, :-1])
                per.append(F.cross_entropy(lg.reshape(-1, V), xb[:, 1:].reshape(-1), reduction="none"))
        per = torch.cat(per).float().cpu().numpy()
        sc = {"val_known": round(float(per[vtg < 3].mean()), 4)}
        for lo, hi_b in [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 10**12)]:
            sel = (~ff) & (pcnt_flat >= lo) & (pcnt_flat < hi_b) & (vtg < 3)
            if sel.sum() >= 50:
                sc[f"pc_{lo}"] = round(float(per[sel].mean()), 4)
        m.train()
        return sc

    ckpts = {}
    grad_corrs = []
    batch = 128
    for step in range(steps):
        sc = min((step + 1) / min(50, max(1, steps // 10)), 1.0, (steps - step) / (0.3 * steps))
        for o, bl in zip(opts, base):
            for g, lr0 in zip(o.param_groups, bl):
                g["lr"] = lr0 * sc
        s0, e0 = step * batch, (step + 1) * batch
        xb = X[s0:e0]

        # If measuring cross-layer gradient correlation on the shared bank, hook per-layer proj_p_v inputs
        if measure_grad_corr and shared_bank and (step + 1) in (100, 200, 300):
            layer_p_inputs = []
            x_emb = m.emb[xb[:, :-1]] * m.escale
            w_raw, p_raw = m.bank.lookup(xb[:, :-1], X_hi[s0:e0, :-1], X_mid[s0:e0, :-1], X_3g[s0:e0, :-1])
            h_cur = x_emb
            for blk in m.blocks:
                p_in = p_raw.detach().clone().requires_grad_(True)
                layer_p_inputs.append(p_in)
                h_cur = blk(h_cur, xb[:, :-1], X_hi[s0:e0, :-1], X_mid[s0:e0, :-1], X_3g[s0:e0, :-1], shared_vecs=(w_raw, p_in))
            h_out = m.norm(h_cur)
            lg = F.linear(h_out, m.head)
            F.cross_entropy(lg.reshape(-1, V), xb[:, 1:].reshape(-1)).backward()
            # Compute pairwise cosine similarity between layer gradients on active pair positions
            active_mask = (X_hi[s0:e0, :-1] > 0) | (X_mid[s0:e0, :-1] > 0)
            g_vecs = [p_in.grad[active_mask].detach().float() for p_in in layer_p_inputs]
            off_diag = []
            for l1 in range(len(g_vecs)):
                for l2 in range(l1 + 1, len(g_vecs)):
                    cos = F.cosine_similarity(g_vecs[l1], g_vecs[l2], dim=-1).mean().item()
                    off_diag.append(cos)
            grad_corrs.append(float(np.mean(off_diag)))
            m.zero_grad(set_to_none=True)

        logits = m(xb[:, :-1], X_hi[s0:e0, :-1], X_mid[s0:e0, :-1], X_3g[s0:e0, :-1])
        F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1)).backward()
        for o in opts:
            o.step()
        m.zero_grad(set_to_none=True)

        if (step + 1) in eval_steps:
            ckpts[str(step + 1)] = eval_now()

    return dict(
        params=sum(w.numel() for w in m.parameters()),
        tab_params=sum(w.numel() for w in word_tabs + pair_tabs),
        ckpts=ckpts,
        mean_interlayer_grad_cos=(round(float(np.mean(grad_corrs)), 4) if grad_corrs else None),
    )


def save_incremental(th, f_block):
    th["F_three_component_tests"] = f_block
    with open(THEORY_JSON, "w", encoding="utf-8") as f:
        json.dump(th, f, indent=2)


def main():
    with open(THEORY_JSON, "r", encoding="utf-8") as f:
        th = json.load(f)

    f_block = th.get("F_three_component_tests", {})

    # 1. KKT sensitivity table
    kkt_tab = kkt_sensitivity_table()
    f_block["test1_kkt_tier_ratio_sensitivity"] = kkt_tab
    save_incremental(th, f_block)
    print("=== Test 1: KKT Tier-Ratio c_hi/c_lo Sensitivity (Variance vs Memory-Rent Regime) ===")
    for k, v in kkt_tab.items():
        print(f"  {k}: {v}")

    data = rus.prepare_rich_data(6000, 300, 128, c3_thresh=30)

    # 2. Preregistered Held-Out M Crossing Test: CTRL_hs(M) vs U123 across M in {6000, 24000, 96000} and steps {75, 150, 225, 300}
    print("\n=== Test 2: Held-Out Table Size M Crossing Trajectory (M in {6000, 24000, 96000}, L=4, w=32, 3 seeds) ===")
    prereg_M = f_block.get("test2_heldout_M_crossing_trajectories", {
        "prediction": (
            "As hash table size M shrinks (96K -> 24K -> 6K), collision variance sigma_coll^2(U(T), M) grows faster with T, "
            "so U123's lead over CTRL_hs(M) must open EARLIER in training steps T and reach a LARGER 300-step margin."
        ),
        "trajectories": {},
    })
    f_block["test2_heldout_M_crossing_trajectories"] = prereg_M
    u123_runs = f_block.get("_u123_runs_cache")
    if not u123_runs:
        u123_runs = [
            train_with_checkpoints(data, [0,1,2,3], d_mem=64, shared_bank=True, ctx_gate=True, inject_res=True,
                                   use_3g=True, bank_mode="mp_ve", pre_head_mem=True, seed=sd)
            for sd in range(3)
        ]
        f_block["_u123_runs_cache"] = u123_runs
        prereg_M["U123_ref"] = {
            st: round(float(np.mean([r["ckpts"][st]["val_known"] for r in u123_runs])), 4)
            for st in ("75", "150", "225", "300")
        }
        save_incremental(th, f_block)
    print("  U123 trajectory (75/150/225/300):", prereg_M["U123_ref"])

    for M in (6000, 24000, 96000):
        m_key = f"M_{M}"
        if m_key in prereg_M["trajectories"]:
            traj = prereg_M["trajectories"][m_key]
            line = " | ".join(f"s={st}: {traj[st]['u123_lead_mean']:+.4f} (pc100:{traj[st]['pc100_lead']:+.4f})" for st in ("75", "150", "225", "300"))
            print(f"  [Cached] CTRL_hs(M={M:5d}) - U123 | {line}")
            continue
        hs_runs = [
            train_with_checkpoints(data, [0,1,2,3], d_mem=64, shared_bank=True, ctx_gate=True, inject_res=True,
                                   use_3g=False, bank_mode="hard_stack", pair_rows=M, pre_head_mem=True, seed=sd)
            for sd in range(3)
        ]
        traj = {}
        for st in ("75", "150", "225", "300"):
            diffs = [h["ckpts"][st]["val_known"] - u["ckpts"][st]["val_known"] for h, u in zip(hs_runs, u123_runs)]
            m_d = float(np.mean(diffs))
            sd_d = float(np.std(diffs, ddof=1))
            traj[st] = {
                "hs_val": round(float(np.mean([h["ckpts"][st]["val_known"] for h in hs_runs])), 4),
                "u123_lead_mean": round(m_d, 4),
                "u123_lead_std": round(sd_d, 4),
                "pc100_lead": round(float(np.mean([h["ckpts"][st]["pc_100"] - u["ckpts"][st]["pc_100"] for h, u in zip(hs_runs, u123_runs)])), 4),
            }
        prereg_M["trajectories"][m_key] = traj
        save_incremental(th, f_block)
        line = " | ".join(f"s={st}: {traj[st]['u123_lead_mean']:+.4f} (pc100:{traj[st]['pc100_lead']:+.4f})" for st in ("75", "150", "225", "300"))
        print(f"  CTRL_hs(M={M:5d}) - U123 | {line}")

    # 3. Depth Sweep L in {2, 4, 6} + Direct Layer-Gradient Correlation measurement
    print("\n=== Test 3: Shared-Bank Depth Scaling (L in {2, 4, 6}) & Inter-Layer Gradient Correlation ===")
    depth_res = f_block.get("test3_shared_bank_depth_and_grad_corr", {})
    f_block["test3_shared_bank_depth_and_grad_corr"] = depth_res
    for L in (2, 4, 6):
        l_key = f"L_{L}"
        if l_key in depth_res:
            print(f"  [Cached] L={L}: {depth_res[l_key]}")
            continue
        all_l = list(range(L))
        last_2 = [L - 2, L - 1]
        indep_runs = [
            train_with_checkpoints(data, last_2, d_mem=32, shared_bank=False, ctx_gate=False, inject_res=False,
                                   use_3g=False, bank_mode="mp_ve", layers=L, eval_steps=(300,), seed=sd)
            for sd in range(3)
        ]
        u1_runs = [
            train_with_checkpoints(data, all_l, d_mem=64, shared_bank=True, ctx_gate=False, inject_res=False,
                                   use_3g=False, bank_mode="mp_ve", layers=L, eval_steps=(300,),
                                   measure_grad_corr=(sd == 0), seed=sd)
            for sd in range(3)
        ]
        u1_3g_runs = [
            train_with_checkpoints(data, all_l, d_mem=64, shared_bank=True, ctx_gate=False, inject_res=False,
                                   use_3g=True, bank_mode="mp_ve", layers=L, eval_steps=(300,), seed=sd)
            for sd in range(3)
        ]
        ind_m = float(np.mean([r["ckpts"]["300"]["val_known"] for r in indep_runs]))
        u1_m = float(np.mean([r["ckpts"]["300"]["val_known"] for r in u1_runs]))
        u1_3g_m = float(np.mean([r["ckpts"]["300"]["val_known"] for r in u1_3g_runs]))
        d_u1 = [r0["ckpts"]["300"]["val_known"] - r1["ckpts"]["300"]["val_known"] for r0, r1 in zip(indep_runs, u1_runs)]
        d_3g = [r0["ckpts"]["300"]["val_known"] - r1["ckpts"]["300"]["val_known"] for r0, r1 in zip(u1_runs, u1_3g_runs)]
        rho_grad = u1_runs[0]["mean_interlayer_grad_cos"]
        depth_res[l_key] = {
            "indep_last2_val": round(ind_m, 4),
            "U1_shared_d64_allL_val": round(u1_m, 4),
            "U1_plus_3g_allL_val": round(u1_3g_m, 4),
            "U1_gain_over_indep_mean": round(float(np.mean(d_u1)), 4),
            "U1_gain_over_indep_std": round(float(np.std(d_u1, ddof=1)), 4),
            "3g_gain_inside_U1_mean": round(float(np.mean(d_3g)), 4),
            "measured_interlayer_grad_cosine": rho_grad,
            "predicted_variance_factor_L_over_1_plus_Lminus1_rho": (
                round(L / (1.0 + (L - 1) * max(0.0, rho_grad)), 2) if rho_grad is not None else None
            ),
        }
        save_incremental(th, f_block)
        print(f"  L={L} | indep_last2={ind_m:.4f} | U1_allL={u1_m:.4f} (gain={np.mean(d_u1):+.4f}±{np.std(d_u1,ddof=1):.4f}) | "
              f"U1+3g={u1_3g_m:.4f} (3g_gain={np.mean(d_3g):+.4f}) | grad_cos_rho={rho_grad}")

    # Ensure 3-seed inter-layer gradient cosine rho_mean +- std is populated across L in {2, 4, 6}
    for L in (2, 4, 6):
        l_key = f"L_{L}"
        if "measured_interlayer_grad_cosine_3seed_mean" in depth_res.get(l_key, {}):
            continue
        all_l = list(range(L))
        rhos = [
            train_with_checkpoints(data, all_l, d_mem=64, shared_bank=True, ctx_gate=False, inject_res=False,
                                   use_3g=False, bank_mode="mp_ve", layers=L, eval_steps=(300,),
                                   measure_grad_corr=True, seed=sd)["mean_interlayer_grad_cos"]
            for sd in range(3)
        ]
        rho_m, rho_sd = float(np.mean(rhos)), float(np.std(rhos, ddof=1))
        depth_res[l_key]["measured_interlayer_grad_cosine_3seed_mean"] = round(rho_m, 4)
        depth_res[l_key]["measured_interlayer_grad_cosine_3seed_std"] = round(rho_sd, 4)
        depth_res[l_key]["measured_interlayer_grad_cosine_per_seed"] = [round(float(x), 4) for x in rhos]
        save_incremental(th, f_block)
        print(f"  [3-seed rho] L={L} | rho={rho_m:.4f}+/-{rho_sd:.4f} (seeds={[round(x, 4) for x in rhos]})")

    f_block.pop("_u123_runs_cache", None)
    save_incremental(th, f_block)
    print("\nSaved F_three_component_tests to", THEORY_JSON)


if __name__ == "__main__":
    main()
