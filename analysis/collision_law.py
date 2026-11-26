#!/usr/bin/env python3
"""Collision-load scaling law: Delta_loss(U(T), M) ~ a * ln(U(T) / M) + b.

Fits the two-parameter log-linear collision-load law on controlled M-sweep runs
(V=6,000, M in {6k, 24k, 96k}) for both unpruned (CTRL_hs) and pruned
(CTRL_hs_pruned) hash controls, computes phase boundary ratio (U/M)* and M*(T),
and evaluates out-of-sample cross-scale predictions at:
  - 30M tokens (V=50,257, M=100k, U123_v2_fullmid)
  - 2.5B tokens (125M, V=50,257, M=1M, u123 exact-tiered)
  - 5.0B tokens (350M, V=50,257, M=2M, preregistered future point)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import run_upgrade_search as rus  # noqa: E402


def fit_log_linear(x: np.ndarray, y: np.ndarray) -> dict:
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(1e-12, ss_tot)
    ratio_star = float(np.exp(-b / a))
    return dict(
        a=round(float(a), 6),
        b=round(float(b), 6),
        r2=round(r2, 5),
        u_over_m_star=round(ratio_star, 3),
        m_star_coeff=round(1.0 / ratio_star, 5),
    )


def main() -> None:
    th = json.load(open(os.path.join(ROOT, "results", "theory_cstar_analysis.json"), encoding="utf-8"))
    lc = json.load(open(os.path.join(ROOT, "results", "lowcost_experiments.json"), encoding="utf-8"))
    ps = json.load(open(os.path.join(ROOT, "results", "pairstat_v50257.json"), encoding="utf-8"))
    v2 = json.load(open(os.path.join(ROOT, "results", "u123_v2_results.json"), encoding="utf-8"))
    br = json.load(open(os.path.join(ROOT, "results", "v50257_bridge_results.json"), encoding="utf-8"))

    # Exact unique bigrams at V=6,000, 300 steps
    data_6k = rus.prepare_rich_data(6000, 300, 128, c3_thresh=30)
    U_v6k = int(data_6k["n_unique_pairs"])

    t2 = th["F_three_component_tests"]["test2_heldout_M_crossing_trajectories"]["trajectories"]
    t2b = lc["part1_test2b_pruned_M_sweep"]
    Ms_6k = [6000, 24000, 96000]
    y_hs_6k = np.array([t2[f"M_{m}"]["300"]["u123_lead_mean"] for m in Ms_6k], dtype=float)
    y_pr_6k = np.array([t2b[f"CTRL_hs_pruned_M_{m}"]["trajectories"]["300"]["u123_300_minus_pruned"] for m in Ms_6k], dtype=float)
    x_6k = np.log(U_v6k / np.array(Ms_6k, dtype=float))

    # FineWeb V=50,257 Heaps' law from exact unique pair counts (1.2M..100M tokens)
    exact_ps = ps["exact"]
    Ts_ps = np.array([e["T"] for e in exact_ps], dtype=float)
    Us_ps = np.array([e["unique_pairs"] for e in exact_ps], dtype=float)
    beta_heaps, log_c_heaps = np.polyfit(np.log(Ts_ps), np.log(Us_ps), 1)
    heaps_pred = np.exp(log_c_heaps) * (Ts_ps ** beta_heaps)
    heaps_r2 = 1.0 - float(np.sum((np.log(Us_ps) - np.log(heaps_pred)) ** 2) /
                             np.sum((np.log(Us_ps) - np.mean(np.log(Us_ps))) ** 2))

    U_30M = float(np.exp(log_c_heaps) * (30e6 ** beta_heaps))
    U_2_5B = float(np.exp(log_c_heaps) * (2.5e9 ** beta_heaps))
    U_5_0B = float(np.exp(log_c_heaps) * (5.0e9 ** beta_heaps))

    # Observed cross-scale deltas (Hash minus Tiered-U123, positive = U123 better)
    u123_30m_vals = [float(s["checkpoints"]["30.0M"]["val"]) for s in v2["arms"]["U123_v2_fullmid"]["seeds"].values()]
    hs_30m_vals = [float(s["checkpoints"]["30.0M"]["val"]) for s in br["arms"]["CTRL_hs_d256_prefix10"]["seeds"].values()]
    obs_30M_hs = float(np.mean(hs_30m_vals) - np.mean(u123_30m_vals))
    obs_30M_pr = float(v2["paired_vs_bridge"]["U123_v2_fullmid_vs_CTRL_hs_pruned_d256_prefix10"]["checkpoints"]["val"]["diff_mean"])
    obs_125M_hs = 0.03384
    obs_125M_pr = 0.02982

    x_30M = float(np.log(U_30M / 100_000.0))
    x_125M = float(np.log(U_2_5B / 1_000_000.0))
    x_350M = float(np.log(U_5_0B / 2_000_000.0))

    fit_hs_3pt = fit_log_linear(x_6k, y_hs_6k)
    fit_pr_3pt = fit_log_linear(x_6k, y_pr_6k)

    # Joint 5-point fit across (V=6k M in {6k,24k,96k}, 30M, 125M)
    x_5pt = np.concatenate([x_6k, [x_30M, x_125M]])
    y_hs_5pt = np.concatenate([y_hs_6k, [obs_30M_hs, obs_125M_hs]])
    y_pr_5pt = np.concatenate([y_pr_6k, [obs_30M_pr, obs_125M_pr]])
    fit_hs_5pt = fit_log_linear(x_5pt, y_hs_5pt)
    fit_pr_5pt = fit_log_linear(x_5pt, y_pr_5pt)

    out = {
        "heaps_law_fineweb": {
            "beta": round(float(beta_heaps), 5),
            "c": round(float(np.exp(log_c_heaps)), 3),
            "log_log_r2": round(heaps_r2, 6),
            "U_30M": round(U_30M),
            "U_2_5B": round(U_2_5B),
            "U_5_0B": round(U_5_0B),
        },
        "v6000_in_sample_points": [
            {
                "label": f"V=6k, M={m//1000}k",
                "M": m,
                "U": U_v6k,
                "U_over_M": round(U_v6k / m, 3),
                "ln_U_over_M": round(float(np.log(U_v6k / m)), 4),
                "lead_vs_ctrl_hs": round(float(yh), 4),
                "lead_vs_ctrl_hs_pruned": round(float(yp), 4),
            }
            for m, yh, yp in zip(Ms_6k, y_hs_6k, y_pr_6k)
        ],
        "fits": {
            "v6k_3pt_vs_ctrl_hs": fit_hs_3pt,
            "v6k_3pt_vs_ctrl_hs_pruned": fit_pr_3pt,
            "joint_5pt_vs_ctrl_hs": fit_hs_5pt,
            "joint_5pt_vs_ctrl_hs_pruned": fit_pr_5pt,
        },
        "cross_scale_predictions": {
            "30M_M100k": {
                "U": round(U_30M),
                "M": 100_000,
                "U_over_M": round(U_30M / 100_000.0, 2),
                "ln_U_over_M": round(x_30M, 4),
                "pred_3pt_vs_ctrl_hs": round(fit_hs_3pt["a"] * x_30M + fit_hs_3pt["b"], 4),
                "obs_vs_ctrl_hs": round(obs_30M_hs, 4),
                "pred_3pt_vs_pruned": round(fit_pr_3pt["a"] * x_30M + fit_pr_3pt["b"], 4),
                "obs_vs_pruned": round(obs_30M_pr, 4),
            },
            "125M_2_5B_M1M": {
                "U": round(U_2_5B),
                "M": 1_000_000,
                "U_over_M": round(U_2_5B / 1_000_000.0, 2),
                "ln_U_over_M": round(x_125M, 4),
                "pred_3pt_vs_ctrl_hs": round(fit_hs_3pt["a"] * x_125M + fit_hs_3pt["b"], 4),
                "obs_vs_ctrl_hs": round(obs_125M_hs, 4),
                "pred_3pt_vs_pruned": round(fit_pr_3pt["a"] * x_125M + fit_pr_3pt["b"], 4),
                "obs_vs_pruned": round(obs_125M_pr, 4),
            },
            "350M_5B_M2M_preregistered": {
                "U": round(U_5_0B),
                "M": 2_000_000,
                "U_over_M": round(U_5_0B / 2_000_000.0, 2),
                "ln_U_over_M": round(x_350M, 4),
                "pred_3pt_vs_ctrl_hs": round(fit_hs_3pt["a"] * x_350M + fit_hs_3pt["b"], 4),
                "pred_3pt_vs_pruned": round(fit_pr_3pt["a"] * x_350M + fit_pr_3pt["b"], 4),
                "pred_5pt_vs_pruned": round(fit_pr_5pt["a"] * x_350M + fit_pr_5pt["b"], 4),
            },
        },
    }

    out_path = os.path.join(ROOT, "results", "collision_law.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[ok] wrote {out_path}")
    print("  3-pt vs CTRL_hs:       ", fit_hs_3pt)
    print("  3-pt vs CTRL_hs_pruned:", fit_pr_3pt)
    print("  5-pt vs CTRL_hs:       ", fit_hs_5pt)
    print("  5-pt vs CTRL_hs_pruned:", fit_pr_5pt)
    print("  30M pred vs obs:       ", out["cross_scale_predictions"]["30M_M100k"])
    print("  125M pred vs obs:      ", out["cross_scale_predictions"]["125M_2_5B_M1M"])
    print("  350M preregistered:    ", out["cross_scale_predictions"]["350M_5B_M2M_preregistered"])


if __name__ == "__main__":
    main()
