#!/usr/bin/env python3
"""Rigorous audit of Lemma 1, Model 1 (collision-load ansatz), Proposition 1, Heuristic 1,
and Empirical Boundary against committed results.

Outputs:
  1. SymPy symbolic integrals of the Zipf bucket-MSE bound (unnormalized fixed-c0 vs
     normalized c0=1/Z_U, and general s != 1).
  2. Full 5-point table (U(T), M, ln(U/M), obs, pred, residual) for CTRL_hs and CTRL_hs_pruned.
  3. Model comparison: ln(U/M) vs ln(M) vs linear (U/M) vs constant null (k, R^2, adj-R^2, AIC, BIC).
  4. Zipf exponent sensitivity for s in {0.8, 1.0, 1.2} (both normalized and fixed-c0).
  5. Honest checks for Proposition 1 (pc_0/pc_2/pc_3/pc_4 + SEs), Heuristic 1 (2.2x vs 6.4x),
     and Empirical Boundary (single-seed plain_eq_weights).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def verify_lemma1_sandwich_mc(n_trials: int = 500, U: int = 4000, M: int = 200, d: int = 16) -> dict:
    """Monte Carlo verification of exact bucket-MSE sandwich:
        sigma_v^2 * (W_b/(p_k+W_b))^2 <= E_v[||v_hat_b - v_k*||^2 | h] <= 2 * sigma_v^2 * (W_b/(p_k+W_b))^2
    """
    rng = np.random.default_rng(42)
    ranks = np.arange(1, U + 1, dtype=np.float64)
    p = 1.0 / ranks
    p /= p.sum()

    h = rng.integers(0, M, size=U)
    bucket_mass = np.bincount(h, weights=p, minlength=M)
    bucket_sq = np.bincount(h, weights=p * p, minlength=M)

    W = bucket_mass[h] - p
    sq_other = np.maximum(0.0, bucket_sq[h] - p * p)
    exact_ratio = (W * W + sq_other) / ((p + W) ** 2)
    lo_ratio = (W / (p + W)) ** 2
    hi_ratio = 2.0 * lo_ratio

    assert np.all(exact_ratio >= lo_ratio - 1e-12)
    assert np.all(exact_ratio <= hi_ratio + 1e-12)

    sample_idx = np.array([0, 10, 50, 200, 1000, 3000])
    emp_mse = np.zeros(len(sample_idx), dtype=np.float64)
    for _ in range(n_trials):
        V = rng.normal(0.0, math.sqrt(1.0 / d), size=(U, d))
        num = np.zeros((M, d), dtype=np.float64)
        np.add.at(num, h, p[:, None] * V)
        v_hat = num / np.maximum(1e-15, bucket_mass[:, None])
        diff = v_hat[h[sample_idx]] - V[sample_idx]
        emp_mse += np.sum(diff * diff, axis=-1) / n_trials

    max_rel_err = float(np.max(np.abs(emp_mse - exact_ratio[sample_idx]) / np.maximum(1e-6, exact_ratio[sample_idx])))
    return {
        "U": U, "M": M, "n_trials": n_trials,
        "all_keys_satisfy_sandwich": True,
        "mean_exact_over_lower_ratio": round(float(np.mean(exact_ratio[W > 0] / lo_ratio[W > 0])), 4),
        "mc_max_rel_err": round(max_rel_err, 4),
    }


def sympy_symbolic_audit() -> dict:
    """Exact symbolic integration of the Zipf bucket-MSE lower bound using SymPy."""
    import sympy as sp

    k, U, M, c0, s = sp.symbols("k U M c0 s", positive=True)
    # s = 1 exact token-weighted integral: int_1^U (c0/k) * ((1/M)/(c0/k + 1/M))^2 dk
    integrand_s1 = (c0 / k) * ((1 / M) / (c0 / k + 1 / M)) ** 2
    exact_s1 = sp.simplify(sp.integrate(integrand_s1, (k, 1, U)))

    # Piecewise min(1, (W/p_k)^2) approximation split at k_c = c0 * M (assuming 1 < c0*M < U):
    kc = c0 * M
    head_s1 = sp.simplify(sp.integrate((c0 / k) * (k / kc) ** 2, (k, 1, kc)))
    tail_s1 = sp.simplify(sp.integrate(c0 / k, (k, kc, U)))
    piecewise_s1 = sp.simplify(head_s1 + tail_s1)

    # Key-averaged (unweighted) integral for s = 1: (1/U) * int_1^U (k/(k + c0*M))^2 dk
    unweighted_s1 = sp.simplify(sp.integrate((k / (k + kc)) ** 2, (k, 1, U)) / U)

    # General s != 1 piecewise tail (k in [kc_s, U] where p_k = c0 * k^(-s) < 1/M -> kc_s = (c0*M)^(1/s))
    kc_s = (c0 * M) ** (1 / s)
    tail_gen_s = sp.simplify(sp.integrate(c0 * k ** (-s), (k, kc_s, U)))

    return {
        "s1_exact_token_weighted": str(exact_s1),
        "s1_piecewise_head": str(head_s1),
        "s1_piecewise_tail": str(tail_s1),
        "s1_piecewise_total": str(piecewise_s1),
        "s1_key_averaged": str(unweighted_s1),
        "general_s_neq_1_tail": str(tail_gen_s),
        "interpretation": (
            "For s=1 with fixed c0 (unnormalized truncated tail on [1, U]), token-weighted integral gives "
            "c0 * (ln((U + c0*M)/(1 + c0*M)) - ... ) ~ c0 * ln(U/(c0*M)) = c0*ln(U/M) - c0*ln(c0) when 1 << c0*M << U. "
            "If normalized on [1, U] so c0 = 1/ln(U), the leading term becomes 1 - ln(M/ln U)/ln U (not pure ln(U/M)). "
            "For general s != 1, the tail scales as power-law (c0/(1-s))*(U^(1-s) - (c0*M)^((1-s)/s)), not ln(U/M). "
            "Key-averaged MSE scales as 1 - O((c0*M/U)*ln(U/(c0*M))), not logarithmic."
        ),
    }


def ols_metrics(x: np.ndarray, y: np.ndarray, constant_only: bool = False) -> dict:
    """Compute OLS fit, residuals, R^2, adjusted R^2, AIC, and BIC (Gaussian likelihood)."""
    n = len(y)
    if constant_only:
        k_params = 1  # intercept only (plus sigma^2 counted consistently below: p = k_params + 1)
        pred = np.full(n, float(np.mean(y)))
        params = {"b": round(float(np.mean(y)), 6)}
    else:
        k_params = 2  # slope + intercept
        a, b = np.polyfit(x, y, 1)
        pred = a * x + b
        params = {"a": round(float(a), 6), "b": round(float(b), 6)}

    resid = y - pred
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / max(1e-15, tss)
    dof = n - k_params
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else float("nan")

    # Standard Gaussian AIC/BIC with p = k_params + 1 (including noise variance sigma^2)
    p_total = k_params + 1
    rss_safe = max(rss, 1e-15)
    aic = n * math.log(rss_safe / n) + 2 * p_total
    bic = n * math.log(rss_safe / n) + p_total * math.log(n)

    return {
        "n": n,
        "k_regression_params": k_params,
        "p_likelihood_params": p_total,
        "params": params,
        "rss": round(rss, 8),
        "rmse": round(math.sqrt(rss / n), 6),
        "r2": round(r2, 5),
        "adj_r2": round(adj_r2, 5),
        "aic": round(aic, 3),
        "bic": round(bic, 3),
        "pred": [round(float(v), 5) for v in pred],
        "residuals": [round(float(v), 5) for v in resid],
    }


def zipf_integral_feature(Us: np.ndarray, Ms: np.ndarray, s: float, normalized: bool) -> np.ndarray:
    """Compute exact token-weighted Zipf integral for exponent s (normalized c0=1/Z_U or fixed c0=1)."""
    out = []
    for u, m in zip(Us, Ms):
        ks = np.geomspace(1.0, float(u), 3000)
        dlnk = math.log(float(u)) / 2999.0
        pk_raw = ks ** (-s)
        Z = float(np.trapezoid(pk_raw * ks, dx=dlnk)) if normalized else 1.0
        pk = pk_raw / Z
        # When fixed c0=1, scale w proportionally using fixed reference Z_ref at U=10^6 so p_k and W=1/M share scale
        w = (1.0 / float(m)) if normalized else (10.0 / float(m))
        val = float(np.trapezoid(pk * ((w / (pk + w)) ** 2) * ks, dx=dlnk))
        out.append(val)
    return np.array(out, dtype=np.float64)


def main() -> None:
    cl = json.load(open(os.path.join(ROOT, "results", "collision_law.json"), encoding="utf-8"))
    th = json.load(open(os.path.join(ROOT, "results", "theory_cstar_analysis.json"), encoding="utf-8"))
    v2 = json.load(open(os.path.join(ROOT, "results", "u123_v2_results.json"), encoding="utf-8"))

    pts = cl["v6000_in_sample_points"]
    cs = cl["cross_scale_predictions"]

    labels = ["V=6k M=6k", "V=6k M=24k", "V=6k M=96k", "30M M=100k", "125M M=1M"]
    Us = np.array([pts[0]["U"], pts[1]["U"], pts[2]["U"], cs["30M_M100k"]["U"], cs["125M_2_5B_M1M"]["U"]], dtype=float)
    Ms = np.array([6000.0, 24000.0, 96000.0, 100000.0, 1000000.0], dtype=float)
    y_hs = np.array([pts[0]["lead_vs_ctrl_hs"], pts[1]["lead_vs_ctrl_hs"], pts[2]["lead_vs_ctrl_hs"],
                     cs["30M_M100k"]["obs_vs_ctrl_hs"], cs["125M_2_5B_M1M"]["obs_vs_ctrl_hs"]], dtype=float)
    y_pr = np.array([pts[0]["lead_vs_ctrl_hs_pruned"], pts[1]["lead_vs_ctrl_hs_pruned"], pts[2]["lead_vs_ctrl_hs_pruned"],
                     cs["30M_M100k"]["obs_vs_pruned"], cs["125M_2_5B_M1M"]["obs_vs_pruned"]], dtype=float)

    ln_um = np.log(Us / Ms)
    ln_m = np.log(Ms)
    lin_um = Us / Ms

    # 1. SymPy symbolic integrals
    sym = sympy_symbolic_audit()

    # 2. Lemma 1 Monte Carlo
    mc = verify_lemma1_sandwich_mc()

    # 3. Model comparison on all 5 points (and 3-pt in-sample -> 2-pt OOS)
    model_comp = {}
    for target_name, y in [("CTRL_hs", y_hs), ("CTRL_hs_pruned", y_pr)]:
        m_log_um = ols_metrics(ln_um, y)
        m_log_m = ols_metrics(ln_m, y)
        m_lin_um = ols_metrics(lin_um, y)
        m_const = ols_metrics(np.zeros_like(y), y, constant_only=True)

        # 3-pt fit evaluated on all 5 points
        a3, b3 = np.polyfit(ln_um[:3], y[:3], 1)
        pred_3pt_all = a3 * ln_um + b3
        resid_3pt_all = y - pred_3pt_all

        point_table = []
        for i, lbl in enumerate(labels):
            point_table.append({
                "point": lbl,
                "U": int(Us[i]),
                "M": int(Ms[i]),
                "U_over_M": round(float(Us[i] / Ms[i]), 3),
                "ln_U_over_M": round(float(ln_um[i]), 4),
                "obs_delta": round(float(y[i]), 5),
                "pred_5pt_log_um": m_log_um["pred"][i],
                "resid_5pt_log_um": m_log_um["residuals"][i],
                "pred_3pt_log_um": round(float(pred_3pt_all[i]), 5),
                "resid_3pt_log_um": round(float(resid_3pt_all[i]), 5),
            })

        # Zipf s sensitivity in {0.8, 1.0, 1.2}
        zipf_sens = {}
        for norm_flag, norm_tag in [(True, "normalized_1_over_ZU"), (False, "fixed_c0_truncated")]:
            zipf_sens[norm_tag] = {}
            for s_val in [0.8, 1.0, 1.2]:
                feat = zipf_integral_feature(Us, Ms, s_val, normalized=norm_flag)
                res5 = ols_metrics(feat, y)
                res3 = ols_metrics(feat[:3], y[:3])
                zipf_sens[norm_tag][f"s={s_val:.1f}"] = {
                    "3pt_r2": res3["r2"],
                    "3pt_adj_r2": res3["adj_r2"],
                    "5pt_r2": res5["r2"],
                    "5pt_adj_r2": res5["adj_r2"],
                    "5pt_aic": res5["aic"],
                    "5pt_bic": res5["bic"],
                }

        model_comp[target_name] = {
            "point_table": point_table,
            "candidates_5pt": {
                "log_U_over_M": m_log_um,
                "log_M_only": m_log_m,
                "linear_U_over_M_null": m_lin_um,
                "constant_null": m_const,
            },
            "zipf_sensitivity": zipf_sens,
        }

    # 4. Proposition 1 (3-term risk + honest 125M per-bucket table with SEs from RESULTS_125M.md)
    # 3-seed paired stats at 2500M for u123 - ctrl_hs_pruned (RESULTS_125M.md):
    # val: -0.02982 +- 0.00325 (SD -> SE = SD/sqrt(3))
    # pc_0: -0.01765 (t=-6.99), pc_2: -0.03265, pc_3: -0.01827, pc_4: -0.03356
    prop1 = {
        "decomposition": "R = R_coll(U, M) + B_rank(r) + V_est(c)",
        "token_ratio_125m_over_30m": round(2.5e9 / 30e6, 2),
        "caveat_cross_scale_confound": "30M vs 125M differs by 83.3x tokens (V_est drops ~83x); clean comparison is same-scale u123 vs ctrl_hs_pruned.",
        "buckets_125m_u123_minus_pruned": {
            "pc_0": {"mean_diff": -0.01765, "p": 0.0198, "note": "Significantly negative -> global non-lookup/adapter-interaction component exists"},
            "pc_2": {"mean_diff": -0.03265, "p": 0.0204, "note": "Statistically indistinguishable from pc_4"},
            "pc_3": {"mean_diff": -0.01827, "p": 0.0374},
            "pc_4": {"mean_diff": -0.03356, "p": 0.0091, "note": "Largest point estimate (includes exact 3g hits)"},
        },
        "30m_same_scale_fullmid_vs_lowrank_hashed": round(
            float(v2["paired_vs_bridge"]["U123_v2_fullmid_vs_U123_d256_prefix10"]["checkpoints"]["val"]["diff_mean"]), 4
        ),
        "30m_same_scale_fullmid_vs_pruned_ns": round(
            float(v2["paired_vs_bridge"]["U123_v2_fullmid_vs_CTRL_hs_pruned_d256_prefix10"]["checkpoints"]["val"]["diff_mean"]), 4
        ),
    }

    # 5. Heuristic 1 (explicitly reporting 2.2x vs 6.4x gap!)
    t3 = th["F_three_component_tests"]["test3_shared_bank_depth_and_grad_corr"]
    Ls = [2, 4, 6]
    rhos = [t3[f"L_{L}"]["measured_interlayer_grad_cosine_3seed_mean"] for L in Ls]
    rho_sds = [t3[f"L_{L}"]["measured_interlayer_grad_cosine_3seed_std"] for L in Ls]
    l_eff = [round(L / (1.0 + (L - 1.0) * r), 3) for L, r in zip(Ls, rhos)]
    g3 = [t3[f"L_{L}"]["3g_gain_inside_U1_mean"] for L in Ls]
    heur1 = {
        "depths": Ls,
        "rho_3seed_mean": rhos,
        "rho_3seed_sd": rho_sds,
        "L_eff": l_eff,
        "L_eff_ratio_L6_over_L2": round(l_eff[-1] / l_eff[0], 2),
        "observed_3g_gain": g3,
        "observed_3g_ratio_L6_over_L2": round(g3[-1] / g3[0], 2),
        "unexplained_multiplier": round((g3[-1] / g3[0]) / (l_eff[-1] / l_eff[0]), 2),
        "caveats": [
            "rho is gradient cosine similarity, not noise correlation (heuristic substitution).",
            "B_conflict(L) is not independently measured.",
            "L=2->4 drop in rho (0.0227) is smaller than L=2 SD (0.0336).",
        ],
    }

    # 6. Empirical Boundary (single seed plain_eq_weights)
    T_ms = np.array([50e6, 100e6, 250e6, 500e6, 1000e6, 2500e6])
    diff_eqw = np.array([-0.06976, -0.07676, -0.03175, +0.00748, +0.03546, +0.07588])
    frac = (0.0 - diff_eqw[2]) / (diff_eqw[3] - diff_eqw[2])
    T_eqw = float(math.exp(math.log(T_ms[2]) + frac * (math.log(T_ms[3]) - math.log(T_ms[2]))))
    emp_bound = {
        "n_seeds_plain_eq_weights": 1,
        "status": "Empirical boundary (single seed; Wave 3 adds seeds 1 & 2)",
        "u123_minus_eqW_s0": diff_eqw.tolist(),
        "crossover_bracket_M": [250, 500],
        "interpolated_T_eqW_M": round(T_eqw / 1e6, 1),
        "u123_b16_speedup_vs_eqW": round(float(np.mean([315104, 315061, 313305])) / 185533.0, 3),
        "u1_b16_speedup_vs_eqW": round(float(np.mean([351328, 352762, 352420])) / 185533.0, 3),
    }

    summary = {
        "frozen_preregistration_commit": "b89068c",
        "lemma1_sandwich_mc": mc,
        "sympy_symbolic_integrals": sym,
        "model1_collision_load_audit": model_comp,
        "proposition1_mid_tier_risk": prop1,
        "heuristic1_cross_layer_snr": heur1,
        "empirical_boundary_eq_weights": emp_bound,
    }

    out_path = os.path.join(ROOT, "results", "theorem_verification.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] wrote {out_path}")
    print("\n=== SYMPY SYMBOLIC INTEGRALS ===")
    for k_s, v_s in sym.items():
        print(f"  {k_s}: {v_s}")

    for tname in ("CTRL_hs", "CTRL_hs_pruned"):
        print(f"\n=== 5-POINT TABLE ({tname}) ===")
        for row in model_comp[tname]["point_table"]:
            print(f"  {row['point']:12s} | U={row['U']:>11,d} M={row['M']:>9,d} ln(U/M)={row['ln_U_over_M']:.4f} | "
                  f"obs={row['obs_delta']:+.5f} pred5={row['pred_5pt_log_um']:+.5f} res5={row['resid_5pt_log_um']:+.5f} | "
                  f"pred3={row['pred_3pt_log_um']:+.5f} res3={row['resid_3pt_log_um']:+.5f}")
        print(f"--- MODEL COMPARISON ({tname}, n=5) ---")
        for cname, cm in model_comp[tname]["candidates_5pt"].items():
            print(f"  {cname:22s} | k={cm['k_regression_params']} R2={cm['r2']:+.4f} adjR2={cm['adj_r2']:+.4f} "
                  f"RMSE={cm['rmse']:.5f} AIC={cm['aic']:+.2f} BIC={cm['bic']:+.2f}")
        print(f"--- ZIPF SENSITIVITY ({tname}) ---")
        for ntag, sd in model_comp[tname]["zipf_sensitivity"].items():
            for sv, sm in sd.items():
                print(f"  {ntag:22s} {sv} | 3pt R2={sm['3pt_r2']:+.4f} | 5pt R2={sm['5pt_r2']:+.4f} adjR2={sm['5pt_adj_r2']:+.4f} AIC={sm['5pt_aic']:+.2f} BIC={sm['5pt_bic']:+.2f}")


if __name__ == "__main__":
    main()
