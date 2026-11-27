#!/usr/bin/env python3
"""Rigorous audit of the re-derived v6.1 theoretical framework:
  - Theorem 1: Finite-sample hash-vs-exact risk identity (under i.i.d. key/noise, sample-mean model;
               zero Zipf assumption) — SymPy exact verification + Monte Carlo verification.
  - Theorem 2: Three-tier spectral truncation thresholds on 1-gram residual targets — SymPy exact
               algebraic verification + debiased eta prediction band.
  - Proposition 1: Properly normalized Sub-Zipf (1/2 < s < 1) exact smooth collision integral and
               Euler-reflection Beta closed-form asymptotic expansion.
  - Model 1: 2-parameter MSE -> NLL linear link across n = 5 empirical points (true M=[6k,24k,96k,100k,1M]),
               with residual SE (df = n - 2 = 3), adj-R^2, AIC/BIC, 125M semi-OOS relative error,
               and pre-unblinding 350M x 5B (M = 2M) predictions with 95% prediction intervals.
  - Heuristic 1: Cross-layer gradient variance reduction diagnostic (2.20x vs 6.43x gap).
  - Boundary 1: Single-seed equal-total-weights crossover empirical boundary.

Does NOT modify frozen b89068c artifacts (results/collision_law.json, proposal tex, run_350m_5b.sh).
Writes:
  - results/subzipf_350m_predictions.json  (for separate pre-unblinding commit)
  - results/theorem_verification.json      (full audit output for docs/THEORY_v6_DRAFT.md)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def verify_theorem1_sympy_and_mc(
    n_trials: int = 800, U: int = 2000, M: int = 100, d: int = 16, sigma_xi2: float = 8.0
) -> dict:
    """Verify Theorem 1 finite-sample identity algebraically (SymPy) and via Monte Carlo."""
    import sympy as sp

    nk, Nb, S2, sv2, sx2 = sp.symbols("n_k N_b S_2 sigma_v2 sigma_xi2", positive=True)
    mse_hash = sv2 * (Nb**2 + S2) / (nk + Nb) ** 2 + sx2 / (nk + Nb)
    mse_exact = sx2 / nk
    diff = sp.simplify(mse_hash - mse_exact)
    expected = sv2 * (Nb**2 + S2) / (nk + Nb) ** 2 - sx2 * Nb / (nk * (nk + Nb))
    sympy_diff_zero = sp.simplify(diff - expected) == 0

    # Harmonic-mean phase boundary exact equivalence:
    # Delta > 0  <=>  (1/n_k) + (1/N_b) < (sigma_v^2 / sigma_xi^2) * (1 + S_2 / N_b^2)
    # Check exact algebraic equivalence: diff * nk * (nk + Nb)**2 / (sx2 * Nb * nk * Nb)
    lhs_harmonic = (sv2 / sx2) * (1 + S2 / Nb**2) - (1 / nk + 1 / Nb)
    scale_pos = sx2 * Nb**2 / (nk + Nb) ** 2
    sympy_harmonic_equiv = sp.simplify(diff - scale_pos * lhs_harmonic) == 0
    # Check exact limit N_b -> oo with S2 / N_b^2 -> 0 reduces to n_k > sigma_xi^2 / sigma_v^2 (Thm 2 full-rank vs prune)
    ratio_s2 = sp.Symbol("rho_2", nonnegative=True)
    boundary_expr = sp.Eq(1 / nk + 1 / Nb, (sv2 / sx2) * (1 + ratio_s2))
    limit_nk_prune = sp.simplify(sp.solve(boundary_expr.subs({1 / Nb: 0, ratio_s2: 0}), nk)[0])
    sympy_prune_limit_exact = sp.simplify(limit_nk_prune - sx2 / sv2) == 0

    # Crossover condition: Delta_k < 0  <=>  n_k < (sigma_xi^2 / sigma_v^2) * N_b * (n_k + N_b) / (N_b^2 + S_2)
    # Monte Carlo check covering BOTH regimes: low-count keys (pooling wins, Delta < 0) & high-count (collision wins, Delta > 0)
    rng = np.random.default_rng(123)
    ranks = np.arange(1, U + 1, dtype=np.float64)
    p = ranks ** (-0.855)
    p /= p.sum()
    n_counts = np.maximum(1, np.round(p * 12000.0).astype(np.int64))

    h = rng.integers(0, M, size=U)
    bucket_n = np.bincount(h, weights=n_counts, minlength=M)
    bucket_n2 = np.bincount(h, weights=n_counts * n_counts, minlength=M)

    Nb_arr = bucket_n[h] - n_counts
    S2_arr = np.maximum(0.0, bucket_n2[h] - n_counts * n_counts)
    sigma_v2 = 1.0

    theory_hash = sigma_v2 * (Nb_arr**2 + S2_arr) / ((n_counts + Nb_arr) ** 2) + sigma_xi2 / (n_counts + Nb_arr)
    theory_exact = sigma_xi2 / n_counts
    theory_diff = theory_hash - theory_exact

    # Pick 6 representative keys: 3 highest positive (collision-dominated) and 3 most negative (pooling-dominated)
    order = np.argsort(theory_diff)
    sample_idx = np.array([order[-1], order[-5], order[-15], order[len(order) // 4], order[10], order[0]])

    emp_hash = np.zeros(len(sample_idx), dtype=np.float64)
    emp_exact = np.zeros(len(sample_idx), dtype=np.float64)
    for _ in range(n_trials):
        V = rng.normal(0.0, math.sqrt(sigma_v2 / d), size=(U, d))
        # Per-key summed observation noise: xi_sum_j ~ N(0, n_j * (sigma_xi2 / d) * I_d)
        xi_sum = rng.normal(0.0, 1.0, size=(U, d)) * np.sqrt(n_counts[:, None] * (sigma_xi2 / d))
        # Exact estimator
        v_ex = V[sample_idx] + xi_sum[sample_idx] / n_counts[sample_idx, None]
        emp_exact += np.sum((v_ex - V[sample_idx]) ** 2, axis=-1) / n_trials
        # Hashed bucket estimator
        num = np.zeros((M, d), dtype=np.float64)
        np.add.at(num, h, n_counts[:, None] * V + xi_sum)
        v_hat = num / bucket_n[:, None]
        emp_hash += np.sum((v_hat[h[sample_idx]] - V[sample_idx]) ** 2, axis=-1) / n_trials

    emp_diff = emp_hash - emp_exact
    max_rel_err_hash = float(np.max(np.abs(emp_hash - theory_hash[sample_idx]) / theory_hash[sample_idx]))
    max_rel_err_exact = float(np.max(np.abs(emp_exact - theory_exact[sample_idx]) / theory_exact[sample_idx]))
    sign_match = bool(np.all(np.sign(emp_diff) == np.sign(theory_diff[sample_idx])))

    return {
        "sympy_identity_diff_is_zero": bool(sympy_diff_zero),
        "sympy_harmonic_phase_boundary_exact": bool(sympy_harmonic_equiv),
        "sympy_infinite_collision_prune_limit_eq_sx2_over_sv2": bool(sympy_prune_limit_exact),
        "harmonic_phase_boundary": "1/n_k + 1/N_b < (sigma_v^2 / sigma_xi^2) * (1 + S_2 / N_b^2)",
        "assumptions": (
            "Idealized i.i.d. zero-mean key targets (E||v_k||^2 = sigma_v^2), i.i.d. observation noise "
            "(E||xi||^2 = sigma_xi^2), uniform hash h, and sample-mean bucket estimator. Zero Zipf / "
            "frequency-distribution assumption."
        ),
        "mc_verification": {
            "U": U,
            "M": M,
            "d": d,
            "sigma_v2": sigma_v2,
            "sigma_xi2": sigma_xi2,
            "n_trials": n_trials,
            "sample_n_counts": [int(n_counts[i]) for i in sample_idx],
            "theory_delta_hash_minus_exact": [round(float(theory_diff[i]), 4) for i in sample_idx],
            "mc_delta_hash_minus_exact": [round(float(x), 4) for x in emp_diff],
            "sign_match_all_sampled_keys": sign_match,
            "max_rel_err_hash_mse": round(max_rel_err_hash, 4),
            "max_rel_err_exact_mse": round(max_rel_err_exact, 4),
            "frac_keys_pooling_wins_negative_delta": round(float(np.mean(theory_diff < 0.0)), 4),
        },
    }


def verify_theorem2_sympy() -> dict:
    """Verify Theorem 2 spectral truncation thresholds on 1-gram residual target."""
    import sympy as sp

    E_top, E_tail, r, d, c, sx2 = sp.symbols("E_top E_tail r d c sigma_xi2", positive=True)
    # Risks defined strictly on 1-gram residual target v_k^{res} = v_k^{bigram} - v_{word(k)}
    R_prune = E_top + E_tail
    R_mid = E_tail + (r / d) * sx2 / c
    R_full = sx2 / c

    c_lo = sp.simplify(sp.solve(sp.Eq(R_prune, R_mid), c)[0])
    c_hi = sp.simplify(sp.solve(sp.Eq(R_mid, R_full), c)[0])
    ratio = sp.simplify(c_hi / c_lo)

    # Prediction band for r = d/4 (so (d-r)/r = 3) across debiased eta in [0.65, 0.80]
    band = {}
    for eta in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        band[f"eta={eta:.2f}"] = round(3.0 * eta / (1.0 - eta), 2)

    return {
        "target_definition": "1-gram residual target v_k^{res} = v_k^{bigram} - v_{word(k)}, so R_prune = E_top + E_tail is exact 1-gram fallback residual risk.",
        "c_lo_formula": str(c_lo),
        "c_hi_formula": str(c_hi),
        "ratio_c_hi_over_c_lo": str(ratio),
        "existence_condition_c_lo_lt_c_hi": "eta_r = E_top / (E_top + E_tail) > r / d",
        "prediction_band_r_eq_d_over_4": band,
        "status_note": (
            "Given a noise-debiased estimate of eta_r (e.g., split-half cross-covariance on ultra-high-count "
            "full-rank keys), the ratio c_hi/c_lo has no additional free parameters. Empirical architecture "
            "thresholds 10/100 (10x) were chosen prior to this formula and are retrospectively consistent "
            "with eta_{d/4} in [0.65, 0.80] (band [5.57x, 12.00x], 9.00x at eta=0.75)."
        ),
    }


def exact_subzipf_R(U_vec: np.ndarray, M_vec: np.ndarray, s: float) -> np.ndarray:
    """Exact properly-normalized sub-Zipf collision risk integral R_coll(U, M) / P_S
    for 0 < s < 1 and arbitrary lambda = U / M > 0:
        R_coll / P_S = int_0^1 1 / (1 + ((1 - s) / lambda) * t^(-s / (1 - s)))^2 dt
    """
    out = []
    t = np.linspace(1e-12, 1.0, 50000)
    t_pow = t ** (-s / (1.0 - s))
    for Uv, Mv in zip(U_vec, M_vec):
        lam = float(Uv) / float(Mv)
        ratio = ((1.0 - s) / lam) * t_pow
        out.append(float(np.trapezoid(1.0 / (1.0 + ratio) ** 2, t)))
    return np.array(out, dtype=np.float64)


def verify_proposition1_subzipf() -> dict:
    """Verify Proposition 1: exact sub-Zipf integral and Euler-reflection Beta closed-form asymptotics."""
    import sympy as sp

    # Verify Beta / Euler reflection integral symbolically for a = (1-s)/s in (0, 1):
    # int_0^inf v^(a-1) * (1 + 2v) / (1 + v)^2 dv = (1 + a) * pi / sin(pi * a)
    a = sp.Symbol("a", positive=True)
    # Check numerically at a = (1-s)/s for s in {0.800, 0.823, 0.855, 0.900}
    checks = []
    for sv in [0.800, 0.823, 0.855, 0.900]:
        a_exp = (1.0 - sv) / sv
        C_exact = ((1.0 - sv) / (sv * sv)) * (math.pi / math.sin(math.pi * a_exp)) * ((1.0 - sv) ** a_exp)
        C_pw = (2.0 * sv / (1.0 + sv)) * ((1.0 - sv) ** a_exp)
        r_num_100 = float(exact_subzipf_R(np.array([100.0]), np.array([1.0]), sv)[0])
        r_asym_100 = 1.0 - C_exact * (100.0 ** (-a_exp))
        r_num_10 = float(exact_subzipf_R(np.array([10.0]), np.array([1.0]), sv)[0])
        r_asym_10 = 1.0 - C_exact * (10.0 ** (-a_exp))
        checks.append({
            "s": sv,
            "alpha_coll": round(a_exp, 4),
            "C_exact": round(C_exact, 4),
            "C_piecewise": round(C_pw, 4),
            "lam_100_exact_integral": round(r_num_100, 5),
            "lam_100_asym_closed_form": round(r_asym_100, 5),
            "lam_100_rel_err_pct": round(abs(r_num_100 - r_asym_100) / r_num_100 * 100.0, 3),
            "lam_10_exact_integral": round(r_num_10, 5),
            "lam_10_asym_closed_form": round(r_asym_10, 5),
        })

    # Also record old s=1 SymPy check for transparency (showing why s=1 normalized failed)
    k, U, M, c0 = sp.symbols("k U M c0", positive=True)
    exact_s1_unnorm = sp.simplify(sp.integrate((c0 / k) * ((1 / M) / (c0 / k + 1 / M)) ** 2, (k, 1, U)))

    return {
        "empirical_bigram_zipf_exponents": {
            "V6000_2_42M_tokens_rank_10_10k": 0.823,
            "FineWeb_30M_tokens_rank_100_100k": 0.855,
            "note": "Measured directly from bigram frequency-rank curves, NEVER fitted to NLL.",
        },
        "exact_integral_formula": "R_coll(U_S, M) / P_S = int_0^1 (1 + ((1-s)/lambda_S) * t^(-s/(1-s)))^(-2) dt, lambda_S = U_S / M",
        "asymptotic_formula_s_in_0_5_to_1": "1 - C_exact(s) * lambda_S^(-(1-s)/s) + O(lambda_S^(-(2-s)/s))",
        "C_exact_formula": "((1-s)/s^2) * (pi / sin(pi*(1-s)/s)) * (1-s)^((1-s)/s)",
        "numerical_checks": checks,
        "legacy_s1_unnorm_sympy_ref": str(exact_s1_unnorm),
    }


def ols_with_pi(x: np.ndarray, y: np.ndarray, x_new: float, constant_only: bool = False) -> dict:
    """OLS fit with residual SE (df = n - k), adj-R^2, AIC/BIC, and 95% prediction interval at x_new."""
    n = len(y)
    if constant_only:
        k_params = 1
        df = n - k_params
        pred = np.full(n, float(np.mean(y)))
        params = {"b": round(float(np.mean(y)), 6)}
        pred_new = float(np.mean(y))
        resid = y - pred
        rss = float(np.sum(resid**2))
        se_res = math.sqrt(rss / df)
        se_pred_new = se_res * math.sqrt(1.0 + 1.0 / n)
    else:
        k_params = 2
        df = n - k_params
        a, b = np.polyfit(x, y, 1)
        pred = a * x + b
        params = {"a": round(float(a), 6), "b": round(float(b), 6)}
        pred_new = float(a * x_new + b)
        resid = y - pred
        rss = float(np.sum(resid**2))
        se_res = math.sqrt(rss / df)
        xbar = float(np.mean(x))
        sxx = float(np.sum((x - xbar) ** 2))
        se_pred_new = se_res * math.sqrt(1.0 + 1.0 / n + ((x_new - xbar) ** 2) / max(1e-15, sxx))

    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / max(1e-15, tss)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df if (not constant_only and df > 0) else 0.0
    p_total = k_params + 1  # Gaussian MLE counts variance sigma^2
    rss_safe = max(rss, 1e-15)
    aic = n * math.log(rss_safe / n) + 2 * p_total
    bic = n * math.log(rss_safe / n) + p_total * math.log(n)
    tcrit = float(stats.t.ppf(0.975, df))
    pi95 = [round(pred_new - tcrit * se_pred_new, 5), round(pred_new + tcrit * se_pred_new, 5)]

    return {
        "n": n,
        "k_regression_params": k_params,
        "df": df,
        "p_likelihood_params": p_total,
        "params": params,
        "rss": round(rss, 9),
        "rmse": round(math.sqrt(rss / n), 6),
        "residual_se": round(se_res, 6),
        "r2": round(r2, 5),
        "adj_r2": round(adj_r2, 5),
        "aic": round(aic, 3),
        "bic": round(bic, 3),
        "pred": [round(float(v), 5) for v in pred],
        "residuals": [round(float(v), 5) for v in resid],
        "pred_350m": round(pred_new, 5),
        "se_pred_350m": round(se_pred_new, 5),
        "tcrit_95": round(tcrit, 4),
        "pi95_350m": pi95,
    }


def analyze_arm(
    labels: list[str],
    U_vec: np.ndarray,
    M_vec: np.ndarray,
    y: np.ndarray,
    U_350: float,
    M_350: float,
) -> dict:
    """Evaluate Exact Sub-Zipf (s=0.855, 0.823), ln(U/M), linear(U/M), ln(M), and constant null."""
    cand_defs = [
        ("exact_subzipf_s0855", lambda u, m: exact_subzipf_R(u, m, 0.855), False),
        ("exact_subzipf_s0823", lambda u, m: exact_subzipf_R(u, m, 0.823), False),
        ("ln_U_over_M_tangent", lambda u, m: np.log(u / m), False),
        ("linear_U_over_M_null", lambda u, m: u / m, False),
        ("ln_M_only_null", lambda u, m: np.log(m), False),
        ("constant_null", lambda u, m: np.zeros_like(m), True),
    ]

    results = {}
    point_tables = {}
    for name, fn, is_const in cand_defs:
        x = fn(U_vec, M_vec)
        x350 = float(fn(np.array([U_350]), np.array([M_350]))[0])
        res5 = ols_with_pi(x, y, x350, constant_only=is_const)

        # 3-pt V=6k in-sample -> 30M (OOS) & 125M (semi-OOS, Heaps-extrapolated U) & 350M (true OOS)
        if not is_const:
            a3, b3 = np.polyfit(x[:3], y[:3], 1)
            pred3_all = a3 * x + b3
            resid3_all = y - pred3_all
            r2_3 = 1.0 - float(np.sum(resid3_all[:3] ** 2)) / float(np.sum((y[:3] - np.mean(y[:3])) ** 2))
            pred3_350 = float(a3 * x350 + b3)
            res5["v6k_3pt_fit"] = {
                "r2_3pt": round(r2_3, 5),
                "a_3pt": round(float(a3), 6),
                "b_3pt": round(float(b3), 6),
                "oos_30m_pred": round(float(pred3_all[3]), 5),
                "oos_30m_obs": round(float(y[3]), 5),
                "oos_30m_rel_err_pct": round(float((pred3_all[3] - y[3]) / abs(y[3]) * 100.0), 1),
                "semi_oos_125m_pred": round(float(pred3_all[4]), 5),
                "semi_oos_125m_obs": round(float(y[4]), 5),
                "semi_oos_125m_rel_err_pct": round(float((pred3_all[4] - y[4]) / abs(y[4]) * 100.0), 1),
                "true_oos_350m_pred": round(pred3_350, 5),
            }
            if name in ("exact_subzipf_s0855", "ln_U_over_M_tangent"):
                rows = []
                for i, lb in enumerate(labels):
                    rows.append({
                        "point": lb,
                        "U_active": int(round(float(U_vec[i]))),
                        "M": int(round(float(M_vec[i]))),
                        "lambda_U_over_M": round(float(U_vec[i] / M_vec[i]), 3),
                        "feature_x": round(float(x[i]), 5),
                        "obs_delta": round(float(y[i]), 5),
                        "pred_5pt": res5["pred"][i],
                        "resid_5pt": res5["residuals"][i],
                        "pred_3pt": round(float(pred3_all[i]), 5),
                        "resid_3pt": round(float(resid3_all[i]), 5),
                        "oos_status": "in-sample" if i < 3 else ("OOS (measured U)" if i == 3 else "semi-OOS (Heaps U)"),
                    })
                point_tables[name] = rows
        results[name] = res5

    return {
        "U_active_5pt": [int(round(float(u))) for u in U_vec],
        "M_5pt": [int(round(float(m))) for m in M_vec],
        "U_active_350m": int(round(float(U_350))),
        "M_350m": int(round(float(M_350))),
        "lambda_350m": round(float(U_350 / M_350), 3),
        "candidates": results,
        "point_tables": point_tables,
    }


def main() -> None:
    cl = json.load(open(os.path.join(ROOT, "results", "collision_law.json"), encoding="utf-8"))
    ps = json.load(open(os.path.join(ROOT, "results", "pairstat_v50257.json"), encoding="utf-8"))["exact"]
    th = json.load(open(os.path.join(ROOT, "results", "theory_cstar_analysis.json"), encoding="utf-8"))
    v2 = json.load(open(os.path.join(ROOT, "results", "u123_v2_results.json"), encoding="utf-8"))

    # Verbatim 5-point dataset from frozen b89068c collision_law.json (TRUE M = [6k, 24k, 96k, 100k, 1M])
    pts = cl["v6000_in_sample_points"]
    cs = cl["cross_scale_predictions"]
    heaps = cl["heaps_law_fineweb"]
    labels = ["V=6k M=6k", "V=6k M=24k", "V=6k M=96k", "30M M=100k", "125M M=1M"]

    U_all = np.array([pts[0]["U"], pts[1]["U"], pts[2]["U"], cs["30M_M100k"]["U"], cs["125M_2_5B_M1M"]["U"]], dtype=float)
    Ms = np.array([6000.0, 24000.0, 96000.0, 100000.0, 1000000.0], dtype=float)
    y_hs = np.array(
        [pts[0]["lead_vs_ctrl_hs"], pts[1]["lead_vs_ctrl_hs"], pts[2]["lead_vs_ctrl_hs"],
         cs["30M_M100k"]["obs_vs_ctrl_hs"], cs["125M_2_5B_M1M"]["obs_vs_ctrl_hs"]],
        dtype=float,
    )
    y_pr = np.array(
        [pts[0]["lead_vs_ctrl_hs_pruned"], pts[1]["lead_vs_ctrl_hs_pruned"], pts[2]["lead_vs_ctrl_hs_pruned"],
         cs["30M_M100k"]["obs_vs_pruned"], cs["125M_2_5B_M1M"]["obs_vs_pruned"]],
        dtype=float,
    )

    # Exact Heaps projections using frozen b89068c parameters (c=23.413, beta=0.71368)
    c_h, beta_h = float(heaps["c"]), float(heaps["beta"])
    U_250M = c_h * (250e6 ** beta_h)    # exact mature_keys count-prefix at 125M (eff_mid = 1 on 250M prefix)
    U_500M = c_h * (500e6 ** beta_h)    # exact mature_keys count-prefix at 350M (eff_mid = 1 on 500M prefix)
    U_5B = float(heaps["U_5_0B"])       # exact frozen U(5B) = 195,612,435

    # Sensitivity definition for U_keep: mature pairs (c >= 10) Heaps law from pairstat_v50257.json
    Ts_ps = np.array([x["T"] for x in ps], dtype=float)
    U_mats_ps = np.array([x["n_hi"] + x["n_mid"] for x in ps], dtype=float)
    beta_mat, lnc_mat = np.polyfit(np.log(Ts_ps), np.log(U_mats_ps), 1)
    U_mat_2_5B = math.exp(float(lnc_mat)) * (2.5e9 ** float(beta_mat))
    U_mat_5B = math.exp(float(lnc_mat)) * (5.0e9 ** float(beta_mat))

    # Active U_keep arrays: V=6k exact U(>=10)=31,106; FineWeb 30M exact U(>=10)=366,259
    U_keep_prefix = np.array([31106.0, 31106.0, 31106.0, 366259.0, U_250M], dtype=float)
    U_keep_mature = np.array([31106.0, 31106.0, 31106.0, 366259.0, U_mat_2_5B], dtype=float)

    # Run all verifications
    thm1 = verify_theorem1_sympy_and_mc()
    thm2 = verify_theorem2_sympy()
    prop1 = verify_proposition1_subzipf()

    arm_hs = analyze_arm(labels, U_all, Ms, y_hs, U_5B, 2000000.0)
    arm_pr_prefix = analyze_arm(labels, U_keep_prefix, Ms, y_pr, U_500M, 2000000.0)
    arm_pr_mature = analyze_arm(labels, U_keep_mature, Ms, y_pr, U_mat_5B, 2000000.0)

    # Empirical scope for Theorem 2 / 125M per-bucket honesty
    emp_buckets_125m = {
        "token_ratio_125m_over_30m": round(2.5e9 / 30e6, 2),
        "caveat_cross_scale_confound": "30M vs 125M differs by 83.3x tokens (V_est drops ~83x); clean comparisons are same-scale only.",
        "buckets_125m_u123_minus_pruned": {
            "pc_0": {"mean_diff": -0.01765, "se": 0.00437, "p": 0.0198, "note": "Significantly negative -> global non-lookup/adapter-interaction component"},
            "pc_2": {"mean_diff": -0.03265, "se": 0.00821, "p": 0.0204, "note": "Statistically indistinguishable from pc_4 (diff=0.0009, SE~0.010)"},
            "pc_3": {"mean_diff": -0.01827, "se": 0.00630, "p": 0.0374},
            "pc_4": {"mean_diff": -0.03356, "se": 0.00558, "p": 0.0091, "note": "Largest point estimate (includes exact 3g hits)"},
        },
        "30m_same_scale_fullmid_vs_lowrank_hashed": round(
            float(v2["paired_vs_bridge"]["U123_v2_fullmid_vs_U123_d256_prefix10"]["checkpoints"]["val"]["diff_mean"]), 4
        ),
        "30m_same_scale_fullmid_vs_pruned_ns": round(
            float(v2["paired_vs_bridge"]["U123_v2_fullmid_vs_CTRL_hs_pruned_d256_prefix10"]["checkpoints"]["val"]["diff_mean"]), 4
        ),
    }

    # Heuristic 1 (explicit 2.20x vs 6.43x gap)
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
            "L=2->4 drop in rho (-0.0227) is smaller than L=2 SD (0.0336).",
        ],
    }

    # Boundary 1 (single-seed plain_eq_weights)
    T_ms = np.array([50e6, 100e6, 250e6, 500e6, 1000e6, 2500e6])
    diff_eqw = np.array([-0.06976, -0.07676, -0.03175, +0.00748, +0.03546, +0.07588])
    frac = (0.0 - diff_eqw[2]) / (diff_eqw[3] - diff_eqw[2])
    T_eqw = float(math.exp(math.log(T_ms[2]) + frac * (math.log(T_ms[3]) - math.log(T_ms[2]))))
    emp_bound = {
        "n_seeds_plain_eq_weights": 1,
        "status": "Empirical boundary (single seed; Phase-2 Wave 3 adds seeds 1 & 2)",
        "u123_minus_eqW_s0": diff_eqw.tolist(),
        "crossover_bracket_M": [250, 500],
        "interpolated_T_eqW_M": round(T_eqw / 1e6, 1),
        "u123_b16_speedup_vs_eqW": round(float(np.mean([315104, 315061, 313305])) / 185533.0, 3),
        "u1_b16_speedup_vs_eqW": round(float(np.mean([351328, 352762, 352420])) / 185533.0, 3),
    }

    # 1. Dedicated Pre-Unblinding 350M Sub-Zipf Prediction Artifact (for its own timestamped commit!)
    prereg_350m = {
        "preregistration_type": "Pre-unblinding 350M x 5B Sub-Zipf OOS predictions (addendum to frozen b89068c; does NOT alter b89068c)",
        "frozen_base_preregistration_commit": "b89068c",
        "run_350m_config": {
            "layers": 24,
            "d": 1024,
            "heads": 16,
            "train_tokens": 5_000_000_000,
            "count_tokens": 500_000_000,
            "pair_rows_M": 2_000_000,
            "n_hi": 1_000_000,
            "n_mid": 3_000_000,
            "n_3g": 300_000,
            "eff_mid_on_500m_prefix": 1,
            "headline_hash_control_in_run_350m_5b_sh": "ctrl_hs_pruned",
        },
        "heaps_inputs": {
            "all_pairs_frozen_b89068c": {"c": c_h, "beta": beta_h, "U_250M": int(round(U_250M)), "U_500M": int(round(U_500M)), "U_5B": int(round(U_5B))},
            "mature_pairs_sensitivity": {"beta_mat": round(float(beta_mat), 6), "U_mat_2_5B": int(round(U_mat_2_5B)), "U_mat_5B": int(round(U_mat_5B))},
        },
        "predictions_350m_ctrl_hs_pruned_primary_U_keep_prefix": {
            "U_keep_350m": int(round(U_500M)),
            "M_350m": 2_000_000,
            "lambda_keep_350m": round(float(U_500M / 2_000_000.0), 3),
            "exact_subzipf_s0855": {
                "pred_5pt": arm_pr_prefix["candidates"]["exact_subzipf_s0855"]["pred_350m"],
                "pi95_5pt": arm_pr_prefix["candidates"]["exact_subzipf_s0855"]["pi95_350m"],
                "se_pred_5pt": arm_pr_prefix["candidates"]["exact_subzipf_s0855"]["se_pred_350m"],
                "pred_3pt_v6k_only": arm_pr_prefix["candidates"]["exact_subzipf_s0855"]["v6k_3pt_fit"]["true_oos_350m_pred"],
            },
            "exact_subzipf_s0823": {
                "pred_5pt": arm_pr_prefix["candidates"]["exact_subzipf_s0823"]["pred_350m"],
                "pi95_5pt": arm_pr_prefix["candidates"]["exact_subzipf_s0823"]["pi95_350m"],
                "se_pred_5pt": arm_pr_prefix["candidates"]["exact_subzipf_s0823"]["se_pred_350m"],
                "pred_3pt_v6k_only": arm_pr_prefix["candidates"]["exact_subzipf_s0823"]["v6k_3pt_fit"]["true_oos_350m_pred"],
            },
            "ln_U_keep_over_M_tangent": {
                "pred_5pt": arm_pr_prefix["candidates"]["ln_U_over_M_tangent"]["pred_350m"],
                "pi95_5pt": arm_pr_prefix["candidates"]["ln_U_over_M_tangent"]["pi95_350m"],
                "pred_3pt_v6k_only": arm_pr_prefix["candidates"]["ln_U_over_M_tangent"]["v6k_3pt_fit"]["true_oos_350m_pred"],
            },
            "frozen_b89068c_ln_U_all_over_M_ref": {
                "pred_5pt": cs["350M_5B_M2M_preregistered"]["pred_5pt_vs_pruned"],
                "pred_3pt": cs["350M_5B_M2M_preregistered"]["pred_3pt_vs_pruned"],
            },
        },
        "predictions_350m_ctrl_hs_pruned_sensitivity_U_mat_5B": {
            "U_keep_350m": int(round(U_mat_5B)),
            "M_350m": 2_000_000,
            "lambda_keep_350m": round(float(U_mat_5B / 2_000_000.0), 3),
            "exact_subzipf_s0855": {
                "pred_5pt": arm_pr_mature["candidates"]["exact_subzipf_s0855"]["pred_350m"],
                "pi95_5pt": arm_pr_mature["candidates"]["exact_subzipf_s0855"]["pi95_350m"],
                "pred_3pt_v6k_only": arm_pr_mature["candidates"]["exact_subzipf_s0855"]["v6k_3pt_fit"]["true_oos_350m_pred"],
            },
        },
        "predictions_350m_ctrl_hs_unpruned": {
            "U_all_350m": int(round(U_5B)),
            "M_350m": 2_000_000,
            "lambda_all_350m": round(float(U_5B / 2_000_000.0), 3),
            "exact_subzipf_s0855": {
                "pred_5pt": arm_hs["candidates"]["exact_subzipf_s0855"]["pred_350m"],
                "pi95_5pt": arm_hs["candidates"]["exact_subzipf_s0855"]["pi95_350m"],
                "se_pred_5pt": arm_hs["candidates"]["exact_subzipf_s0855"]["se_pred_350m"],
                "pred_3pt_v6k_only": arm_hs["candidates"]["exact_subzipf_s0855"]["v6k_3pt_fit"]["true_oos_350m_pred"],
            },
            "exact_subzipf_s0823": {
                "pred_5pt": arm_hs["candidates"]["exact_subzipf_s0823"]["pred_350m"],
                "pi95_5pt": arm_hs["candidates"]["exact_subzipf_s0823"]["pi95_350m"],
                "pred_3pt_v6k_only": arm_hs["candidates"]["exact_subzipf_s0823"]["v6k_3pt_fit"]["true_oos_350m_pred"],
            },
            "ln_U_all_over_M_tangent": {
                "pred_5pt": arm_hs["candidates"]["ln_U_over_M_tangent"]["pred_350m"],
                "pi95_5pt": arm_hs["candidates"]["ln_U_over_M_tangent"]["pi95_350m"],
                "pred_3pt_v6k_only": arm_hs["candidates"]["ln_U_over_M_tangent"]["v6k_3pt_fit"]["true_oos_350m_pred"],
            },
        },
        "falsification_criteria": [
            "Observed 350M (ctrl_hs_pruned - u123) must remain strictly positive (Theorem 1 collision-dominated regime at lambda_keep ~ 18.9).",
            "Observed 350M (ctrl_hs_pruned - u123) falls inside the 95% PI [+0.0169, +0.0369] (point estimate +0.0269 vs frozen b89068c +0.0225).",
            "Non-trivial directional test: because M doubles (1M -> 2M) while U grows only ~1.64x (2^0.7137), pure collision-load models predict 350M gap <= 125M gap (+0.0298), NOT growing proportionally with T or N.",
        ],
    }

    prereg_path = os.path.join(ROOT, "results", "subzipf_350m_predictions.json")
    with open(prereg_path, "w", encoding="utf-8") as f:
        json.dump(prereg_350m, f, indent=2)

    # 2. Full Theory Verification Summary
    summary = {
        "version": "v6.1-calibrated",
        "frozen_preregistration_commit": "b89068c",
        "theorem1_finite_sample_identity": thm1,
        "theorem2_spectral_thresholds": thm2,
        "proposition1_subzipf_integral": prop1,
        "model1_collision_load_fits": {
            "CTRL_hs_unpruned_U_all": arm_hs,
            "CTRL_hs_pruned_U_keep_prefix": arm_pr_prefix,
            "CTRL_hs_pruned_U_keep_mature_sensitivity": arm_pr_mature,
        },
        "empirical_scope_125m_buckets": emp_buckets_125m,
        "heuristic1_cross_layer_snr": heur1,
        "boundary1_eq_weights": emp_bound,
    }

    out_path = os.path.join(ROOT, "results", "theorem_verification.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote {prereg_path}")
    print(f"[ok] wrote {out_path}")
    print("\n=== THEOREM 1 (SYMPY + MC) ===")
    print("  sympy_diff_zero:", thm1["sympy_identity_diff_is_zero"])
    print("  MC:", thm1["mc_verification"])
    print("\n=== THEOREM 2 (SYMPY) ===")
    print("  c_lo:", thm2["c_lo_formula"], "| c_hi:", thm2["c_hi_formula"], "| ratio:", thm2["ratio_c_hi_over_c_lo"])
    print("  band (r=d/4):", thm2["prediction_band_r_eq_d_over_4"])
    print("\n=== PROPOSITION 1 (EXACT SUB-ZIPF + BETA ASYMPTOTICS) ===")
    for row in prop1["numerical_checks"]:
        print(" ", row)
    for tag, arm in [
        ("CTRL_hs (U_all)", arm_hs),
        ("CTRL_hs_pruned (U_keep=prefix)", arm_pr_prefix),
        ("CTRL_hs_pruned (U_keep=mature sens.)", arm_pr_mature),
    ]:
        print(f"\n=== MODEL 1: {tag} ===")
        if "exact_subzipf_s0855" in arm["point_tables"]:
            print("  [5-pt table: exact_subzipf_s0855]")
            for r in arm["point_tables"]["exact_subzipf_s0855"]:
                print(f"    {r['point']:12s} | U={r['U_active']:>11,d} M={r['M']:>9,d} lam={r['lambda_U_over_M']:>7.3f} x={r['feature_x']:.5f} | "
                      f"obs={r['obs_delta']:+.5f} pred5={r['pred_5pt']:+.5f} res5={r['resid_5pt']:+.5f} | "
                      f"pred3={r['pred_3pt']:+.5f} res3={r['resid_3pt']:+.5f} [{r['oos_status']}]")
        for cname, cm in arm["candidates"].items():
            line = (
                f"  {cname:22s} | k={cm['k_regression_params']} df={cm['df']} R2={cm['r2']:+.4f} "
                f"adjR2={cm['adj_r2']:+.4f} s_e={cm['residual_se']:.5f} AIC={cm['aic']:+6.2f} BIC={cm['bic']:+6.2f} "
                f"| 350M={cm['pred_350m']:+.5f} PI95={cm['pi95_350m']}"
            )
            if "v6k_3pt_fit" in cm:
                f3 = cm["v6k_3pt_fit"]
                line += f" | 3pt->125M={f3['semi_oos_125m_pred']:+.5f}({f3['semi_oos_125m_rel_err_pct']:+.1f}%)"
            print(line)


if __name__ == "__main__":
    main()
