r"""Run preregistered held-out width test (d=24 and d=48) for fine-bucket c*(d) (pair_ve M=24000 vs word, V=6000, L=2, 150s, 3 seeds).
Tests the preregistered prediction locked in theory_cstar_analysis.json['D_reduced_model_and_prereg_widths']:
  - Sublinear model (alpha=0.319) predicts:
      d=24: c* ≈ 476.3 (crosses in b_250_500 -> b_500_1000; b_500_1000 gap > 0)
      d=48: c* ≈ 594.2 (crosses in b_250_500 -> b_500_1000; b_500_1000 gap > 0)
  - Naive linear scaling (alpha=1.0 from d=16 c*=408.2) predicts:
      d=48: c* = 1224.6 (> 726.6, so b_500_1000 MUST be < 0!)
Updates theory_cstar_analysis.json and p0_results.json.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json, math, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_local_suite as rls
import run_step1_fine_cstar as s1f

THEORY_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "theory_cstar_analysis.json")
P0_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "p0_results.json")


def main():
    data = rls.load_data(6000, 150, 128)
    with open(P0_JSON, "r", encoding="utf-8") as f:
        p0 = json.load(f)
    with open(THEORY_JSON, "r", encoding="utf-8") as f:
        th = json.load(f)

    b_c = p0["step1_fine_cstar"]["fine_bucket_c"]
    keys = [f"b_{lo}_{hi}" for lo, hi in zip(s1f.FINE_EDGES[:-1], s1f.FINE_EDGES[1:])]
    inner_keys = [k for k in keys if not k.startswith("b_0_") and not k.startswith("b_1000_")]

    heldout_res = {}
    for d in (24, 48):
        H = 2 if d <= 32 else 4
        w_runs = [s1f.train_and_eval_fine(data, [0, 1], "word", seed=sd, width=d, heads=H, pair_rows=24000) for sd in range(3)]
        p_runs = [s1f.train_and_eval_fine(data, [0, 1], "pair_ve", seed=sd, width=d, heads=H, pair_rows=24000) for sd in range(3)]

        gap_curve = {}
        for k in keys:
            if k in w_runs[0]:
                diffs = [w[k] - p[k] for w, p in zip(w_runs, p_runs)]
                gap_curve[k] = {
                    "c_geom": b_c[k][1],
                    "gap_mean": round(float(np.mean(diffs)), 5),
                    "gap_sd": round(float(np.std(diffs, ddof=1)), 5),
                }

        pts_c = [(gap_curve[k]["c_geom"], gap_curve[k]["gap_mean"]) for k in inner_keys]
        cstar = None
        cross_bracket = None
        for i in range(len(pts_c) - 1):
            c0, g0 = pts_c[i]
            c1, g1 = pts_c[i + 1]
            if g0 <= 0 < g1:
                frac = -g0 / (g1 - g0)
                cstar = round(float(math.exp(math.log(c0) + frac * (math.log(c1) - math.log(c0)))), 1)
                cross_bracket = f"{inner_keys[i]} -> {inner_keys[i+1]}"
                break

        p0["step1_fine_cstar"]["widths"][str(d)] = {
            "word_runs": w_runs, "pair_runs": p_runs, "gap_curve": gap_curve, "cstar_mean_curve": cstar
        }
        prereg = th["D_reduced_model_and_prereg_widths"]["preregistered_heldout_widths"][str(d)]
        b500_gap = gap_curve["b_500_1000"]["gap_mean"]
        heldout_res[str(d)] = {
            "preregistered_cstar_sublinear": prereg["predicted_cstar_sublinear"],
            "naive_linear_alpha1_cstar": prereg["naive_linear_alpha1_prediction"],
            "measured_cstar": cstar,
            "measured_cross_bracket": cross_bracket,
            "measured_b_500_1000_gap": b500_gap,
            "sublinear_error_pct": round(abs(cstar - prereg["predicted_cstar_sublinear"]) / prereg["predicted_cstar_sublinear"] * 100, 1) if cstar else None,
            "linear_null_rejected_at_b500_1000": bool(b500_gap > 0),
        }
        g_line = " | ".join(f"{k.split('_')[1]}:{gap_curve[k]['gap_mean']:+.3f}" for k in inner_keys)
        print(f"Held-out d={d:2d} | pred_c*={prereg['predicted_cstar_sublinear']:.1f} (linear_null={prereg['naive_linear_alpha1_prediction']:.1f}) | "
              f"MEASURED c*={cstar} ({cross_bracket}) | b_500_1000={b500_gap:+.4f} | {g_line}")

    # Re-fit 5-width power law over all 5 weakly identified widths d in {16, 24, 32, 48, 64}
    five_d = [16, 24, 32, 48, 64]
    five_c = [p0["step1_fine_cstar"]["widths"][str(d)]["cstar_mean_curve"] for d in five_d]
    if all(x is not None for x in five_c):
        lx, ly = np.log(five_d), np.log(five_c)
        sl5, ic5 = np.polyfit(lx, ly, 1)
        r2_5 = 1 - np.sum((ly - (sl5 * lx + ic5)) ** 2) / np.sum((ly - ly.mean()) ** 2)
        print(f"\n5-width (16,24,32,48,64) log(c*) ~ {sl5:+.3f} * log(d) + {ic5:.3f}, R2={r2_5:.3f}")
        heldout_res["five_width_fit_16_24_32_48_64"] = {
            "widths": five_d, "cstars": [round(float(x), 1) for x in five_c],
            "slope": round(float(sl5), 4), "intercept": round(float(ic5), 4), "r2": round(float(r2_5), 4)
        }

    th["D_reduced_model_and_prereg_widths"]["heldout_verification_results"] = heldout_res
    with open(P0_JSON, "w", encoding="utf-8") as f:
        json.dump(p0, f, indent=2)
    with open(THEORY_JSON, "w", encoding="utf-8") as f:
        json.dump(th, f, indent=2)
    print("Saved held-out width results to", THEORY_JSON)


if __name__ == "__main__":
    main()
