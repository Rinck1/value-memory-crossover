#!/usr/bin/env python3
"""Generate publication-quality Harmonic-Mean Phase Diagram (figs_v5/fig10_phase_diagram.pdf/.png)
visualizing Theorem 1's exact phase boundary:
    Delta_k^{hash - exact} > 0  <=>  1/n_k + 1/N_{b\k} < SNR * (1 + S_2 / N_{b\k}^2)
where SNR = sigma_v^2 / sigma_xi^2.

Panel A: Linear phase boundary in (1/n_k, 1/N_{b\k}) coordinates with Monte Carlo Sub-Zipf key
         scatter colored by Delta_k^{hash - exact} (blue = pooling wins, red = exact lookup wins).
Panel B: Exact hyperbolic boundary in log-log (n_k, N_{b\k}) space, illustrating the two asymptotes
         (n_k = 1/SNR pruning limit as N_b -> inf; N_b = 1/SNR light-collision limit as n_k -> inf).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "figs_v5")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Idealized Sub-Zipf (s = 0.855, matching FineWeb 30M) key-count & hash realization
    rng = np.random.default_rng(7)
    U = 4000
    M = 120
    ranks = np.arange(1, U + 1, dtype=np.float64)
    p = ranks ** (-0.855)
    p /= p.sum()
    n_counts = np.maximum(1, np.round(p * 30000.0).astype(np.int64))
    h = rng.integers(0, M, size=U)
    bucket_n = np.bincount(h, weights=n_counts, minlength=M)
    bucket_n2 = np.bincount(h, weights=n_counts * n_counts, minlength=M)

    Nb = bucket_n[h] - n_counts
    S2 = np.maximum(0.0, bucket_n2[h] - n_counts * n_counts)
    valid = Nb > 0

    nk_v = n_counts[valid].astype(np.float64)
    Nb_v = Nb[valid].astype(np.float64)
    S2_v = S2[valid].astype(np.float64)

    sigma_v2 = 1.0
    sigma_xi2 = 8.0
    snr = sigma_v2 / sigma_xi2  # 0.125 -> 1/SNR = 8.0 sightings

    delta = sigma_v2 * (Nb_v**2 + S2_v) / ((nk_v + Nb_v) ** 2) - sigma_xi2 * Nb_v / (nk_v * (nk_v + Nb_v))

    # Add a small set of light-collision synthetic points (M larger) so Panel A/B show both regimes richly
    M_hi = 1800
    h2 = rng.integers(0, M_hi, size=U)
    bn2 = np.bincount(h2, weights=n_counts, minlength=M_hi)
    bn2_sq = np.bincount(h2, weights=n_counts * n_counts, minlength=M_hi)
    Nb2 = bn2[h2] - n_counts
    S2_2 = np.maximum(0.0, bn2_sq[h2] - n_counts * n_counts)
    v2 = Nb2 > 0
    nk_all = np.concatenate([nk_v, n_counts[v2].astype(np.float64)])
    Nb_all = np.concatenate([Nb_v, Nb2[v2].astype(np.float64)])
    S2_all = np.concatenate([S2_v, S2_2[v2].astype(np.float64)])
    delta_all = np.concatenate([
        delta,
        sigma_v2 * (Nb2[v2] ** 2 + S2_2[v2]) / ((n_counts[v2] + Nb2[v2]) ** 2)
        - sigma_xi2 * Nb2[v2] / (n_counts[v2] * (n_counts[v2] + Nb2[v2])),
    ])

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0), dpi=300)
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=1.0)
    cmap = plt.cm.RdBu_r

    # --- Panel A: Linear Phase Boundary in (1/n_k, 1/N_{b\k}) ---
    inv_n = 1.0 / nk_all
    inv_N = 1.0 / Nb_all
    # Subsample slightly for visual clarity while preserving exact extremes
    keep = (inv_n <= 0.35) & (inv_N <= 0.35)
    sc1 = ax1.scatter(
        inv_n[keep], inv_N[keep], c=np.clip(delta_all[keep], -1.5, 1.0),
        cmap=cmap, norm=norm, s=11, alpha=0.72, edgecolors="none", rasterized=True,
    )

    xs_lo = np.linspace(0.0, snr, 200)
    ys_lo = snr - xs_lo
    xs_hi = np.linspace(0.0, 2.0 * snr, 200)
    ys_hi = 2.0 * snr - xs_hi

    ax1.fill_between(xs_hi, np.maximum(0.0, snr - xs_hi), ys_hi, color="#f2c94c", alpha=0.22,
                     label=r"Collision-skew band ($S_2/N_b^2 \in [0, 1]$)")
    ax1.plot(xs_lo, ys_lo, color="#111111", lw=1.8,
             label=r"Uniform collision limit: $\frac{1}{n_k}+\frac{1}{N_{b\setminus k}}=\mathrm{SNR}$")
    ax1.plot(xs_hi, ys_hi, color="#444444", lw=1.2, ls="--",
             label=r"Single-dominator limit: $\frac{1}{n_k}+\frac{1}{N_{b\setminus k}}=2\,\mathrm{SNR}$")

    lbl_box = dict(facecolor="white", alpha=0.86, edgecolor="none", pad=1.8)
    # Mark Theorem 2 pruning limit on horizontal axis (1/N_b -> 0)
    ax1.scatter([snr], [0.002], color="#b91c1c", s=44, zorder=6, marker="^")
    ax1.annotate(
        r"Thm. 2 prune limit" + "\n" + r"($N_b\to\infty:\; n_k^\star = \mathrm{SNR}^{-1}$)",
        xy=(snr, 0.005), xytext=(snr + 0.045, 0.068),
        fontsize=7.8, color="#7f1d1d", bbox=lbl_box,
        arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.0),
    )
    ax1.text(0.008, 0.025, "Exact lookup wins\n($\\Delta > 0$)", fontsize=8.3, fontweight="bold", color="#991b1b", bbox=lbl_box)
    ax1.text(0.150, 0.165, "Hash variance-pooling wins\n($\\Delta < 0$)", fontsize=8.3, fontweight="bold", color="#1e3a8a", bbox=lbl_box)

    ax1.set_xlim(-0.005, 0.34)
    ax1.set_ylim(-0.005, 0.34)
    ax1.set_xlabel(r"Inverse key count $1 / n_k$  (left = mature, right = rare)")
    ax1.set_ylabel(r"Inverse competing mass $1 / N_{b\setminus k}$  (bottom = heavy collision)")
    ax1.set_title(r"(a) Exact Linear Boundary in $(1/n_k,\, 1/N_{b\setminus k})$")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.94, edgecolor="#cccccc")
    ax1.grid(True, ls=":", alpha=0.35)

    # --- Panel B: Hyperbolic Boundary in log-log (n_k, N_{b\k}) ---
    sc2 = ax2.scatter(
        nk_all, Nb_all, c=np.clip(delta_all, -1.5, 1.0),
        cmap=cmap, norm=norm, s=11, alpha=0.72, edgecolors="none", rasterized=True,
    )

    n_grid_lo = np.geomspace(1.0 / snr * 1.015, 2500.0, 400)
    N_hyp_lo = 1.0 / (snr - 1.0 / n_grid_lo)
    n_grid_hi = np.geomspace(1.0 / (2.0 * snr) * 1.015, 2500.0, 400)
    N_hyp_hi = 1.0 / (2.0 * snr - 1.0 / n_grid_hi)

    ax2.plot(n_grid_lo, N_hyp_lo, color="#111111", lw=1.8, label=r"Hyperbola ($S_2 \ll N_b^2$)")
    ax2.plot(n_grid_hi, N_hyp_hi, color="#444444", lw=1.2, ls="--", label=r"Hyperbola ($S_2 = N_b^2$)")
    ax2.axvline(1.0 / snr, color="#b91c1c", lw=0.9, ls=":", alpha=0.75)
    ax2.axhline(1.0 / snr, color="#1d4ed8", lw=0.9, ls=":", alpha=0.75)

    ax2.text(1.0 / snr * 1.15, 1.12, r"$n_k = \mathrm{SNR}^{-1}$", fontsize=7.8, color="#991b1b", bbox=lbl_box)
    ax2.text(320, 1.0 / snr * 1.25, r"$N_b = \mathrm{SNR}^{-1}$", fontsize=7.8, color="#1e40af", bbox=lbl_box)
    ax2.text(90, 520, "Exact lookup wins\n($\\Delta > 0$)", fontsize=8.3, fontweight="bold", color="#991b1b", bbox=lbl_box)
    ax2.text(1.05, 520, "Pooling wins\n(low $n_k$)", fontsize=8.0, fontweight="bold", color="#1e3a8a", bbox=lbl_box)
    ax2.text(42, 2.0, "Pooling wins (low $N_b$)", fontsize=7.8, fontweight="bold", color="#1e3a8a", bbox=lbl_box)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(0.85, 2500)
    ax2.set_ylim(0.85, 2500)
    ax2.set_xlabel(r"Key sightings $n_k$ (log scale)")
    ax2.set_ylabel(r"Competing bucket sightings $N_{b\setminus k}$ (log scale)")
    ax2.set_title(r"(b) Hyperbolic Phase Boundary in $(n_k,\, N_{b\setminus k})$")
    ax2.legend(loc="center right", bbox_to_anchor=(0.99, 0.42), frameon=True, framealpha=0.94, edgecolor="#cccccc")
    ax2.grid(True, which="major", ls=":", alpha=0.35)

    cbar = fig.colorbar(sc2, ax=[ax1, ax2], fraction=0.024, pad=0.025)
    cbar.set_label(r"Excess MSE $\Delta_k^{\mathrm{hash}-\mathrm{exact}}$ (clipped to $[-1.5, 1.0]$)", fontsize=8.8)

    fig.suptitle(r"Theorem 1 Unified Phase Diagram: Hash Variance-Pooling Bonus vs. Collision Bias ($\mathrm{SNR} = \sigma_v^2/\sigma_\xi^2 = 1/8$)",
                 fontsize=10.8, y=1.01)
    fig.subplots_adjust(wspace=0.28, right=0.88)

    pdf_path = os.path.join(OUT_DIR, "fig10_phase_diagram.pdf")
    png_path = os.path.join(OUT_DIR, "fig10_phase_diagram.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"[ok] wrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
