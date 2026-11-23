#!/usr/bin/env python3
"""Publication-quality figures (v5) for the value-memory paper.

Reads the JSON result files in this directory and writes PDF + PNG figures to
figs_v5/.  All labels are in English; the style is a compact two-column ML
conference style (Times/STIX, 8 pt, colour-blind-safe palette).

Usage:  python make_figs_v5.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
OUT = ROOT / "figs_v5"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.1,
    "lines.markersize": 4,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.linewidth": 0.35,
    "grid.alpha": 0.45,
})

# Okabe-Ito palette (colour-blind safe)
C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#8C8C8C",
    "black": "#222222",
}

BUCKETS = ["pc_0", "pc_1", "pc_10", "pc_100", "pc_1000"]
BUCKET_LABELS = ["0", "1–9", "10–99", "100–999", "1000+"]
BUCKET_CENTERS = np.array([0.5, 5.0, 50.0, 500.0, 5000.0])


def load(name: str):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def panel_label(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def save(fig, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / (stem + '.pdf')}")


def seed_vals(arm, key):
    return np.array([r["scores"][key] for r in arm["raw"]], dtype=float)


def gap_mean_sem(a, b, key):
    d = seed_vals(a, key) - seed_vals(b, key)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def arm_bucket_vals(arm, b):
    if "raw" in arm and arm["raw"]:
        return np.array([r["scores"][b] for r in arm["raw"]], dtype=float)
    return np.array([arm["buckets"][b]], dtype=float)


# ----------------------------------------------------------------------------
# Fig. 1  Crossover by sighting bucket (Block E, three vocabularies)
# ----------------------------------------------------------------------------
def fig1():
    E = load("local_suite_results.json")["block_E_heldout_3seed"]
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    styles = [
        ("1500", C["sky"], "o", r"$V=1{,}500$"),
        ("3000", C["green"], "s", r"$V=3{,}000$"),
        ("6000", C["orange"], "^", r"$V=6{,}000$"),
    ]
    x = np.arange(len(BUCKETS))
    for V, col, mk, lab in styles:
        means, sems = [], []
        for b in BUCKETS:
            m, se = gap_mean_sem(E[V]["300"]["engram_full"], E[V]["300"]["word"], b)
            means.append(m)
            sems.append(se)
        ax.errorbar(x, means, yerr=sems, color=col, marker=mk, mfc="white",
                    mec=col, mew=0.9, capsize=1.8, label=lab)
    ax.axhline(0.0, color=C["black"], lw=0.7, ls="--")
    ax.axvspan(2.5, 4.5, color=C["grey"], alpha=0.10, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Pair sightings in training")
    ax.set_ylabel("Pair $-$ word loss gap (nats)")
    ax.set_xlim(-0.4, 4.4)
    ax.text(0.97, 0.96, "word better", transform=ax.transAxes, fontsize=6.5,
            color=C["black"], va="top", ha="right")
    ax.text(0.97, 0.05, "pair better", transform=ax.transAxes, fontsize=6.5,
            color=C["black"], va="bottom", ha="right")
    ax.legend(loc="lower left", ncol=1, handlelength=1.6)
    save(fig, "fig1_crossover")


# ----------------------------------------------------------------------------
# Fig. 2  (M, T) phase diagram (Block B)
# ----------------------------------------------------------------------------
def fig2():
    B = load("local_suite_results.json")["block_B_phase_diagram"]
    steps = ["75", "150", "300", "600"]
    Ms = sorted({int(k.split("_")[1]) for k in B["75"] if k.startswith("M_")})
    tokens = np.array([int(s) * 128 * 64 / 1e6 for s in steps])
    gap = np.zeros((len(Ms), len(steps)))
    for j, s in enumerate(steps):
        w = B[s]["word"]["mean"]
        for i, M in enumerate(Ms):
            gap[i, j] = B[s][f"M_{M}"]["mean"] - w

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    vmax = max(abs(gap).max(), 0.01)
    im = ax.imshow(gap, origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    for i in range(len(Ms)):
        for j in range(len(steps)):
            ax.text(j, i, f"{gap[i, j]:+.3f}", ha="center", va="center",
                    fontsize=5.8, color=C["black"])
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels([f"{t:.2f}M" for t in tokens])
    ax.set_yticks(np.arange(len(Ms)))
    ax.set_yticklabels([f"{m:,}" for m in Ms])
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Table rows $M$")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Pair $-$ word gap (nats; <0: pair better)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig2_phase")


# ----------------------------------------------------------------------------
# Fig. 3  Width scaling and c*(d)
# ----------------------------------------------------------------------------
def fig3():
    L = load("local_suite_results.json")
    Cw = L["block_C_width_scaling"]
    P = load("p0_results.json")["step1_fine_cstar"]

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.25))
    ax = axes[0]
    ds = sorted(int(d) for d in Cw)
    means, sems = [], []
    for d in ds:
        m, se = gap_mean_sem(Cw[str(d)]["word"], Cw[str(d)]["engram_full"], "pc_100")
        means.append(m)
        sems.append(se)
    ax.errorbar(ds, means, yerr=sems, color=C["blue"], marker="o", mfc="white",
                mec=C["blue"], capsize=1.8, label=r"100–999 bucket")
    ax.axhline(0.0, color=C["black"], lw=0.7, ls="--")
    ax.text(0.97, 0.96, "positive: pair better", transform=ax.transAxes,
            fontsize=6.2, va="top", ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ds)
    ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlabel("Model width $d$")
    ax.set_ylabel("Pair advantage (nats)")
    ax.set_title("(a) High-frequency pair advantage\n(word $-$ Engram, 100–999)",
                 fontsize=8, loc="left")

    ax = axes[1]
    widths = np.array([w for w in (16, 32, 64, 128) if str(w) in P["widths"]])
    cstar = np.array([P["widths"][str(w)]["cstar_mean_curve"] for w in widths])
    ax.plot(widths, cstar, "o", color=C["red"], mfc="white", mec=C["red"],
            label=r"estimated $c^\star$")
    sub = widths <= 64
    coef = np.polyfit(np.log(widths[sub]), np.log(cstar[sub]), 1)
    dg = np.logspace(np.log10(widths.min()), np.log10(widths.max()), 100)
    ax.plot(dg, np.exp(coef[1]) * dg ** coef[0], color=C["red"], lw=1.0,
            label=fr"$d=16$–$64$: $\alpha={coef[0]:.2f}$")
    ax.plot([128], [cstar[-1]], "o", mfc="white", mec=C["grey"], mew=1.0)
    ax.annotate("not\nidentified", xy=(128, cstar[-1]), xytext=(80, 110),
                fontsize=6.2, color=C["grey"], ha="right",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C["grey"]))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("Model width $d$")
    ax.set_ylabel(r"Estimated crossover $c^\star$")
    ax.set_title("(b) Crossover vs width", fontsize=8, loc="left")
    ax.text(0.03, 0.96, r"estimated $c^\star$" + "\n" + r"solid: $d=16$–$64$, $\alpha=0.32$",
            transform=ax.transAxes, fontsize=6.2, va="top")
    ax.text(0.03, 0.03, r"full fit (16–128): $\alpha=-0.94$, $R^2=0.46$",
            transform=ax.transAxes, fontsize=5.6, color=C["grey"], va="bottom")
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig3_width_cstar")


# ----------------------------------------------------------------------------
# Fig. 4  Method comparison at L=4 (loss and memory)
# ----------------------------------------------------------------------------
def fig4():
    D = load("local_suite_results.json")["block_D_ablations_L4"]
    U = load("upgrade_search_results.json")

    rows = [
        ("Plain $w{=}32$", D["plain_w32_eqFLOP"], C["grey"], "o"),
        ("Word, last 2", D["word_last2"], C["blue"], "s"),
        ("Hard stack", D["hard_stack_decoupled"], C["orange"], "^"),
        ("MP-VE", D["mp_ve_last2_dense"], C["green"], "D"),
        ("U1 shared", U["U1_shared_d64_all4"], C["sky"], "v"),
        ("U123 + prehead", U["U123_plus_prehead_d64"], C["red"], "P"),
        ("Control (hash)", U["CTRL_hs_d64_dual_prehead"], C["purple"], "X"),
        ("Plain $w{=}88$\n(eq. weights)", D["plain_w88_eqWeights"], C["black"], "*"),
    ]
    rows.sort(key=lambda r: r[1]["mean"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    y = np.arange(len(rows))
    for i, (lab, r, col, mk) in enumerate(rows):
        ax.errorbar(r["mean"], i, xerr=r["std"] / np.sqrt(3),
                    fmt=mk, color=col, mfc="white" if mk != "*" else col,
                    mec=col, capsize=1.8, ms=4.2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Known-token loss (nats)")
    ax.set_xlim(5.265, 5.435)
    ax.set_title("(a) Loss at equal active FLOPs", fontsize=8, loc="left")
    ax.axvline(D["plain_w88_eqWeights"]["mean"], color=C["black"], lw=0.7, ls=":")

    ax = axes[1]
    table_rows = [
        ("Word", D["word_last2"], C["blue"], "s", 500, 5.387),
        ("Hard stack", D["hard_stack_decoupled"], C["orange"], "^", 1880, 5.367),
        ("MP-VE", D["mp_ve_last2_dense"], C["green"], "D", 1050, 5.348),
        ("U1", U["U1_shared_d64_all4"], C["sky"], "v", 1050, 5.335),
        ("U123", U["U123_plus_prehead_d64"], C["red"], "P", 1300, 5.315),
        ("Control (hash)", U["CTRL_hs_d64_dual_prehead"], C["purple"], "X", 1880, 5.337),
    ]
    for lab, r, col, mk, xk, yk in table_rows:
        ax.errorbar(r["tab_params"] / 1e3, r["mean"], yerr=r["std"] / np.sqrt(3),
                    fmt=mk, color=col, mfc="white", mec=col, capsize=1.8, ms=4.2)
        ax.annotate(lab, xy=(r["tab_params"] / 1e3, r["mean"]),
                    xytext=(xk, yk), fontsize=5.8, color=col,
                    ha="left" if xk > r["tab_params"] / 1e3 else "right",
                    va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=col))
    ax.axhline(D["plain_w88_eqWeights"]["mean"], color=C["black"], lw=0.7, ls=":")
    ax.text(2050, D["plain_w88_eqWeights"]["mean"] + 0.0012,
            "plain, eq. weights\n(2.2× FLOPs)", fontsize=5.6, color=C["black"],
            ha="right", va="bottom")
    ax.set_xlim(250, 2100)
    ax.set_ylim(5.275, 5.392)
    ax.set_xlabel("Table parameters (thousands)")
    ax.set_ylabel("Known-token loss (nats)")
    ax.set_title("(b) Quality vs table memory", fontsize=8, loc="left")
    fig.subplots_adjust(left=0.255, right=0.985, top=0.88, bottom=0.17, wspace=0.42)
    save(fig, "fig4_method")


# ----------------------------------------------------------------------------
# Fig. 5  Preregistered held-out prediction (P0 step 2)
# ----------------------------------------------------------------------------
def fig5():
    P = load("p0_results.json")["step2_prereg_heldout"]
    pred_b = np.array([0.0247, 0.0389, 0.0077, -0.0218, 0.0075])
    meas_b, sem_b = [], []
    for b in BUCKETS:
        m, se = gap_mean_sem(P["engram_full"], P["word_last2"], b)
        meas_b.append(m)
        sem_b.append(se)
    meas_b = np.array(meas_b)
    sem_b = np.array(sem_b)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.35))
    ax = axes[0]
    ax.errorbar(pred_b, meas_b, xerr=0, yerr=sem_b, fmt="o", color=C["blue"],
                mfc="white", mec=C["blue"], capsize=1.8)
    lim = [-0.035, 0.055]
    ax.plot(lim, lim, color=C["grey"], lw=0.7, ls="--")
    for i, lab in enumerate(BUCKET_LABELS):
        ax.annotate(lab, (pred_b[i], meas_b[i]), textcoords="offset points",
                    xytext=(3, 3), fontsize=5.8, color=C["black"])
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted gap (nats)")
    ax.set_ylabel("Measured gap (nats)")
    ax.set_title("Per-bucket, Engram $-$ word", fontsize=7.5)
    ax.text(0.04, 0.94, "5/5 signs match", transform=ax.transAxes, fontsize=6.5,
            va="top")

    ax = axes[1]
    preds = np.array([0.0099, 0.0209, 0.0203])
    labels = ["Engram $-$ word", "Word $-$ hard stack", "Hard stack $-$ MP-VE"]
    meas, sems = [], []
    for a, b in [("engram_full", "word_last2"),
                 ("word_last2", "hard_stack_decoupled"),
                 ("hard_stack_decoupled", "mp_ve_last2_dense")]:
        d = seed_vals(P[a], "val_known") - seed_vals(P[b], "val_known")
        meas.append(float(d.mean()))
        sems.append(float(d.std(ddof=1) / np.sqrt(len(d))))
    ax.errorbar(preds, meas, yerr=sems, fmt="o", color=C["red"], mfc="white",
                mec=C["red"], capsize=1.8)
    lim = [0.0, 0.028]
    ax.plot(lim, lim, color=C["grey"], lw=0.7, ls="--")
    handles = [Line2D([], [], marker="o", color=C["red"], mfc="white",
                      mec=C["red"], lw=0, label=lab) for lab in labels]
    ax.legend(handles=handles, loc="lower left", fontsize=6.0, handlelength=1.2,
              borderaxespad=0.4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted overall gap (nats)")
    ax.set_ylabel("Measured overall gap (nats)")
    ax.set_title("Overall gaps", fontsize=7.5)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig5_heldout")


# ----------------------------------------------------------------------------
# Fig. 6  Learned bucket gates and table-only beta2 sweep
# ----------------------------------------------------------------------------
def fig6():
    L = load("local_suite_results.json")
    E = L["block_E_heldout_3seed"]
    F = L["block_F_3seed_diags"]["beta2_table_only"]

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.25))
    ax = axes[0]
    styles = [("1500", C["sky"], "o", r"$V=1{,}500$"),
              ("3000", C["green"], "s", r"$V=3{,}000$"),
              ("6000", C["orange"], "^", r"$V=6{,}000$")]
    x = np.arange(len(BUCKETS))
    for V, col, mk, lab in styles:
        b5 = E[V]["300"]["bucket5"]["ve_b5"]
        ax.errorbar(x, b5["mean"], yerr=b5["std"], color=col, marker=mk,
                    mfc="white", mec=col, capsize=1.8, label=lab)
    ax.axhline(0.5, color=C["grey"], lw=0.7, ls=":")
    ax.text(0.02, 0.505, "uniform init.", fontsize=5.8, color=C["grey"], va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Pair sightings bucket")
    ax.set_ylabel("Learned gate value")
    ax.set_title("Zero-prior bucket gates (300 steps)", fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=6.2)

    ax = axes[1]
    b2s = ["0.9", "0.95", "0.99", "0.999"]
    cols = [C["sky"], C["green"], C["orange"], C["red"]]
    for b2, col in zip(b2s, cols):
        w = F[b2]["word"]["buckets"]
        e = F[b2]["engram_full"]["buckets"]
        gaps = [e[b] - w[b] for b in BUCKETS]
        ax.plot(x, gaps, marker="o", color=col, mfc="white", mec=col,
                label=fr"$\beta_2={b2}$")
    ax.axhline(0.0, color=C["black"], lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Pair sightings bucket")
    ax.set_ylabel("Engram $-$ word gap (nats)")
    ax.set_title(r"Table-only $\beta_2$ sweep", fontsize=7.5)
    ax.legend(loc="lower right", fontsize=6.0, ncol=2)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig6_gates_beta2")


# ----------------------------------------------------------------------------
# Fig. 7  V=50,257 bridge
# ----------------------------------------------------------------------------
def fig7():
    B = load("v50257_bridge_results.json")
    arms = B["arms"]
    styles = [
        ("plain_d256_eqFLOP", "Plain, $d=256$", C["grey"], "o", "-"),
        ("word_d256_last2", "Word, $d=256$", C["blue"], "s", "-"),
        ("mp_ve_d256_prefix10", "MP-VE, $d=256$", C["green"], "D", "-"),
        ("plain_d384_eqWeights", "Plain, $d=384$ (eq. weights)", C["black"], "*", "--"),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for key, lab, col, mk, ls in styles:
        ck = arms[key]["checkpoints"]
        xs = sorted(float(k.replace("M", "")) for k in ck)
        ys = [ck[f"{x:g}M"]["val"] if f"{x:g}M" in ck else ck[f"{x:.1f}M"]["val"] for x in xs]
        ax.plot(xs, ys, marker=mk, color=col, ls=ls, mfc="white" if mk != "*" else col,
                mec=col, label=lab)
    ax.set_xlabel("Training tokens (millions)")
    ax.set_ylabel("Validation loss (nats)")
    ax.set_xlim(6, 32)
    ax.legend(loc="upper right", fontsize=6.0)
    ax.annotate("30M tokens: d384 4.857\nvs MP-VE 4.870", xy=(30, 4.87),
                xytext=(19.5, 4.905), fontsize=5.8, color=C["black"],
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C["grey"]))
    save(fig, "fig7_bridge")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
