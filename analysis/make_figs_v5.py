#!/usr/bin/env python3
"""Spacious, uncluttered publication figures (v5.3) for Conditional Value Memory.

Design principles (aligned with clean NeurIPS / ICML scaling figures):
  - Generous panel size (~5.0 x 3.6 in per panel; 10.8 x 3.8 in for 1x2 layouts)
  - High subplot spacing (wspace=0.32) so axis labels never crowd adjacent panels
  - Maximum 3-5 core curves per plot; redundant minor variants moved to ablation dot-plot
  - Open top/right L-frame spines, light solid grid (#EAECEF), small filled markers
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
OUT = ROOT / "figs_v5"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Clean, Spacious Academic Style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.0,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.0,
    "axes.titleweight": "normal",
    "axes.titlepad": 8.0,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "axes.edgecolor": "#4B5563",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.45,
    "lines.markersize": 4.8,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#D1D5DB",
    "legend.framealpha": 0.96,
    "legend.borderpad": 0.45,
    "legend.labelspacing": 0.35,
    "legend.handlelength": 1.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.color": "#EAECEF",
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
})

PAL = {
    "cyan": "#4EAEC4",
    "teal": "#12947F",
    "coral": "#D95338",
    "indigo": "#3B5B8C",
    "amber": "#D4882A",
    "purple": "#7A6888",
    "grey": "#8C96A0",
    "dark": "#1F2937",
}

BUCKETS = ["pc_0", "pc_1", "pc_10", "pc_100", "pc_1000"]
BUCKET_LABELS = ["0", "1–9", "10–99", "100–999", "1000+"]


def load(name: str):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


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


# ----------------------------------------------------------------------------
# Fig. 0  Spacious Minimalist Architecture Schematic
# ----------------------------------------------------------------------------
def fig0_architecture():
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")
    ax.grid(False)

    def node(x, y, w, h, text, fc="#F9FAFB", ec="#CBD5E1", lw=0.95, fs=8.8, ls="-", title_col="#111827"):
        p = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.25,rounding_size=0.8",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3,
        )
        ax.add_patch(p)
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=title_col, zorder=4, linespacing=1.28,
        )

    def conn(x1, y1, x2, y2, col="#64748B", lw=1.05, ls="-", rad=0.0):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.45,head_width=0.26",
                color=col, lw=lw, ls=ls,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=2,
        )

    ax.text(9.0, 34.2, "Causal Routing", ha="center", va="center", fontsize=9.8, color=PAL["dark"])
    ax.text(39.0, 34.2, "Prefix-Tiered Shared Bank (All Layers)", ha="center", va="center", fontsize=9.8, color=PAL["indigo"])
    ax.text(80.0, 34.2, r"Layer-$\ell$ Injection & Readout", ha="center", va="center", fontsize=9.8, color=PAL["coral"])

    ax.add_patch(patches.FancyBboxPatch((19.8, 2.0), 38.4, 30.2, boxstyle="round,pad=0.2,rounding_size=1.0",
                                        facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=0.75, zorder=1))
    ax.add_patch(patches.FancyBboxPatch((61.0, 2.0), 37.8, 30.2, boxstyle="round,pad=0.2,rounding_size=1.0",
                                        facecolor="#FAFAFC", edgecolor="#E2E8F0", linewidth=0.75, zorder=1))

    node(1.2, 3.0, 15.8, 5.8, "Input Stream\n" r"$(\dots, w_{t-1}, w_t)$", fc="#F3F4F6", ec="#9CA3AF")
    node(1.2, 11.8, 15.8, 7.2, "10% Prefix Router\n" r"($c_{\mathrm{lo}}{=}10, c_{\mathrm{hi}}{=}100$)" "\nZero Future Leak",
         fc="#FEF7EC", ec=PAL["amber"])
    node(1.2, 22.4, 15.8, 8.2, "Causal Keys\n" r"1g: $w_t$, 2g: $(w_{t-1},w_t)$" "\n" r"3g: $(w_{t-2},w_{t-1},w_t)$",
         fc="#F8FAFC", ec="#94A3B8")
    conn(9.1, 9.1, 9.1, 11.5)
    conn(9.1, 19.3, 9.1, 22.1)

    node(21.6, 25.0, 15.8, 5.4, "1g Word Bank\n" r"$E_{\mathrm{word}} \in \mathbb{R}^{V \times d}$",
         fc="#EFF7FA", ec=PAL["cyan"], lw=1.0)
    node(21.6, 18.2, 15.8, 5.2, "High-Freq 2g\n" r"$c \geq 100$, rank $d$",
         fc="#FDF2EF", ec=PAL["coral"], lw=1.0)
    node(21.6, 11.4, 15.8, 5.2, "Mid Low-Rank 2g\n" r"$10 \leq c < 100, r{=}d/4$",
         fc="#EEF8F6", ec=PAL["teal"], lw=1.0)
    node(21.6, 3.8, 15.8, 5.0, r"Rare 2g ($c<10$)" "\nPruned ($k^\\star=0$)",
         fc="#F9FAFB", ec="#CBD5E1", ls=":", title_col="#6B7280")

    node(41.2, 11.4, 15.2, 5.2, "Mature 3g Bank\n" r"$c_3 \geq 30$ (Exact)",
         fc="#F5F3F7", ec=PAL["purple"], lw=1.0)
    node(41.2, 20.6, 15.2, 7.8, "Shared Vectors\n" r"$\mathbf{w}_{\mathrm{mem}} = E_{\mathrm{word}}[w_t]$" "\n" r"$\mathbf{p}_{\mathrm{mem}} = \mathbf{e}_{hi}{+}\mathbf{e}_{mid}{+}\mathbf{e}_{3g}$",
         fc="white", ec=PAL["indigo"], lw=1.15)

    conn(17.2, 27.7, 21.3, 27.7, col=PAL["cyan"])
    conn(17.2, 24.8, 21.3, 20.8, col=PAL["coral"])
    conn(17.2, 16.5, 21.3, 14.0, col=PAL["teal"])
    conn(17.2, 13.8, 21.3, 6.6, col="#9CA3AF", ls="--")
    ax.plot([17.2, 39.2, 39.2], [17.3, 17.3, 14.0], color=PAL["purple"], lw=1.0, zorder=2)
    conn(39.2, 14.0, 40.9, 14.0, col=PAL["purple"])

    conn(37.6, 27.7, 40.9, 25.5, col=PAL["cyan"])
    conn(37.6, 20.8, 40.9, 23.8, col=PAL["coral"])
    conn(37.6, 14.0, 40.9, 21.8, col=PAL["teal"])
    conn(48.8, 16.8, 48.8, 20.3, col=PAL["purple"])

    node(62.8, 3.0, 15.4, 4.4, r"Block Input $\mathbf{h}^{(\ell-1)}$", fc="#F3F4F6", ec="#9CA3AF")
    node(62.8, 9.8, 15.4, 6.2,
         r"$V$-Injection" "\n" r"$\mathbf{v}'{=}\mathbf{v} + g_v(\mathbf{h}) \odot W_v[\mathbf{w};\mathbf{p}]$",
         fc="#EFF7FA", ec=PAL["indigo"], lw=1.0)
    node(62.8, 18.6, 15.4, 5.0, "Causal Self-Attn\n" r"$\mathrm{Attn}(Q, K, \mathbf{v}')$", fc="white", ec="#94A3B8")
    node(82.4, 18.6, 14.8, 5.0, "Gated Residual\n" r"$+ g_r W_r[\mathbf{w};\mathbf{p}]$",
         fc="#EEF8F6", ec=PAL["teal"], lw=1.0)
    node(82.4, 9.8, 14.8, 5.4, "SwiGLU FFN", fc="#F3F4F6", ec="#9CA3AF")
    node(72.0, 26.2, 25.2, 5.0,
         r"Pre-Head Readout ($\ell{=}L$)" "\n"
         r"$\mathbf{h}_L + g_h W_h(\mathbf{w}{+}\mathbf{p}) \to \mathrm{LM}$",
         fc="#FDF2EF", ec=PAL["coral"], lw=1.0, fs=8.3)

    conn(70.5, 7.6, 70.5, 9.5)
    conn(70.5, 16.2, 70.5, 18.3)
    conn(78.4, 21.1, 82.1, 21.1)
    conn(89.8, 18.3, 89.8, 15.5)
    conn(89.8, 23.8, 89.8, 26.0)

    conn(56.6, 22.8, 62.5, 12.9, col=PAL["indigo"], lw=1.05, rad=-0.06)
    conn(56.6, 24.4, 82.1, 21.8, col=PAL["teal"], lw=1.0, rad=-0.16)
    conn(56.6, 26.2, 71.7, 28.7, col=PAL["coral"], lw=1.0, rad=-0.08)

    save(fig, "fig0_architecture")


# ----------------------------------------------------------------------------
# Fig. 1  Sighting-Count Crossover & <UNK>-Reconciled 1000+ Bucket
# ----------------------------------------------------------------------------
def fig1():
    E = load("local_suite_results.json")["block_E_heldout_3seed"]
    U = load("unk_filtered_eval.json")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    fig.subplots_adjust(wspace=0.30)

    ax = axes[0]
    styles = [
        ("1500", PAL["cyan"],  "o", r"$V=1{,}500$"),
        ("3000", PAL["teal"],  "s", r"$V=3{,}000$"),
        ("6000", PAL["coral"], "^", r"$V=6{,}000$"),
    ]
    x = np.arange(len(BUCKETS))
    for V, col, mk, lab in styles:
        means, sems = [], []
        for b in BUCKETS:
            m, se = gap_mean_sem(E[V]["300"]["engram_full"], E[V]["300"]["word"], b)
            means.append(m)
            sems.append(se)
        ax.errorbar(x, means, yerr=sems, color=col, marker=mk, mfc=col, mec=col,
                    capsize=2.5, elinewidth=1.0, label=lab)
    ax.axhline(0.0, color="#6B7280", lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Pair Sightings in Training")
    ax.set_ylabel("Pair $-$ Word Gap (nats)")
    ax.set_ylim(-0.040, 0.080)
    ax.legend(loc="lower left")
    ax.set_title("(a) Sighting-Count Crossover")

    ax = axes[1]
    vocs = ["1500", "3000", "6000"]
    xv = np.arange(len(vocs))
    w = 0.24
    all_eng = [U[v]["eng_minus_word_ALL"]["pc_1000"] for v in vocs]
    nonunk_eng = [U[v]["eng_minus_word_NON_UNK_PAIR"]["pc_1000"] for v in vocs]
    nonunk_mpve = [U[v]["mpve_minus_word_NON_UNK_PAIR"]["pc_1000"] for v in vocs]

    ax.bar(xv - w, all_eng, width=w, color=PAL["cyan"], alpha=0.88, label="Engram (All 1000+)")
    ax.bar(xv, nonunk_eng, width=w, color=PAL["teal"], alpha=0.92, label=r"Engram (Non-$\langle\mathrm{UNK}\rangle$)")
    ax.bar(xv + w, nonunk_mpve, width=w, color=PAL["coral"], alpha=0.92, label=r"MP-VE (Non-$\langle\mathrm{UNK}\rangle$)")
    ax.axhline(0.0, color="#6B7280", lw=0.9, ls="--")
    ax.set_xticks(xv)
    ax.set_xticklabels([r"$1{,}500$", r"$3{,}000$", r"$6{,}000$"])
    ax.set_xlabel("Truncated Vocabulary Size $V$")
    ax.set_ylabel("1000+ Bucket Gap (nats)")
    ax.set_ylim(-0.068, 0.056)
    ax.legend(loc="upper right")
    ax.set_title(r"(b) Resolving 1000+ via $\langle\mathrm{UNK}\rangle$ Filtering")

    save(fig, "fig1_crossover")


# ----------------------------------------------------------------------------
# Fig. 2  (M, T) Collision-Variance Phase Diagram
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

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    cmap = LinearSegmentedColormap.from_list(
        "ref_div", ["#2B7A90", "#A6D4DE", "#F9FAFB", "#F2A899", "#C84B36"]
    )
    vmax = max(abs(gap).max(), 0.01)
    im = ax.imshow(gap, origin="lower", aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(steps) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(Ms) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6, alpha=1.0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(Ms)):
        for j in range(len(steps)):
            val = gap[i, j]
            txt_col = "white" if abs(val) > 0.65 * vmax else "#1F2937"
            fw = "bold" if val < -0.002 else "normal"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8.8, fontweight=fw, color=txt_col)
            if val < -0.002:
                ax.add_patch(patches.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94,
                                               fill=False, edgecolor="#1F2937", lw=1.2))

    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels([f"{t:.2f}M" for t in tokens])
    ax.set_yticks(np.arange(len(Ms)))
    ax.set_yticklabels([f"{m:,}" for m in Ms])
    ax.set_xlabel("Training Tokens $T$")
    ax.set_ylabel("Pair Hash Rows $M$")
    ax.set_title(r"$(M, T)$ Phase Diagram ($V=1{,}500$)")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_linewidth(0.6)
    cb.set_label("Pair $-$ Word Gap (nats)", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    save(fig, "fig2_phase")


# ----------------------------------------------------------------------------
# Fig. 3  Width Scaling & Weakly-Identified c*(d)
# ----------------------------------------------------------------------------
def fig3():
    L = load("local_suite_results.json")
    Cw = L["block_C_width_scaling"]
    P = load("p0_results.json")["step1_fine_cstar"]
    T = load("theory_cstar_analysis.json")["D_reduced_model_and_prereg_widths"]
    hv = T["heldout_verification_results"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    fig.subplots_adjust(wspace=0.30)

    ax = axes[0]
    ds = sorted(int(d) for d in Cw)
    means, sems = [], []
    for d in ds:
        m, se = gap_mean_sem(Cw[str(d)]["word"], Cw[str(d)]["engram_full"], "pc_100")
        means.append(m)
        sems.append(se)
    ax.errorbar(ds, means, yerr=sems, color=PAL["indigo"], marker="o", mfc=PAL["indigo"],
                mec=PAL["indigo"], capsize=2.5, elinewidth=1.0, label=r"Word $-$ Engram ($\mathrm{pc}_{100}$)")
    ax.axhline(0.0, color="#6B7280", lw=0.9, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ds)
    ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlabel("Backbone Width $d$")
    ax.set_ylabel("Mature Pair Advantage (nats)")
    ax.legend(loc="upper right")
    ax.set_title(r"(a) Pair Advantage vs. Width ($d=8\text{–}128$)")

    ax = axes[1]
    fit_w = np.array([16, 32, 64], dtype=float)
    fit_c = np.array([P["widths"][str(int(w))]["cstar_mean_curve"] for w in fit_w])
    c128 = P["widths"]["128"]["cstar_mean_curve"]
    ho_w = np.array([24, 48], dtype=float)
    ho_c = np.array([hv["24"]["measured_cstar"], hv["48"]["measured_cstar"]])

    coef = np.polyfit(np.log(fit_w), np.log(fit_c), 1)
    dg = np.logspace(np.log10(14), np.log10(136), 120)
    ax.plot(dg, np.exp(coef[1]) * dg ** coef[0], color=PAL["coral"], lw=1.45,
            label=fr"Sublinear fit ($\alpha={coef[0]:.2f}$)")
    ax.plot(dg, fit_c[0] * (dg / 16.0), color=PAL["cyan"], lw=1.35, ls="--",
            label=r"Linear null ($\alpha=1.0$)")
    ax.plot(fit_w, fit_c, "o", color=PAL["coral"], ms=5.6, label=r"$d \in \{16,32,64\}$")
    ax.plot(ho_w, ho_c, "D", color=PAL["teal"], ms=5.4, label=r"Held-out $d \in \{24,48\}$")
    ax.plot([128], [c128], "s", mfc="white", mec=PAL["grey"], mew=1.3, ms=5.6,
            label=r"$d=128$ (unidentified)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([16, 24, 32, 48, 64, 128])
    ax.set_xticklabels(["16", "24", "32", "48", "64", "128"])
    ax.set_xlim(13.5, 148)
    ax.set_ylim(28, 7200)
    ax.set_xlabel("Backbone Width $d$")
    ax.set_ylabel(r"Estimated $c^\star$")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.set_title(r"(b) Weakly-Identified $c^\star(d)$")

    save(fig, "fig3_width_cstar")


# ----------------------------------------------------------------------------
# Fig. 4  Clean Quality-Memory Frontier (Core Milestones) & Ablations
# ----------------------------------------------------------------------------
def fig4():
    D = load("local_suite_results.json")["block_D_ablations_L4"]
    U = load("upgrade_search_results.json")
    X = load("extra_ablations_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    fig.subplots_adjust(wspace=0.42)

    # (a) Spacious Frontier (ONLY 7 core milestones — detailed 11-arm breakdown lives in (b))
    ax = axes[0]
    eq_w = D["plain_w88_eqWeights"]["mean"]
    ax.axhline(eq_w, color=PAL["amber"], lw=1.1, ls="--", label=r"Plain $w=88$ ($6\times$ FLOPs)")

    frontier = [
        ("Plain $w{=}32$",  D["plain_w32_eqFLOP"],        PAL["grey"],  "o", (120,  5.422), "left"),
        ("Word (BoV)",      D["word_last2"],              PAL["cyan"],  "s", (490,  5.388), "left"),
        ("MP-VE",           D["mp_ve_last2_dense"],       PAL["teal"],  "D", (900,  5.352), "right"),
        ("U1 (Shared)",     U["U1_shared_d64_all4"],      PAL["indigo"],"v", (880,  5.339), "right"),
        ("U12 (no 3g)",     X["U12_plus_prehead_no3g_d64"], PAL["indigo"], "^", (880, 5.323), "right"),
        ("U123 (Full)",     U["U123_plus_prehead_d64"],   PAL["coral"], "o", (1240, 5.304), "center"),
        ("U123 (Mid-Full)", X["U123_mid_fullrank_d64"],   PAL["coral"], "*", (2630, 5.304), "right"),
    ]
    anchors = [
        ("Hard-Stack",      D["hard_stack_decoupled"],            PAL["amber"],  "^", (2000, 5.365), "left"),
        ("CTRL_hs",         U["CTRL_hs_d64_dual_prehead"],        PAL["purple"], "s", (2000, 5.336), "left"),
    ]
    fx = [r["tab_params"] / 1e3 for _, r, *_ in frontier]
    fy = [r["mean"] for _, r, *_ in frontier]
    ax.plot(fx, fy, color=PAL["indigo"], ls=":", lw=1.2, label="Pareto Frontier", zorder=2)

    for lab, rec, col, mk, (tx, ty), ha in frontier + anchors:
        xk, yk, se = rec["tab_params"] / 1e3, rec["mean"], rec["std"] / np.sqrt(3)
        ax.errorbar(xk, yk, yerr=se, fmt=mk, color=col, mfc=col, mec=col,
                    capsize=2.2, elinewidth=1.0, ms=5.2 if mk != "*" else 7.0, zorder=4)
        ax.text(tx, ty, lab, fontsize=8.2, color=col, ha=ha, va="center",
                fontweight="bold" if lab == "U123 (Full)" else "normal", zorder=5)

    ax.set_xlim(-120, 2820)
    ax.set_ylim(5.272, 5.434)
    ax.set_xlabel("Lookup Table Parameters (K)")
    ax.set_ylabel("Known-Token Validation Loss (nats)")
    ax.legend(loc="upper right")
    ax.set_title(r"(a) Quality–Memory Frontier ($V=6{,}000$)")

    # (b) Spacious Cleveland dot plot of all 11 ablations
    ax = axes[1]
    ab_items = [
        ("U123 (Full)",              U["U123_plus_prehead_d64"],            PAL["coral"]),
        ("w/o V Ctx-Gate",           X["U123_no_ctxgate_d64"],              PAL["teal"]),
        ("w/o Pre-Head",             U["U123_shared_d64_dual_3g"],          PAL["cyan"]),
        ("w/o 3-Gram",               X["U12_plus_prehead_no3g_d64"],        PAL["indigo"]),
        ("w/o Residual Inj.",        X["U123_no_res_d64"],                  PAL["indigo"]),
        ("CTRL_hs_pruned",           X["CTRL_hs_pruned_d64_dual_prehead"],  PAL["teal"]),
        ("w/o Mid Low-Rank",         X["U123_no_mid_tier_d64"],             PAL["amber"]),
        ("CTRL_hs (Unpruned)",       U["CTRL_hs_d64_dual_prehead"],         PAL["purple"]),
        ("CTRL_hs + Hashed 3g",      X["CTRL_hs_plus_3g_d64_dual_prehead"], PAL["grey"]),
        ("w/o Freq. Routing",        X["U123_random_tier_d64"],             PAL["grey"]),
        ("CTRL_word (1g)",           U["CTRL_word_d64_dual_prehead"],       PAL["grey"]),
    ]
    ys = np.arange(len(ab_items))[::-1]
    u123_ref = U["U123_plus_prehead_d64"]["mean"]
    ctrl_ref = U["CTRL_hs_d64_dual_prehead"]["mean"]
    ax.axvline(u123_ref, color=PAL["coral"], lw=1.1, ls="--", alpha=0.85, label="U123 Ref")
    ax.axvline(ctrl_ref, color=PAL["purple"], lw=1.1, ls=":", alpha=0.85, label="CTRL_hs Ref")

    for yi, (lab, rec, col) in zip(ys, ab_items):
        m, se = rec["mean"], rec["std"] / np.sqrt(3)
        ax.hlines(yi, 5.310, m, color="#E5E7EB", lw=1.1, zorder=1)
        ax.errorbar(m, yi, xerr=se, fmt="o", color=col, mfc=col, mec=col,
                    ms=5.2, capsize=2.4, elinewidth=1.1, zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels([x[0] for x in ab_items], fontsize=8.4)
    ax.set_xlim(5.310, 5.378)
    ax.set_xlabel("Known-Token Validation Loss (nats)")
    ax.legend(loc="upper right")
    ax.set_title("(b) Controlled & Leave-One-Out Ablations")

    save(fig, "fig4_method")


# ----------------------------------------------------------------------------
# Fig. 5  Preregistered Round 30 Held-Out Verification
# ----------------------------------------------------------------------------
def fig5():
    P = load("p0_results.json")["step2_prereg_heldout"]
    pred_b = np.array([0.0247, 0.0389, 0.0077, -0.0218, 0.0075])
    meas_b, sem_b = [], []
    for b in BUCKETS:
        m, se = gap_mean_sem(P["engram_full"], P["word_last2"], b)
        meas_b.append(m)
        sem_b.append(se)
    meas_b, sem_b = np.array(meas_b), np.array(sem_b)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    fig.subplots_adjust(wspace=0.30)

    ax = axes[0]
    lim = np.array([-0.034, 0.054])
    ax.fill_between(lim, lim - 0.015, lim + 0.015, color="#F3F4F6", label=r"$\pm 0.015$ break band")
    ax.plot(lim, lim, color="#6B7280", lw=1.0, ls="--", label="$y=x$")
    ax.axhline(0.0, color="#D1D5DB", lw=0.75)
    ax.axvline(0.0, color="#D1D5DB", lw=0.75)
    ax.errorbar(pred_b, meas_b, yerr=sem_b, fmt="o", color=PAL["indigo"],
                mfc=PAL["indigo"], mec=PAL["indigo"], capsize=2.4, zorder=5)
    offsets = [(6, 4), (6, -10), (6, 4), (6, -10), (-6, 7)]
    for i, lab in enumerate(BUCKET_LABELS):
        ax.annotate(lab, (pred_b[i], meas_b[i]), textcoords="offset points",
                    xytext=offsets[i], fontsize=8.4, color=PAL["dark"])
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted Gap (nats)")
    ax.set_ylabel("Measured Gap (nats)")
    ax.legend(loc="lower right")
    ax.set_title(r"(a) Per-Bucket Held-Out ($V{=}4{,}500, d{=}48$)")

    ax = axes[1]
    preds = np.array([0.0099, 0.0209, 0.0203])
    labels = ["Engram $-$ Word", "Word $-$ Hard Stack", "Hard Stack $-$ MP-VE"]
    cols = [PAL["cyan"], PAL["teal"], PAL["coral"]]
    mks = ["o", "s", "D"]
    lim2 = np.array([0.0, 0.028])
    ax.fill_between(lim2, lim2 - 0.015, lim2 + 0.015, color="#F3F4F6")
    ax.plot(lim2, lim2, color="#6B7280", lw=1.0, ls="--")
    for idx, (a, b) in enumerate([("engram_full", "word_last2"),
                                  ("word_last2", "hard_stack_decoupled"),
                                  ("hard_stack_decoupled", "mp_ve_last2_dense")]):
        d = seed_vals(P[a], "val_known") - seed_vals(P[b], "val_known")
        ax.errorbar(preds[idx], float(d.mean()), yerr=float(d.std(ddof=1) / np.sqrt(len(d))),
                    fmt=mks[idx], color=cols[idx], mfc=cols[idx], mec=cols[idx],
                    capsize=2.4, label=f"{labels[idx]}", zorder=5)
    ax.set_xlim(lim2)
    ax.set_ylim(lim2)
    ax.set_xlabel("Predicted Overall Gap (nats)")
    ax.set_ylabel("Measured Overall Gap (nats)")
    ax.legend(loc="lower right")
    ax.set_title("(b) Overall Held-Out Gaps")

    save(fig, "fig5_heldout")


# ----------------------------------------------------------------------------
# Fig. 6  Zero-Prior 5-Bucket Gates vs. Engram Full + beta2 Sweep
# ----------------------------------------------------------------------------
def fig6():
    L = load("local_suite_results.json")
    E = L["block_E_heldout_3seed"]
    F_diag = L["block_F_3seed_diags"]["diag_engram"]["300"]["full"]["bucket_means"]
    F_b2 = L["block_F_3seed_diags"]["beta2_table_only"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    fig.subplots_adjust(wspace=0.30)
    x = np.arange(len(BUCKETS))

    ax = axes[0]
    styles = [("1500", PAL["cyan"],  "o", r"5-Bucket ($V=1{,}500$)"),
              ("3000", PAL["teal"],  "s", r"5-Bucket ($V=3{,}000$)"),
              ("6000", PAL["coral"], "^", r"5-Bucket ($V=6{,}000$)")]
    for V, col, mk, lab in styles:
        b5 = E[V]["300"]["bucket5"]["ve_b5"]
        ax.errorbar(x, b5["mean"], yerr=b5["std"], color=col, marker=mk,
                    mfc=col, mec=col, capsize=2.4, elinewidth=1.0, label=lab)
    eng_gates = [F_diag[b] for b in BUCKETS]
    ax.plot(x, eng_gates, color=PAL["grey"], marker="D", ls="--", mfc=PAL["grey"],
            mec=PAL["grey"], label=r"Engram RMSNorm ($r{=}-0.05$)")
    ax.axhline(0.5, color="#9CA3AF", lw=0.9, ls=":", label="Init (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Pair Sightings Bucket")
    ax.set_ylabel("Learned Gate Opening")
    ax.set_ylim(0.0, 1.06)
    ax.legend(loc="upper left", fontsize=7.6)
    ax.set_title("(a) Zero-Prior Gate Emergence")

    ax = axes[1]
    b2s = ["0.9", "0.95", "0.99", "0.999"]
    cols = [PAL["cyan"], PAL["teal"], PAL["amber"], PAL["coral"]]
    mks = ["o", "s", "^", "D"]
    for b2, col, mk in zip(b2s, cols, mks):
        w = F_b2[b2]["word"]["buckets"]
        e = F_b2[b2]["engram_full"]["buckets"]
        ax.plot(x, [e[b] - w[b] for b in BUCKETS], marker=mk, color=col, mfc=col, mec=col,
                label=fr"$\beta_2={b2}$")
    ax.axhline(0.0, color="#6B7280", lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Pair Sightings Bucket")
    ax.set_ylabel("Engram $-$ Word Gap (nats)")
    ax.legend(loc="lower left", ncol=2)
    ax.set_title(r"(b) Table-Only $\beta_2$ Sweep")

    save(fig, "fig6_gates_beta2")


# ----------------------------------------------------------------------------
# Fig. 7  V=50,257 (30M) Bridge — Spacious 2-Panel Layout (4 Curves + Frontier)
# ----------------------------------------------------------------------------
def fig7():
    B = load("v50257_bridge_results.json")
    V2 = load("u123_v2_results.json")["arms"]
    LAT = load("latency_v50257.json")["arms"]
    arms = {**B["arms"], **V2}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    fig.subplots_adjust(wspace=0.30)

    def get_curve(key):
        info = arms[key]
        ms = info.get("multiseed_summary", {})
        if ms.get("n_seeds", 1) > 1:
            ck = ms["checkpoints"]
            xs = sorted(float(k.replace("M", "")) for k in ck)
            ys = np.array([ck[f"{x:.1f}M"]["val_mean"] for x in xs])
            es = np.array([ck[f"{x:.1f}M"]["val_std"] for x in xs])
            return np.array(xs), ys, es, ms["n_seeds"]
        ck = info["checkpoints"]
        xs = sorted(float(k.replace("M", "")) for k in ck)
        ys = np.array([ck[f"{x:g}M"]["val"] if f"{x:g}M" in ck else ck[f"{x:.1f}M"]["val"] for x in xs])
        return np.array(xs), ys, None, 1

    # (a) Spacious 5-curve convergence showing low-rank fix & parity with CTRL_hs_pruned
    ax = axes[0]
    core = [
        ("plain_d256_eqFLOP",             "Plain $d=256$ (Eq. FLOPs)",            PAL["grey"],   "o", "--"),
        ("plain_d384_eqWeights",          r"Plain $d=384$ ($2.25\times$ FLOPs)",  PAL["amber"],  "D", "--"),
        ("U123_d256_prefix10",            "U123 (Low-Rank Mid)",                  PAL["cyan"],   "s", "-"),
        ("CTRL_hs_pruned_d256_prefix10",  "CTRL_hs_pruned",                       PAL["indigo"], "^", "-."),
        ("U123_v2_fullmid",               "U123 v2 (FullMid)",                    PAL["coral"],  "o", "-"),
    ]
    for key, lab, col, mk, ls in core:
        if key not in arms:
            continue
        xs, ys, es, ns = get_curve(key)
        lbl = f"{lab} ({ns}s)" if ns > 1 else lab
        if es is not None:
            ax.errorbar(xs, ys, yerr=es, marker=mk, color=col, ls=ls, mfc=col, mec=col,
                        capsize=2.5, elinewidth=1.0, label=lbl)
        else:
            ax.plot(xs, ys, marker=mk, color=col, ls=ls, mfc=col, mec=col, label=lbl)
    ax.set_xlabel("Training Tokens (M)")
    ax.set_ylabel("Validation Loss (nats)")
    ax.set_xlim(6.5, 31.2)
    ax.legend(loc="upper right", fontsize=8.0)
    ax.set_title(r"(a) Scaling Convergence ($V=50{,}257$, 30M)")

    # (b) Spacious Throughput vs 30M Quality Frontier
    ax = axes[1]
    if "U123_v2_fullmid" in arms:
        _, fm_ys, fm_es, _ = get_curve("U123_v2_fullmid")
        ax.axhline(float(fm_ys[-1]), color=PAL["coral"], lw=1.0, ls="--",
                   label=fr"U123 v2 FullMid (3s): ${fm_ys[-1]:.3f}\pm{fm_es[-1]:.3f}$")
    pareto_pts = [
        ("Plain 256",       "plain_d256_eqFLOP",             PAL["grey"],   "o", (345, 4.982), "right", False),
        ("Word",            "word_d256_last2",               PAL["cyan"],   "s", (376, 4.918), "right", True),
        ("MP-VE",           "mp_ve_d256_prefix10",           PAL["teal"],   "D", (332, 4.892), "right", True),
        ("U1 Shared",       "U1_shared_d512_all6",           PAL["indigo"], "v", (348, 4.844), "left",  True),
        ("Plain 384",       "plain_d384_eqWeights",          PAL["amber"],  "D", (264, 4.876), "left",  False),
        ("CTRL_pruned",     "CTRL_hs_pruned_d256_prefix10",  PAL["indigo"], "^", (202, 4.844), "center",True),
        ("U123 (NoGate)",   "U123_no_ctxgate_d256_prefix10", PAL["grey"],   "P", (173, 4.864), "left",  False),
        ("U123 (Low-Rank)", "U123_d256_prefix10",            PAL["coral"],  "o", (264, 4.832), "left",  True),
    ]
    fx, fy = [], []
    for lab, key, col, mk, (tx, ty), ha, on_f in pareto_pts:
        if key not in arms or key not in LAT:
            continue
        _, ys, es, _ = get_curve(key)
        loss_v = float(ys[-1])
        err_v = float(es[-1]) if es is not None else None
        thr = LAT[key]["b16_infer_tok_per_s"] / 1e3
        if err_v is not None:
            ax.errorbar([thr], [loss_v], yerr=[err_v], fmt=mk, color=col,
                        mfc=col if on_f else "white", mec=col, mew=1.2, ms=5.6,
                        capsize=2.5, elinewidth=1.0, zorder=4)
        else:
            ax.scatter([thr], [loss_v], marker=mk, color=col, s=38,
                       facecolors=col if on_f else "white", edgecolors=col, linewidths=1.2, zorder=4)
        ax.text(tx, ty, lab, fontsize=8.0, color=col, ha=ha, va="center",
                fontweight="bold" if "CTRL_pruned" in lab or "U123 (Low-Rank)" in lab else "normal", zorder=5)
        if on_f:
            fx.append(thr)
            fy.append(loss_v)
    ord_f = np.argsort(fx)
    ax.plot(np.array(fx)[ord_f], np.array(fy)[ord_f], color=PAL["indigo"], ls=":", lw=1.2,
            label="Throughput–Quality Frontier", zorder=2)

    ax.set_xlabel(r"Batch-16 Inference Speed ($\times 10^3$ tok/s $\uparrow$)")
    ax.set_ylabel(r"30M Validation Loss (nats $\downarrow$)")
    ax.set_xlim(165, 405)
    ax.set_ylim(4.792, 5.002)
    ax.legend(loc="upper left", fontsize=7.8)
    ax.set_title("(b) Inference Throughput vs. Quality")

    save(fig, "fig7_bridge")


# ----------------------------------------------------------------------------
# Fig. 8  Theory Tests (Spacious 2-Panel Line Plots)
# ----------------------------------------------------------------------------
def fig8_theory():
    F = load("theory_cstar_analysis.json")["F_three_component_tests"]
    t2 = F["test2_heldout_M_crossing_trajectories"]["trajectories"]
    t3 = F["test3_shared_bank_depth_and_grad_corr"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    steps = [75, 150, 225, 300]
    m_styles = [
        ("M_6000",  r"$M=6{,}000$ (Heavy Coll.)",   PAL["coral"],  "o", "-"),
        ("M_24000", r"$M=24{,}000$ (Moderate)",     PAL["indigo"], "s", "-"),
        ("M_96000", r"$M=96{,}000$ (Low Coll.)",    PAL["teal"],   "D", "-"),
    ]
    for mk_key, lab, col, mk, ls in m_styles:
        ys = [t2[mk_key][str(s)]["u123_lead_mean"] for s in steps]
        es = [t2[mk_key][str(s)]["u123_lead_std"] / np.sqrt(3) for s in steps]
        ax.errorbar(steps, ys, yerr=es, marker=mk, color=col, ls=ls, mfc=col, mec=col,
                    capsize=2.4, elinewidth=1.0, label=lab)
    pc100_96 = [t2["M_96000"][str(s)]["pc100_lead"] for s in steps]
    ax.plot(steps, pc100_96, marker="^", color=PAL["cyan"], ls=":", mfc=PAL["cyan"], mec=PAL["cyan"],
            label=r"$M=96{,}000$ ($\mathrm{pc}_{100}$)")
    ax.axhline(0.0, color="#6B7280", lw=0.9, ls="--")
    ax.set_xticks(steps)
    ax.set_xlabel("Training Steps $T$ ($V=6{,}000$)")
    ax.set_ylabel(r"$\mathrm{CTRL\_hs}(M) - \mathrm{U123}$ Lead (nats)")
    ax.set_ylim(-0.025, 0.062)
    ax.legend(loc="upper right", fontsize=7.6)
    ax.set_title(r"(a) Table Size $M$ Shifts $T_{\mathrm{cross}}$")

    ax = axes[1]
    Ls = np.array([2, 4, 6])
    u1_gain = np.array([t3[f"L_{L}"]["U1_gain_over_indep_mean"] for L in Ls])
    tot_gain = np.array([t3[f"L_{L}"]["U1_gain_over_indep_mean"] + t3[f"L_{L}"]["3g_gain_inside_U1_mean"] for L in Ls])
    rho_m = np.array([t3[f"L_{L}"]["measured_interlayer_grad_cosine_3seed_mean"] for L in Ls])
    rho_sd = np.array([t3[f"L_{L}"]["measured_interlayer_grad_cosine_3seed_std"] for L in Ls])

    ax.plot(Ls, u1_gain, marker="o", color=PAL["cyan"], mfc=PAL["cyan"], label="Pure 2g U1 Gain")
    ax.plot(Ls, tot_gain, marker="s", color=PAL["coral"], mfc=PAL["coral"], label="U1 + Sparse 3g Gain")
    ax.set_xticks(Ls)
    ax.set_xticklabels([f"$L={L}$" for L in Ls])
    ax.set_xlim(1.5, 6.5)
    ax.set_ylim(0.0, 0.022)
    ax.set_xlabel("Model Depth $L$ ($V=6{,}000, w=32$)")
    ax.set_ylabel("Gain over Indep. Last-2 (nats)")

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(PAL["amber"])
    ax2.grid(False)
    ax2.errorbar(Ls, rho_m, yerr=rho_sd, marker="D", color=PAL["amber"], ls="--",
                 mfc=PAL["amber"], mec=PAL["amber"], capsize=2.4, elinewidth=1.0,
                 label=r"Grad. Cosine $\rho_{\ell,\ell'}$ (Right)")
    ax2.set_ylabel(r"Inter-Layer Grad. Cosine $\rho_{\ell,\ell'}$", color=PAL["amber"])
    ax2.tick_params(axis="y", colors=PAL["amber"])
    ax2.set_ylim(0.0, 0.22)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.6)
    ax.set_title(r"(b) Depth Scaling & Gradient Corr. $\rho$")

    save(fig, "fig8_theory_tests")


if __name__ == "__main__":
    fig0_architecture()
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8_theory()
