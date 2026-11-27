# Standalone Theory Draft (v6.1 — Re-Derived & Audit-Calibrated)

**Status:** Standalone draft (`docs/THEORY_v6_DRAFT.md`). **Not merged into `docs/ICML_Research_Proposal_v5.tex`.**  
**Preregistration Freeze:** Frozen base `350M × 5B` preregistration at commit **`b89068c`** is untouched. Pre-unblinding Sub-Zipf `350M` predictions and $95\%$ PIs are committed separately in [docs/PREREG_350M_SUBZIPF.md](PREREG_350M_SUBZIPF.md) and [results/subzipf_350m_predictions.json](../results/subzipf_350m_predictions.json).  
**Reproducibility:** All SymPy algebraic verifications, Monte Carlo checks, exact numerical quadratures, 5-point residual tables, adjusted $R^2$, residual SEs, AIC/BIC, and $95\%$ prediction intervals are generated verbatim by [analysis/verify_theorems.py](../analysis/verify_theorems.py) and stored in [results/theorem_verification.json](../results/theorem_verification.json).

---

## Core Mechanism (Recommended Abstract Framing)

> **Variance-Pooling Bonus vs. Collision Bias Crossover.**  
> On low-frequency $n$-grams ($n_k$ small), hash tables *outperform* exact tables because bucket sharing pools training observations across colliding keys, shrinking finite-sample estimation variance ($\mathcal{V}_{\text{est}}$). On high-frequency $n$-grams ($n_k$ large), variance vanishes as $O(1/n_k)$ while hash collision bias persists, causing exact frequency-partitioned tables to decisively overtake hashing. The crossover is governed by the balance between per-key count $n_k$ and competing bucket mass $N_{b\setminus k}$ (Theorem 1), while optimal exact-table rank allocation follows the spectral energy concentration of the 1-gram residual target (Theorem 2).

---

## Master Audit Table (Tiered by Proof Strength & Falsifiability)

| Tier | Name | Explicit Assumptions | Proof Status | Empirical Support | Falsification Test |
|---|---|---|---|---|---|
| **Theorem 1** | Finite-sample hash vs. exact risk identity | Idealized i.i.d. zero-mean keys, i.i.d. noise, uniform hash, sample-mean estimator (zero Zipf assumption) | SymPy algebra verified (`diff == 0`) + MC verified (`max rel err 2.1%`, `100%` sign match) | Explains sign competition: negative gap at `V=6k M=96k` (`-0.0025`) & early steps (`-0.0140`), while `pc_100` stays `+0.0055 > 0` | Sign of aggregate $\Delta_{350\text{M}}$ and `pc_4`/`pc_0` bucket split at `350M × 5B` |
| **Theorem 2** | 3-tier spectral truncation thresholds | **1-gram residual target** $v_k^{\text{res}} = v_k^{\text{bigram}} - v_{\text{word}(k)}$, isotropic noise, exact lookup ($\mathcal{R}_{\text{coll}}=0$) | SymPy algebra verified (`c_lo`, `c_hi`, ratio) | Given $\eta_{d/4} \in [0.65, 0.80]$, predicts $c_{\text{hi}}/c_{\text{lo}} \in [5.57\times, 12.00\times]$ ($9.00\times$ at $\eta=0.75$); retrospectively consistent with architecture `10/100` | Noise-debiased split-half SVD of ultra-high-count full-rank `ve_hi` (non-circular) + `350M` bucket gains |
| **Proposition 1** | Properly normalized Sub-Zipf collision integral | Empirically measured $s \in (0.823, 0.855) \subset (1/2, 1)$ (not fitted to NLL), continuous rank approximation | Exact integral on $[0,1]$ + Euler-reflection Beta closed form (`0.10%` err at $\lambda=100$) | Eliminates `s=1` normalization failure; yields scale-invariant function of $\lambda_{\mathcal{S}} = U_{\mathcal{S}}/M$; explains $\ln(U_{\mathcal{S}}/M)$ as small-$\alpha_{\text{coll}}$ tangent | True OOS `350M × 5B` (`M=2M`) |
| **Model 1** | 2-parameter MSE $\to$ NLL linear link | $\Delta = a \cdot x + b$ ($k=2, df=3$ on $n=5$) | Empirical regression (not a theorem) | `CTRL_hs` $R^2=0.953$, $s_e=0.00336$; `CTRL_hs_pruned` ($U_{\text{keep}}$) $R^2=0.967$, $s_e=0.00252$, $\Delta\text{AIC}=-9.14$ vs linear; `125M` semi-OOS err `-7.5% / -9.6%` | Pre-unblinding `350M` 95% PI `[+0.0169, +0.0369]` (`PREREG_350M_SUBZIPF.md`) |
| **Heuristic 1** | Cross-layer shared-bank variance reduction | $\rho \approx$ inter-layer gradient cosine; unmeasured $\mathcal{B}_{\text{conflict}}(L)$ | Unclosed diagnostic (`2.20×` vs `6.43×`, `2.92×` unexplained) | Direction matches `L = 2, 4, 6` 3-gram gain growth | Independent measurement of $\mathcal{B}_{\text{conflict}}(L)$ + `125M u123_no3g` ($L=12$) |
| **Boundary 1** | Equal-total-weights crossover horizon | Single seed (`plain_eq_weights` s0 at `125M`) | Empirical boundary | `u123` leads through `250M`, `eqW` overtakes in `250M–500M` (`~438M`), while `u123` is `1.695×` faster at `B16` | Multi-seed `plain_eq_weights` (Phase-2 Wave 3 `s1, s2`) |

---

## 1. Theorem 1 (Exact Finite-Sample Hash-vs.-Exact Risk Identity under i.i.d. Key/Noise Model)

**Idealized Risk Model & Explicit Assumptions (Zero Zipf Assumption):**  
Let $\mathcal{S} \subseteq [U]$ be the active hashed key set. Assume:
1. *(Independent zero-mean targets)* Key value targets $v_k^\star \in \mathbb{R}^d$ are mutually independent with $\mathbb{E}[v_k^\star] = 0$ and $\mathbb{E}\|v_k^\star\|^2 = \sigma_v^2$.
2. *(Independent observation noise)* Each of the $n_k \ge 1$ occurrences of key $k$ yields a noisy target $y_{k,t} = v_k^\star + \xi_{k,t}$ with independent zero-mean noise $\mathbb{E}[\xi_{k,t}] = 0$, $\mathbb{E}\|\xi_{k,t}\|^2 = \sigma_\xi^2$.
3. *(Uniform hash & sample-mean estimator)* Keys are mapped by $h: \mathcal{S} \to [M]$, and each bucket $b$ stores its sample mean $\hat{v}_b = \frac{1}{n_k + N_{b\setminus k}}\sum_{j \in \mathcal{S}:\, h(j)=b}\sum_{t=1}^{n_j} y_{j,t}$, where $N_{b\setminus k} := \sum_{j \in \mathcal{S}\setminus\{k\},\, h(j)=b} n_j$ and $S_{2,b\setminus k} := \sum_{j \in \mathcal{S}\setminus\{k\},\, h(j)=b} n_j^2$. An exact lookup table stores $\hat{v}_k^{\text{ex}} = \frac{1}{n_k}\sum_{t=1}^{n_k} y_{k,t}$.

*(Modeling caveat: Real SGD/Adam training exhibits correlated keys, non-stationary gradients across steps, and adaptive preconditioning rather than exact sample averaging; Theorem 1 is an exact identity for this idealized risk model, not a distribution-free claim about Adam.)*

**Statement.** Conditional on any hash assignment $h$ and count vector $\{n_j\}_{j \in \mathcal{S}}$ (for **arbitrary** frequency distributions, without assuming Zipf), the finite-sample MSEs satisfy:
$$\mathbb{E}_{v,\xi}\!\left[\|\hat{v}_{h(k)} - v_k^\star\|^2 \,\middle|\, h, \{n_j\}\right] = \sigma_v^2 \frac{N_{b\setminus k}^2 + S_{2,b\setminus k}}{(n_k + N_{b\setminus k})^2} + \frac{\sigma_\xi^2}{n_k + N_{b\setminus k}}, \qquad \mathbb{E}_{v,\xi}\!\left[\|\hat{v}_k^{\text{ex}} - v_k^\star\|^2 \,\middle|\, \{n_j\}\right] = \frac{\sigma_\xi^2}{n_k}.$$
Subtracting gives the exact excess risk of hashing over exact lookup:
$$\Delta_k^{\text{hash}-\text{exact}} = \underbrace{\sigma_v^2 \frac{N_{b\setminus k}^2 + S_{2,b\setminus k}}{(n_k + N_{b\setminus k})^2}}_{\text{Collision Bias } (\ge 0)} \;-\; \underbrace{\sigma_\xi^2 \frac{N_{b\setminus k}}{n_k(n_k + N_{b\setminus k})}}_{\text{Variance-Pooling Bonus } (\ge 0)}.$$

**Proof.** Write $\hat{v}_b - v_k^\star = \frac{-N_{b\setminus k} v_k^\star + \sum_{j \in b\setminus k} n_j v_j^\star + \sum_{j \in b}\sum_{t=1}^{n_j}\xi_{j,t}}{n_k + N_{b\setminus k}}$. By mutual independence and zero mean of $\{v_j^\star\}$ and $\{\xi_{j,t}\}$, all cross-terms vanish and $\mathbb{E}\|\sum_{j\in b}\sum_t \xi_{j,t}\|^2 = (n_k + N_{b\setminus k})\sigma_\xi^2$, yielding the first display. Subtracting $\sigma_\xi^2/n_k$ and simplifying $\frac{1}{n_k + N_{b\setminus k}} - \frac{1}{n_k} = -\frac{N_{b\setminus k}}{n_k(n_k + N_{b\setminus k})}$ gives the second display. $\square$

**Verification (`verify_theorems.py`):** SymPy confirms algebraic residual `== 0`. Monte Carlo ($U=2{,}000, M=100, d=16, \sigma_v^2=1, \sigma_\xi^2=8$, $800$ trials) matches both regimes simultaneously (`sample n_k = [31, 16, 41, 2, 1, 1]`, theory $\Delta = [+1.4558, +0.8796, +0.6846, -2.8672, -6.8750, -6.8869]$ vs. MC $[+1.4482, +0.8641, +0.6701, -2.8552, -6.8282, -6.8136]$, `100%` sign match, max relative MSE error `2.10%`).

**What Theorem 1 does—and does not—claim:** The identity proves the **exact sign-competition mechanism** (why hashing beats exact lookup when $n_k$ is small or $N_{b\setminus k}$ is small relative to noise, while mature high-$n_k$ keys flip to exact winning because the pooling bonus decays as $1/n_k$). The exact crossover point $\Delta_k = 0$ depends on the empirical signal-to-noise ratio $\sigma_v^2 / \sigma_\xi^2$ and $(N_{b\setminus k}, S_{2,b\setminus k})$; the identity provides the decomposition, not a zero-input sign prediction.

---

## 2. Theorem 2 (Exact Three-Tier Spectral Truncation Thresholds on 1-Gram Residual Targets)

**Architecture-Faithful Setup (1-Gram Residual Target):**  
In our tiered memory architecture, pruning a bigram key ($r = 0$) does **not** predict the zero vector in raw embedding space—it falls back to the 1-gram word table `ve_word` (plus backbone). Therefore, define all risks on the **1-gram residual target**
$$v_k^{\text{res}} := v_k^{\text{bigram}} - v_{\text{word}(k)} \in \mathbb{R}^d,$$
with target covariance $\Sigma_{\text{res}} = \mathbb{E}[v_k^{\text{res}}(v_k^{\text{res}})^\top]$ having ordered eigenvalues $\lambda_1 \ge \dots \ge \lambda_d > 0$. For a mid-tier rank $r \in \{1, \dots, d-1\}$, denote $E_{\text{top-}r} := \sum_{i=1}^r \lambda_i$, $E_{\text{tail}} := \sum_{i=r+1}^d \lambda_i$, and spectral energy concentration $\eta_r := \frac{E_{\text{top-}r}}{E_{\text{top-}r} + E_{\text{tail}}} \in (0, 1)$. Under isotropic observation noise $\mathbb{E}[\xi\xi^\top] = \frac{\sigma_\xi^2}{d} I_d$ and exact lookup ($\mathcal{R}_{\text{coll}} = 0$), a key seen $c$ times incurs exact risk:
- **Prune / 1-gram fallback ($r = 0$):** $\mathcal{R}_0(c) = E_{\text{top-}r} + E_{\text{tail}}$ *(exact 1-gram residual risk)*
- **Exact rank-$r$ ($r$):** $\mathcal{R}_r(c) = E_{\text{tail}} + \frac{r}{d}\frac{\sigma_\xi^2}{c}$ *(spectral truncation bias + reduced $r$-dim variance)*
- **Exact full-rank ($d$):** $\mathcal{R}_d(c) = \frac{\sigma_\xi^2}{c}$ *(zero truncation bias + full $d$-dim variance)*

**Statement.** Equating $\mathcal{R}_0(c_{\text{lo}}) = \mathcal{R}_r(c_{\text{lo}})$ and $\mathcal{R}_r(c_{\text{hi}}) = \mathcal{R}_d(c_{\text{hi}})$ yields exact thresholds (SymPy verified):
$$c_{\text{lo}} = \frac{r}{d}\cdot\frac{\sigma_\xi^2}{E_{\text{top-}r}}, \qquad c_{\text{hi}} = \frac{d-r}{d}\cdot\frac{\sigma_\xi^2}{E_{\text{tail}}}.$$
A non-empty mid-rank tier ($c_{\text{lo}} < c_{\text{hi}}$) exists if and only if $\eta_r > r/d$ (anisotropic residual spectrum), and **given $\eta_r$, the threshold ratio has no additional free parameters**:
$$\frac{c_{\text{hi}}}{c_{\text{lo}}} = \frac{d-r}{r}\cdot\frac{\eta_r}{1 - \eta_r}.$$

**Honest Interpretation & Non-Circular Validation Protocol:**
1. **Retrospective consistency, not prospective prediction of `10/100`:** Our architecture's `mid/hi` thresholds (`10/100`, ratio $10\times$) were chosen before this derivation. For $r = d/4$ ($\frac{d-r}{r} = 3$), Theorem 2 yields the prediction band:
   - $\eta_{d/4} = 0.60 \Rightarrow 4.50\times$
   - $\eta_{d/4} = 0.65 \Rightarrow 5.57\times$
   - $\eta_{d/4} = 0.70 \Rightarrow 7.00\times$
   - $\eta_{d/4} = 0.75 \Rightarrow 9.00\times$
   - $\eta_{d/4} = 0.80 \Rightarrow 12.00\times$
   - $\eta_{d/4} = 0.85 \Rightarrow 17.00\times$
   Thus `10/100` is retrospectively consistent with $\eta_{d/4} \in [0.65, 0.80]$ (`[5.57×, 12.00×]`).
2. **Avoiding SVD circularity:** Raw SVD of a trained `ve_hi` matrix measures the *estimated* spectrum (polluted by $\sigma_\xi^2/c$ estimation noise and optimization regularization), NOT $\Sigma_{\text{res}}$. A non-circular test requires **noise-debiased spectral estimation**—specifically, split-half cross-covariance $\hat{\Sigma}_{\text{debiased}} = \frac{1}{2}(\hat{V}_A^\top \hat{V}_B + \hat{V}_B^\top \hat{V}_A)$ computed strictly on ultra-high-count full-rank keys ($c \gg c_{\text{hi}}$) trained on independent data halves (or Marchenko–Pastur bulk subtraction), to estimate $\eta_{d/4}$ independently of the `10/100` choice.
3. **Empirical Scope & Confounds at `125M × 2.5B` (`u123 - ctrl_hs_pruned`):**
   - **Same-scale comparisons only:** `125M` (`2.5B` tokens) sees $83.3\times$ more tokens than `30M`, shrinking $\mathcal{V}_{\text{est}}$ by $\sim 83\times$; never attribute `30M` vs. `125M` differences solely to hash vs. exact `mid`.
   - **Global non-lookup component (`pc_0`):** Observed NLL gain decomposes as $\Delta\mathcal{L}_{\text{obs}}(c) = \Delta\mathcal{L}_{\text{direct}}(c) + \Delta\mathcal{L}_{\text{global}}$. At `2500M`, `pc_0 = -0.01765 ± 0.00437` (`p = 0.0198`, `t = -6.99`) is significantly negative on unseen bigrams, proving a non-lookup backbone/adapter interaction component outside Theorem 2.
   - **`pc_2` vs. `pc_4` non-separation:** `pc_2 = -0.03265 ± 0.00821` (`p = 0.0204`) and `pc_4 = -0.03356 ± 0.00558` (`p = 0.0091`) differ by `0.0009` (SE $\approx 0.010$); they are statistically indistinguishable, and `pc_3 = -0.01827 ± 0.00630` (`p = 0.0374`). Do NOT claim `pc_2` is the unique peak.

---

## 3. Proposition 1 (Properly Normalized Sub-Zipf Collision Integral for $1/2 < s < 1$)

**Empirical Grounding (Zero Shape Fitting on NLL):**  
Direct log-log rank-frequency measurements on our training corpora show bigram frequencies are strictly **Sub-Zipf ($s < 1$)**, not $s = 1$:
- `V=6,000` ($T = 2.42\text{M}$, $U = 411{,}030$, $U_{\ge 10} = 31{,}106$): measured $s(10\text{..}10\text{k}) = \mathbf{0.823}$.
- `FineWeb 30M` ($U = 5{,}190{,}373$, $U_{\ge 10} = 366{,}259$): measured $s(100\text{..}100\text{k}) = \mathbf{0.855}$.

**Statement.** For an active key set $\mathcal{S}$ of size $U_{\mathcal{S}}$ carrying total probability mass $P_{\mathcal{S}}$ under a **properly normalized** continuous Sub-Zipf density ($0 < s < 1$):
$$p(k) = P_{\mathcal{S}}\,\frac{1-s}{U_{\mathcal{S}}^{1-s}}\,k^{-s}, \qquad \int_0^{U_{\mathcal{S}}} p(k)\,dk = P_{\mathcal{S}},$$
with expected uniform-hash competing bucket mass $W = P_{\mathcal{S}}/M$, define the active collision load $\lambda_{\mathcal{S}} := U_{\mathcal{S}} / M > 0$. Substituting $t = (k / U_{\mathcal{S}})^{1-s} \in [0, 1]$ eliminates $P_{\mathcal{S}}$ from $p(k)/W$ and yields the **exact smooth collision-bias integral** valid for **all** $\lambda_{\mathcal{S}} > 0$ (including $\lambda_{\mathcal{S}} < 1$):
$$\frac{\mathcal{R}_{\text{coll}}^{\text{exact}}(U_{\mathcal{S}}, M)}{P_{\mathcal{S}}} = \frac{1}{P_{\mathcal{S}}}\int_0^{U_{\mathcal{S}}} p(k)\left(\frac{W}{p(k) + W}\right)^2 dk = \int_0^1 \frac{dt}{\left(1 + \frac{1-s}{\lambda_{\mathcal{S}}}\,t^{-\frac{s}{1-s}}\right)^2}.$$
Moreover, for $s \in (1/2, 1)$ in the high-collision regime $\lambda_{\mathcal{S}} \gg 1$, applying the Beta integral and Euler's reflection formula $\Gamma(a)\Gamma(1-a) = \pi/\sin(\pi a)$ with $a = \frac{1-s}{s} \in (0, 1)$ gives the exact closed-form asymptotic law:
$$\frac{\mathcal{R}_{\text{coll}}^{\text{exact}}(U_{\mathcal{S}}, M)}{P_{\mathcal{S}}} = 1 - C_{\text{exact}}(s)\,\lambda_{\mathcal{S}}^{-\frac{1-s}{s}} + O\!\left(\lambda_{\mathcal{S}}^{-\frac{2-s}{s}}\right), \qquad C_{\text{exact}}(s) = \frac{1-s}{s^2}\cdot\frac{\pi}{\sin\!\left(\pi\frac{1-s}{s}\right)}\cdot(1-s)^{\frac{1-s}{s}}.$$

**Numerical Verification (`verify_theorems.py`):**

| $s$ | $\alpha_{\text{coll}} = \frac{1-s}{s}$ | $C_{\text{exact}}(s)$ | $\lambda=100$ Exact Integral | $\lambda=100$ Closed-Form Asym. | Rel. Error |
|---:|---:|---:|---:|---:|---:|
| `0.800` | `0.2500` | `0.9285` | `0.70772` | `0.70639` | `0.188%` |
| `0.823` | `0.2151` | `0.9045` | `0.66501` | `0.66404` | `0.146%` |
| `0.855` | `0.1696` | `0.8842` | `0.59567` | `0.59508` | `0.099%` |
| `0.900` | `0.1111` | `0.8780` | `0.47389` | `0.47364` | `0.053%` |

**Why `ln(U/M)` worked empirically without needing an $s=1$ unnormalized hack:** Because $\alpha_{\text{coll}} \approx 0.170$ is small and $\lim_{\alpha \to 0^+}\frac{1 - \lambda^{-\alpha}}{\alpha} = \ln\lambda$, $\ln(U_{\mathcal{S}}/M)$ is the small-$\alpha_{\text{coll}}$ tangent of the bounded, properly normalized Sub-Zipf collision integral.

---

## 4. Model 1 (2-Parameter MSE $\to$ NLL Linear Link — Full 5-Point Audit)

*(Note on true $M$ values: All tables below use the verbatim committed values from frozen `b89068c` `results/collision_law.json`, with $M = [6\text{k}, 24\text{k}, 96\text{k}, 100\text{k}, 1\text{M}]$ and $V=6\text{k}, M=24\text{k}$ pruned gap `+0.00900`. All 2-parameter models fit $\Delta = a\cdot x + b$ with $k=2$, residual $df = n - k = 3$, Gaussian likelihood parameter count $p = k + 1 = 3$.)*

### 4.1 Target A: `CTRL_hs - U123` (Unpruned Hash, Active Set $U_{\text{all}}$)

**5-Point Table (`Exact Sub-Zipf s=0.855`, 0 fitted $s$):**

| Point | OOS Status | $U_{\text{all}}$ | $M$ | $\lambda = U/M$ | $x = \mathcal{R}_{\text{coll}}/P$ | Obs $\Delta$ | 5-pt Pred | 5-pt Resid | 3-pt Pred ($V=6\text{k}$) | 3-pt Resid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `V=6k M=6k` | in-sample | 411,030 | 6,000 | 68.505 | 0.56911 | `+0.02500` | `+0.02629` | `-0.00129` | `+0.02623` | `-0.00123` |
| `V=6k M=24k` | in-sample | 411,030 | 24,000 | 17.126 | 0.45726 | `+0.01590` | `+0.01336` | `+0.00254` | `+0.01365` | `+0.00225` |
| `V=6k M=96k` | in-sample | 411,030 | 96,000 | 4.282 | 0.32258 | `-0.00250` | `-0.00219` | `-0.00031` | `-0.00148` | `-0.00102` |
| `30M M=100k` | OOS (measured $U$) | 5,078,209 | 100,000 | 50.782 | 0.54693 | `+0.01970` | `+0.02372` | `-0.00402` | `+0.02373` | `-0.00403` |
| `125M M=1M` | **semi-OOS (Heaps $U$)** | 119,277,299 | 1,000,000 | 119.277 | 0.60750 | `+0.03380` | `+0.03072` | `+0.00308` | `+0.03054` | `+0.00326` |

**Model Comparison (`CTRL_hs`, $n=5, df=3$):**

| Candidate Model | $k$ | $R^2$ | adj-$R^2$ | Resid SE $s_e$ | AIC | BIC | 3-pt $\to$ `125M` semi-OOS err | 5-pt `350M` Pred & 95% PI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Exact Sub-Zipf ($s=0.855$)** | 2 | 0.9533 | +0.9378 | `0.00336` | -53.52 | -54.69 | `-9.6%` (`+0.03054` vs `+0.03380`) | `+0.02918` `[+0.01667, +0.04168]` |
| Exact Sub-Zipf ($s=0.823$) | 2 | 0.9517 | +0.9356 | `0.00342` | -53.34 | -54.51 | `-10.7%` (`+0.03019`) | `+0.02903` `[+0.01633, +0.04174]` |
| $\ln(U_{\text{all}}/M)$ tangent | 2 | 0.9565 | +0.9419 | `0.00325` | -53.86 | -55.03 | `-5.2%` (`+0.03205`) | `+0.02973` `[+0.01757, +0.04189]` |
| Linear $U_{\text{all}}/M$ null | 2 | 0.8250 | +0.7667 | `0.00651` | -46.91 | -48.08 | `+32.1%` (`+0.04463`) | `+0.03069` `[+0.00573, +0.05564]` |
| $\ln(M)$ only null | 2 | 0.0431 | -0.2759 | `0.01521` | -38.41 | -39.58 | `-171.6%` (`-0.02419`) | `+0.02337` `[-0.04501, +0.09174]` |
| Constant null | 1 | 0.0000 | 0.0000 | `0.01347` | -40.19 | -40.97 | — | `+0.01838` `[-0.02258, +0.05934]` |

### 4.2 Target B: `CTRL_hs_pruned - U123` (Pruned Hash, Active Set $U_{\text{keep}}$)

In `train_125m.py`, `ctrl_hs_pruned` only hashes keys in `cd.mature_keys` (`V=6k`: $31{,}106$; `30M`: $366{,}259$; `125M`: `eff_mid = 1` on `250M` prefix $\Rightarrow U_{\text{keep}} = U(250\text{M}) = 23{,}062{,}102$). Switching from the mismatched $U_{\text{all}}$ to the actual hashed set $U_{\text{keep}}$ resolves the previous linear-null anomaly:

**5-Point Table (`Exact Sub-Zipf s=0.855`, Active Set $U_{\text{keep}}$):**

| Point | OOS Status | $U_{\text{keep}}$ | $M$ | $\lambda_{\text{keep}}$ | $x = \mathcal{R}_{\text{coll}}/P$ | Obs $\Delta$ | 5-pt Pred | 5-pt Resid | 3-pt Pred ($V=6\text{k}$) | 3-pt Resid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `V=6k M=6k` | in-sample | 31,106 | 6,000 | 5.184 | 0.34233 | `+0.01650` | `+0.01752` | `-0.00102` | `+0.01744` | `-0.00094` |
| `V=6k M=24k` | in-sample | 31,106 | 24,000 | 1.296 | 0.19636 | `+0.00900` | `+0.00646` | `+0.00254` | `+0.00695` | `+0.00205` |
| `V=6k M=96k` | in-sample | 31,106 | 96,000 | 0.324 | 0.07194 | `-0.00310` | `-0.00296` | `-0.00014` | `-0.00199` | `-0.00111` |
| `30M M=100k` | OOS (measured $U$) | 366,259 | 100,000 | 3.663 | 0.30627 | `+0.01180` | `+0.01479` | `-0.00299` | `+0.01485` | `-0.00305` |
| `125M M=1M` | **semi-OOS (Heaps $U$)** | 23,062,102 | 1,000,000 | 23.062 | 0.48326 | `+0.02980` | `+0.02820` | `+0.00160` | `+0.02757` | `+0.00223` |

**Model Comparison (`CTRL_hs_pruned`, $n=5, df=3$):**

| Candidate Model | $k$ | $R^2$ | adj-$R^2$ | Resid SE $s_e$ | AIC | BIC | 3-pt $\to$ `125M` semi-OOS err | 5-pt `350M` Pred & 95% PI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Exact Sub-Zipf ($s=0.855$)** | 2 | 0.9667 | +0.9556 | `0.00252` | -56.40 | -57.57 | `-7.5%` (`+0.02757` vs `+0.02980`) | **`+0.02689` `[+0.01689, +0.03689]`** |
| Exact Sub-Zipf ($s=0.823$) | 2 | 0.9650 | +0.9533 | `0.00258` | -56.15 | -57.32 | `-7.8%` (`+0.02747`) | `+0.02688` `[+0.01663, +0.03714]` |
| $\ln(U_{\text{keep}}/M)$ tangent | 2 | 0.9734 | +0.9646 | `0.00225` | -57.53 | -58.70 | `-6.7%` (`+0.02782`) | `+0.02684` `[+0.01792, +0.03577]` |
| Linear $U_{\text{keep}}/M$ null | 2 | 0.7927 | +0.7237 | `0.00628` | -47.26 | -48.43 | `+164.5%` (`+0.07882`) | `+0.02670` `[+0.00121, +0.05219]` |
| $\ln(M)$ only null | 2 | 0.1466 | -0.1378 | `0.01274` | -40.18 | -41.35 | `-163.4%` (`-0.01890`) | `+0.02096` `[-0.03632, +0.07824]` |
| Constant null | 1 | 0.0000 | 0.0000 | `0.01195` | -41.39 | -42.17 | — | `+0.01280` `[-0.02354, +0.04914]` |

**Sensitivity & Statistical Calibration Notes:**
1. **Alternative $U_{\text{keep}}$ definition (`U_mat(2.5B) = 27.34M`, `U_mat(5B) = 54.38M`):** Results are invariant—Exact Sub-Zipf ($s=0.855$) gives $R^2=0.9719$, adj-$R^2=+0.9626$, $s_e=0.00231$, $\Delta\text{AIC}=-10.54$ over linear null, 3-pt `125M` semi-OOS err `-4.0%`, and `350M` pred `+0.02859` (`95% PI [+0.01914, +0.03804]`).
2. **Honest sample-size limitation:** With $n=5$ ($df=3$), Exact Sub-Zipf and its small-$\alpha$ tangent $\ln(U_{\mathcal{S}}/M)$ are statistically indistinguishable ($\Delta\text{AIC} = 0.34\text{–}1.13$), whereas both strongly beat linear $U_{\mathcal{S}}/M$ ($\Delta\text{AIC} = -6.6$ to $-10.3$) and $\ln M$ alone ($\Delta\text{AIC} = -15.1$ to $-17.4$). Furthermore, `125M` is **semi-OOS** (because $U(2.5\text{B})$ and $U(250\text{M})$ are Heaps-extrapolated, not directly counted, and 3-pt predictions underpredict `125M` by `-5.2%` to `-9.6%` while overpredicting `30M` by `+20%` to `+26%`). Therefore `adj-R² ≈ 0.94–0.96` is empirical model support, **never** theorem-level proof; true OOS validation awaits `350M × 5B`.

---

## 5. Heuristic 1 (Cross-Layer Shared-Bank Gradient Variance Diagnostic — Appendix)

Sharing one memory bank across $L$ layers suggests an effective sample-size diagnostic:
$$L_{\text{eff}}(L) = \frac{L}{1 + (L - 1)\rho}, \qquad \mathcal{E}_{\text{shared}}(L; c) \approx \frac{\sigma_g^2}{c\,L_{\text{eff}}(L)} + \mathcal{B}_{\text{conflict}}(L).$$

**Why this remains a Heuristic (not a Theorem):**
- Measured 3-seed inter-layer gradient cosines $\rho = [0.1498, 0.1271, 0.1142]$ at $L \in \{2, 4, 6\}$ give $L_{\text{eff}} = [1.739, 2.896, 3.819]$ (**`2.20×`** from $L=2$ to $L=6$), while observed 3-gram gain grows **`6.43×`** (`+0.0014 -> +0.0030 -> +0.0090`), leaving an unexplained **`2.92×`** factor.
- $\rho$ is gradient cosine similarity rather than noise correlation $\rho_{\text{noise}}$; $\mathcal{B}_{\text{conflict}}(L)$ is unmeasured; and the $L=2\to 4$ drop (`-0.0227`) is smaller than the $L=2$ SD (`0.0336`).

---

## 6. Boundary 1 (Equal-Total-Weights Crossover Horizon — Limitations)

At `125M × 2.5B`, single-seed `plain_eq_weights` (`459.7M` dense body) trails `u123` through `250M` (`50M: -0.06976`, `100M: -0.07676`, `250M: -0.03175`) and overtakes between `250M` and `500M` (`500M: +0.00748`, `1000M: +0.03546`, `2500M: +0.07588`; log-interpolated crossing $T_{\text{eqW}} \approx 438.1\text{M}$), while `u123` and `u1` retain **`1.695×`** and **`1.898×`** higher `B16` inference throughput. Remains strictly an **empirical single-seed boundary** until Phase-2 Wave 3 finishes seeds `1` and `2`.
