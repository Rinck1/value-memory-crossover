# Standalone Theory Draft (v6 — Calibrated Analytical Models & Audit)

**Status:** Standalone draft (`docs/THEORY_v6_DRAFT.md`). **Not merged into `docs/ICML_Research_Proposal_v5.tex`.**  
**Preregistration Freeze:** All `350M × 5B` preregistered targets remain frozen at commit **`b89068c`** (`results/collision_law.json`, `experiments/run_350m_5b.sh`, `CLOUD_AGENT_HANDOFF_PHASE2.md`) and are NOT altered by this draft.  
**Reproducibility:** All numbers, SymPy symbolic integrals, Monte Carlo bounds, residuals, adjusted $R^2$, AIC/BIC, and Zipf sensitivity checks are generated directly by [analysis/verify_theorems.py](../analysis/verify_theorems.py) and saved in [results/theorem_verification.json](../results/theorem_verification.json).

---

## Taxonomy & Placement Policy

| Label in This Draft | Scientific Status | Intended Paper Placement | Promotion Gate (Post-350M) |
|---|---|---|---|
| **Lemma 1** | Rigorous algebraic inequality (proof + MC verified) | Appendix (referenced from Method) | Already valid; stays Lemma 1 |
| **Model 1** | Phenomenological log-linear ansatz (NOT derived from first principles without extra truncation assumptions) | Appendix / Analysis (with AIC/BIC & sensitivity table) | Promote functional form only if `350M × 5B` out-of-sample confirms $\ln(U/M)$ over linear $U/M$ |
| **Proposition 1** | Three-term risk decomposition (collision + rank + estimation variance) | Main text (short paragraph, explicit assumptions & caveats) | Strengthen once same-scale `125M` / `350M` ablations finish |
| **Heuristic 1** | Cross-layer gradient variance heuristic (`2.20×` predicted vs `6.43×` observed) | Appendix (explicitly noting unexplained `2.92×` factor) | Re-check after `125M` `u123_no3g` ($L=12$) |
| **Empirical Boundary** | Single-seed `plain_eq_weights` crossover (`250M–500M`) | Main text Limitations / Pareto Frontier | Upgrade to multi-seed boundary after Phase-2 Wave 3 (`s1, s2`) |

---

## Lemma 1 (Exact Two-Sided Bucket-Collision MSE Bound)

**Setup.** Consider $U$ distinct keys with probabilities $\{p_k\}_{k=1}^U$ ($\sum_k p_k \le 1$) and zero-mean orthogonal target value vectors $v_k^\star \in \mathbb{R}^d$ satisfying $\mathbb{E}[v_k^\star] = 0$ and $\mathbb{E}[\langle v_i^\star, v_j^\star \rangle] = \sigma_v^2 \delta_{ij}$. Let $h: [U] \to [M]$ be any hash assignment, and assume each bucket $b \in [M]$ converges to its frequency-weighted optimal mean
$$\hat{v}_b = \frac{\sum_{j:\,h(j)=b} p_j v_j^\star}{\sum_{j:\,h(j)=b} p_j}.$$
For any key $k$ mapped to $b = h(k)$, define the competing bucket probability mass $W_b := \sum_{j \neq k,\, h(j)=b} p_j$.

**Statement.** Conditional on any hash assignment $h$, the expected squared representation error for key $k$ obeys the exact closed-form identity and strict two-sided bound:
$$\mathbb{E}_v\left[\|\hat{v}_{h(k)} - v_k^\star\|^2 \,\middle|\, h\right] = \sigma_v^2 \cdot \frac{W_b^2 + \sum_{j \neq k,\, h(j)=b} p_j^2}{(p_k + W_b)^2},$$
and therefore
$$\sigma_v^2 \left(\frac{W_b}{p_k + W_b}\right)^2 \;\le\; \mathbb{E}_v\left[\|\hat{v}_{h(k)} - v_k^\star\|^2 \,\middle|\, h\right] \;\le\; 2\,\sigma_v^2 \left(\frac{W_b}{p_k + W_b}\right)^2.$$

**Proof.** Rewrite $\hat{v}_b - v_k^\star = \frac{-W_b v_k^\star + \sum_{j \neq k,\, h(j)=b} p_j v_j^\star}{p_k + W_b}$. Expanding the squared norm and taking expectation over $v$ using $\mathbb{E}[\langle v_i^\star, v_j^\star \rangle] = \sigma_v^2 \delta_{ij}$ directly yields the exact identity. Because $p_j \ge 0$, we have $0 \le \sum_{j \neq k,\, h(j)=b} p_j^2 \le \left(\sum_{j \neq k,\, h(j)=b} p_j\right)^2 = W_b^2$, giving both bounds immediately. $\square$

**Monte Carlo Check (`verify_theorems.py`):** Verified on $U=4{,}000$, $M=200$, $500$ trials: `100%` of keys satisfy the sandwich (`mean exact/lower = 1.2345`, `max MC rel err = 0.0247`).

---

## Model 1 (Collision-Load Analytical Ansatz — Full Symbolic & Statistical Audit)

### 1. What SymPy Symbolic Integration Actually Shows (Honest Scope)

Integrating Lemma 1's lower bound over rank $k \in [1, U]$ with $W_b \approx 1/M$ and Zipf law $p_k = c_0 k^{-s}$ yields **different functional forms depending on weighting, normalization, and $s$** (`sympy_symbolic_integrals` in `results/theorem_verification.json`):

- **Token-weighted, $s=1$, fixed $c_0$ (unnormalized truncated tail on $[1, U]$):**
  $$\int_1^U \frac{c_0}{k}\left(\frac{1/M}{c_0/k + 1/M}\right)^2 dk = c_0 \frac{(M c_0 + 1)(M c_0 + (M c_0 + U)\ln(M c_0 + U)) - (M c_0 + U)(M c_0 + (M c_0 + 1)\ln(M c_0 + 1))}{(M c_0 + 1)(M c_0 + U)}.$$
  Piecewise splitting at $k_c = c_0 M$ gives head $= \frac{c_0}{2} - \frac{1}{2 M^2 c_0}$ and tail $= c_0 \ln\!\left(\frac{U}{c_0 M}\right) = c_0 \ln(U/M) - c_0 \ln c_0$ (when $1 \ll c_0 M \ll U$).
- **Token-weighted, $s=1$, normalized on $[1, U]$ ($c_0 = 1 / H_U \approx 1 / \ln U$):**
  Dividing by $\ln U$ converts the leading term into $1 - \frac{\ln(M/\ln U)}{\ln U}$, which is **not** $\ln(U/M)$ and degrades 5-point fit severely ($R^2 = 0.626$ / $0.428$).
- **Token-weighted, general $s \neq 1$:**
  Piecewise tail is $\frac{c_0}{s - 1}\left((M c_0)^{(1-s)/s} - U^{1-s}\right)$ (power-law, **not** logarithmic).
- **Key-averaged (unweighted), $s=1$:**
  $\frac{1}{U}\int_1^U \left(\frac{k}{k+c_0 M}\right)^2 dk = 1 - \mathcal{O}\!\left(\frac{c_0 M}{U}\ln\frac{U}{c_0 M}\right)$ (**not** logarithmic).

**Conclusion:** $\Delta \approx a \ln(U(T)/M) + b$ is therefore an **empirical / analytical ansatz (Model 1)** motivated by fixed-$c_0$ $s \approx 1$ Zipf truncation or an additive interference-channel approximation, **not** an assumption-free theorem derived from Lemma 1.

### 2. Full 5-Point Data Table, Predictions, and Residuals

**Target A: `CTRL_hs - U123` (Unpruned Hash vs. Tiered Exact)**

| Point | $U(T)$ | $M$ | $U/M$ | $\ln(U/M)$ | Obs $\Delta$ | 5-pt Pred | 5-pt Resid | 3-pt Pred ($V=6\text{k}$) | 3-pt Resid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `V=6k, M=6k` | 411,030 | 6,000 | 68.505 | 4.2269 | `+0.02500` | `+0.02618` | `-0.00118` | `+0.02655` | `-0.00155` |
| `V=6k, M=24k` | 411,030 | 24,000 | 17.126 | 2.8406 | `+0.01590` | `+0.01233` | `+0.00357` | `+0.01280` | `+0.00310` |
| `V=6k, M=96k` | 411,030 | 96,000 | 4.282 | 1.4543 | `-0.00250` | `-0.00151` | `-0.00099` | `-0.00095` | `-0.00155` |
| `30M, M=100k` | 5,078,209 | 100,000 | 50.782 | 3.9275 | `+0.01970` | `+0.02319` | `-0.00349` | `+0.02358` | `-0.00388` |
| `125M, M=1M` | 119,277,299 | 1,000,000 | 119.277 | 4.7815 | `+0.03380` | `+0.03171` | `+0.00209` | `+0.03205` | `+0.00175` |

**Target B: `CTRL_hs_pruned - U123` (Pruned Hash vs. Tiered Exact)**

| Point | $U(T)$ | $M$ | $U/M$ | $\ln(U/M)$ | Obs $\Delta$ | 5-pt Pred | 5-pt Resid | 3-pt Pred ($V=6\text{k}$) | 3-pt Resid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `V=6k, M=6k` | 411,030 | 6,000 | 68.505 | 4.2269 | `+0.01650` | `+0.01943` | `-0.00293` | `+0.01727` | `-0.00077` |
| `V=6k, M=24k` | 411,030 | 24,000 | 17.126 | 2.8406 | `+0.00900` | `+0.00766` | `+0.00134` | `+0.00747` | `+0.00153` |
| `V=6k, M=96k` | 411,030 | 96,000 | 4.282 | 1.4543 | `-0.00310` | `-0.00411` | `+0.00101` | `-0.00233` | `-0.00077` |
| `30M, M=100k` | 5,078,209 | 100,000 | 50.782 | 3.9275 | `+0.01180` | `+0.01689` | `-0.00509` | `+0.01515` | `-0.00335` |
| `125M, M=1M` | 119,277,299 | 1,000,000 | 119.277 | 4.7815 | `+0.02980` | `+0.02414` | `+0.00566` | `+0.02119` | `+0.00861` |

### 3. Model Comparison: Adjusted $R^2$, AIC, and BIC ($n = 5$)

| Target | Candidate Model | $k$ (reg. params) | $R^2$ | Adjusted $R^2$ | RMSE | AIC | BIC |
|---|---|---:|---:|---:|---:|---:|---:|
| `CTRL_hs` | **$\ln(U/M)$ ansatz** | 2 | **0.9565** | **0.9419** | **0.00251** | **-53.86** | **-55.03** |
| `CTRL_hs` | Linear $U/M$ null | 2 | 0.8250 | 0.7667 | 0.00504 | -46.91 | -48.08 |
| `CTRL_hs` | $\ln(M)$ only | 2 | 0.0431 | -0.2759 | 0.01178 | -38.41 | -39.58 |
| `CTRL_hs` | Constant null | 1 | 0.0000 | 0.0000 | 0.01205 | -40.19 | -40.97 |
| `CTRL_hs_pruned` | $\ln(U/M)$ ansatz | 2 | 0.8785 | 0.8380 | 0.00372 | -49.93 | -51.10 |
| `CTRL_hs_pruned` | **Linear $U/M$ null** | 2 | **0.9289** | **0.9052** | **0.00285** | **-52.61** | **-53.78** |
| `CTRL_hs_pruned` | $\ln(M)$ only | 2 | 0.1466 | -0.1378 | 0.00987 | -40.18 | -41.35 |
| `CTRL_hs_pruned` | Constant null | 1 | 0.0000 | 0.0000 | 0.01069 | -41.39 | -42.17 |

**Honest Interpretation:**
1. Both `ln(U/M)` and `linear(U/M)` strongly beat `ln(M)` alone (`adj-R² < 0`) and the constant null — confirming **collision load $U(T)/M$** (not raw table size $M$) is the right governing coordinate.
2. On unpruned `CTRL_hs`, `ln(U/M)` beats `linear(U/M)` cleanly ($\Delta\text{AIC} = -6.95$, `adj-R² = 0.942` vs `0.767`).
3. On pruned `CTRL_hs_pruned`, **`linear(U/M)` actually outperforms `ln(U/M)` on $n=5$ ($\Delta\text{AIC} = -2.68$, `adj-R² = 0.905` vs `0.838`)**, driven by the positive residual `+0.00566` at `125M` (where `u123` also eliminated `mid`-tier hashing collisions compared to `V=6k` and `30M`). With $n=5$ points, `ln(U/M)` vs linear $U/M$ on pruned controls is **not decisively separated**; the `350M × 5B` point ($U/M \approx 97.8$) will test this out-of-sample.

### 4. Zipf Exponent Sensitivity ($s \in \{0.8, 1.0, 1.2\}$)

| Target | Integral Normalization | $s$ | 3-pt $R^2$ | 5-pt $R^2$ | 5-pt Adj-$R^2$ | 5-pt AIC | 5-pt BIC |
|---|---|---:|---:|---:|---:|---:|---:|
| `CTRL_hs` | Normalized ($c_0 = 1/Z_U$) | `0.8` | 0.9860 | 0.9297 | 0.9063 | -51.47 | -52.64 |
| `CTRL_hs` | Normalized ($c_0 = 1/Z_U$) | `1.0` | 0.9617 | 0.6264 | 0.5018 | -43.11 | -44.28 |
| `CTRL_hs` | Normalized ($c_0 = 1/Z_U$) | `1.2` | 0.9342 | 0.0444 | -0.2742 | -38.42 | -39.59 |
| `CTRL_hs` | Fixed-$c_0$ truncated | `0.8` | 0.9637 | 0.7254 | 0.6338 | -44.65 | -45.82 |
| `CTRL_hs` | Fixed-$c_0$ truncated | `1.0` | 0.9612 | 0.9559 | 0.9412 | -53.80 | -54.97 |
| `CTRL_hs` | Fixed-$c_0$ truncated | `1.2` | 0.9343 | 0.0364 | -0.2848 | -38.38 | -39.55 |
| `CTRL_hs_pruned` | Normalized ($c_0 = 1/Z_U$) | `0.8` | 0.9963 | 0.7907 | 0.7209 | -47.21 | -48.38 |
| `CTRL_hs_pruned` | Normalized ($c_0 = 1/Z_U$) | `1.0` | 0.9808 | 0.4275 | 0.2366 | -42.18 | -43.35 |
| `CTRL_hs_pruned` | Normalized ($c_0 = 1/Z_U$) | `1.2` | 0.9600 | 0.0010 | -0.3320 | -39.39 | -40.56 |
| `CTRL_hs_pruned` | Fixed-$c_0$ truncated | `0.8` | 0.9823 | 0.8684 | 0.8246 | -49.53 | -50.70 |
| `CTRL_hs_pruned` | Fixed-$c_0$ truncated | `1.0` | 0.9805 | 0.8798 | 0.8397 | -49.98 | -51.15 |
| `CTRL_hs_pruned` | Fixed-$c_0$ truncated | `1.2` | 0.9601 | 0.0001 | -0.3332 | -39.39 | -40.56 |

**Takeaway:** Cross-scale fit is highly sensitive to both $s$ and normalization ($s=1.2$ collapses completely to $R^2 \approx 0$). Only fixed-$c_0$ at $s=1.0$ (or normalized $s=0.8$) matches the cross-scale behavior. This confirms Model 1 must be presented as an **ansatz** with explicit sensitivity, never a universal first-principles theorem.

---

## Proposition 1 (Three-Term Mid-Tier Risk Decomposition — Main-Text Scope)

For any mid-frequency bigram key seen $c$ times, its excess value-memory risk decomposes into three distinct terms:
$$\mathcal{R}_{\text{mid}}(c, r, U, M) \;=\; \mathcal{R}_{\text{coll}}(U, M) \;+\; \mathcal{B}_{\text{rank}}(r) \;+\; \mathcal{V}_{\text{est}}(c),$$
where $\mathcal{R}_{\text{coll}}(U, M)$ is hash-collision distortion (zero for exact lookup), $\mathcal{B}_{\text{rank}}(r) \propto \sum_{i > r} \lambda_i$ is low-rank spectral truncation bias (zero for full rank $r = d_{\text{mem}}$), and $\mathcal{V}_{\text{est}}(c) \propto \sigma^2 / c$ is finite-sample estimation variance (present in **both** exact and hashed tables).

**Strict Empirical Scope & Caveats:**
1. **Do NOT directly attribute `30M` vs `125M` solely to exact-vs-hash `mid`:** `125M` (`2.5B` tokens) trains on **$83.3\times$ more tokens** than `30M`, reducing $\mathcal{V}_{\text{est}}(c)$ by roughly $83\times$. Clean causal comparisons are **same-scale only**:
   - At `30M`: `U123_v2_fullmid` beats `U123` low-rank hashed by `+0.0284` (`t=+2.54`, isolating $\mathcal{B}_{\text{rank}}(r)$ when both pay $\mathcal{R}_{\text{coll}}$), and ties `CTRL_hs_pruned` (`+0.0118, t=+1.25`).
   - At `125M × 2.5B`: `u123` (exact low-rank `mid` + exact `3g`) beats `ctrl_hs_pruned` by **`-0.02982 ± 0.00325` (`t = -15.885, p = 0.0039`)**.
2. **Per-bucket honesty at `125M × 2.5B` (`u123 - ctrl_hs_pruned`):**
   - `pc_2`: `-0.03265` (`p = 0.0204`) and `pc_4`: `-0.03356` (`p = 0.0091`) are **statistically indistinguishable** (diff `0.0009`, SE `~0.010`); do NOT claim `pc_2` is the unique peak (`pc_4` also receives exact `3g` hits).
   - `pc_0`: `-0.01765` (`p = 0.0198`, `t = -6.99`) is **significantly negative** even on unseen bigrams where neither table hits a mature pair, proving a **global non-lookup / representation-interaction component** (cleaner gradients / less corrupted shared adapters & backbone) beyond direct table lookup hits.

---

## Heuristic 1 (Cross-Layer Shared-Bank Gradient Variance Reduction — Appendix)

Sharing one memory bank across $L$ layers yields an effective sample-size heuristic:
$$L_{\text{eff}}(L) = \frac{L}{1 + (L - 1)\rho}, \qquad \mathcal{E}_{\text{shared}}(L; c) \approx \frac{\sigma_g^2}{c\,L_{\text{eff}}(L)} + \mathcal{B}_{\text{conflict}}(L).$$

**Explicit Limitations (Why This Is a Heuristic, Not a Theorem):**
1. **Quantitative mismatch (`2.20×` vs `6.43×`):** Substituting measured 3-seed inter-layer gradient cosines $\rho = [0.1498, 0.1271, 0.1142]$ at $L \in \{2, 4, 6\}$ gives $L_{\text{eff}} = [1.739, 2.896, 3.819]$ — an increase of **`2.20×`** from $L=2$ to $L=6$, whereas observed `3-gram` gain grows **`6.43×`** (`+0.0014 -> +0.0030 -> +0.0090`), leaving an unexplained **`2.92×` multiplier**.
2. **Proxy & identifiability gaps:** $\rho$ measures gradient cosine similarity (not stochastic noise correlation $\rho_{\text{noise}}$); $\mathcal{B}_{\text{conflict}}(L)$ is unmeasured; and the $L=2 \to 4$ change in $\rho$ (`-0.0227`) is smaller than the $L=2$ SD (`0.0336`).

---

## Empirical Boundary (Equal-Total-Weights Crossover Horizon — Limitations)

At `125M × 2.5B`, single-seed `plain_eq_weights` (`459.7M` dense body) trails `u123` early (`50M: -0.06976`, `100M: -0.07676`, `250M: -0.03175`) and overtakes between `250M` and `500M` tokens (`500M: +0.00748`, `1000M: +0.03546`, `2500M: +0.07588`; log-interpolated crossing $T_{\text{eqW}} \approx 438.1\text{M}$ tokens), while `u123` and `u1` maintain **`1.695×`** and **`1.898×`** higher `B16` inference throughput.

**Status:** Strictly reported as an **empirical single-seed boundary** until Phase-2 Wave 3 completes `plain_eq_weights` seeds `1` and `2`.
