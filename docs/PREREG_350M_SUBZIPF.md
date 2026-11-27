# Pre-Unblinding 350M × 5B Sub-Zipf Prediction Addendum

**Status:** Pre-unblinding preregistration addendum, committed independently prior to running `experiments/run_350m_5b.sh`.  
**Frozen Base Preregistration:** Commit **`b89068c`** (`results/collision_law.json`, `docs/ICML_Research_Proposal_v5.tex`, `experiments/run_350m_5b.sh`) is **NOT modified**. This addendum records the out-of-sample (OOS) predictions of the properly normalized Sub-Zipf collision-load model (`s = 0.855` measured directly from FineWeb 30M bigram ranks; zero shape fitting on NLL) together with $95\%$ prediction intervals ($df = n - k = 5 - 2 = 3$, $t_{0.975,3} = 3.1824$).  
**Machine-Readable Artifact:** [results/subzipf_350m_predictions.json](../results/subzipf_350m_predictions.json).

---

## 1. Fixed 350M × 5B Experimental Inputs (`experiments/run_350m_5b.sh`)

- **Model & Budget:** $L=24, d=1024, H=16$, $T = 5.0\times 10^9$ tokens, count prefix $T_{\text{cnt}} = 5.0\times 10^8$ tokens (`eff_mid = max(1, round(10 * 0.10)) = 1`), hash table rows $M = 2{,}000{,}000$.
- **Headline Hash Control:** `ctrl_hs_pruned` vs. `u123` (2 seeds in Waves 1–2; optional seed 2 in Wave 4).
- **Heaps-Projected Active Key Counts** (using frozen `b89068c` FineWeb Heaps law $U(T) = 23.413 \cdot T^{0.71368}$):
  - `CTRL_hs_pruned` primary active set (`mature_keys` from `500M` prefix, `eff_mid = 1`): $U_{\text{keep}}(350\text{M}) = U(500\text{M}) = 37{,}821{,}433 \Rightarrow \lambda_{\text{keep}} = U_{\text{keep}}/M = \mathbf{18.911}$.
  - `CTRL_hs_pruned` sensitivity active set (full-horizon mature $c \ge 10$ Heaps $\beta_{\text{mat}}=0.99221$): $U_{\text{mat}}(5\text{B}) = 54{,}377{,}266 \Rightarrow \lambda_{\text{mat}} = \mathbf{27.189}$.
  - `CTRL_hs` unpruned active set: $U_{\text{all}}(5\text{B}) = 195{,}612{,}435 \Rightarrow \lambda_{\text{all}} = \mathbf{97.806}$.

---

## 2. Pre-Unblinding Point Predictions & 95% Prediction Intervals

Target is $\Delta \equiv \text{NLL}(\text{hash control}) - \text{NLL}(\text{u123}) > 0$ (positive means tiered exact `u123` beats the hash control).

| Target Arm | Active Set $U_{\mathcal{S}}$ | Model ($k=2$ linear params, $df=3$) | 5-pt Pred $\hat{\Delta}_{350\text{M}}$ | $\text{SE}_{\text{pred}}$ | 95% Prediction Interval | 3-pt ($V=6\text{k}$ only) OOS Pred |
|---|---|---|---:|---:|---|---:|
| **`CTRL_hs_pruned`** *(headline)* | $U(500\text{M}) = 37.82\text{M}$ | **Exact Sub-Zipf ($s=0.855$)** | **`+0.02689`** | `0.00314` | **`[+0.01689, +0.03689]`** | `+0.02634` |
| **`CTRL_hs_pruned`** *(headline)* | $U(500\text{M}) = 37.82\text{M}$ | Exact Sub-Zipf ($s=0.823$) | `+0.02688` | `0.00322` | `[+0.01663, +0.03714]` | `+0.02627` |
| **`CTRL_hs_pruned`** *(headline)* | $U(500\text{M}) = 37.82\text{M}$ | $\ln(U_{\text{keep}}/M)$ tangent | `+0.02684` | `0.00280` | `[+0.01792, +0.03577]` | `+0.02641` |
| `CTRL_hs_pruned` *(sensitivity)* | $U_{\text{mat}}(5\text{B}) = 54.38\text{M}$ | Exact Sub-Zipf ($s=0.855$) | `+0.02859` | `0.00297` | `[+0.01914, +0.03804]` | `+0.02857` |
| `CTRL_hs_pruned` *(frozen ref)* | $U_{\text{all}}(5\text{B}) = 195.6\text{M}$ | Frozen `b89068c` $\ln(U_{\text{all}}/M)$ | `+0.02250` | — | — | `+0.01980` |
| `CTRL_hs` *(unpruned ref)* | $U_{\text{all}}(5\text{B}) = 195.6\text{M}$ | **Exact Sub-Zipf ($s=0.855$)** | **`+0.02918`** | `0.00393` | **`[+0.01667, +0.04168]`** | `+0.02904` |
| `CTRL_hs` *(unpruned ref)* | $U_{\text{all}}(5\text{B}) = 195.6\text{M}$ | Exact Sub-Zipf ($s=0.823$) | `+0.02903` | `0.00399` | `[+0.01633, +0.04174]` | `+0.02879` |
| `CTRL_hs` *(unpruned ref)* | $U_{\text{all}}(5\text{B}) = 195.6\text{M}$ | $\ln(U_{\text{all}}/M)$ tangent | `+0.02973` | `0.00382` | `[+0.01757, +0.04189]` | `+0.03008` |

---

## 3. Explicit Falsification Criteria (Evaluated Upon 350M Unblinding)

1. **Sign test (Theorem 1 collision-dominated regime):** Because $\lambda_{\text{keep}} \approx 18.9 \gg 1$, observed $\Delta_{350\text{M}} = \text{NLL}(\text{ctrl\_hs\_pruned}) - \text{NLL}(\text{u123})$ must remain strictly positive across seeds.
2. **Interval calibration:** Observed $\Delta_{350\text{M}}$ falls inside the $95\%$ PI **`[+0.01689, +0.03689]`**.
3. **Non-trivial collision-load vs. scale test:** Because $M$ doubles from `125M` to `350M` (`1M -> 2M`) while $U(T) \propto T^{0.71368}$ grows by only $2^{0.71368} = 1.640\times$, collision load $\lambda_{\text{keep}}$ **decreases slightly** (`23.06 -> 18.91`). Thus pure collision-load models predict $\Delta_{350\text{M}} \approx +0.0269 \le \Delta_{125\text{M}} (+0.0298)$, rather than growing linearly with $T$ or model parameters. Any significant positive excess above `+0.037` will falsify the pure-collision-load approximation and quantify second-order effects from reduced $\mathcal{V}_{\text{est}}$ ($2\times$ tokens) or architectural depth/width ($L=24, d=1024$).
4. **Active-set head-to-head (`U_keep` vs. frozen `b89068c` `U_all`):** Unblinding directly compares the active-set Sub-Zipf prediction (`+0.0269`) against the frozen `b89068c` `U_all` prediction (`+0.0225`).
