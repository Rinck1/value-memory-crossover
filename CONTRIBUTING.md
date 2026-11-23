# Collaborator Guide (`CONTRIBUTING.md`)

Welcome! This repository is set up so two people (plus a cloud GPU agent) can work in parallel without merge conflicts, broken paths, or unattributed claims.

---

## 1. Git & Branching Hygiene

- **Never push broken/untested changes or large binaries straight to `main`.** Use short feature/experiment branches (`feat/...`, `exp/...`, `docs/...`) and merge cleanly.
- **Never commit** `data/`, `runs/`, `runs_125m/`, `*.pt`, `*.ckpt`, `*.log`, or `__pycache__/`. They are ignored in `.gitignore`.
- **DO commit** code changes (`train_125m.py`, `experiments/*.py`, `analysis/*.py`), lightweight reproducible JSON results (`results/*.json`), and regenerated figures (`figs_v5/*`).

---

## 2. Where Everything Goes

| Change Type | Target Folder | Notes |
|---|---|---|
| Cloud `125M` training / data fetch | Root (`train_125m.py`, `download_data.py`, `run_125m_experiments.sh`) | Keep CLI interfaces backward-compatible with `CLOUD_AGENT_HANDOFF.md` |
| Local GPU / `V=50,257` bridge scripts | `experiments/` | Resolve paths relative to `ROOT = Path(__file__).resolve().parent.parent` |
| Zero-GPU fits, tables, plotting | `analysis/` | Keep figure styles & scripts deterministic |
| Experimental outputs (`.json`) | `results/` | Every JSON in `results/` must correspond to a committed script |
| Research proposal, experiment logs & literature | `docs/` | Keep `docs/PROPOSAL.md`, `docs/RESULTS.md`, and `ICML_Research_Proposal_v5.pdf` updated |
| Historical Paper-06 872-run files | `legacy/` | Read-only historical reference |

---

## 3. Code & Experiment Conventions

1. **Always check causal alignment before GPU runs:** Never allow target token `tokens[:, j+1]` to enter any lookup key at position `j`. `train_125m.py` has been verified with a causal-invariance check (`--smoke`).
2. **Always run `--smoke` after editing `train_125m.py`:**
   ```bash
   for arm in plain plain_eq_weights word hard_stack mp_ve u1 u123 ctrl_word ctrl_hs ctrl_hs_pruned; do
       python train_125m.py --smoke --arm "${arm}" --device cpu --out "/tmp/smoke_${arm}"
   done
   ```
3. **Never hardcode local user paths** (e.g. `C:\Users\...`, `/home/...`) in any tracked script or markdown doc. Always resolve via `Path(__file__).resolve()` or CLI flags.
4. **Scientific reporting discipline:**
   - Report **3-seed or 5-seed paired differences (`mean ± std`, paired `t`)** for headline comparisons — never single-seed cherry-picks.
   - Always keep **Equal Active FLOPs** (`plain` vs `u123`) strictly separate from **Equal Total Weights** (`plain_eq_weights`).
   - Decompose `u123` gains transparently into: adapter architecture (`ctrl_hs - word`), maturity pruning (`ctrl_hs_pruned - ctrl_hs`), and exact `2g/3g` de-collision (`u123 - ctrl_hs_pruned`).
   - Report **both** training throughput (`train_tok_s`) and inference latency (`b1_seq_ms`, `b16_infer_tok_s`).
