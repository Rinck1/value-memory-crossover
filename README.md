# Word- vs Pair-Keyed Value Memories in Small Language Models

Code, data-processing scripts, and result files for a controlled study of
lookup-table memory in GPT-style language models.  A learned vector is added to
the attention value stream for each token (word table) or each two-word
sequence (pair table); the central question is when the pair table overtakes
the word table as a function of how often the pair was seen in training.

**Status:** research code and preliminary results.  The manuscript is a work in
progress; the 125M-parameter runs and downstream evaluation are not included in
this repository.  Some early internal drafts are deliberately not included
because their claims were superseded by the current results (see *Caveats*).

## Key findings

- **Sighting-count crossover.** Word tables lead on positions whose pair was
  seen fewer than about 100 times; the pair-table advantage peaks in the
  100--999 bucket.  The 1000+ bucket is mixed in small vocabularies and is
  partly an `<UNK>`-folding artifact (about half of the 1000+ input pairs
  contain `<UNK>` at V=6,000).
- **Preregistered held-out prediction.** At an unseen configuration
  (V=4,500, d=48, L=4, 250 steps), the predicted per-bucket signs matched 5/5;
  the overall gaps were within 0.012 nats of the preregistered values.
- **Method.** Maturity-gated dual-granularity value memory (MP-VE) and its
  shared-table / context-gated variant (U1/U123).  At L=4, w=32, V=6,000,
  U123 improves known-token loss by +0.045 nats over a tuned hard stack
  (3 seeds, paired t=8.0).  U1 is the lower-overhead variant; a dedicated
  latency measurement is pending.
- **Full-vocabulary bridge.** At V=50,257, 30M tokens, single seed, U123
  reaches 4.8234 versus 4.9855 for an equal-active-FLOPs plain model and
  4.8961 for the word table.  A wider plain model (d=384, 2.25x active FLOPs)
  reaches 4.8570, so the method is positioned for equal-active-FLOPs,
  compute-limited inference, not equal total weights.
- **Boundary conditions.** An equal-weight plain model (w=88) remains the
  best-loss model; our per-row sparse Adam variant is worse than dense Adam;
  applying a maturity gate to a standalone Engram table does not help.

## Repository layout

| Path | Description |
|---|---|
| `pw.py` | Core model / training script: word and pair value tables, Engram-style baselines, per-bucket evaluation |
| `run_local_suite.py` | Three-seed local suite (orthogonality, phase diagram, width scaling, L=4 ablations, held-out vocabulary, beta2 diagnostics) |
| `diag_week1.py` | First-week diagnostics (Engram gate vs frequency, go/no-go comparison, beta2) |
| `run_upgrade_search.py` | Twelve-arm upgrade search (shared tables, dual-path gating, 3-gram routing, controls) |
| `theory_cstar_analysis.py` | Crossover-count estimation and interval-censored analysis |
| `pairstat_scale.py` | Pair-sighting statistics at V=50,257 |
| `make_figs_v5.py` | Generates every figure in `figs_v5/` |
| `download_data.py` | Downloads FineWeb shards and builds `data/counts.npy` |
| `results.jsonl` | 872-run log from the original small-scale grid |
| `RESULTS.md` | Lab notebook for the original grid |
| `LITERATURE.md` | Literature notes |
| `local_suite_results.json` | Results for `run_local_suite.py` |
| `p0_results.json` | Results for the P0 checks (fine c*, held-out prediction, shape search, ablations, 5 seeds, prefix/AdamW) |
| `upgrade_search_results.json` | Results for the upgrade search |
| `v50257_bridge_results.json` | Full-vocabulary 30M-token bridge |
| `theory_cstar_analysis.json` | Numerical output of the c* analysis |
| `pairstat_v50257.json` | Pair-sighting shares at V=50,257 |
| `figures/` | Figures for the original manuscript |
| `figs_v5/` | Current figures (PDF + PNG), captions, and figure README |
| `PAPER.md`, `paper.pdf` | Manuscript draft (may lag the current results) |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
```

`download_data.py` fetches two FineWeb GPT-2-tokenized shards (~380 MB) from
`kjj0/fineweb10B-gpt2` and builds `data/counts.npy`.  The data directory is not
committed.  The script uses `curl` (`curl.exe` on Windows).

## Reproduce

```bash
# Single run: plain model
python pw.py --dev cuda --V 6000 --width 16 --layers 2 --steps 150 --batch 128 --elr 0.1 --tag repro-plain

# Single run: word value table
python pw.py --dev cuda --V 6000 --width 16 --layers 2 --steps 150 --batch 128 --elr 0.1 --ve 0,1 --tlr 0.1 --tag repro-word

# Three-seed local suite (about 15-25 min on an RTX 4070)
python run_local_suite.py --dev cuda

# Figures
python make_figs_v5.py
```

Use `--dev cpu` if CUDA is unavailable.

## Caveats

- **`<UNK>` contamination.** At small vocabularies, positions whose *input*
  pair contains `<UNK>` are currently kept in the evaluation.  About half of
  the 1000+ bucket at V=6,000 contains `<UNK>`; filter those positions before
  interpreting the 1000+ bucket.
- **Crossover exponent.** `c*` is only weakly identified over d=16--64; the
  d=128 estimate is not identified (the residual advantage is at the noise
  floor and the gap curve changes sign five times).  The conservative
  interval-censored bound includes alpha=1; the sublinear exponent is a
  weakly identified empirical trend, not a proven scaling law.
- **Bridge scale.** The V=50,257 results are 30M-token, single-seed directional
  checks, not 125M-parameter results.
- **Equal weights.** Equal-weight plain models remain strong; the method's
  claim is about equal active FLOPs / compute-limited inference.
- **Manuscript.** `PAPER.md` and `paper.pdf` are drafts.  The current results
  and caveats are in the JSON files and `figs_v5/README.md`; where they
  disagree, the JSON files and `figs_v5/README.md` take precedence.

## License

Code is released under the MIT License (see `LICENSE`).  The manuscript text
and figures are copyright the authors; please cite the manuscript if you use
them.
