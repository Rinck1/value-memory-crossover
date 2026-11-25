# Word- vs Pair-Keyed Value Memories in Small Language Models

Code, data-processing scripts, and empirical results for a controlled study of lookup-table value memories in GPT-style language models. A learned vector is added to the attention value stream for single tokens (`word` table) or token bigrams/trigrams (`pair` / `U123` tables); we characterize when pair tables overtake word tables as a function of training sightings, table collisions, model width, and architecture.

**Status:** Cleaned two-person research & cloud-scaling repository. All local multi-seed experiments (`V=1,500–6,000`), full-vocabulary (`V=50,257`) `30M`-token bridge runs, and causal-verified `125M` cloud training pipeline are ready.

---

## Key Verified Findings

- **Sighting-count crossover:** Word tables lead on pairs seen `<100` times; pair tables overtake in `100–999`. In truncated-vocab (`V=1,500–6,000`) evaluations, the `1000+` bucket is mixed because `52%` of pair types (and `68–72%` of positions) involve `<UNK>`; excluding `<UNK>` input pairs flips `1000+` cleanly to a pair-table win (`V=6,000: -0.024` nats, see `results/unk_filtered_eval.json`).
- **Held-out predictions:** Preregistered held-out vocabulary (`V=4,500, d=48, L=4, 250 steps`, 3 seeds) matched `5/5` per-bucket signs within `<0.012` nats (`docs/RESULTS.md` Round 30). Held-out fine-bucket width tests (`d=24, 48`) reject naive linear `c* ~ d` (`alpha=1.0`), supporting a weakly identified sublinear trend over `d=16–64`.
- **Upgraded architecture (`U123`) & causal ablations (`results/upgrade_search_results.json`, `results/extra_ablations_results.json`):** Sharing a wider tiered bank (`1+2+3-gram`) across all layers with dual `V` + residual + pre-head readout improves `L=4, w=32, V=6,000` by `+0.0658` over `Word` (`t=7.2`) and `+0.0451` over `Hard-Stack` (`t=8.0`), beating a matched-adapter hashed control (`CTRL_hs`) by `+0.0159` (`t=4.96`) with `35%` fewer table params. Matched 3-seed ablations confirm: frequency-based selection beats identical-budget random keys by `+0.0454` (`t=63.6`), removing the mid low-rank tier (`U123_no_mid_tier`, `5.3329±0.0066`) drops `-0.0139` (`t=-2.5`) while making `mid` full-rank (`5.3157`, `2.63M` params) adds only `+0.0033` (`t=1.5`), dropping residual injection (`U123_no_res`, `5.3245±0.0011`) trims `-0.0055`, pure 2-gram `U12` (`5.3229`) still beats `CTRL_hs` (`+0.0120, t=10.7`), and `CTRL_hs_pruned` (`5.3280`) isolates `+0.0068` (`t=6.1`) pruning gains.
- **Theory tests (`results/theory_cstar_analysis.json`, `F_three_component_tests`):** Held-out table-size sweeps (`M=6k/24k/96k`) confirm `U123`'s 300-step lead over `CTRL_hs(M)` shrinks monotonically with lower collision load (`+0.0250 -> +0.0159 -> -0.0025`, while `pc_100` keeps `+0.0055`). Depth sweeps (`L=2/4/6`) confirm low 3-seed inter-layer gradient cosine (`rho = 0.1498±0.0336 -> 0.1271±0.0048 -> 0.1142±0.0039`), driving `6.4x` growth in sparse `3-gram` gain (`+0.0014 -> +0.0030 -> +0.0090`) even as denser 2-gram `U1` saturates by `L=4`.
- **Full-vocabulary `V=50,257` bridge (`30M tokens`, `10%` prefix counts, 3-seed paired checks & latest latency):**
  - `plain_d256_eqFLOP` (1s): `4.9855` (`10.63 ms` b1, `367.0k tok/s` b16-infer)
  - `word_d256_last2` (1s): `4.8961` (`11.13 ms` b1, `383.8k tok/s` b16-infer)
  - `mp_ve_d256_prefix10` (1s): `4.8697` (`11.67 ms` b1, `347.7k tok/s` b16-infer)
  - `U1_shared_d512_all6` (1s): `4.8658` (`12.97 ms` b1, `335.1k tok/s` b16-infer)
  - `plain_d384_eqWeights` (`2.25x` active FLOPs, **3s**): `4.8599 ± 0.0030` (`11.73 ms` b1, `252.7k tok/s` b16-infer)
  - `U123_no_ctxgate_d256_prefix10` (Seed 0): `4.8304` (`-0.0070` behind Seed 0 `U123` `4.8234`, showing context gating is required at `V=50,257`)
  - `CTRL_hs_d256_prefix10` (**3s**): `4.8269 ± 0.0106` (`20.14 ms` b1, `233.7k tok/s` b16-infer)
  - `U123_d256_prefix10` (**3s**, low-rank mid): `4.8355 ± 0.0136` (`23.31 ms` b1, **`238.0k tok/s`** b16-infer; paired vs `plain_d384`: `+0.0243 ± 0.0106, t=+3.96`)
  - **`CTRL_hs_pruned_d256_prefix10` (3s, co-main baseline):** **`4.8189 ± 0.0074`** (`22.43 ms` b1, `202.7k tok/s` b16-infer; pruning alone gains `+0.0440 ± 0.0022, t=+34.15` over `CTRL_hs` on `pc_0`)
  - **`U123_v2_fullmid` (3s, param-matched full-rank mid `50K x 512`, `76.4M` tab params, `results/u123_v2_results.json`, `docs/U123_V2_RESULT.md`):** **`4.8072 ± 0.0099`** (seeds `4.8033, 4.7998, 4.8184`); beats `plain_d384` by `+0.0527 ± 0.0110` (`t=+8.27`), beats original low-rank `U123` by `+0.0284 ± 0.0193` (`t=+2.54`, marginal two-sided `p≈0.085`; `initfix` recovered only `+0.0058` on seed 1), and is **not significant** vs `CTRL_hs_pruned` (`+0.0118 ± 0.0163, t=+1.25`; per-seed `+0.0231, +0.0192, -0.0069`). Do not claim `fullmid`/maturity routing beats pruned hashing at 30M.
- **Strict boundaries & caveats:** Equal-total-weights plain models still dominate at `L=4, w=88`; dense Adam outperforms per-row sparse Adam by `0.018` nats (`results/p0_results.json` after `index_add_` fix); maturity gating on a standalone Engram table fails without a 1-gram word fallback.

---

## Repository Structure

```text
.
├── README.md                    # Project overview & quick start
├── CONTRIBUTING.md              # Collaborator workflow, branch rules, and reporting standards
├── CLOUD_AGENT_HANDOFF.md       # Step-by-step cloud 4-GPU execution manual (125M x 2.5B)
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── .gitignore
│
├── download_data.py             # Multi-threaded FineWeb10B-GPT2 shard downloader & verifier
├── train_125m.py                # Cloud-scale bf16 125M trainer (10 arms, causal-verified)
├── run_125m_experiments.sh      # 4-GPU multi-wave cloud launch script
│
├── experiments/                 # Reproducible local & bridge training/eval scripts
│   ├── run_local_suite.py       # 6-block 3-seed local experimental suite
│   ├── run_p0_blocks.py         # P0 checks (shape search, 5-seed, prefix, AdamW, SparseAdam)
│   ├── run_step1_fine_cstar.py  # Fine-bucket c*(d) evaluation
│   ├── run_heldout_widths_cstar.py
│   ├── run_theory_tests_L_M_rho.py
│   ├── run_upgrade_search.py    # U1-U4 + matched-adapter controls (L=4, V=6,000)
│   ├── run_extra_ablations.py   # 5 extra 3-seed causal/component ablations (no3g, random-tier, no-ctxgate, pruned, +3g)
│   ├── run_v50257_bridge.py     # V=50,257 30M bridge (plain, word, mp_ve, d384)
│   ├── run_v50257_upgrades.py   # V=50,257 30M upgrades (U1, U123, CTRL_hs)
│   ├── run_v50257_multiseed.py  # V=50,257 multi-seed & CTRL_hs_pruned runner
│   ├── measure_v50257_latency.py
│   ├── eval_non_unk_v6000.py
│   ├── diag_week1.py
│   └── pairstat_scale.py
│
├── analysis/                    # Theory analysis & publication figure generation
│   ├── theory_cstar_analysis.py
│   └── make_figs_v5.py          # Generates all figs_v5/*.pdf & *.png from results/*.json
│
├── results/                     # Committed JSON outputs (1-to-1 with experiments/ & analysis/)
├── figs_v5/                     # Current publication figures (PDF + PNG + captions.tex + README)
├── docs/                        # Research proposal (PROPOSAL.md, ICML_Research_Proposal_v5.pdf/.tex), RESULTS.md, LITERATURE.md
└── legacy/                      # Original 872-run small-scale baseline scripts & results.jsonl
```

---

## Setup & Quick Start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Fetch validation + 100M-token training shard (~380 MB) and verify
python download_data.py --shards 1 --workers 4 --verify
```

### Reproduce Local Benchmarks & Figures

```bash
# Regenerate all publication figures in figs_v5/ from committed JSONs (no GPU needed)
python analysis/make_figs_v5.py

# Re-run c*(d) & <UNK>-composition analysis (no GPU needed)
python analysis/theory_cstar_analysis.py

# Run 3-seed local suite on GPU (~20 min on RTX 4070)
python experiments/run_local_suite.py --dev cuda

# Smoke-test 125M trainer across all 10 arms (CPU, <20 s)
python train_125m.py --smoke --arm u123 --device cpu --out /tmp/smoke_u123
```

### Cloud 125M Scaling Runs

See `CLOUD_AGENT_HANDOFF.md` and `run_125m_experiments.sh` for the 4-GPU `2.5B`-token workflow.

---

## License

Code is released under the MIT License (`LICENSE`). Manuscript text and figures are copyright the authors.
