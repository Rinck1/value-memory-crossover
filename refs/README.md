# References

Directly relevant papers for the value-memory / sighting-count project.
Run `bash refs/download_refs.sh` to fetch the PDFs into this directory
(PDFs are git-ignored).  arXiv IDs for 2026 papers follow `docs/LITERATURE.md`;
if a link is stale, search the title on arXiv.

## Core comparisons (word vs pair / n-gram memory)

| Paper | Why it matters | Links |
|---|---|---|
| Engram (Cheng et al. 2026) | Hashed 2/3-gram tables + context gate; main pair-table baseline | [abs](https://arxiv.org/abs/2601.07372) · [pdf](https://arxiv.org/pdf/2601.07372) |
| X-GRAM (Chen et al. 2026) | Frequency-aware token tables; Appendix C.3's ">100 updates / 7.3%" is the closest prior | [abs](https://arxiv.org/abs/2604.21724) · [pdf](https://arxiv.org/pdf/2604.21724) |
| LongCat-Flash-Lite (Liu et al. 2026) | N-gram tables overtake per-token tables after 300B tokens | [abs](https://arxiv.org/abs/2601.21204) · [pdf](https://arxiv.org/pdf/2601.21204) |
| Bank of Values (He et al. 2026) | Per-token value table in late layers; direct predecessor of the word table | [abs](https://arxiv.org/abs/2606.02780) · [pdf](https://arxiv.org/pdf/2606.02780) |
| MoME (Li et al. 2026) | Value embedding vs bigram table tie at 135M | [abs](https://arxiv.org/abs/2609.15126) · [pdf](https://arxiv.org/pdf/2609.15126) |
| MoVE (Li 2026) | Shared value bank + soft gate | [abs](https://arxiv.org/abs/2601.22887) · [pdf](https://arxiv.org/pdf/2601.22887) |
| Scone (Yu et al. 2025) | N-gram input tables; scaling embedding layers | [abs](https://arxiv.org/abs/2502.01637) · [pdf](https://arxiv.org/pdf/2502.01637) |
| Over-Tokenized Transformers (Huang et al. 2025) | 2-gram-only table vs 1-gram table | [abs](https://arxiv.org/abs/2501.16975) · [pdf](https://arxiv.org/pdf/2501.16975) |
| Hash Layers (Roller et al. 2021) | Token vs bigram-hash routing | [abs](https://arxiv.org/abs/2106.04426) · [pdf](https://arxiv.org/pdf/2106.04426) |
| JTok (Yang et al. 2026) | Token-indexed vector gating in the MLP | [abs](https://arxiv.org/abs/2602.00800) · [pdf](https://arxiv.org/pdf/2602.00800) |
| Memory Attention (Kang 2026) | Per-layer token memory for values | [abs](https://arxiv.org/abs/2609.28399) · [pdf](https://arxiv.org/pdf/2609.28399) |
| STEM (Sadhukhan et al. 2026) | Per-token table in the FFN | [abs](https://arxiv.org/abs/2601.10639) · [pdf](https://arxiv.org/pdf/2601.10639) |

## Frequency / tying / input-output decoupling

| Paper | Why it matters | Links |
|---|---|---|
| Adaptive Input Representations (Baevski & Auli 2019) | Frequency-dependent input width | [abs](https://arxiv.org/abs/1809.10853) · [pdf](https://arxiv.org/pdf/1809.10853) |
| Using the Output Embedding to Improve Language Models (Press & Wolf 2017) | Weight tying | [abs](https://arxiv.org/abs/1608.05859) · [pdf](https://arxiv.org/pdf/1608.05859) |
| Tying Word Vectors and Word Classifiers (Inan et al. 2017) | Loss-framework view of tying | [abs](https://arxiv.org/abs/1611.01462) · [pdf](https://arxiv.org/pdf/1611.01462) |
| Weight Tying Biases Token Embeddings (Lopardo et al. 2026) | Gradient cause of the tying cost | [abs](https://arxiv.org/abs/2603.26663) · [pdf](https://arxiv.org/pdf/2603.26663) |
| Leviathan (Batley 2026) | Decoupled input/output; rare-token gains | [abs](https://arxiv.org/abs/2601.22040) · [pdf](https://arxiv.org/pdf/2601.22040) |
| Exploiting Vocabulary Frequency Imbalance (2025) | Frequency axis of vocabulary scaling | [abs](https://arxiv.org/abs/2508.15390) · [pdf](https://arxiv.org/pdf/2508.15390) |
| Value Residual Learning / ResFormer (Zhou et al. 2024) | Cross-layer value reuse | [abs](https://arxiv.org/abs/2410.17897) · [pdf](https://arxiv.org/pdf/2410.17897) |

## Background

| Paper / resource | Why it matters | Links |
|---|---|---|
| Kaplan et al. 2020, Scaling Laws | `6ND` and compute budgets | [abs](https://arxiv.org/abs/2001.08361) · [pdf](https://arxiv.org/pdf/2001.08361) |
| Hoffmann et al. 2022, Chinchilla | Tokens-per-parameter | [abs](https://arxiv.org/abs/2203.15556) · [pdf](https://arxiv.org/pdf/2203.15556) |
| AdamW (Loshchilov & Hutter 2017) | Dense vs sparse, weight decay | [abs](https://arxiv.org/abs/1711.05101) · [pdf](https://arxiv.org/pdf/1711.05101) |
| Hash Embeddings (Svenstrup et al. 2017) | Hashing and collisions | [abs](https://arxiv.org/abs/1709.03933) · [pdf](https://arxiv.org/pdf/1709.03933) |
| Feature Hashing (Weinberger et al. 2009) | Original hashing trick | [abs](https://arxiv.org/abs/0902.2206) · [pdf](https://arxiv.org/pdf/0902.2206) |
| modded-nanogpt PR #201 | Bigram-hash speedrun | [GitHub](https://github.com/KellerJordan/modded-nanogpt/pull/201) |
