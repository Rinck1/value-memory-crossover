# Literature: per-word weights, word frequency and training length

Checked 2026-11-15 (web searches; abstracts and summaries, not full texts unless noted).

## Sources

| Paper | What it does | What it says about frequency or training length |
|---|---|---|
| Press & Wolf 2017, "Using the Output Embedding to Improve Language Models" | Ties input and output tables (one row per word). | Tying saves weights and helps small LSTM models; no split by frequency or training length. |
| Baevski & Auli 2019, "Adaptive Input Representations" | Frequent words get wide input vectors, rare words narrow ones. | Assumes frequent words need more per-word capacity; no test of how this changes with training length. |
| Lopardo et al. 2026, "Weight Tying Biases Token Embeddings Towards the Output Space" (arXiv 2603.26663, Findings ACL 2026) | Tied tables end up close to the untied output table; output gradients dominate early. | Scaling the input gradient reduces the bias. No per-frequency gain curve along training. |
| Batley 2026, "Leviathan: Decoupling Input and Output Representations" (arXiv 2601.22040) | Replaces the input table by a smooth learned map from token index. | Fully untied tables trail tied ones over 8.4B tokens; the Leviathan gain sits in **rare** tokens and is near zero for frequent ones. |
| Our trial-1 (`projects/ai-research-factory-trial-1`) | Tied head; untie only the top-K input rows. | The tying cost sits in **common** words in short runs (top 25 rows recover half); at 640 steps (37 tokens per weight) the top 100 recover nothing and the gap moves to ranks 100-2,000. Opposite of Leviathan. |
| Li 2026, "MoVE: Mixture of Value Embeddings" (arXiv 2601.22887) | A shared bank of value vectors mixed into the attention values by a gate. | Claims a new memory-scaling axis; no split by word frequency or tokens per weight. |
| Yang et al. 2026, "JTok" (arXiv 2602.00800) | A token-indexed vector gates the MLP output in every layer. | Power law in the size of token-indexed weights; gains reported to persist with backbone size. No frequency split, no tokens-per-weight sweep. |
| Yu et al. 2025, "Scone: Scaling Embedding Layers in Language Models" (arXiv 2502.01637) | Adds embeddings for frequent n-grams, computed by a small model and offloaded. | Gains grow with the number of n-gram rows; frequency enters only through which n-grams get rows. |
| "Memory Attention" 2026 (arXiv 2609.28399) | Values built from a per-layer token memory plus keys; no value projection. | Fixed 10B-token budget; values beat the residual stream at 336M+. No training-length curve. |
| "Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training" 2025 (arXiv 2508.15390) | Larger vocabularies help mainly by making frequent words easier to predict. | Frequency is the axis of the gain; no per-word extra weights. |
| Our value-embeddings paper (`conference-papers/llm-value-embeddings`) | Value embeddings v = W_V h + lam E[token]. | Gain 0.038 / 0.032 / 0.026 at 80 / 160 / 240 steps (short runs only); gain grows with vocabulary. |
| Our hash-routing paper (`conference-papers/llm-hash-routing`) | Per-word fixed expert table vs learned router. | The per-word table's lead fades to a tie at about 100 tokens per weight. |

Haiku check (2026-11-15): a Haiku agent searched for the 5 closest published papers, and I read each abstract on arXiv. They are:
- Bank of Values, arXiv 2606.02780, a per-token value table in the last third of the layers at 135M and 780M weights;
- MoVE, arXiv 2601.22887, a shared bank of value vectors mixed into the value projection by a soft gate;
- Memory Attention, arXiv 2609.28399, values built from a per-layer token memory plus the keys, with no value projection;
- Value Residual Learning (ResFormer), arXiv 2410.17897, the first layer's values added to later layers' values;
- Engram, arXiv 2601.07372, hashed n-gram tables looked up at chosen layers.

Verdict: Bank of Values does nearly the same computation, a token-indexed value table in late layers. It replaces the value instead of adding to it, and its abstract says adding the context part back gives little extra. So the module itself is not new. What is new is the comparison with a bigger plain model at equal compute, the head-to-head with Engram at equal rows and the 100-sightings mechanism, the last-two-layers placement against other placements, and the hybrid table. None of the other four uses an added per-word row: MoVE gates a shared bank by context, Memory Attention needs the keys, ResFormer's added term is computed from context, and Engram is keyed by n-grams. The idea is kept.

## Done and not done

| Question | Done? |
|---|---|
| Per-word extra tables help at a fixed budget (MoVE, JTok, Scone, Memory Attention) | Done |
| Tying cost and its gradient cause | Done (Lopardo; trial-1) |
| Which frequency band gains from decoupled inputs, at one training length | Done, and the two answers disagree (Leviathan: rare; trial-1: common) |
| **How the gain of per-word weights changes with tokens per weight, split by word frequency** | **Not done** |
| **Whether the band that gains moves from common to rare words as training grows** (would reconcile Leviathan and trial-1) | **Not done** |
| **Same compute (free lookup) vs equal weights (wider plain model) along training** | **Not done** |
| **Top-K by share of text covered vs random-K, along training** | **Not done** |

The paper fills the four "not done" rows with one test bed (`pw.py`).

## Links
- https://arxiv.org/abs/2601.22040
- https://arxiv.org/abs/2601.22887
- https://arxiv.org/abs/2602.00800
- https://arxiv.org/abs/2603.26663
- https://arxiv.org/abs/2609.28399
- https://arxiv.org/abs/2508.15390
- https://arxiv.org/abs/2502.01637
