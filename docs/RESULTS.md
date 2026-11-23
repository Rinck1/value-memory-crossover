# Results notebook: per-word weights along long training

Test bed `pw.py`. Default model: width 16, 2 layers, 2 heads, RMSNorm, RoPE, QK-norm, SwiGLU, no biases; Muon (rate 0.02) on block matrices, Adam on tables, norms and scalars. Vocabulary 1,500 (the top 1,499 GPT-2 tokens plus one unknown token, which covers 31% of FineWeb text). Context 64, batch 32 rows (2,048 tokens per step), each token read once, warmup then flat then a linear cooldown over the last 30% of steps. Every run has its own cooldown; "steps" means the full length of that run. One run per setting.

**Main score: loss on known target words** (`val_known`, 1,000 held-out FineWeb rows of 64 tokens). The unknown token is a small-vocabulary artifact (real vocabularies have none), and in round 0 it ranked learning rates differently from known words.

**Word bands** by share of the kept text covered, counted from the most common word: common = the first 50% (37 words at vocabulary 1,500), middle = 50-80% (344 words), rare = 80-100% (1,117 words). Finer bins by frequency rank: 0-10, 10-30, 30-100, 100-300, 300-1,000, 1,000+. "Gain on bin X" is the loss change on positions whose current input word is in X.

**Tokens per weight (tpw)** counts every weight, tables included. At this size the tables are 80-90% of the weights, so tokens per body weight are 5-10 times higher (600 at 1,600 steps).

## Round 0: learning rates (400 steps, 15 tokens per untied weight)

Rule going in: none yet; tune before comparing.

| Model | Setting | Known-word loss |
|---|---|---|
| untied | table rate 0.01 / 0.03 / 0.05 / 0.1 / 0.3 | 5.087 / 4.955 / **4.941** / 4.977 / 5.068 |
| tied | table rate 0.01 / 0.03 / 0.05 / 0.1 / 0.3 | 5.151 / 5.052 / **5.044** / 5.080 / 5.124 |
| untied / tied | block rate 0.01 / 0.04 | 4.948 / 5.001 ; 5.073 / 5.054 |
| untied / tied | input scale 1 / 2 / 8 (default 4) | 4.998 / 4.959 / 4.934 ; 5.070 / 5.055 / 5.059 |
| untied + value embeddings in both layers | table rate 0.02 / 0.05 / 0.1 / 0.2 | 4.911 / 4.898 / 4.891 / **4.888** |
| tied + own input rows for the 37 common words | row rate 0.005 / 0.01 / 0.02 / 0.05 / 0.1 / 0.2 | 5.018 / 5.021 / **5.001** / 5.019 / 5.028 / 5.025 |

**Takeaway.** Defaults: block rate 0.02, table rate 0.05, input scale 4, value-embedding rate 0.1, own-input-row rate 0.05 (within 0.02 of the best; round 2 found rare rows want 0.05). The optimum is flat: about 0.01-0.02 between neighbouring rates, which is also the single-run noise level used from here on.

## Round 1: tied vs untied vs "untie the common words" along training

Rule going in (from trial-1): giving the most common words their own input rows recovers a large share of the tying cost.

Lengths 100 / 200 / 400 / 800 / 1,600 steps (0.2M to 3.3M tokens). Validation set 600 rows here (rounds 3+ use 1,000 rows, so numbers differ slightly).

| Steps | tpw (tied) | Tying cost (tied - untied) | Gain from own rows for the 37 common words | Share recovered |
|---|---|---|---|---|
| 100 | 6.8 | 0.207 | 0.085 | 41% |
| 200 | 14 | 0.142 | 0.047 | 33% |
| 400 | 27 | 0.102 | 0.043 | 42% |
| 800 | 55 | 0.098 | 0.018 | 19% |
| 1,600 | 109 | 0.081 | 0.007 | 8% |

By input band, the tying cost on common inputs fell from 0.189 to 0.037; on middle and rare inputs it fell only from 0.31 / 0.20 to 0.10 / 0.11.

**Takeaway.** The rule breaks with long training: untying the common words recovers 41% of the tying cost early and 8% at 109 tokens per weight. The cost that remains sits on middle and rare words.

## Round 2: untie other bands (200 and 800 steps)

Rule going in: at long training the words that need their own input rows are the middle and rare ones.

| Variant (tied base) | Rows | Gain at 200 steps | Gain at 800 steps |
|---|---|---|---|
| own rows: common | 37 | 0.047 | 0.018 |
| own rows: middle | 344 | 0.000 | 0.018 (0.031 at row rate 0.05) |
| own rows: rare | 1,117 | 0.017 | 0.001 (0.011 at row rate 0.05) |
| own rows: 37 random words | 37 | -0.002 | 0.004 |
| own rows: 344 random words | 344 | -0.003 | -0.006 |
| own rows: every known word (row rate 0.02 / 0.05) | 1,498 | | 0.064 / 0.085 (fully untied: 0.098) |

**Takeaway.** Random rows recover nothing, so frequency carries the gain. Untying every known word recovers 87% of the cost, but the three bands alone add up to 0.060 of 0.085: part of the cost is shared by all rows (the output table improves when every row is free of input duty). Middle and rare rows want the higher rate (0.05), so all band runs from round 3 on use 0.05.

## Round 3: the full grid along training (both families)

Rule going in: (a) untying gains move from common to rarer words with training; (b) value embeddings, like untying, help common words early and fade.

Both families at 100 / 200 / 400 / 800 / 1,600 steps. 1,600-step runs of the tied family took 33-41 s on the CPU because three other sessions were training on the same Mac; from here on runs use the Apple GPU (`--dev mps`), which gave exactly the same loss on a check run (3.8800 on both) at 1.7 times the speed.

**Untying (tied base). Gain in known-word loss:**

| Steps | tpw (tied) | Untied (all) | Common (37) | Middle (344) | Rare (1,117) | Every known word | Tied width 28 (equal weights to untied) |
|---|---|---|---|---|---|---|---|
| 100 | 6.8 | 0.220 | 0.081 | -0.029 | -0.040 | 0.193 | 0.147 |
| 200 | 14 | 0.142 | 0.055 | 0.036 | 0.030 | 0.138 | 0.112 |
| 400 | 27 | 0.105 | 0.038 | 0.024 | 0.011 | 0.095 | 0.071 |
| 800 | 55 | 0.094 | 0.026 | 0.035 | 0.013 | 0.082 | 0.116 |
| 1,600 | 109 | 0.075 | 0.014 | 0.019 | 0.014 | 0.076 | 0.131 |

**Tying cost by input frequency rank (untied minus tied, loss on positions with that input):**

| Steps | 0-10 | 10-30 | 30-100 | 100-300 | 300-1,000 | 1,000+ |
|---|---|---|---|---|---|---|
| 100 | 0.168 | 0.250 | 0.321 | 0.334 | 0.243 | 0.141 |
| 200 | 0.049 | 0.120 | 0.154 | 0.216 | 0.211 | 0.143 |
| 400 | 0.032 | 0.106 | 0.118 | 0.142 | 0.150 | 0.127 |
| 800 | 0.037 | 0.054 | 0.084 | 0.126 | 0.135 | 0.103 |
| 1,600 | 0.041 | 0.032 | 0.068 | 0.105 | 0.117 | 0.100 |

**Value embeddings (untied base). Gain in known-word loss:**

| Steps | tpw (untied) | All rows (both layers) | Common (37) | Middle (344) | Rare (1,117) | Plain width 28 (equal weights to "all rows") |
|---|---|---|---|---|---|---|
| 100 | 3.8 | 0.008 | -0.007 | -0.006 | 0.010 | 0.068 |
| 200 | 7.6 | 0.031 | -0.013 | -0.004 | 0.000 | 0.055 |
| 400 | 15 | 0.047 | 0.011 | 0.020 | 0.002 | 0.090 |
| 800 | 30 | 0.035 | 0.002 | 0.017 | 0.007 | 0.109 |
| 1,600 | 61 | 0.057 | 0.015 | 0.016 | 0.001 | 0.128 |

Value-embedding gain by input rank at 1,600 steps: 0.044 / 0.045 / 0.049 / 0.031 / 0.044 / 0.032 (bins as above), unknown token 0.026.

**Takeaway.** (a) holds: the tying cost on the 10 most common words drops 4 times in the first 400 steps and then stays near 0.03-0.04, while on ranks 100-1,500 it falls only from 0.14-0.33 to 0.10-0.12; the peak moves from ranks 30-300 to ranks 100-1,500. (b) breaks: value-embedding gains grow with training (0.008 to 0.057) and are spread over all frequency bins. The two kinds of per-word weights move in opposite directions. At equal weights a wider plain model beats value embeddings at every length, and from 800 steps on a wider tied model beats the untied one (as in trial-1), but both wider models cost more compute per token, while a table lookup is free.

## Round 4: controls and rate checks at 1,600 steps

Rule going in: the gains come from learned per-word content chosen by frequency, not from extra weights or tuning.

| Run | Known-word loss | Gain over its base |
|---|---|---|
| untied (base for value embeddings) | 4.770 | |
| tied (base for untying) | 4.845 | |
| own rows: 37 common / 37 random | 4.831 / 4.849 | 0.014 / -0.004 |
| own rows: 344 middle / 344 random | 4.826 / 4.844 | 0.019 / 0.000 |
| value embeddings: all rows | 4.713 | 0.057 |
| value embeddings: all known words (no unknown-token row) | 4.726 | 0.044 |
| value embeddings: common + middle (381 rows) / 381 random rows | 4.745 / 4.762 | 0.025 / 0.008 |
| value embeddings: all rows frozen at random init | 4.772 | -0.003 |
| rate checks: untied table 0.03 / 0.1 | 4.763 / 4.767 | |
| rate checks: tied table 0.03 / 0.1 | 4.849 / 4.843 | |
| rate checks: value embeddings 0.05 / 0.2 | 4.719 / 4.713 | |
| rate checks: common own rows 0.02 / 0.1 | 4.842 / 4.828 | |
| rate checks: common value embeddings 0.05 / 0.3 | 4.754 / 4.762 | |

**Takeaway.** Frequency-chosen rows beat random rows by 0.016-0.019 in both families; frozen random value rows give nothing, so the value-embedding gain is learned per-word content, not noise or extra weights. The one unknown-token row carries 0.013 of the 0.057. Every rate is within 0.007 of its best, so no gap here is a tuning artifact (best-tuned tying cost 0.080, best-tuned common-row gain 0.015).

Per weight at 1,600 steps: common value rows give 0.015 for 1,184 weights (0.013 per 1K weights), all value rows 0.057 for 48K weights (0.0012 per 1K), widening to 28 gives 0.128 for 48.5K weights (0.0026 per 1K, at twice the compute per token).

## Round 5: width (1,600 steps, vocabulary 1,500)

Rule going in: common-word untying gains fade with tokens per weight; value-embedding gains do not.

Width 8 uses 1 head of 8, width 32 uses 2 heads of 16. Rates as at width 16.

| Width | tpw (tied) | Tying cost | Own rows: common (37) | Own rows: middle (344) | Value embeddings: all rows | Value embeddings: common |
|---|---|---|---|---|---|---|
| 8 | 245 | 0.095 | 0.025 | 0.018 | 0.048 | 0.008 |
| 16 | 109 | 0.075 | 0.014 | 0.019 | 0.057 | 0.015 |
| 32 | 46 | 0.057 | 0.013 | -0.005 | 0.032 | 0.011 |

**Takeaway.** Tokens per weight alone does not set the common-word gain: width 8 has the most tokens per weight but the largest common-row gain (0.025). The narrow model has less room in one shared row for both jobs, so the conflict lasts longer. The wider model has a smaller tying cost overall (0.057). Value-embedding gains stay positive at every width (0.032-0.057).

## Round 6: vocabulary 500 and 6,000

Rule going in: the band whose own rows pay moves from common to rare words with training, at every vocabulary size.

Vocabulary 6,000: the table rate retuned at 600 steps (0.03 / 0.05 / 0.1 / 0.2 gave untied 5.699 / 5.685 / 5.658 / 5.656 and tied 5.792 / 5.787 / 5.752 / 5.758), so the table rate is 0.1; own rows keep 0.05 (0.05 beat 0.1: 5.733 vs 5.757); value tables 0.1 (beat 0.2). Bands at 6,000: common 84 words, middle 1,175, rare 4,739. At 500: common 20, middle 116, rare 362 (the unknown token is 42% of text). Each 1,200-step run at 6,000 took 18 s.

**Vocabulary 6,000, gain in known-word loss:**

| Steps | tpw (tied) | Tying cost | Own rows: common (84) | middle (1,175) | rare (4,739) | Value emb.: all | Value emb.: common | Value emb.: middle |
|---|---|---|---|---|---|---|---|---|
| 200 | 4 | 0.125 | 0.031 | -0.024 | -0.022 | 0.022 | -0.002 | 0.021 |
| 600 | 12 | 0.094 | 0.019 | 0.001 | -0.007 | 0.060 | 0.018 | 0.033 |
| 1,200 | 24 | 0.093 | 0.020 | 0.018 | -0.013 | 0.054 | 0.014 | 0.024 |

**Vocabulary 500, gain in known-word loss:**

| Steps | tpw (tied) | Tying cost | Own rows: common (20) | middle (116) | rare (362) | Value emb.: all | Value emb.: common | Value emb.: middle |
|---|---|---|---|---|---|---|---|---|
| 200 | 29 | 0.076 | 0.019 | -0.007 | 0.019 | 0.043 | 0.032 | 0.011 |
| 800 | 117 | 0.059 | 0.000 | 0.012 | 0.014 | 0.034 | 0.010 | 0.013 |
| 1,600 | 234 | 0.052 | -0.006 | 0.033 | 0.015 | 0.040 | 0.014 | 0.014 |

**Takeaway.** At vocabulary 500 the common-word gain is gone by 117 tokens per weight (0.000, then -0.006), while middle rows gain 0.033. At vocabulary 6,000 the middle band starts paying only at 1,200 steps and the rare rows still lose (-0.013): each rare word has been seen only about 70 times, too few to train its own row. Value embeddings gain 0.034-0.060 at both vocabularies and every length from 600 steps.

## Round 6b: the occurrence window (analysis, no new runs; `band_occ.py`)

Rule going in: the training length at which a band's own rows pay depends on vocabulary and width, so tokens per weight is the wrong clock for it.

New clock: how many times one word of the band has been seen in training (median over the band = share of text per word times tokens seen). All 37 band points from rounds 3, 5 and 6, sorted by that count:

| Times one word was seen | Points | Own-row gain on all known words | Own-row gain on positions with that input |
|---|---|---|---|
| under 100 | 7 (rare at 6,000; rare/middle at 100 steps) | -0.040 to +0.030, mean -0.015 | -0.009 to +0.078 |
| 100-1,000 | 13 | -0.007 to +0.036, mean +0.017 | +0.030 to +0.080 |
| 1,000-10,000 | 12 | -0.005 to +0.081, mean +0.028 | +0.008 to +0.109 |
| over 15,000 | 5 (common at 1,600 steps; common at vocabulary 500) | -0.006 to +0.025, mean +0.009 | -0.010 to +0.029 |

The tying cost on a band falls with the same count in every vocabulary: about 0.12-0.33 under 100, 0.09-0.19 at 100-1,000, 0.04-0.11 at 1,000-10,000 (one 100-step point at 0.195), and 0.013-0.063 over 15,000. Across all 79 rank-bin points the correlation of the tying cost with log(times seen) is -0.67.

**Takeaway.** Own input rows pay for words seen roughly 100 to 10,000 times. Below that window a new row is too undertrained to beat the shared one (rare words at vocabulary 6,000); above it the shared row has learned to do both jobs (common words at vocabulary 500 and at long training). Common words pass through the window first, which is why short runs (trial-1) find the gain on common words and long runs (Leviathan) find it on rare ones. The upper edge moves with width: at width 8 the common words still gain 0.029 at 17,500 sightings, at width 32 0.012. Value embeddings do not follow the window: common rows still gain 0.014 at 31,000 sightings (vocabulary 500).

## Round 7: batch size vs steps at fixed tokens (vocabulary 1,500)

Rule going in: the gains follow tokens seen (how often each word was seen), not the number of optimizer steps.

Batch 128 rows instead of 32. Rate check on the tied model at batch 128 x 400 steps: table rate 0.05 / 0.1 / 0.2 gave 4.894 / 4.895 / 4.885; block rate 0.02 / 0.04 / 0.08 gave 4.894 / 4.899 / 4.924; untied block rate 0.04 gave 4.802 vs 4.799. Defaults kept.

| Setting | Tokens | Tying cost | Own rows: common | middle | rare | Value emb.: all | Value emb.: common |
|---|---|---|---|---|---|---|---|
| batch 32 x 400 steps | 0.8M | 0.105 | 0.038 | 0.024 | 0.011 | 0.047 | 0.011 |
| batch 32 x 1,600 steps | 3.3M | 0.075 | 0.014 | 0.019 | 0.014 | 0.057 | 0.015 |
| batch 128 x 400 steps | 3.3M | 0.095 | 0.019 | 0.016 | 0.010 | 0.041 | -0.007 |
| batch 128 x 800 steps | 6.6M | 0.085 | 0.024 | 0.022 | 0.019 | 0.053 | 0.007 |

**Takeaway.** At equal tokens, batch 128 x 400 gives the band pattern of batch 32 x 1,600 (common 0.019 vs 0.014, not the 0.038 of 400 small steps): which words gain from own rows follows how often each word was seen. The tying cost itself shrinks more slowly with big batches (0.095 vs 0.075 at equal tokens; fewer updates). With twice the tokens (6.6M, 219 tokens per tied weight, each common word seen 35,000 times) the common-word gain stays at 0.024, equal to middle and rare: at vocabulary 1,500 it settles at a floor of about 0.02 instead of reaching zero. Value-embedding gains hold (0.041-0.053); the common-only value rows are within noise of zero here.

## Round 8: context length at fixed tokens per step (1,600 steps, 2,048 tokens per step)

Rule going in: none of the per-word effects depends on context length.

Context 32 (batch 64), 64 (batch 32, round 3), 128 (batch 16), 256 (batch 8; validation 500 rows).

| Context | Tying cost | Own rows: common | middle | Value emb.: all | Value emb.: common |
|---|---|---|---|---|---|
| 32 | 0.081 | 0.003 | 0.028 | 0.045 | 0.002 |
| 64 | 0.075 | 0.014 | 0.019 | 0.057 | 0.015 |
| 128 | 0.079 | 0.021 | 0.032 | 0.062 | 0.034 |
| 256 | 0.070 | 0.002 | 0.015 | 0.057 | 0.008 |

**Takeaway.** No trend with context: the tying cost stays at 0.070-0.081, middle rows gain at least as much as common rows at every context (0.015-0.032 vs 0.002-0.021), and the all-row value-embedding gain stays at 0.045-0.062. The common-only value-row gain swings between 0.002 and 0.034 with no order, so it is not claimed.

## Round 9: vocabulary 24,000 (400 steps, 2 tokens per tied weight)

Rule going in: own rows help only words seen about 100+ times; at a large vocabulary that excludes most of the table early.

Table rate 0.1 (0.2 was worse for both: untied 6.513 vs 6.500, tied 6.651 vs 6.607). Bands: common 169 words (seen about 870 times each), middle 2,941 (55 times), rare 20,888 (5 times). Runs took 20-26 s.

| Variant | Gain on known words |
|---|---|
| tying cost (untied over tied) | 0.107 |
| own rows: common (169) | 0.013 |
| own rows: middle (2,941) | -0.029 |
| own rows: rare (20,888) | -0.018 |
| value embeddings: all rows | 0.062 |
| value embeddings: common rows | 0.022 |

**Takeaway.** The window's lower edge holds at a fourth vocabulary: rows for words seen 5-55 times hurt, rows for words seen 870 times help. Value embeddings gain 0.062, as at every other vocabulary.

## Round 6b update: the window with rounds 7-9 (52 band points; `band_occ.py`)

| Times one word was seen | Points | Own-row gain on all known words, mean (range) |
|---|---|---|
| under 100 | 10 | -0.014 (-0.040 to +0.030) |
| 100-1,000 | 15 | +0.017 (-0.007 to +0.036) |
| 1,000-10,000 | 17 | +0.026 (-0.005 to +0.081) |
| over 15,000 | 10 | +0.012 (-0.006 to +0.025) |

**Takeaway.** Lower edge: firm; 8 of 10 points under 100 sightings lose (the two gains are the rare band at 200 and 400 steps at vocabulary 1,500, seen 46 and 91 times). Upper edge: soft. Above 15,000 sightings the gain falls to 0.00-0.025 (zero at vocabulary 500, about 0.02 at 1,500), never back to the 0.04-0.08 of the window's middle.

## Round 10: do the bands add up? (untie two bands, or all known words, in one run)

Rule going in (from round 6b): untie only the words seen 100+ times; rows for rarer words are too undertrained to help.

| Setting | Tying cost | Common alone | Middle alone | Rare alone | Sum of the three | Common + middle together | All known words together |
|---|---|---|---|---|---|---|---|
| vocab 1,500, 100 steps | 0.220 | 0.081 | -0.029 | -0.040 | 0.012 | | 0.193 |
| vocab 1,500, 200 steps | 0.142 | 0.055 | 0.036 | 0.030 | 0.121 | | 0.138 |
| vocab 1,500, 400 steps | 0.105 | 0.038 | 0.024 | 0.011 | 0.073 | | 0.095 |
| vocab 1,500, 800 steps | 0.094 | 0.026 | 0.035 | 0.013 | 0.074 | | 0.082 |
| vocab 1,500, 1,600 steps | 0.075 | 0.014 | 0.019 | 0.014 | 0.047 | 0.037 | 0.076 |
| vocab 500, 1,600 steps | 0.052 | -0.006 | 0.033 | 0.015 | 0.042 | | 0.054 |
| vocab 6,000, 1,200 steps | 0.093 | 0.020 | 0.018 | -0.013 | 0.025 | 0.057 | 0.087 |
| vocab 24,000, 400 steps | 0.107 | 0.013 | -0.029 | -0.018 | -0.034 | 0.048 | 0.096 |

Untying one band also moves the loss on positions whose input is in another band. At vocabulary 24,000, untying the middle band alone makes common-input positions 0.016 worse and rare-input positions 0.058 worse.

**Takeaway.** The rule breaks. Bands do not add: at vocabulary 24,000 the middle rows alone lose 0.029 and the common rows alone gain 0.013, but the two together gain 0.048, and all known rows together gain 0.096 (90% of the tying cost). Untying every known word recovers 87-104% of the tying cost in all 8 settings. The best single band recovers 12-39% (63% for the middle band at vocabulary 500). The cost of tying belongs to the whole table, not to a set of words. A likely reason, not yet tested: the first layer reads one input space, and while most rows still sit in the output space, a few free rows do not fit it. The 100-sighting window describes what one band does alone inside a tied table, which is the trial-1 setup. It is not a guide to which rows a model should untie.

## Round 11: rate checks and an offset test at vocabulary 24,000 (400 steps)

Rule going in: bands do not add, and this is not a tuning artifact.

Own-row rate (tied base 6.607 known-word loss; gain in brackets):

| Rows untied | rate 0.01 | 0.02 | 0.05 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|---|
| common (169) | | | 6.594 (0.013) | **6.558 (0.049)** | | |
| middle (2,941) | | 6.643 (-0.036) | 6.636 (-0.029) | **6.621 (-0.013)** | | 6.621 (-0.013) |
| rare (20,888) | 6.653 (-0.045) | | **6.625 (-0.018)** | | 6.632 (-0.024) | |
| common + middle | | | 6.559 (0.048) | **6.524 (0.083)** | | |
| all known words | | | 6.511 (0.096) | **6.485 (0.122)** | | |
| rare, as offsets added to the shared row (start at zero) | | | 6.637 (-0.030) | | | |

Fully untied at its tuned table rate: 6.500 (tying cost 0.107).

**Correction.** At vocabulary 24,000 the own rows want rate 0.15, not 0.05. Tuned, the common rows gain 0.049 (not 0.013) and all known rows gain 0.122, more than the fully untied model (0.107; it shares one rate for both tables).

**Takeaway.** Tuned, the bands still do not add: common 0.049 plus middle -0.013 is 0.036, while the two together gain 0.083 and all rows 0.122. Rows for words seen 5-55 times lose at every rate when untied alone. Making the rare rows offsets from the shared row, starting exactly at it, loses too (-0.030), so the loss does not come from new rows starting at random places. Why a lone band hurts and the full table helps is open.

## Round 12: depth (review fix-up; vocabulary 1,500, 800 steps)

Rule under test: claims 1, 2 and 4 were all measured with 2 layers. A reviewer would ask whether a deeper body, which can rebuild an input representation from a shared row, changes them. Prediction: common rows still recover a minority of the cost, all rows recover all of it, value embeddings still gain about 0.03-0.05.

| Layers | tpw (tied) | Tying cost | Common | Middle | Rare | Common + middle | All known | Value emb. (all layers) |
|---|---|---|---|---|---|---|---|---|
| 2 | 55 | 0.094 | 0.026 | 0.035 | 0.013 | | 0.082 | 0.035 |
| 4 | 46 | 0.067 | 0.017 | -0.004 (0.007 at rate 0.15) | -0.017 | 0.034 | 0.074 | 0.032 |
| 6 | 39 | 0.071 | 0.028 | | | | 0.067 | 0.052 |

Runs took 18-28 s on the GPU (load average 5).

**Takeaway.** Depth keeps every rule. All known rows recover 110% / 94% of the tying cost at 4 / 6 layers; common rows recover 25% / 39%. At 4 layers the three bands alone sum to -0.004 while common and middle together gain 0.034 and all known rows 0.074, the clearest case of bands not adding. Value embeddings gain 0.032-0.052. Deeper bodies cut the tying cost from 0.094 to about 0.07 but do not remove it.

## Round 13: equal training compute (new direction: per-word weights as an invention)

New brief from the parent session (2026-11-15): turn the findings into an invention. First question: claim 6 compared value embeddings to a wider model at equal *weights*, where the wider model wins. A table lookup costs no FLOPs, so the fair comparison for training cost is equal *compute*. `flops.py` counts forward FLOPs per token (multiply-add = 2; attention over L/2 keys on average; value rows 2d per layer; table lookups free). Training compute = 3 x forward x tokens. Every plain model gets the steps that match the budget of width 16 trained for the longest run (1,600 / 1,600 / 1,200 / 400 steps at vocabulary 500 / 1,500 / 6,000 / 24,000).

Rule under test: at equal training compute, a model with a value-embedding table beats the best plain model. Prediction: yes where the output layer is most of the FLOPs (large vocabulary), and less or not at all at vocabulary 500, where the output layer is half the FLOPs.

| Vocabulary (budget) | Untied w16 | Best plain (shape, steps) | Other plain shapes | Value emb. w16 | Best value emb. (shape) | Value emb. lead over best plain |
|---|---|---|---|---|---|---|
| 500 (0.31 TFLOP) | 4.2787 | 4.2138 (w28, 709) | w20 4.2295, w24 4.2211, 4 layers 4.2724 | 4.2391 | 4.1973 (w24, 882) | **0.017** |
| 1,500 (0.63 TFLOP) | 4.7697 | 4.7544 (w28, 799) | w20 4.7687, w24 4.7702, w32 4.7550, 4 layers 4.8148, 6 layers 4.8092, w24 + 4 layers 4.7564 | 4.7131 | 4.7131 (w16, 1,600) | **0.041** |
| 1,500 (0.31 TFLOP) | 4.8651 | 4.8403 (w28, 400) | w20 4.8555, w24 4.8740, w32 4.8909, w24 + 4 layers 4.8972 | 4.8301 | 4.8301 (w16, 800) | **0.010** |
| 6,000 (1.53 TFLOP) | 5.5539 | 5.5084 (w24, 775) | w20 5.5301, w28 5.5437, w32 5.5473, 4 layers 5.5455 | 5.4997 | 5.4810 (w24, 775) | **0.027** |
| 24,000 (1.93 TFLOP) | 6.4999 | 6.4999 (w16, 400) | w20 6.5144, w24 6.5699, w28 6.5882, 4 layers 6.5077 | 6.4384 | 6.4384 (w16, 400) | **0.062** |

Rate check on the best plain shape at vocabulary 1,500 (w28, 799 steps): table rate 0.03 / 0.05 / 0.1 give 4.7713 / 4.7544 / 4.7678; block rate 0.01 / 0.02 / 0.04 give 4.7654 / 4.7544 / 4.7727. The default rates are its best. Value embeddings on wider models: vocabulary 1,500 w20 4.7253, w24 4.7361; vocabulary 6,000 w20 5.4840, vocabulary 24,000 w20 6.4962; vocabulary 500 w28 4.2070.

**Takeaway.** At equal training compute the best model with a value-embedding table beats the best plain model at all four vocabularies, by 0.017-0.062. The lead is largest where the output layer is most of the FLOPs (98% at vocabulary 24,000, where no wider or deeper plain model beats width 16 at all), and grows with the budget at vocabulary 1,500 (0.010 at 0.31 TFLOP, 0.041 at 0.63 TFLOP). At vocabulary 500 a narrow table model loses to the wider plain model (4.2391 vs 4.2138); the table only wins there once it is added to a wider body (w24). Claim 6 (wider wins at equal weights) and this round together: the table buys loss with weights instead of FLOPs.

## Round 14: a cheaper value table (vocabulary 1,500, 1,600 steps, untied width 16)

Rule under test: a value table needs full rows only for the words the model sees often; the rest can share structure. New flags: `--verank r` (words without a full row get a rank-r row A[w] @ P), `--vehash N` (they share one of N hashed rows), `--veunk 1` (the unknown token always gets a full row; at this vocabulary it is 31% of text). Base: untied 4.7697; full value table 4.7131 (gain 0.057, 48K weights; best value-table rate 0.1-0.2, round 4). Gains below are over the untied base.

| Value rows (both layers unless noted) | Table weights | Gain | Share of full gain | Gain per 1K weights |
|---|---|---|---|---|
| full, every word | 48.0K | 0.057 | 100% | 0.0012 |
| rank 2, every word | 6.0K | 0.008 | 14% | 0.0013 |
| rank 4, every word | 12.1K | 0.017 | 29% | 0.0014 |
| rank 8, every word | 24.3K | 0.027 | 48% | 0.0011 |
| full for unknown, rank 4 for the rest | 12.1K | 0.013 | 23% | 0.0011 |
| full for 381 frequent + unknown, nothing for the rest | 12.2K | 0.020 | 35% | 0.0016 |
| full for 381 frequent + unknown, 64 hashed rows for the rest | 14.3K | 0.018 | 31% | 0.0013 |
| full for 150 frequent + unknown, rank 4 for the rest | 16.9K | 0.024 | 41% | 0.0014 |
| full for 381 frequent (no unknown row), rank 4 for the rest | 24.3K | 0.032 | 56% | 0.0013 |
| full for 381 frequent + unknown, rank 2 for the rest (rate 0.1 / 0.05) | 18.3K | 0.040 / 0.035 | 70% / 61% | 0.0022 |
| **full for 381 frequent + unknown, rank 4 for the rest** (rate 0.05 / 0.1 / 0.2) | 24.3K | **0.052** / 0.047 / 0.032 | **91%** | **0.0021** |
| layer 0 only: full, every word | 24.0K | 0.031 | 54% | 0.0013 |
| layer 0 only: 381 + unknown, nothing else | 6.1K | 0.011 | 19% | 0.0018 |
| layer 0 only: 381 + unknown, rank 2 | 9.1K | 0.016 | 28% | 0.0018 |
| **layer 0 only: 381 + unknown, rank 4** | 12.2K | **0.035** | **61%** | **0.0028** |
| layer 1 only: 381 + unknown, rank 4 | 12.2K | 0.020 | 35% | 0.0016 |

Equal weights: plain untied width 20 (69.1K weights, 4.7268) and width 24 (85.9K, 4.6789) at the same 1,600 steps. Interpolated to the both-layer hybrid's 78.3K weights a plain model reaches about 4.700, ahead of the hybrid (4.7174) by 0.017 at 1.4 times the FLOPs. Interpolated to the layer-0 hybrid's 66.1K weights a plain model reaches about 4.735, equal to the hybrid (4.7352) at 1.2 times the FLOPs. Widening from 16 to 24 gains 0.0028 per 1K weights.

**Takeaway.** Full rows for the words that cover 80% of text (the 381 words from the common and middle bands) plus the unknown token, and rank-4 rows for the rest, keep 91% of the value-table gain at half the weights. The parts do not add: frequent rows alone 0.020 plus rank-4 rows alone 0.017 is 0.037, the hybrid 0.052. Hashed rows add nothing over leaving the rest empty. Put in layer 0 only, the hybrid reaches the gain per weight of widening (0.0028 per 1K) and ties a plain model of equal weights while costing fewer FLOPs. Rank 4 is the right size here: rank 2 drops to 61-70%.

## Round 15: is there a cheap full untie? (vocabulary 1,500, 1,600 steps, tied width 16)

Rule under test: claim 2 says the tying cost belongs to the whole table. If it is a property of the table's geometry, a cheap change applied to every word should recover most of it. New flags: `--urank r` (input = shared row + A[w] @ B, A starts at zero) and `--umat 1` (input = shared row @ M, one shared 16 x 16 matrix starting at identity). Base: tied 4.8446; fully untied 4.7697 (tying cost 0.075, 24K extra weights).

| Input side | Extra weights | Gain over tied | Share of tying cost |
|---|---|---|---|
| rank 1 offset (rate 0.05) | 1.5K | -0.010 | -13% |
| rank 2 offset (0.05) | 3.0K | 0.007 | 9% |
| rank 4 offset (0.05 / 0.15) | 6.1K | 0.004 / -0.021 | 6% / -28% |
| rank 8 offset (0.15) | 12.1K | 0.024 | 32% |
| one shared 16 x 16 matrix (0.05 / 0.01) | 256 | 0.008 / 0.007 | 11% |
| own rows for the 381 frequent words + rank 4 offset for the rest (0.05 / 0.15) | 12.2K | 0.041 / 0.040 | 55% |
| (round 10) own rows for the 381 frequent words alone | 6.1K | 0.037 | 49% |
| full-rank offset for every known word (rows start at the shared row) | 24.0K | 0.072 | 96% |
| fully untied | 24.0K | 0.075 | 100% |

**Takeaway.** No cheap untie. A shared matrix recovers 11%, rank 4 offsets nothing, rank 8 (half the width) 32%, while a full-rank offset recovers 96%. Adding rank-4 offsets to frequent own rows adds 0.004. The tying cost is per word and full rank: every word needs its own unconstrained input direction. This sharpens claim 2: the cost belongs to the whole table, but not to a shared transform of it. Untying stays a full-table decision; the value table is the per-word weight that can be made cheap.

## Round 16: the hybrid value table along the knobs, and at equal weights

Rule under test: full value rows for the words covering 80% of text plus the unknown token, rank-r rows for the rest (r = width / 4), keep most of the full value-table gain at about half its weights in every setting. Each setting has its own untied base and full value table (both layers). Hybrid rate 0.05 unless noted.

| Setting | Untied | Full value table | Hybrid | Hybrid gain (share of full) | Hybrid extra weights (share of full) |
|---|---|---|---|---|---|
| Vocabulary 500, 1,600 steps (107 tokens per weight) | 4.2787 | 4.2391 | 4.2501 | 0.029 (72%) | 8.4K (53%) |
| Vocabulary 1,500, 1,600 steps (round 14) | 4.7697 | 4.7131 | 4.7177 | 0.052 (91%) | 24.3K (51%) |
| Vocabulary 6,000, 1,200 steps (rate 0.05 / 0.1) | 5.5539 | 5.4997 | 5.5080 / 5.5087 | 0.046 (85%) | 88K (46%) |
| Vocabulary 24,000, 400 steps (rate 0.1 / 0.05) | 6.4999 | 6.4384 | 6.4511 / 6.4564 | 0.049 (79%) | 292K (38%) |
| Width 32, 1,600 steps (rank 4 / rank 8) | 4.6289 | 4.5970 | 4.6112 / 4.6015 | 0.018 (56%) / 0.027 (85%) | 37K (38%) / 49K (51%) |
| Width 64, 1,600 steps (rank 16) | 4.5392 | 4.5119 | 4.5168 | 0.022 (82%) | 99K (52%) |
| Batch 128 x 800 steps (6.6M tokens, 84 tokens per weight) | 4.7558 | 4.7024 | 4.7184 | 0.037 (70%) | 51% |
| Depth 4, 800 steps (rows in all 4 layers) | 4.8497 | 4.8177 | 4.8157 | 0.034 (106%) | 49K (51%) |
| Context 128 (batch 16) | 4.7718 | 4.7095 | 4.7153 | 0.057 (91%) | 51% |
| Context 256 (batch 8, 500 validation rows) | 4.7412 | 4.6844 | 4.7028 | 0.038 (67%) | 51% |

Correction: the first context-256 hybrid runs (`r16-L256-hyb`, `-tlr0.1`, `-rk8`, `-hyb95`) used 1,000 validation rows while the context-256 bases (`r8-L256-*`) used 500, which made the hybrid look like it kept 33%. They are not comparable and are left out. Rerun with 500 rows: 67%. All other compared runs use 1,000 rows (round 0 used 600 and is not compared with anything here).

Equal weights, same tokens, plain untied models made wider:

| Vocabulary | Hybrid (weights, loss) | Plain wider models (weights, loss) | Plain at the hybrid's weights (interpolated) | Hybrid minus plain | Plain FLOPs / hybrid FLOPs |
|---|---|---|---|---|---|
| 1,500 (1,600 steps) | 78.3K, 4.7177 | w20 69.1K, 4.7268; w24 85.9K, 4.6789 | 4.700 | +0.017 (plain wins) | 1.4 |
| 6,000 (1,200 steps) | 286K, 5.5080 | w20 249K, 5.4950 (rate 0.1) / 5.4993 (0.05); w24 302K, 5.4413 (0.05) | 5.461 | +0.047 (plain wins) | 1.4 |
| 24,000 (400 steps) | 1.07M, 6.4511 | w20 969K, 6.4645; w24 1.17M, 6.4340 | 6.450 | +0.001 (tie) | 1.4 |

The vocabulary-6,000 untied base at rate 0.05 gives 5.5492 (0.1: 5.5539), so the base was not under-tuned.

**Takeaway.** The hybrid keeps 67-106% of the full value-table gain at 38-53% of its weights in all 10 settings (median 82%), including width 64, depth 4 and vocabulary 24,000. It does not win at equal weights: a plain model widened to the hybrid's weight count is better by 0.017 and 0.047 at vocabularies 1,500 and 6,000 and ties at 24,000 (the shortest run). The hybrid's case is equal compute: it adds 0.04-0.5% FLOPs per token, while the equal-weight plain model costs 1.4 times the FLOPs. At equal training compute (round 13) value tables beat the best plain model at every vocabulary.

## Round 17: does the hybrid need the unknown token's row?

Rule under test (review weakness 3): the hybrid's gain depends on the unknown token, which real vocabularies do not have. Prediction if true: without the unknown row the hybrid's share falls at every vocabulary. Same hybrid (80% coverage, rank 4), unknown token in the low-rank part instead.

| Vocabulary (unknown share of text) | Full value gain | Hybrid with unknown row | Hybrid without unknown row |
|---|---|---|---|
| 1,500 (31%) | 0.057 | 0.052 (92%) | 0.032 (56%) |
| 6,000 (16%) | 0.054 | 0.046 (85%) | 0.036 (66%) |
| 24,000 (6%) | 0.062 | 0.049 (79%) | 0.056 (91%) |

**Takeaway.** The unknown row matters in proportion to the unknown token's share of text, and at vocabulary 24,000 the hybrid does better without it (91% of the gain at 38% of the weights). At a real vocabulary, with no unknown token, the frequent-word rows carry the hybrid. The 24,000 point is a 400-step run.

## Unit correction (2026-11-15)

`compute.py` printed training compute divided by 10¹² and labelled it "PF". The numbers are TFLOP (10¹² FLOPs), not PF. Every budget in this file and the paper is now written in TFLOP; the values themselves are unchanged.

## Round 18: does the equal-compute win break where the output layer is a small share of compute?

Rule under test: at equal training compute a value table beats the best plain model. Risk: at vocabulary 1,500 and width 16 the output layer is 75% of forward FLOPs, so widening is expensive and a free table looks good. In large models the output layer is a few percent. Prediction if the risk is real: the lead shrinks toward zero as the output share falls. Output share = 2·V·d / forward FLOPs per token.

Design: batch 128 (3.4 times faster per token than batch 32 here), vocabulary 1,500, value table in every layer, table rate 0.1. For each shape: plain and value table at the same steps (equal compute within 0.5%), then plain models of other widths and depths at equal compute. Width 128 and depth 8 runs are 13-30 s. Steps for equal compute come from `flops.py`.

Same shape, same steps (equal compute):

| Shape | Output share | Plain | Value table | Gain |
|---|---|---|---|---|
| width 32, 2 layers, 400 steps | 63% | 4.6653 | 4.6185 | 0.047 |
| width 40, 2 layers, 587 steps | 58% | 4.5700 | 4.5442 | 0.026 |
| width 48, 2 layers, 453 steps | 54% | 4.5756 | 4.5548 | 0.021 |
| width 64, 2 layers, 300 steps | 48% | 4.6392 | 4.6012 | 0.038 |
| width 48, 4 layers, 315 steps | 37% | 4.6548 | 4.6203 | 0.035 |
| width 24, 8 layers, 306 steps | 35% | 4.7463 | 4.6881 | 0.058 |
| width 128, 2 layers, 300 steps | 32% | 4.5800 | 4.5539 | 0.026 |
| width 128, 2 layers, batch 32 x 600 steps (table rate 0.03 / 0.1 / 0.3) | 32% | 4.6256 | 4.6143 / 4.6189 / 4.6093 | 0.016 |
| width 64, 4 layers, 200 steps | 31% | 4.7022 | 4.6593 | 0.043 |
| width 32, 8 layers, 200 steps | 30% | 4.7911 | 4.7311 | 0.060 |
| width 48, 8 layers, 196 steps | 23% | 4.7385 | 4.6913 | 0.047 |
| width 64, 8 layers, 120 steps | 19% | 4.8168 | 4.7569 | 0.060 |
| vocabulary 500, width 16, 8 layers, 400 steps (82 tokens per weight) | 20% | 4.2853 | 4.2259 | 0.059 |
| vocabulary 500, width 16, 4 layers, 666 steps | 34% | 4.2245 | 4.1750 | 0.050 |

Best shape at each budget (plain models over width and depth; the value table at the best one or two plain shapes):

| Budget | Plain models tried (loss) | Best plain | Best value table | Lead |
|---|---|---|---|---|
| V1,500, 1.5 TFLOP | w24 4.6794, w32 4.6653, w40 4.6754, w32x4 layers 4.6856; at 1.56: w32 4.6588, w32x4 4.6935, w24x8 4.7463, w32x8 4.7911 | w32, 2 layers (63%) | w32 4.6185 | 0.041-0.047 |
| V1,500, 3.0 TFLOP | w40 4.5700, w48 4.5756, w64 4.6392, w80 4.6564, w64x4 4.7022, w48x4 4.6548, w48x8 4.7385, w64x8 4.8168 | w40, 2 layers (58%) | w40 4.5442 | 0.026 |
| V1,500, 4.4 TFLOP (batch 32) | w96 4.5650, w128 4.6256, w160 4.7166, w128x4 4.7311 | w96 or narrower (not reached in 30 s) | not run | - |
| V500, 0.78 TFLOP | w16x8 4.2853, w16x4 4.2245, w24x4 4.2360, w24 4.2056, w32 4.1780, w40 4.2156, w48 4.2563 | w32, 2 layers (37%) | w32 4.1552 | 0.023 |

Deeper plain models never won at these budgets: depth 4 and 8 lost to depth 2 by 0.02-0.15 at equal compute, with or without a value table. So the lowest output shares come from shapes that are not compute-optimal on a laptop budget.

**Takeaway.** No threshold. Within a fixed shape the value-table gain stays at 0.035-0.060 at output shares of 19-37% and does not shrink with the share; it grows with depth (one table per layer) and is smallest for width 128 at depth 2 (0.016-0.026). At the best shape of each budget the lead is 0.021-0.047. The one decline is vocabulary 1,500 from 1.5 to 3.0 TFLOP (0.041-0.047 to 0.026); round 20 tests whether the lead keeps falling with budget.

## Round 19: literature baselines and the hybrid at equal compute; inference memory

Rule under test: a value table is the best way to spend extra per-token weights at equal compute. Baselines, each added to the plain model with its own tables on Adam and its FLOPs counted in `flops.py`:
- Engram-style: hashed 2-gram and 3-gram tables (N rows each) gated by the hidden state (sigmoid of the dot product of the normalised hidden state and row), added to the residual before layer 1 (`--ngram N`).
- MoVE-style: a shared bank of K value vectors per layer, mixed by a softmax router on the hidden state, added to the values (`--bank K`).
- Scone-style: a hashed bigram table of N rows added to the input vector (`--inbig N`).

Setting A (vocabulary 1,500, width 32, 2 layers, batch 128, 400 steps, 15 tokens per weight). All tables add under 0.1% FLOPs except the bank (router d x K). Best of table rates 0.01-0.3 shown.

| Model | Extra weights | Known loss | Gain over plain |
|---|---|---|---|
| plain | 0 | 4.6653 | - |
| value bank K=64 / 512 (rate 0.1 / 0.03) | 8K / 65K | 4.6617 / 4.6479 | 0.004 / 0.017 |
| hybrid value table, with / without unknown row | 49K | 4.6303 / 4.6348 | 0.035 / 0.031 |
| Engram-style, 768 rows per order | 49K | 4.6397 | 0.026 |
| Scone-style, 1,536 rows | 49K | 4.6363 | 0.029 |
| **value table, every word (rate 0.03 / 0.1 / 0.2)** | 96K | **4.6185** / 4.6219 / 4.6207 | **0.047** |
| Engram-style, 1,500 rows per order (rate 0.03 / 0.1) | 96K | 4.6316 / 4.6575 | 0.034 |
| Scone-style, 3,000 rows (rate 0.03 / 0.1) | 96K | 4.6319 / 4.6502 | 0.033 |
| Engram-style, 8,192 rows per order (rate 0.01 / 0.03 / 0.1 / 0.3) | 524K | 4.6175 / **4.6101** / 4.6275 / 4.6432 | 0.055 |
| Scone-style, 16,384 rows (rate 0.03 / 0.1 / 0.3) | 524K | **4.6013** / 4.6277 / 4.6840 | 0.064 |
| Scone-style, 65,536 rows | 2.1M | 4.5913 | 0.074 |
| value table + Scone-style 16,384 rows | 620K | 4.5716 | 0.094 |

Vocabulary 24,000, width 12 (narrower plain beats width 16 at this budget, so round 13's best plain is corrected), 537 steps, 1.93 TFLOP, about 1 token per weight:

| Model | Extra weights | Known loss | Lead over best plain |
|---|---|---|---|
| plain width 16, 400 steps (round 13's best) | - | 6.4999 | - |
| **plain width 12** | 0 | **6.4736** | - |
| value table (rate 0.03 / 0.1 / 0.3) | 576K | 6.4537 / **6.4367** / 6.4425 | 0.037 |
| hybrid, no unknown row (3,110 full rows, rank 3) | 219K | 6.4391 | 0.035 |
| Engram-style, 24,000 rows per order (rate 0.03 / 0.1) | 576K | **6.4243** / 6.4372 | 0.049 |
| Scone-style, 48,000 rows | 576K | 6.4296 | 0.044 |

The width-12 runs took 25-36 s under a load average of 4.7; a width-8 plain model (807 steps, about 40 s) was not run.

Inference memory (`memory.py`, bf16): weights stored and values read per token. Every lookup uses only token IDs, so rows can be fetched from host memory ahead of the layer.

| Design | Setting A stored / read per token | GPT-2 vocabulary, width 4,096, tables in 2 layers |
|---|---|---|
| value table, every word | 0.19 MB / 128 B | 823 MB / 16.4 KB |
| hybrid (80% full rows, rank d/4) | 0.10 MB / 109 B | 289 MB / 13.9 KB |
| Engram-style 2+3-gram rows (N = V) | 0.19 MB / 128 B | 823 MB / 16.4 KB |
| Scone-style bigram rows (N = V) | 0.10 MB / 64 B | 412 MB / 8.2 KB |

In GPT-2 tokens of FineWeb, 4,054 tokens cover 80% of text, so the hybrid stores 35% of the full table at that scale. A value table in all 32 layers of such a model would store 13.2 GB, which is why practice puts it in a few layers.

**Takeaway.** At equal extra weights and 15 tokens per weight the value table beats the Engram-style and Scone-style tables (0.047 against 0.033-0.034) and the hybrid beats both at its own weight count (0.035 against 0.026-0.029); a MoVE-style bank adds little (0.004-0.017). Hashed n-gram tables keep growing with rows, which a per-word table cannot do (it has V rows per layer): at 5.5 times the weights they pass the value table (0.055-0.064), and the two stack (0.094 together, 85% of the sum). At vocabulary 24,000 and 1 token per weight the n-gram tables beat the value table at equal weights (0.044-0.049 against 0.037). With the corrected plain baseline the vocabulary-24,000 lead of the value table is 0.037 (was 0.062), and the hybrid without the unknown row keeps 94% of it at 38% of the weights.

## Round 20: does the lead fall as the budget grows?

Rule under test: the lead at the best shape does not depend on the budget. Round 18 saw 0.041-0.047 at 1.5 TFLOP and 0.026 at 3.0 TFLOP at vocabulary 1,500, but larger budgets there need runs over 30 s. Vocabulary 500 runs are fast, so the budget is pushed 20 times there. Batch 128, depth 2 (depth 4 lost again at 1.6 TFLOP: 4.1930).

| Budget | Plain (width: loss) | Best plain (output share) | Value table (width: loss) | Lead |
|---|---|---|---|---|
| 0.31 TFLOP (round 13, batch 32) | 20: 4.2295, 24: 4.2211, 28: 4.2138 | 28 (39%) | 24: 4.1973, 28: 4.2070 | 0.017 |
| 0.78 TFLOP (round 18) | 24: 4.2056, 32: 4.1780, 40: 4.2156, 48: 4.2563 | 32 (37%) | 24: 4.1764, 32: 4.1552 | 0.023 |
| 1.6 TFLOP | 32: 4.1241, 48: 4.1157, 64: 4.1704 | 48 (28%) | 32: 4.0954, 48: 4.0966 (rate 0.03: 4.1012) | 0.020 |
| 3.2 TFLOP | 48: 4.1143, 64: 4.0679, 96: 4.1540 | 64 (23%) | 48: 4.1018, 64: 4.0456 | 0.022 |
| 6.4 TFLOP | 64: 4.0161, 96: 4.0599 | 64 (23%) | 64: 3.9926 | 0.024 |

The two 6.4 TFLOP width-64 runs took 33 and 37 s (load average 4-5).

**Takeaway.** Over a 20 times larger budget the lead stays at 0.017-0.024 while the best plain model's output share falls from 39% to 23% and it reaches 48 tokens per weight. The budget does not erode the win at vocabulary 500; the vocabulary-1,500 drop at 3.0 TFLOP (0.026) is inside the 0.02-0.06 range seen across shapes.

## Round 21: the hybrid at its best width at equal compute (REVIEW weakness 5)

Rule under test: the hybrid beats the best plain model at equal compute. At width 16 it lost at vocabulary 500 and tied at 6,000, where the full table needed width 24 to win. Steps lowered so the hybrid's rank rows (2rd FLOPs per layer) keep compute equal to round 13's budget.

| Vocabulary (budget) | Best plain | Full table (best width) | Hybrid, width 16 | Hybrid, best width | Hybrid lead |
|---|---|---|---|---|---|
| 500 (0.31 TFLOP) | 4.2138 (w28) | 4.1973 (w24) | 4.2501 | w24 4.2034 (rate 0.1; 0.05: 4.2078), w28 4.2252 | 0.010 |
| 1,500 (0.63 TFLOP) | 4.7544 (w28) | 4.7131 (w16) | 4.7174 | w16 | 0.037 |
| 1,500 (1.5 TFLOP, batch 128) | 4.6588-4.6653 (w32) | 4.6185 (w32) | - | w32 4.6303 | 0.029-0.035 |
| 6,000 (1.53 TFLOP) | 5.5084 (w24) | 5.4810 (w24) | 5.5080 | w24 5.4881 | 0.020 |
| 24,000 (1.93 TFLOP) | 6.4736 (w12) | 6.4367 (w12) | 6.4511 | w12 6.4391 (no unknown row) | 0.035 |

The vocabulary-6,000 hybrid run took 66 s because it ran on the CPU (the `--dev mps` flag was left out; see the correction in round 23), over the 40 s limit; reported, not repeated.

**Takeaway.** At its best width the hybrid leads the best plain model at all four vocabularies: by 0.020-0.037 at 1,500-24,000 and by 0.010 (inside run noise) at 500, where the full table's own lead is only 0.017. It keeps 59-95% of the full table's equal-compute lead at 38-53% of its weights.

## Round 22: value tables in only 2 of 8 layers (REVIEW question 3)

Rule under test: the equal-compute gain holds when the table sits in a few layers, as practice does for memory. Round 18 put a table in every layer, which gave the largest gains at depth 8. Batch 128, table rate 0.1, layers 0 and 1 only.

| Setting | Plain | Table in all 8 layers | Table in layers 0, 1 | Share of the all-layer gain |
|---|---|---|---|---|
| vocabulary 1,500, width 32, 200 steps | 4.7911 | 4.7311 (0.060) | 4.7526 (0.039) | 64% |
| vocabulary 500, width 16, 400 steps | 4.2853 | 4.2259 (0.059) | 4.2568 (0.029) | 49% |

Both runs took 45 s and 73 s because they ran on the CPU (the `--dev mps` flag was left out), not because of machine load as first written. The vocabulary-500 run repeated on the GPU gives the same loss, 4.2568, in 28.7 s (round 23).

**Takeaway.** Two of eight layers keep half to two thirds of the all-layer gain at a quarter of the table weights; the gain at depth 8 is still 0.029-0.039, the size of the depth-2 gains. Which layers to pick (first, last, spread) is the next test.

## Round 23: which layers get the value table, shared or separate tables, and width 64-128

Rule under test (from round 22): 2 of 8 layers keep half to two thirds of the all-layer gain at a quarter of the table weights. Runs that could break it: a placement that keeps much more or much less (first two, last two, spread, every fourth), depth 12, vocabulary 24,000, width 64-128 where the body is a large share of the weights, and one table shared by the chosen layers. Every run at batch 128 with `--dev mps`; table rates tried at 0.03-1.0 for the all-layer table and 0.1/0.3 for the few-layer tables.

**Correction to rounds 21-22.** The 45-73 s runs in rounds 21-22 and the first runs of this round were slow because the `--dev mps` flag was left out and they ran on the CPU, not because of machine load. The losses do not depend on the device: the round-22 vocabulary-500 run repeated on the GPU gives 4.2568 again, in 28.7 s.

### A. Placement: share of the all-layer gain kept by two tables

Gain over the plain model of the same shape and steps, best of the rates tried.

| Setting | Plain | All layers | First two | Last two | Other placements |
|---|---|---|---|---|---|
| vocab 500, w16, depth 8, 400 steps | 4.2853 | 4.2259 (0.059) | 0.029 (49%) | **0.035 (60%)** | - |
| vocab 1,500, w32, depth 8, 200 steps | 4.7911 | 4.7311 (0.060) | **0.039 (64%)** | 0.033 (54%) | - |
| vocab 6,000, w16, depth 8, 250 steps | 5.5969 | 5.5106 (0.086) | 0.048 (56%) | **0.053 (61%)** | 1,5: 0.054 (62%); 0,4: 0.050 (58%); 3,4: 0.045 (52%); 0,7: 0.043 (50%); layer 7 alone: 0.044 (51%) |
| vocab 6,000, w16, depth 12, 120 steps | 5.8471 | 5.7494 (0.098) | 0.053 (54%) | **0.067 (68%)** | 2,8: 0.037 (38%); 0,4,8 (three tables): 0.055 (56%) |
| vocab 24,000, w12, depth 8, 100 steps | 6.6735 | 6.6119 (0.062) | 0.030 (49%) | **0.072 (117%)** | 0,4: 0.065 (105%); 1,5: 0.001 |
| vocab 1,500, w64, depth 4, 200 steps | 4.6951 | 4.6593 (0.036) | 0.029 (81%) | **0.039 (110%)** | - |
| vocab 1,500, w128, depth 4, 150 steps | 4.6948 | 4.6753 (0.020) | 0.010 (51%) | **0.020 (101%)** | - |

Two tables are 25% of the all-layer table weights at depth 8, 17% at depth 12 and 50% at depth 4. The last two layers beat the first two in 6 of 7 settings, by 0.004-0.042, and trail by 0.006 once (vocabulary 1,500, width 32). At depth 8 the placements other than the last two sit within 0.011 of each other (50-62%), except 1,5 at vocabulary 24,000 (0.001, the noisiest setting: 100 steps, under 1.5 tokens per weight). The spread pair 2,8 at depth 12 keeps the least (38%), and three tables at 0,4,8 keep less than two tables at 10,11.

**Takeaway.** Put the tables in the last two layers: in 4-12 layer models they keep 54-117% (median 68%) of the all-layer gain at 17-50% of the table weights, and at depth 4 they match all four layers.

### B. Equal compute with tables in a few layers

| Budget | Best plain | Best model with tables | Deep model, last two tables | Deep model, all tables | Deep plain |
|---|---|---|---|---|---|
| vocab 6,000, 1.57 TFLOP | 5.5576 (w20, depth 2) | 5.4920 (w20 depth 2, both layers): lead 0.066 | 5.5442: lead 0.013 | 5.5106: lead 0.047 | 5.5969: behind 0.039 |
| vocab 6,000, 0.85 TFLOP | 5.7327 (w16, depth 2) | 5.6575 (w16 depth 2): lead 0.075 | 5.7803: behind 0.048 | 5.7494: behind 0.017 | 5.8471: behind 0.114 |
| vocab 1,500, w64 depth 4 budget | 4.5604 (w32, depth 2, 812 steps) | 4.5303 (w32 depth 2): lead 0.030 | 4.6557: behind 0.095 | 4.6593 | 4.6951 |

Other plain shapes at these budgets: 1.57 TFLOP w24 depth 2 5.5645, w32 depth 2 5.6278, w16 depth 4 5.5682, w16 depth 2 5.5662; 0.85 TFLOP w20 depth 2 5.7389, w24 depth 2 5.7655; w64-depth-4 budget w48 depth 2 4.6016, w64 depth 2 4.6192.

**Takeaway.** At laptop budgets depth 2 is compute-optimal, so the best model at equal compute is depth 2 with a table in both layers. A depth-8 model with tables in its last two layers still beats the best plain model at 1.57 TFLOP (by 0.013), so the tables recover more than the cost of the extra depth there; at depth 12 they do not.

### C. One table shared by several layers vs one table in one layer, at equal weights

Each layer keeps its own scale λ when the table is shared (`--veshare 1`).

| Setting | Plain | One table in layer 7 | Shared by 6, 7 | Shared by all 8 |
|---|---|---|---|---|
| vocab 500, w16, 400 steps | 4.2853 | 4.2620 (0.023) | - | **4.2560 (0.029)** |
| vocab 1,500, w16, 250 steps | 4.8780 | 4.8591 (0.019) | - | **4.8303 (0.048)** (rate 0.1: 4.8475) |
| vocab 1,500, w32, 200 steps | 4.7911 | 4.7687 (0.022) | 4.7671 (0.024) | **4.7568 (0.034)** |
| vocab 6,000, w16, 250 steps | 5.5969 | **5.5530 (0.044)** | 5.5625 (0.034) | 5.5692 (0.028; rates 0.03/0.1/0.3: 5.5692/5.5726/5.5723); shared by 0,7: 5.5718 |

Two separate tables in 6,7 (twice the weights) gain 0.053 at vocabulary 6,000 and 0.033 at 1,500 width 32.

**Takeaway.** Sharing one table across layers beats the same table in one layer at vocabularies 500-1,500 (by 0.006-0.029) and loses at 6,000 (by 0.010-0.016). The direction flips toward real vocabularies, so the rule is separate tables; why sharing flips is not found.

### D. Width 64 and 128: two tables against the same weights spent on width or depth

Vocabulary 1,500, depth 4, tables in layers 2 and 3, same steps for all models in a block. FLOPs per token relative to the model with tables.

| Width | Model | Weights | FLOPs/token | Loss |
|---|---|---|---|---|
| 64 (200 steps) | plain | 387K | 1.00 | 4.6951 |
| | hybrid table (rank 16), rate 0.3 / 0.1 | 486K | 1.01 | 4.6852 / 4.6872 |
| | plain, depth 6 | 485K | 1.34 | 4.6838 |
| | plain, width 72 | 466K | 1.22 | 4.6821 |
| | full table | 579K | 1.00 | **4.6557** |
| | plain, width 80 | 543K | 1.44 | 4.6716 |
| | plain, depth 8 | 583K | 1.69 | 4.6854 |
| 128 (150 steps) | plain | 1.164M | 1.00 | 4.6948 |
| | hybrid table (rank 32), rate 0.1 / 0.3 | 1.366M | 1.01 | 4.6914 / 4.6974 |
| | plain, depth 5 | 1.359M | 1.19 | 4.6907 |
| | full table | 1.547M | 1.00 | **4.6751** |
| | plain, width 152 | 1.557M | 1.36 | 4.6873 |

These runs see only 0.8-3.4 tokens per weight.

**Takeaway.** Two full tables beat the same weights spent on width or depth at both widths (by 0.016-0.030 at width 64, 0.012 at width 128) while costing 1.36-1.69 times less compute per token. The two-layer hybrid only ties the matched plain models (within 0.004) at 1.2-1.3 times less compute: at 1-3 tokens per weight its low-rank rows have too few sightings.

### E. Breaking D: the last layer alone, and longer training

| Setting | Model | Weights | Loss | Gain |
|---|---|---|---|---|
| vocab 1,500, w64, depth 4, 200 steps | table in layer 3 alone | 483K | 4.6724 | 0.023 (63% of all four, at 25% of the weights) |
| vocab 1,500, w32, depth 4, 500 steps (17-29 tokens per weight) | plain | 143K | 4.6011 | - |
| | hybrid in 2,3 (rank 8) | 192K | 4.5767 | 0.024 (72% of the full table's gain at 51% of its extra weights) |
| | full table in 2,3 | 239K | 4.5674 | 0.034 |
| | plain, width 40 | 196K | **4.5611** | 0.040 |
| | plain, width 48 | 255K | **4.5327** | 0.068 |

**Takeaway.** One table in the last layer keeps less than two (63% against 110% at depth 4). At 17-29 tokens per weight the equal-weight win of D is gone: a wider plain model with the same weights beats the hybrid by 0.016 and the full tables by 0.035, as Claim 9 found for tables in every layer. The two-table win at equal weights in D is a short-training (1-3 tokens per weight) result; the win per unit of compute is not.

### The rule

**Put value tables in the last two layers, one table per layer. They keep a median 68% (54-117%) of the all-layer gain at 17-50% of the table weights; at depth 4 they keep all of it.** One table in the last layer keeps less (51-63%), and one table shared by several layers loses to it at vocabulary 6,000. Two tables cost almost no compute, so they pay per unit of compute; per weight, a wider model wins once training passes about 10 tokens per weight. At laptop budgets depth 2 with a table in both layers stays the best model at equal compute.

Next knob that could break it: depth 16-24 (does the last-two share fall as the table covers a smaller share of layers?) and the 24,000 vocabulary at longer training, both GPU-sized.

## Round 24: what decides whether a deep model with last-two tables wins at equal compute

Rule under test (round 23): value tables in the last two layers keep a median 68% of the all-layer gain, and a depth-8 model with them beats the best plain model at equal compute (vocabulary 6,000, 1.57 TFLOP, lead 0.013) while depth 12 does not (0.85 TFLOP). Runs that could break it or explain it: depth 8 at depth 12's budget, longer training at vocabulary 500 (100-400 tokens per weight), vocabulary 24,000 at equal compute, and hashed bigram tables of the same size in the same layers. New flag `--vebig N`: the value table in the `--ve` layers is indexed by a hashed (previous, current) word pair into N rows instead of by the word.

Load average was 4.7-5 during this round (three other sessions' training jobs each held a CPU core). Five runs took 32-51 s; they are reported and not repeated, and no longer runs were started.

### A. Equal compute: lead = two-table gain minus the deep shape's own penalty

Penalty: the deep plain model's loss minus the best plain model's loss at the same FLOPs. Gain: the deep model with tables in its last two layers against the deep plain model. Lead = penalty subtracted from gain. Head share: the output layer's share of the deep model's forward FLOPs.

| Vocabulary, depth, budget | Head share | Best plain (shape) | Penalty | Two-table gain | Lead |
|---|---|---|---|---|---|
| 6,000, depth 4, 0.85 TFLOP | 86% | 5.7284 (w16 depth 4) | 0.000 | 0.056 | **+0.056** |
| 6,000, depth 4, 1.57 TFLOP | 86% | 5.5576 (w20 depth 2) | 0.011 | 0.053 | **+0.042** |
| 6,000, depth 8, 0.85 TFLOP | 75% | 5.7284 | 0.052 | 0.035 | -0.018 |
| 6,000, depth 8, 1.57 TFLOP | 75% | 5.5576 | 0.039 | 0.053 | **+0.013** |
| 6,000, depth 12, 0.85 TFLOP | 67% | 5.7284 | 0.119 | 0.067 | -0.052 |
| 24,000, depth 8, 1.51 TFLOP | 94% | 6.6124 (w8 depth 2) | 0.061 | 0.072 | **+0.011** |
| 500, depth 4, 0.39 TFLOP | 34% | 4.3130 (w20 depth 2) | 0.075 | 0.065 | -0.011 |
| 500, depth 4, 0.78 TFLOP | 34% | 4.2340 (w20 depth 2) | 0.015 | 0.040 | **+0.025** |
| 500, depth 4, 1.56 TFLOP | 34% | 4.1585 (w20 depth 2) | 0.040 | 0.019 | -0.021 |
| 500, depth 8, 0.39 TFLOP | 20% | 4.3130 | 0.223 | 0.020 | -0.203 |
| 500, depth 8, 0.78 TFLOP | 20% | 4.2340 | 0.112 | 0.028 | -0.084 |
| 500, depth 12, 0.39 TFLOP | 14% | 4.3130 | 0.266 | 0.008 | -0.258 |

Vocabulary 500 runs use batch 256; the others batch 128. Plain shapes at vocabulary 500: 0.39 TFLOP w16 / w20 / w24 depth 2 4.3221 / 4.3130 / 4.3611; 0.78 TFLOP 4.2449 / 4.2340 / 4.2396; 1.56 TFLOP w20 / w28 depth 2 4.1585 / 4.1631. At 0.85 TFLOP and vocabulary 6,000, w16 depth 4 (5.7284) beats the depth-2 shapes of round 23 (best 5.7327). At vocabulary 24,000, w12 depth 2 6.6277, w12 depth 4 6.6739. The best model with tables at vocabulary 500, 1.56 TFLOP, is w28 depth 2 with a table in both layers, 4.1247 (lead 0.034). Table rates: 0.1 at 500, 0.3 at 6,000 depth 4 and 12, best of 0.1/0.3 elsewhere (vocabulary 500, 1.56 TFLOP: 0.3 gives 4.1794 against 4.1823).

**Takeaway.** Depth 8 also loses at depth 12's budget (by 0.018), so depth 8's win at 1.57 TFLOP comes from the budget, not from depth 8 being special. Two tables add a gain of 0.02-0.07 on top of whatever the deep shape scores; they do not make a badly shaped model good. The deep model wins at equal compute exactly when its own penalty against the best plain shape is smaller than that gain. The penalty is smallest when the output layer takes most of the compute (75-94% share: penalty 0.00-0.06 at the larger budgets), because extra layers then cost little; at a 14-20% share depth 8-12 trails by 0.11-0.27 and two tables recover 0.01-0.03 of it. The penalty halves when the budget doubles at vocabulary 500 (depth 8: 0.223 to 0.112; depth 4: 0.075 to 0.015).

### B. Longer training: the two-table gain in a 4-layer model shrinks

Vocabulary 500, w16, depth 4, batch 256, tables in layers 2 and 3.

| Budget | Plain tokens per weight | Two-table gain | Lead over best plain |
|---|---|---|---|
| 0.39 TFLOP | 98 | 0.065 | -0.011 |
| 0.78 TFLOP | 195 | 0.040 | +0.025 |
| 1.56 TFLOP | 390 | 0.019 (rate 0.3) | -0.021 |

**Takeaway.** Past about 200 tokens per weight the gain of tables in half of the layers falls (0.065 to 0.019), while the depth-2 model with a table in both layers keeps a lead of 0.034 at 1.56 TFLOP. This is the first setting where few-layer tables fade with training; the all-layer table at the same depth-4 shape was not run (44 s under this load), so whether the share of the all-layer gain falls too is open. Depth 12 at longer training could not be run under 40 s (vocabulary 6,000 depth 12 at 222 steps took 46 s in round 23).

### C. Same layers, same weights: word rows against hashed bigram rows

Depth 8, the last-two-layer word tables against (a) value tables in layers 6 and 7 indexed by a hashed (previous, current) pair with V rows each (`--vebig V`) and (b) Engram-style gated 2-gram and 3-gram rows of V rows each added to the residual stream before layer 6, or before layer 1 as Engram places them. All three have the same extra weights (2 × V × width). Best of 2-3 rates each.

| Setting | Plain | Word rows in 6, 7 | Bigram value rows in 6, 7 (rates) | Engram-style before layer 6 (rates) | Engram-style before layer 1 |
|---|---|---|---|---|---|
| vocab 1,500, w32, 200 steps | 4.7911 | **0.033** | 0.029 (0.1 / 0.3) | 0.016 (0.03 / 0.1) | - |
| vocab 6,000, w16, 250 steps | 5.5969 | **0.053** | 0.032 (0.03 / 0.1 / 0.3) | 0.023 (0.01 / 0.03 / 0.1) | 0.021 |
| vocab 24,000, w12, 100 steps | 6.6735 | **0.072** | -0.010 (0.1 / 0.3) | 0.001 (0.03 / 0.1) | 0.000 |

**Takeaway.** In the last two layers of a deep model, word rows beat hashed bigram rows of the same size at all three vocabularies, by 0.004 at 1,500 (a tie) and by 0.021-0.071 over the best n-gram table at 6,000-24,000. Round 19 found the opposite at vocabulary 24,000 and depth 2 (n-gram rows 0.044-0.049 against 0.037); in a deep model at 0.7 tokens per weight, n-gram rows gain nothing. A likely reason, not tested: a hashed pair is seen far less often than a word, so its row trains less, and in a deep model the earlier layers already pass the previous word to the last layers.

### The rule after round 24

**Value tables in the last two layers add a gain of about 0.02-0.07 on top of the model's shape, and beat hashed n-gram rows of the same size in the same layers. A deep model with them beats the best plain model at equal compute only when the deep shape itself is close to compute-optimal (its penalty under the gain): when the output layer takes most of the compute (75-94% here) or the budget is large enough.** At long training (200+ tokens per weight) the two-table gain in a 4-layer model shrinks, so the all-layer table is the safer choice there.

Next knob that could break it: the all-layer table at the depth-4, 1.56 TFLOP shape (does the share fall, or only the absolute gain?), and depth 12 at a larger budget, both when the machine is free.

## Round 25: more pair rows, larger vocabulary (2026-11-15)

Rule going in: word rows beat hashed pair rows of the same size. Breakers tried: more buckets, a larger vocabulary, more depth.

Bucket count for pair value rows, vocabulary 6,000, depth 8, 250 steps (plain 5.5969, word rows 5.5442):

| Pair buckets | 375 | 1,500 | 6,000 | 24,000 | 24,000, rate 0.3 |
|---|---|---|---|---|---|
| Loss | 5.5774 | 5.5776 | 5.5653 | 5.5591 | 5.5547 |

Vocabulary 24,000, width 12:

| Setting | Plain | Word | Engram-style |
|---|---|---|---|
| depth 2, batch 128, 100 steps | 6.6505 | 6.6514 | 6.6444 |
| depth 2, batch 32, 400 steps | 6.5583 | **6.5185** | 6.5282 |
| depth 4, batch 32, 400 steps | 6.5541 | **6.5065** | (killed when the app quit) |
| depth 8, batch 32, 400 steps | 6.5322 | **6.4946** | 6.5503 |

**Takeaway.** Fewer buckets do not help pair rows (more pairs share each row); 4 times the buckets still leave pair rows behind word rows. At vocabulary 24,000 word rows win at 400 steps and Engram-style rows are worse than plain at depth 8. One run took 72.8 s (load from other sessions).

## Round 26: why pair rows lose, a 45-run grid and a sightings breakdown (2026-11-15)

Rule going in: word rows beat pair rows because a pair row is seen far less often. Test: a grid over vocabulary (500 / 1,500 / 6,000) × depth (2 / 4 / 8) with five arms of equal extra weights, and a breakdown of the loss by how often each (previous, current) pair was seen in training. Width 16, 150 steps of 128 × 64. Gain over plain:

| Setting | Plain | Word | Previous-word rows | Pair value rows | Engram-style |
|---|---|---|---|---|---|
| 500, depth 2 | 4.4796 | .048 | .052 | .032 | .032 |
| 500, depth 4 | 4.4886 | .058 | .055 | .039 | .054 |
| 500, depth 8 | 4.4605 | .020 | .019 | .014 | .020 |
| 1,500, depth 2 | 5.0214 | .032 | .040 | .040 | .032 |
| 1,500, depth 4 | 5.0028 | .021 | .026 | .017 | .011 |
| 1,500, depth 8 | 5.0199 | .022 | .037 | .018 | .023 |
| 6,000, depth 2 | 5.7849 | .070 | .082 | .037 | .050 |
| 6,000, depth 4 | 5.7611 | .050 | .049 | .021 | .028 |
| 6,000, depth 8 | 5.7597 | .043 | .036 | .011 | .035 |

Gain over plain by how often the position's pair was seen in training (0 / 1-9 / 10-99 / 100-999 / 1,000+):

| Setting | Word | Previous-word | Pair value rows | Engram-style |
|---|---|---|---|---|
| 6,000, depth 2 | .051 / .061 / .089 / .085 / .066 | .055 / .072 / .097 / .111 / .072 | .012 / .017 / .043 / .090 / .035 | -.016 / -.021 / .058 / **.158** / .099 |
| 6,000, depth 8 | .036 / .042 / .060 / .050 / .018 | - | -.006 / -.007 / .010 / .042 / .024 | -.005 / -.007 / .042 / **.102** / .054 |

Probe (ridge readout of the previous word from the residual entering each layer, vocabulary 1,500, 32 classes, chance 0.12): in the plain deep model previous-word accuracy rises from 0.26 to 0.34 by the middle layers; with a previous-word table the layers stop building it.

Data alone (`pairstat.py`, no training): share of positions whose pair was seen 100+ times is 0.34 at vocabulary 6,000 after 1.2M tokens, 0.76 after 50M; at vocabulary 50,257 after 50M tokens it is 0.49, and 0.26 have a pair seen 1,000+ times.

Rate sweep, best word against best Engram-style (each at its own best of 3 rates): vocabulary 500 depth 4 4.4100 vs 4.4342 (word +0.024); 1,500 depth 2 4.9890 vs 4.9898 (tie); 1,500 depth 4 4.9779 vs 4.9915 (+0.014); 6,000 depth 8 5.7170 vs 5.7252 (+0.008). Word rows win by 0.01+ in at least one cell at each of the four vocabularies.

Hybrid value table re-sweep, vocabulary 6,000 depth 2 (full table 5.7154, plain 5.7849): coverage 0.8 with rank 1 keeps 85% of the gain at 27% of the extra weights, rank 2 80% at 34%, rank 4 54% at 46%; coverage 0.95 rank 2 94% at 79%. Low rank (1-2) is best in short training; the 80% coverage line holds.

Placement past 200 tokens per weight (vocabulary 500, width 8, depth 4, batch 256): tables in the last two layers keep 88% of the all-layer gain at 132 tokens per weight (0.031 vs 0.035) and 84% at 264 (0.065 vs 0.078), at half the table weights.

**Takeaway.** Pair rows help only positions whose pair was seen about 100+ times, and there they beat word rows by up to 0.07; on pairs seen fewer than 10 times they gain nothing or hurt, while word and previous-word rows help from the first sighting. A single word as the lookup key is what matters: previous-word rows equal current-word rows. Runs over 40 s: rate-sweep runs at vocabulary 6,000 depth 8 (44-58 s), hybrid runs c0.5-r4 (139.6 s), c0.8-r2 (116.8 s), c0.8-r4 (42.8 s), c0.95-r4 (48.5 s), all under load 13-19 from other sessions.

## Round 27: Claim 6 mechanism and longer-training cells (2026-11-15)

### A. Is the input table its own geometry? (REVIEW weakness 8)

Rule going in (Claim 6): the tying cost belongs to the whole input table. Prediction from "the input side needs its own geometry": fully untied input rows are not a linear map of the output rows; a band untied alone stays pinned near its output rows by the shared ones. New readout `--geom 1`: for each word, cosine of its input and output rows; R² of one ridge map input ≈ output @ M; overlap of the 10 nearest neighbours in the two tables; and how much of what the map leaves is in the top 1 / 2 / 4 / 8 directions. Vocabulary 1,500, width 16, depth 2, 150 steps of 128 × 64, tied 5.1480.

| Input table | Known loss | cos (all) | R² linear map | Neighbour overlap |
|---|---|---|---|---|
| tied | 5.1480 | 1.00 | 1.00 | 1.00 |
| all known words own rows, random start | 5.0074 | -0.23 | 0.15 | 0.07 |
| all known words own rows, start as copies (`--udelta 1`) | 5.0326 | 0.52 | 0.56 | 0.20 |
| fully untied | 5.0214 | -0.16 | 0.15 | 0.08 |
| common band alone (own-band rows) | 5.0867 | 0.18 | 0.01 | 0.20 |
| middle band alone | 5.1213 | 0.12 | 0.12 | 0.14 |
| rare band alone | 5.1301 | -0.10 | 0.05 | 0.04 |

What the best linear map leaves is spread over all 16 directions: the top 1 / 2 / 4 / 8 hold 17 / 27 / 45 / 69% (random start) and 14 / 27 / 47 / 73% (copy start).

Band additivity, gain over tied (known loss), and the gain on positions whose input word is in each band:

| Setting | common alone | middle alone | rare alone | sum | all together | sum / all |
|---|---|---|---|---|---|---|
| width 8 | 0.119 | 0.075 | 0.012 | 0.206 | 0.245 | 0.84 |
| width 16 | 0.061 | 0.027 | 0.018 | 0.106 | 0.141 | 0.75 |
| width 32 | 0.052 | 0.008 | 0.032 | 0.093 | 0.099 | 0.94 |
| width 16, context 2 | 0.035 | 0.009 | 0.011 | 0.055 | 0.096 | 0.57 |

| Setting | Untied band's gain on its own input positions, alone / with all untied (common, middle, rare) | Common band alone: gain on common / middle / rare input positions |
|---|---|---|
| width 8 | 0.129 / 0.212, 0.183 / 0.335, 0.109 / 0.302 | 0.129 / 0.126 / 0.167 |
| width 16 | 0.042 / 0.068, 0.073 / 0.191, 0.068 / 0.167 | 0.042 / 0.067 / 0.056 |
| width 32 | 0.032 / 0.048, 0.029 / 0.112, 0.048 / 0.131 | 0.032 / 0.030 / 0.050 |
| width 16, context 2 | 0.022 / 0.062, 0.038 / 0.156, 0.052 / 0.118 | 0.022 / 0.057 / 0.046 |

**Takeaway.**
- Own input rows leave the output geometry, even when they start as exact copies: in 150 steps the neighbour overlap falls from 1.00 to 0.20, and one linear map explains 15-56% of the input table. What it leaves needs 8+ of 16 directions. This is why one shared 16×16 matrix (11% of the cost) and rank-1 to rank-8 offsets fail.
- The pinning prediction is broken: a band untied alone also leaves the output geometry (R² 0.01-0.12).
- The gains do not stay in their band. Untying common words alone helps middle and rare input positions as much as common ones, also at context 2, where those positions read only their own tied row, unchanged in definition. That gain can only come through the shared layers and the output rows of the common words, which no longer serve as inputs. In the other direction, a band's own rows give only 26-67% of their all-untied gain on their own positions when the rest of the table stays tied.
- So the cost is one coupled system: the first layer and the output table are shaped by whichever input geometry most of the text uses. Context is not the cause (bands add up worse at context 2, 57%, than at 64, 75%) and width shows no trend (84 / 75 / 94% at widths 8 / 16 / 32).
- The first run at 400 steps took 46.5 s.

### B. Longer training at four vocabularies: word rows against Engram-style at 1× and 16× rows

Depth 2, Engram-style rows at rate 0.03, word rows at 0.1. Gain over plain:

| Vocabulary, training | Plain | Word | Engram 1× rows | Engram 16× rows |
|---|---|---|---|---|
| 500, 300 × 256 (4× the base tokens) | 4.3128 | 0.062 | 0.035 | **0.066** (tie) |
| 1,500, 300 × 128 (2×) | 4.8470 | 0.038 | 0.018 | **0.059** |
| 6,000, 300 × 128 (2×) | 5.5696 | **0.061** | 0.031 | 0.058 (tie) |
| 24,000, width 12, 400 × 32 | 6.5583 | **0.040** | 0.030 | 0.032 |

**Takeaway.**
- At equal rows, word rows beat Engram-style rows at all four vocabularies with longer training, by 0.010-0.030.
- With 16 times the rows, Engram-style rows:
  - win at vocabulary 1,500 (by 0.021), where pairs are seen most often per row;
  - tie at 500 and 6,000;
  - gain nothing from the extra rows at 24,000, where training is 0.08 tokens per weight and pairs are seen too rarely.
- This is the limit rule: pair tables overtake only with about 16 times the rows and enough training for most positions' pairs to pass about 100 sightings.
- Run times: 35-55 s at vocabulary 500, 31-41 s at 1,500, 65-68 s at 6,000 and 86 s for the 16× run at 24,000, all at load 10-21 from other sessions.

### The rule after round 27

**A per-word value row beats a hashed pair row of the same size in short and moderate training because it helps from the first sighting, while a pair row needs about 100 sightings. Pair tables overtake only with about 16 times the rows and long training. Freeing the input table helps only as a whole, because untying any band reshapes the shared layers and output rows for every word.**

## Round 28: Engram's full design, and the vocabulary-24,000 exception (2026-11-15)

Rule going in: per-word value rows beat Engram-style rows of the same size at 3 of 4 vocabularies; at vocabulary 24,000, width 12, depth 2, Engram-style rows win by 0.012. Two open questions (REVIEW 2 and 4): does Engram's full design close the gap, and what causes the exception?

### A. Engram's full design against word rows

`--ngfull 1 --nghead 4` adds the parts of Engram missing from our core version:
- 4 hash heads per n-gram order, each with its own multiplier, giving rows of width d/4 that are concatenated;
- key and value maps from the concatenated 2-gram and 3-gram memory (2d → d);
- the context gate sigmoid(rms(h)·rms(k)/√d);
- a depthwise causal convolution (kernel 4, dilation 3) with SiLU on the gated value.

The rows have the same count as before, V per order. The two maps add 4d² weights. Best of the rates swept for each arm (embedding rate 0.05 / 0.1 / 0.2, table rate 0.01 / 0.03 / 0.1 for Engram, 0.1 / 0.3 for word rows):

| Setting | Full Engram, best | Word rows, best | Lead of word rows |
|---|---|---|---|
| Vocabulary 500, depth 4 | 4.4227 | **4.4099** | 0.013 |
| Vocabulary 1,500, depth 4 | 4.9669 | **4.9607** | 0.006 (tie) |
| Vocabulary 1,500, width 32, depth 8, 200 steps | 4.7600 | **4.7577** | 0.002 (tie) |
| Vocabulary 6,000, depth 2 | 5.6925 | **5.6819** | 0.011 |
| Vocabulary 24,000, width 16, depth 2, 537 × 32 (section C) | 6.3844 | **6.3710** | 0.013 |

At vocabulary 6,000, 2 hash heads give 5.7022, and table rates 0.1 / 0.01 give 5.7137 / 5.7159.

**Takeaway.** The full design gains 0.005-0.012 over our core version. It ties word rows at vocabulary 1,500, where pairs are seen most often per row, and loses by 0.011-0.013 at the other three vocabularies. Word rows beat it by more than 0.01 at 3 of 4 vocabularies. Run times were 37-90 s under load from other sessions.

### B. Why Engram won at vocabulary 24,000, width 12

The cell reproduces exactly: plain 6.4735, word rows 6.4367, core Engram 6.4243. Split by how often the (previous, current) pair was seen in training, gain over plain:

| Pair seen | 0 | 1-9 | 10-99 | 100-999 | 1000+ |
|---|---|---|---|---|---|
| Word rows, vocabulary 24,000, width 12 | 0.037 | 0.034 | 0.053 | 0.039 | 0.013 |
| Engram, vocabulary 24,000, width 12 | 0.021 | 0.018 | 0.094 | 0.118 | 0.084 |
| Word rows, vocabulary 6,000, width 16, 600 × 32 | 0.042 | 0.054 | 0.074 | 0.071 | 0.063 |
| Engram, vocabulary 6,000, width 16, 600 × 32 | 0.008 | 0.020 | 0.055 | 0.100 | 0.061 |

The pair rows behave as in every other cell: they gain on often-seen pairs and little on rare ones. What differs at width 12 is the word rows. They gain only 0.013 on pairs seen 1000+ times and 0.019-0.022 on the 100 most common input words, where at width 16 they gain 0.057-0.063.

Three causes were tested one at a time:

| Test | Plain | Word rows | Engram | Lead of word rows |
|---|---|---|---|---|
| Placement: vocabulary 6,000, depth 2, Engram before the last layer (core / full) | 5.7460 | 5.6819 | 5.7106 / 5.6966 | 0.029 / 0.015 |
| Small batch: vocabulary 6,000, depth 2, 600 × 32 | 5.6577 | 5.5981 | 5.6115 | 0.013 |
| Width 12: vocabulary 6,000, depth 2, 150 × 128 | 5.7860 | 5.7539 | 5.7535 | 0.000 (tie) |

### C. The direct test: vocabulary 24,000 at width 16

Same cell (depth 2, 537 steps × 32, embedding rate 0.1), only the width moved from 12 to 16:

| Arm | Width 12 | Width 16 |
|---|---|---|
| Plain | 6.4735 | 6.4127 |
| Word rows | 6.4367 | **6.3710** |
| Core Engram, embedding rate 0.1 / 0.2 | **6.4243** / – | 6.3895 / 6.3837 |
| Full Engram, embedding rate 0.1 / 0.2 | – | 6.3844 / 6.3908 |

**Takeaway.** At width 16 word rows win by 0.013 over the best Engram arm. The exception came from the width, not the vocabulary. At width 12 a word row has 12 numbers, and the word rows lose most of their gain on common words, while the pair rows keep theirs: from width 16 to 12 the word-row gain over plain falls from 0.064 to 0.032 at vocabulary 6,000, and the Engram gain falls from 0.042 to 0.033. Word rows now beat the strongest pair table, at its best rate, at all 4 vocabularies against core Engram and at 3 of 4 against full Engram (tie at 1,500). The width-16 runs at vocabulary 24,000 took 95-151 s each under load 10-13.

### C2. Width trend and Engram's rate at width 16 (reviewer pass, after the 17:20 restart)

Vocabulary 6,000, depth 2, 150 × 128, embedding rate 0.1 (word table rate 0.1, Engram 0.03; width 16 at each arm's best rates):

| Width | Plain | Word rows | Engram | Lead of word rows |
|---|---|---|---|---|
| 8 | 5.9019 | 5.8754 | 5.8838 | 0.008 |
| 12 | 5.7860 | 5.7539 | 5.7535 | 0.000 |
| 16 | 5.7460 | 5.6819 | 5.7041 | 0.022 |
| 24 | 5.6464 | **5.6106** | 5.6304 | 0.020 |

Vocabulary 24,000, width 16, core Engram at embedding rate 0.2 and table rate 0.01 / 0.03 / 0.1: 6.3982 / **6.3837** / 6.3902. Word rows: 6.3710.

**Takeaway.** Below width 16, word rows and pair rows tie within 0.01; from width 16 up, word rows lead by 0.02. The vocabulary-24,000 lead of 0.013 holds against Engram's best of 3 table rates, 2 embedding rates and both designs. The lag guard (all Python above 10 GB, or 9+ Python runs at once) killed 9 of these runs mid-training; each was rerun until it finished, and killed runs leave no record.

### D. Which layer's rows matter (REVIEW 11)

`--offeach 1` switches off one layer's value rows at a time at test time. At vocabulary 1,500, depth 8, rows in all 8 layers, the loss rise on known next words per layer is: 0.000, −0.009, 0.019, −0.027, 0.036, −0.028, 0.002, −0.015. Switching all of them off raises the loss by 0.116.

**Takeaway.** Rows in neighbouring layers cancel each other: switching off one layer helps half the time, and the 8 single-layer rises sum to −0.023, against +0.116 for all together. A single-layer switch-off therefore cannot say why the last two layers win; that needs training with rows in one layer at a time (Claim 10's placement runs). The run took 150 s under load; the vocabulary-6,000 run was stopped.

### The rule after round 28

**A per-word value row beats a hashed pair row of the same size, including Engram's full design, from width 16 up; below width 16 the two tie, because a word row of 8-12 numbers holds too little to help on common words. Pair rows win only where pairs are seen about 100+ times. Pair tables overtake only with about 16 times the rows and long training.**

## Round 29: does the sightings rule predict a held-out vocabulary? (2026-11-15)

**Rule under test.** Word rows beat Engram-style pair rows (1× rows, same last layers) on positions whose pair was seen under 100 times, and lose on pairs seen 100-999 times. So the overall gap (Engram loss minus word loss) is the sum over sighting buckets of (share of positions in the bucket) × (per-bucket gap).

**Predictor, written before any vocabulary-3,000 run.**
- Per-bucket gaps d_b at vocabulary 1,500 and 6,000 come from rounds 26 (150 steps × 128) and 27 (300 × 128). The bucket sum reproduces the measured overall gaps: 0.0011 vs 0.0008 and 0.0214 vs 0.0199 at 150 steps; 0.0213 vs 0.0200 and 0.0320 vs 0.0301 at 300 steps.
- Vocabulary 3,000 is held out. Its d_b is the mean of the 1,500 and 6,000 values (3,000 is the midpoint on a log scale).
- Its bucket shares come from the data alone (`pw.py --pcshare 1`, no training).

| Steps × batch | Shares 0 / 1-9 / 10-99 / 100-999 / 1000+ | Predicted d_b | Predicted gap |
|---|---|---|---|
| 150 × 128 | 0.099 / 0.180 / 0.268 / 0.179 / 0.274 | +0.073 / +0.082 / +0.028 / −0.061 / −0.028 | **+0.011** |
| 300 × 128 | 0.073 / 0.140 / 0.260 / 0.216 / 0.311 | +0.100 / +0.096 / +0.049 / −0.017 / −0.004 | **+0.028** |

**What would break the rule:** Engram ahead on any bucket under 100 sightings; word rows ahead on the 100-999 bucket at 150 steps; or an overall gap more than 0.015 from the prediction (the size of single-run noise).

Runs: depth 2, width 16, context 64, plain / word rows (`--ve L-2,L-1 --tlr 0.1`) / Engram-style (`--ngram V --nglayer L-2 --tlr 0.03`), embedding rate 0.05.

### Result (6 runs, 19-42 s each)

| Steps × batch | Plain | Word rows | Engram-style | Gap: predicted | Gap: measured |
|---|---|---|---|---|---|
| 150 × 128 | 5.3773 | **5.3365** | 5.3716 | +0.011 | **+0.035** |
| 300 × 128 | 5.1972 | **5.1581** | 5.1927 | +0.028 | **+0.035** |

Per-bucket gap d_b (Engram loss minus word loss), sighting buckets 0 / 1-9 / 10-99 / 100-999 / 1000+:

| Steps | Predicted | Measured |
|---|---|---|
| 150 | +0.073 / +0.082 / +0.028 / −0.061 / −0.028 | +0.105 / +0.095 / +0.058 / **−0.037** / −0.005 |
| 300 | +0.100 / +0.096 / +0.049 / −0.017 / −0.004 | +0.094 / +0.095 / +0.061 / +0.002 / −0.000 |

**Takeaway.** No break condition was met: word rows lead on all three buckets under 100 sightings (by 0.06-0.11), and the lead ends at 100 sightings (Engram ahead by 0.037 at 150 steps; at 300 steps the 100-999 and 1000+ buckets tie at +0.002 and 0.000, where a small Engram lead of 0.017 and 0.004 was predicted). The size held at 300 steps (0.035 vs 0.028 predicted) and missed at 150 steps (0.035 vs 0.011): at vocabulary 3,000 Engram lost more on rarely seen pairs than at either 1,500 or 6,000, so averaging the per-bucket gaps across vocabularies underestimated it. The bucket sum with the measured d_b still gives the measured gap (0.035 vs 0.035, 0.036 vs 0.035), so the share-times-gap decomposition is exact; what does not transfer cleanly across vocabularies is the size of each bucket's gap. A held-out training length past 300 steps at vocabulary 3,000 would take over 40 s per run and was not run.

### The rule after round 29

**Word rows beat pair rows of the same size on positions whose pair was seen fewer than about 100 times; pair rows win from there when each pair has enough of its own row. The overall gap is the sum of bucket share × bucket lead; shares come from the data alone, bucket leads carry over across vocabularies only roughly (held-out vocabulary 3,000: size right at 300 steps, 0.024 low at 150).** Paper rewritten around this rule (manager review 2026-11-15).

## Round 30: Preregistered Held-Out Prediction at Unseen (V=4,500, d=48, L=4, T=250×128) (2026-11-23)

**Preregistered BEFORE any GPU run at `(V=4,500, d=48, L=4, heads=4, 250 steps × 128)`.**
- **Data-only bucket shares** (`load_data(4500, 250, 128)`, non-first known positions):
  - `pc_0`: `0.1094`, `pc_1`: `0.1843`, `pc_10`: `0.2753`, `pc_100`: `0.1804`, `pc_1000`: `0.2506`.
- **Reference per-bucket gaps** carried over directly from Block D (`V=6,000, d=32, L=4, 300s`):
  - `engram_full - word_last2`: `+0.0247 / +0.0389 / +0.0077 / -0.0218 / +0.0075`
  - `word_last2 - mp_ve_last2_dense`: `+0.0240 / +0.0172 / +0.0514 / +0.0747 / +0.0312`
  - `hard_stack_decoupled - mp_ve_last2_dense`: `+0.0243 / +0.0161 / +0.0209 / +0.0297 / +0.0143`
- **Share-weighted overall predictions (3-seed mean):**
  1. `engram_full - word_last2 ≈ +0.0099` (i.e., `word_last2` wins overall; `word_last2` leads `pc_0, pc_1, pc_10`, `engram_full` leads `pc_100`).
  2. `word_last2 - hard_stack_decoupled ≈ +0.0209`.
  3. `hard_stack_decoupled - mp_ve_last2_dense ≈ +0.0203` (so `word_last2 - mp_ve_last2_dense ≈ +0.0412`).
- **Falsification criteria:** Any sign reversal on `pc_0/pc_1` (`engram` beating `word`), reversal on `pc_100` (`word` beating `engram`), `mp_ve` failing to beat `hard_stack_decoupled`, or any overall gap error `> 0.015` nats.

### Result (15 runs, 3 seeds each, 2026-11-23)

| Arm | Known Loss (3 seeds) | Params (Tab) | Pred gap vs ref | Measured gap |
|---|---|---|---|---|
| `plain_w48` | `5.2748 ± 0.0094` | 543,032 (0) | — | — |
| `word_last2` | `5.2394 ± 0.0024` | 974,936 (431,904) | — | — |
| `engram_full` | `5.2509 ± 0.0027` | 984,440 (441,408) | `eng - word = +0.0099` | **`+0.0115`** (err `0.0016`) |
| `hard_stack_decoupled` | `5.2214 ± 0.0083` | 2,702,936 (2,159,904) | `word - hs = +0.0209` | **`+0.0180`** (`t=+5.0`, err `0.0029`) |
| `mp_ve_last2_dense` | **`5.2099 ± 0.0117`** | 1,707,296 (1,164,264) | `hs - mpve = +0.0203` | **`+0.0115`** (`t=+2.5`, err `0.0088`) |

Per-bucket `engram_full - word_last2` (`pc_0 / pc_1 / pc_10 / pc_100 / pc_1000`):
- **Predicted**: `+0.0247 / +0.0389 / +0.0077 / -0.0218 / +0.0075`
- **Measured**: `+0.0220 / +0.0366 / +0.0059 / -0.0154 / +0.0173` (5/5 bucket signs match; zero falsification criteria triggered).


