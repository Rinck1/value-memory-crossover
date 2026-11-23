"""Inference memory of per-word and per-n-gram tables: weights stored and bytes read per token (bf16, 2 bytes).
Lookups use only token IDs, so every row can be fetched from host memory before the layer that needs it.
usage: python3 memory.py   (prints our width-32 setting A and a large-model example with the GPT-2 vocabulary)"""
import os, numpy as np
counts = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "counts.npy")).astype(np.float64)
c = np.sort(counts[counts > 0])[::-1]; cov = np.cumsum(c) / c.sum()
K80 = int(np.searchsorted(cov, 0.8)) + 1


def designs(V, d, nl, k80, share80=0.8, ngram_rows=None):
    r = d // 4; n = ngram_rows or V
    return {   # name: (weights stored, values read per token)
        "value table, every word": (nl * V * d, nl * d),
        "hybrid (80% full, rank d/4)": (nl * (k80 * d + V * r + r * d), nl * (share80 * d + (1 - share80) * r)),
        "Engram-style 2+3-gram rows": (2 * n * d, 2 * d),
        "Scone-style input bigram rows": (n * d, d),
        "untied input table": (V * d, d),
    }


def show(title, V, d, nl, k80, share80=0.8, ngram_rows=None):
    print(f"\n{title}: vocabulary {V}, width {d}, {nl} layers with a value table, words covering 80% = {k80}")
    print(f"{'design':34s} {'stored MB':>10s} {'read KB/token':>14s}")
    for k, (w, rd) in designs(V, d, nl, k80, share80, ngram_rows).items():
        print(f"{k:34s} {2 * w / 1e6:10.2f} {2 * rd / 1e3:14.3f}")


if __name__ == "__main__":
    show("Setting A (vocabulary 1,500, unknown token kept)", 1500, 32, 2, 382, 0.8)
    show("Large model, GPT-2 vocabulary of FineWeb", 50257, 4096, 2, K80, 0.8)
    show("Large model, value table in all 32 layers", 50257, 4096, 32, K80, 0.8)
