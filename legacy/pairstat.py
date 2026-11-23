"""Data statistics behind the word-vs-pair result, no training.

For vocabulary V and T training tokens: share of validation positions (known next word, as in pw.py) whose current
word, and whose (previous, current) pair, was seen at least k times in training. Pair rows only help positions
whose pair was seen about 100+ times (round 26), word rows help from about 1 sighting.
python3 pairstat.py
"""
import os, numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
counts = np.load(os.path.join(DATA, "counts.npy")).astype(np.float64)
tr = np.memmap(os.path.join(DATA, "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
va = np.memmap(os.path.join(DATA, "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)
EOT = 50256
K = (1, 10, 100, 1000)


def stats(Vw, T):
    if Vw >= 50257: look = np.arange(65536); V = 50258; UNK = 50257
    else:
        keep = np.unique(np.concatenate([np.argsort(counts)[-(Vw - 2):], [EOT]])); V = len(keep) + 1; UNK = V - 1
        look = np.full(65536, UNK, np.int64); look[keep] = np.arange(len(keep))
    t = look[np.asarray(tr[:T]).astype(np.int64)]; v = look[np.asarray(va[:1_000_000]).astype(np.int64)]
    wc = np.bincount(t, minlength=V); u, c = np.unique(t[:-1] * V + t[1:], return_counts=True)
    vp = v[:-2] * V + v[1:-1]; i = np.clip(np.searchsorted(u, vp), 0, len(u) - 1); pc = np.where(u[i] == vp, c[i], 0)
    known = v[2:] != UNK; wcv = wc[v[1:-1]][known]; pcv = pc[known]
    return len(u), [round(float((wcv >= k).mean()), 2) for k in K], [round(float((pcv >= k).mean()), 2) for k in K]


if __name__ == "__main__":
    print("V, tokens, distinct pairs, share of positions whose word / pair was seen >= 1, 10, 100, 1000 times")
    for Vw in (500, 1500, 6000, 24000, 50257):
        for T in (1_200_000, 10_000_000, 50_000_000):
            n, w, p = stats(Vw, T); print(Vw, f"{T/1e6:g}M", n, "word", w, "pair", p, flush=True)
