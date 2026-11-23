"""Own-row gain on each band vs the median number of times a word of that band was seen in training.
usage: python3 band_occ.py   (reads results.jsonl and data/counts.npy)"""
import json, re, numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
counts = np.load(os.path.join(HERE, "data", "counts.npy")).astype(np.float64)
R = {}
for l in open(os.path.join(HERE, "results.jsonl")):
    r = json.loads(l); R[r["tag"]] = r
def bands(Vsz):
    keep = np.argsort(counts)[-(Vsz - 2):]; f = np.sort(counts[keep])[::-1]; share = f / counts.sum()
    cov = np.cumsum(f) / f.sum() - f / f.sum()
    return {b: share[(cov >= lo) & (cov < hi)] for b, lo, hi in (("common", 0, .5), ("middle", .5, .8), ("rare", .8, 1.01))}
rows = []
for tag in R:
    m = re.fullmatch(r"(r3-|r5-w\d+-|r6-V\d+-|r7-b128-|r8-L\d+-|r9-V\d+-)tied-s(\d+)", tag)
    if not m: continue
    pre, s = m.group(1), m.group(2); t = R[tag]; Vsz = t["args"]["V"]; tok = t["tokens"]; B = bands(Vsz)
    for b, short in (("common", "uc"), ("middle", "um"), ("rare", "ur")):
        v = R.get(f"{pre}{short}-s{s}"); u = R.get(f"{pre}untied-s{s}")
        if not v or not u: continue
        rows.append(dict(set=pre[:-1], V=Vsz, w=t["args"]["width"], s=int(s), band=b, n=len(B[b]), occ=float(np.median(B[b]) * tok),
                         cost=t["scores"][f"in_{b}"] - u["scores"][f"in_{b}"], gain=t["scores"][f"in_{b}"] - v["scores"][f"in_{b}"],
                         gain_known=t["scores"]["val_known"] - v["scores"]["val_known"]))
json.dump(rows, open(os.path.join(HERE, "band_occ.json"), "w"), indent=0)
for r in sorted(rows, key=lambda r: r["occ"]):
    print(f"{r['set']:9s} s{r['s']:5d} {r['band']:7s} n{r['n']:5d} occ/word {r['occ']:9.0f}  tying cost on band {r['cost']:+.3f}  own-row gain on band {r['gain']:+.3f}  on all known {r['gain_known']:+.3f}")
