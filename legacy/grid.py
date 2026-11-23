"""Scaling grid for word vs pair tables: runs cells one at a time in the foreground.

python3 grid.py BLOCK   (blocks defined below; finished tags in results.jsonl are skipped)
Arms at equal extra weights: plain, word (value rows in the last two layers), pair (hashed-bigram value rows
in the same layers, V rows each), engram (gated 2-gram + 3-gram rows, V each, before the first of those layers).
"""
import json, subprocess, sys, time

done = {json.loads(l)['tag'] for l in open('results.jsonl')}


def arms(L, V, which=("plain", "word", "pair", "engram")):
    last = f"{L-2},{L-1}"
    a = dict(plain=[], word=["--ve", last, "--tlr", "0.1"], pair=["--ve", last, "--vebig", str(V), "--tlr", "0.1"],
             engram=["--ngram", str(V), "--nglayer", str(L - 2), "--tlr", "0.03"],
             prev=["--ve", last, "--veprev", "1", "--tlr", "0.1"],
             word01=["--ve", "0,1", "--tlr", "0.1"], prev01=["--ve", "0,1", "--veprev", "1", "--tlr", "0.1"],
             pair01=["--ve", "0,1", "--vebig", str(V), "--tlr", "0.1"], engram0=["--ngram", str(V), "--nglayer", "0", "--tlr", "0.03"])
    return [(k, a[k]) for k in which]


def cell(prefix, V, L, width=16, steps=150, batch=128, ctx=64, extra=(), which=("plain", "word", "pair", "engram")):
    out = []
    for k, ar in arms(L, V, which):
        tag = f"{prefix}-V{V}-L{L}-w{width}-s{steps}-b{batch}-c{ctx}-{k}"
        out.append((tag, ["--V", str(V), "--layers", str(L), "--width", str(width), "--steps", str(steps), "--batch", str(batch), "--L", str(ctx), *extra, *ar]))
    return out


B = {}
A5 = ("plain", "word", "prev", "pair", "engram")
B["mech"] = [c for L in (2, 4, 8) for c in cell("r26m", 1500, L, extra=["--probe", "32"], which=A5)] + \
    [c for L in (4, 8) for c in cell("r26m", 1500, L, extra=["--probe", "32"], which=("word01", "prev01", "pair01", "engram0"))]
B["pc"] = [c for V in (1500, 6000) for L in (2, 8) for c in cell("r26pc", V, L, which=("plain", "word", "prev", "pair", "engram"))]
B["vd"] = [c for V in (500, 1500, 6000) for L in (2, 4, 8) for c in cell("r26vd", V, L, which=A5)]
B["tpw"] = [c for L in (2, 8) for s in (75, 300, 600) for c in cell("r26tpw", 1500, L, steps=s)]
B["width"] = [c for L in (2, 8) for w in (32, 64) for c in cell("r26w", 1500, L, width=w)]
B["ctx"] = [c for L in (2, 8) for cx, b in ((32, 256), (128, 64), (256, 32)) for c in cell("r26ctx", 1500, L, batch=b, ctx=cx)]

def long(V, L, width, steps, batch):
    last = f"{L-2},{L-1}"; base = ["--V", str(V), "--layers", str(L), "--width", str(width), "--steps", str(steps), "--batch", str(batch)]
    ar = dict(plain=[], word=["--ve", last, "--tlr", "0.1"], engram=["--ngram", str(V), "--nglayer", str(L - 2), "--tlr", "0.03"],
              engram16=["--ngram", str(16 * V), "--nglayer", str(L - 2), "--tlr", "0.03"])
    return [(f"r27long-V{V}-L{L}-w{width}-s{steps}-b{batch}-{k}", base + v) for k, v in ar.items()]


B["long"] = long(500, 2, 16, 300, 256) + long(1500, 2, 16, 300, 128) + long(6000, 2, 16, 300, 128) + long(24000, 2, 12, 300, 128)

B["pred"] = [c for s in (150, 300) for c in cell("r29pred", 3000, 2, steps=s, which=("plain", "word", "engram"))]

if __name__ == "__main__":
    n = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 10**9
    for blk in [x for x in sys.argv[1:] if not x.isdigit()]:
        todo = [c for c in B[blk] if c[0] not in done][:n]
        for tag, ar in todo:
            if tag in done: continue
            t = time.time()
            r = subprocess.run(["nice", "-n", "20", "python3", "pw.py", "--dev", "mps", "--threads", "1", "--tag", tag, *ar], capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith(tag)]
            print(line[-1] if line else (tag, "FAILED", r.stderr[-300:]), flush=True)
            done.add(tag)
