"""Print gains (baseline loss minus variant loss, positive = variant better) by word bucket.
usage: python3 show.py BASE_PREFIX VARIANT_PREFIX [VARIANT_PREFIX ...]   (prefix up to the '-s<steps>' part)
Matches runs '<prefix>-s<steps>' with the same steps and prints known-word loss gain and gain by input / target bucket.
"""
import json, sys, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for line in open(os.path.join(HERE, "results.jsonl")):
    r = json.loads(line); R[r["tag"]] = r
base, vars_ = sys.argv[1], sys.argv[2:]
cols = ["val_known", "in_common", "in_middle", "in_rare", "in_unk", "tg_common", "tg_middle", "tg_rare"] if len(sys.argv) < 4 or "--fine" not in sys.argv else ["val_known", "rin_0", "rin_10", "rin_30", "rin_100", "rin_300", "rin_1000", "in_unk"]
sys.argv = [x for x in sys.argv if x != "--fine"]
print(f"{'variant':28s} {'steps':>5s} {'tpw':>6s} " + " ".join(f"{c:>9s}" for c in cols))
for v in vars_:
    for tag, r in R.items():
        m = re.fullmatch(re.escape(v) + r"-s(\d+)", tag)
        if not m: continue
        b = R.get(f"{base}-s{m.group(1)}")
        if not b: continue
        g = [b["scores"].get(c, float("nan")) - r["scores"].get(c, float("nan")) for c in cols]
        print(f"{v:28s} {m.group(1):>5s} {r['tpw']:6.1f} " + " ".join(f"{x:+9.3f}" for x in g))
