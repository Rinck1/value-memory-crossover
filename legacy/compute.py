"""Print known-word loss, weights, forward FLOPs per token and total training FLOPs (3 x forward x tokens) for runs
whose tag matches a regex.  usage: python3 compute.py 'r3-(untied|ve|w28)-s'"""
import json, re, sys
from flops import flops_tok
pat = re.compile(sys.argv[1])
for l in open("results.jsonl"):
    r = json.loads(l)
    if not pat.search(r["tag"]): continue
    f = r.get("flops_tok") or flops_tok(r["args"])
    print(f"{r['tag']:34s} known {r['scores']['val_known']:.4f}  params {r['params']:7d}  flops/tok {f/1e3:7.1f}K  train TFLOP {3*f*r['tokens']/1e12:7.3f}  tpw {r['tpw']:6.1f}  {r['train_s']}s")
