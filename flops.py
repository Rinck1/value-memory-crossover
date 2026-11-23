"""Forward FLOPs per token for a pw.py run (multiply-add = 2 FLOPs). Training costs about 3x this.
Table lookups cost nothing; adding a value row costs 2*d per layer; a rank-r row costs 2*r*d more."""


def flops_tok(a):
    d, L, V = a["width"], a["L"], a["V"]; f = int(8 * d / 3) // 8 * 8
    ve = [x for x in str(a.get("ve", "")).split(",") if x != ""]
    layer = 2 * (4 * d * d + 3 * d * f) + 2 * L * d          # q,k,v,o + SwiGLU, then QK^T and AV over L/2 keys on average
    extra = len(ve) * (2 * d + 2 * a.get("verank", 0) * d) + 2 * a.get("urank", 0) * d + 2 * a.get("umat", 0) * d * d
    bank = a.get("bank", 0)
    if bank: extra += len(ve) * (4 * d * bank - 2 * d)              # router d x K and mix K x d replace the row add
    extra += 2 * 4 * d * (a.get("ngram", 0) > 0) + d * (a.get("inbig", 0) > 0)   # two gated n-gram rows; one input row
    if a.get("ngfull", 0) and a.get("ngram", 0): extra += 2 * 2 * (2 * d * d) + 2 * 4 * d + 6 * d   # key and value maps from the 2d memory, conv, gate
    return a["layers"] * layer + 2 * V * d + extra
