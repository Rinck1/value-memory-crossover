"""Run 3 controlled incremental upgraded arms on REAL V=50,257 FineWeb (30M tokens, RTX 4070 8GB, bf16).
Preserves the 4 existing arms in v50257_bridge_results.json and adds:
  1. U1_shared_d512_all6   : Shared d_mem=512 bank projected to all 6 layers (fast-latency deployment variant)
  2. U123_d256_prefix10    : Shared d_mem=512 + exact mature 3g (c3>=10 in 10% prefix) + ctx-gated V + gated Res + prehead
  3. CTRL_hs_d256_prefix10 : Identical dense architecture/adapters/gates/prehead, unpruned hashed 2g bank (no maturity routing)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_v50257_bridge as bridge

V_GPT2 = bridge.V_GPT2
OUT_JSON = bridge.OUT_JSON
RMSNorm = bridge.RMSNorm
Muon = bridge.Muon
rope = bridge.rope


def prepare_data_with_3g(total_tokens, seq_len=256, prefix_frac=0.10, nval_seqs=512, c3_prefix_thresh=10):
    data = bridge.prepare_data(total_tokens, seq_len=seq_len, prefix_frac=prefix_frac, nval_seqs=nval_seqs)
    t0 = time.time()
    rows = data["rows"]
    val = data["val"]
    n_pref = max(1, int(len(rows) * prefix_frac))
    pref = rows[:n_pref]

    # V_GPT2 = 50257 -> V^3 = 1.27e14 << 9.22e18 (exact in signed int64)
    k3_pref = (pref[:, :-2] * V_GPT2 + pref[:, 1:-1]) * V_GPT2 + pref[:, 2:]
    u3, c3 = np.unique(k3_pref.reshape(-1), return_counts=True)
    hi3_keys = u3[c3 >= c3_prefix_thresh]

    def lookup_3g(arr):
        out = np.zeros_like(arr, dtype=np.int32)
        if len(hi3_keys) == 0:
            return out
        k3 = (arr[:, :-2] * V_GPT2 + arr[:, 1:-1]) * V_GPT2 + arr[:, 2:]
        ix = np.clip(np.searchsorted(hi3_keys, k3), 0, len(hi3_keys) - 1)
        out[:, 2:] = np.where(hi3_keys[ix] == k3, ix + 1, 0)
        return out

    data["tr_3g"] = lookup_3g(rows)
    data["val_3g"] = lookup_3g(val)
    data["n_3g"] = int(len(hi3_keys))
    print(f"Prepared 3-gram prefix tier in {time.time() - t0:.1f}s | c3>={c3_prefix_thresh}: {data['n_3g']:,} exact 3-grams")
    return data


class SharedBridgeBank(nn.Module):
    """Shared maturity-tiered or hashed bigram/trigram memory bank at V=50,257."""
    def __init__(self, V, d_mem, n_hi, mid_hash_rows=200000, n_3g=0, use_3g=True,
                 bank_mode="mp_ve", hs_pair_rows=100000):
        super().__init__()
        self.V = V
        self.d_mem = d_mem
        self.mid_hash_rows = mid_hash_rows
        self.hs_pair_rows = hs_pair_rows
        self.bank_mode = bank_mode

        self.ve_word = nn.Parameter(torch.randn(V, d_mem) * d_mem ** -0.5)
        self.ve_hi = self.ve_mid_a = self.ve_mid_p = self.ve_3g = self.ve_pair = None

        if bank_mode == "mp_ve":
            self.ve_hi = nn.Parameter(torch.randn(n_hi + 1, d_mem) * d_mem ** -0.5)
            with torch.no_grad():
                self.ve_hi[0].zero_()
            r = max(4, d_mem // 4)
            self.ve_mid_a = nn.Parameter(torch.randn(mid_hash_rows, r) * r ** -0.5)
            self.ve_mid_p = nn.Parameter(torch.randn(r, d_mem) * (r * d_mem) ** -0.5)
            if use_3g and n_3g > 0:
                self.ve_3g = nn.Parameter(torch.randn(n_3g + 1, d_mem) * d_mem ** -0.5)
                with torch.no_grad():
                    self.ve_3g[0].zero_()
        elif bank_mode == "hard_stack":
            self.ve_pair = nn.Parameter(torch.randn(hs_pair_rows, d_mem) * d_mem ** -0.5)
        else:
            raise ValueError(f"Unknown bank_mode: {bank_mode}")

    def lookup(self, ids, hi_id, is_mid, g3_id, out_dtype):
        w_vec = self.ve_word[ids].to(out_dtype)
        p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
        if self.bank_mode == "hard_stack":
            h_idx = (p1 * 1000003 + ids) % self.hs_pair_rows
            return w_vec, self.ve_pair[h_idx].to(out_dtype)

        hi_vec = self.ve_hi[hi_id].to(out_dtype) * (hi_id > 0).unsqueeze(-1).to(out_dtype)
        h_idx = (p1 * 1000003 + ids) % self.mid_hash_rows
        mid_vec = (self.ve_mid_a[h_idx].to(out_dtype) @ self.ve_mid_p.to(out_dtype)) * is_mid.unsqueeze(-1).to(out_dtype)
        p_vec = hi_vec + mid_vec
        if self.ve_3g is not None and g3_id is not None:
            g3_vec = self.ve_3g[g3_id].to(out_dtype) * (g3_id > 0).unsqueeze(-1).to(out_dtype)
            p_vec = p_vec + g3_vec
        return w_vec, p_vec


class UpgradedBridgeBlock(nn.Module):
    def __init__(self, d, H, V, d_mem, ctx_gate=False, inject_res=False):
        super().__init__()
        self.H, self.dk, self.V = H, d // H, V
        self.ctx_gate = ctx_gate
        self.inject_res = inject_res

        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qn, self.kn = RMSNorm(d // H), RMSNorm(d // H)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        f = max(8, int(8 * d / 3) // 8 * 8)
        self.gate, self.up, self.down = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

        self.proj_w_v = nn.Linear(d_mem, d, bias=False)
        self.proj_p_v = nn.Linear(d_mem, d, bias=False)
        self.lam_w = nn.Parameter(torch.tensor(1.0))
        self.lam_p = nn.Parameter(torch.tensor(1.0))

        self.gate_v = self.proj_res = self.gate_res = None
        if ctx_gate:
            self.gate_v = nn.Linear(d, d, bias=True)
            nn.init.zeros_(self.gate_v.weight)
            nn.init.zeros_(self.gate_v.bias)  # 2 * sigmoid(0) = 1.0 at init
        if inject_res:
            self.proj_res = nn.Linear(d_mem, d, bias=False)
            self.gate_res = nn.Linear(d, d, bias=True)
            nn.init.zeros_(self.gate_res.weight)
            nn.init.constant_(self.gate_res.bias, -1.0)

    def forward(self, x, shared_vecs):
        b, t, d = x.shape
        h = self.n1(x)
        sp = lambda z: z.view(b, t, self.H, self.dk).transpose(1, 2)
        v = self.v(h)

        w_raw, p_raw = shared_vecs
        wv = self.proj_w_v(w_raw)
        pv = self.proj_p_v(p_raw)
        ve_sum = self.lam_w.to(v.dtype) * wv + self.lam_p.to(v.dtype) * pv
        if self.ctx_gate:
            g_v = 2.0 * torch.sigmoid(self.gate_v(h))
            v = v + g_v * ve_sum
        else:
            v = v + ve_sum

        q, k = rope(self.qn(sp(self.q(h)))), rope(self.kn(sp(self.k(h))))
        y = F.scaled_dot_product_attention(q, k, sp(v), is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(b, t, d))

        if self.inject_res:
            h_mid = self.n2(x)
            g_r = torch.sigmoid(self.gate_res(h_mid))
            x = x + g_r * self.proj_res(w_raw + p_raw)

        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class UpgradedBridgeModel(nn.Module):
    def __init__(self, V, d, layers, heads, d_mem, n_hi, mid_hash_rows, n_3g, use_3g,
                 bank_mode="mp_ve", hs_pair_rows=100000, ctx_gate=False, inject_res=False, pre_head_mem=False):
        super().__init__()
        self.V, self.d = V, d
        self.escale = 4.0
        self.pre_head_mem = pre_head_mem

        self.emb = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        self.head = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        self.bank = SharedBridgeBank(
            V, d_mem, n_hi, mid_hash_rows=mid_hash_rows, n_3g=n_3g, use_3g=use_3g,
            bank_mode=bank_mode, hs_pair_rows=hs_pair_rows,
        )
        self.blocks = nn.ModuleList(
            UpgradedBridgeBlock(d, heads, V, d_mem, ctx_gate=ctx_gate, inject_res=inject_res)
            for _ in range(layers)
        )
        self.norm = RMSNorm(d)
        if pre_head_mem:
            self.head_proj = nn.Linear(d_mem, d, bias=False)
            self.head_gate = nn.Linear(d, d, bias=True)
            nn.init.zeros_(self.head_gate.weight)
            nn.init.constant_(self.head_gate.bias, -1.0)

    def forward(self, ids, hi_id=None, is_mid=None, g3_id=None, targets=None):
        x = self.emb[ids] * self.escale
        out_dtype = x.dtype
        shared_vecs = self.bank.lookup(ids, hi_id, is_mid, g3_id, out_dtype=out_dtype)
        for blk in self.blocks:
            x = blk(x, shared_vecs)
        h = self.norm(x)
        if self.pre_head_mem:
            w_raw, p_raw = shared_vecs
            g_h = torch.sigmoid(self.head_gate(h))
            h = h + g_h * self.head_proj(w_raw + p_raw)
        if targets is None:
            return F.linear(h, self.head.to(h.dtype))

        h_flat = h.reshape(-1, h.shape[-1])
        t_flat = targets.reshape(-1)
        head_w = self.head.to(h.dtype)
        loss_sum = torch.zeros((), device=h.device, dtype=torch.float32)
        chunk = 2048
        for s0 in range(0, h_flat.shape[0], chunk):
            lg = F.linear(h_flat[s0:s0 + chunk], head_w)
            loss_sum = loss_sum + F.cross_entropy(lg.float(), t_flat[s0:s0 + chunk], reduction="sum")
        return loss_sum / t_flat.numel()


def evaluate_upg(m, data, dev="cuda"):
    m.eval()
    val = data["val"]
    val_hi = data["val_hi"]
    val_mid = data["val_mid_mask"]
    val_3g = data["val_3g"]
    per = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for s0 in range(0, len(val), 16):
            xb = torch.from_numpy(val[s0:s0 + 16]).to(dev, dtype=torch.long)
            hi = torch.from_numpy(val_hi[s0:s0 + 16, :-1]).to(dev, dtype=torch.long)
            md = torch.from_numpy(val_mid[s0:s0 + 16, :-1]).to(dev)
            g3 = torch.from_numpy(val_3g[s0:s0 + 16, :-1]).to(dev, dtype=torch.long)
            lg = m(xb[:, :-1], hi, md, g3)
            per.append(F.cross_entropy(lg.float().reshape(-1, V_GPT2), xb[:, 1:].reshape(-1), reduction="none").cpu())
    per = torch.cat(per).numpy()
    pc_flat = data["val_pcnt"][:, :-1].reshape(-1)
    fm = np.zeros_like(val[:, :-1], dtype=bool)
    fm[:, 0] = True
    ff = fm.reshape(-1)
    scores = {"val": float(per.mean())}
    for lo, hi_b in [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 10**12)]:
        sel = (~ff) & (pc_flat >= lo) & (pc_flat < hi_b)
        if sel.sum() >= 50:
            scores[f"pc_{lo}"] = round(float(per[sel].mean()), 4)
    m.train()
    return scores


def measure_b1_latency(m, data, upgraded=True, dev="cuda"):
    m.eval()
    xb1 = torch.from_numpy(data["val"][:1, :-1]).to(dev, dtype=torch.long)
    hi1 = torch.from_numpy(data["val_hi"][:1, :-1]).to(dev, dtype=torch.long)
    md1 = torch.from_numpy(data["val_mid_mask"][:1, :-1]).to(dev)
    g31 = torch.from_numpy(data["val_3g"][:1, :-1]).to(dev, dtype=torch.long)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(10):
            _ = m(xb1, hi1, md1, g31) if upgraded else m(xb1)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            _ = m(xb1, hi1, md1, g31) if upgraded else m(xb1)
        torch.cuda.synchronize()
    ms = round((time.time() - t0) / 50.0 * 1000.0, 3)
    m.train()
    return ms


def run_upg_arm(name, data, d_mem=512, mid_hash_rows=200000, hs_pair_rows=100000,
                use_3g=True, bank_mode="mp_ve", ctx_gate=False, inject_res=False, pre_head_mem=False,
                batch_seqs=32, accum=2, elr=0.05, tlr_w=0.05, tlr_p=0.03, lr=0.015,
                eval_fracs=(0.25, 0.5, 0.75, 1.0), seed=0, dev="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.reset_peak_memory_stats()

    d, L, H = 256, 6, 8
    m = UpgradedBridgeModel(
        V_GPT2, d, L, H, d_mem, data["n_hi"], mid_hash_rows, data["n_3g"], use_3g=use_3g,
        bank_mode=bank_mode, hs_pair_rows=hs_pair_rows, ctx_gate=ctx_gate,
        inject_res=inject_res, pre_head_mem=pre_head_mem,
    ).to(dev)

    word_tabs = [m.bank.ve_word]
    pair_tabs = [z for z in (m.bank.ve_hi, m.bank.ve_mid_a, m.bank.ve_mid_p, m.bank.ve_3g, m.bank.ve_pair) if z is not None]
    all_tabs = set(id(w) for w in word_tabs + pair_tabs)
    mats = [
        w for n, w in m.named_parameters()
        if w.ndim == 2 and (n.startswith("blocks") or n.startswith("head_proj") or n.startswith("head_gate"))
        and id(w) not in all_tabs
    ]
    mat_ids = set(id(w) for w in mats)
    rest = [w for w in m.parameters() if w.requires_grad and id(w) not in all_tabs and id(w) not in mat_ids]

    opts = [
        Muon(mats, lr),
        torch.optim.AdamW([dict(params=rest, lr=elr, weight_decay=0.0)], betas=(0.9, 0.95)),
        torch.optim.Adam([dict(params=word_tabs, lr=tlr_w), dict(params=pair_tabs, lr=tlr_p)], betas=(0.9, 0.95)),
    ]
    base = [[g["lr"] for g in o.param_groups] for o in opts]

    n_seqs = len(data["rows"])
    steps = n_seqs // batch_seqs
    mb = batch_seqs // accum
    eval_steps = sorted(set(max(1, min(steps, int(round(steps * fr)))) for fr in eval_fracs))
    ckpts = {}

    t0 = time.time()
    m.train()
    for step in range(steps):
        sc = min((step + 1) / max(1, min(100, steps // 10)), 1.0, (steps - step) / max(1.0, 0.25 * steps))
        for o, bl in zip(opts, base):
            for g, lr0 in zip(o.param_groups, bl):
                g["lr"] = lr0 * sc
        for a in range(accum):
            s0 = step * batch_seqs + a * mb
            s1 = s0 + mb
            xb = torch.from_numpy(data["rows"][s0:s1]).to(dev, dtype=torch.long)
            hi = torch.from_numpy(data["tr_hi"][s0:s1, :-1]).to(dev, dtype=torch.long)
            md = torch.from_numpy(data["tr_mid_mask"][s0:s1, :-1]).to(dev)
            g3 = torch.from_numpy(data["tr_3g"][s0:s1, :-1]).to(dev, dtype=torch.long)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = m(xb[:, :-1], hi, md, g3, targets=xb[:, 1:]) / accum
            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss encountered in {name} at step {step}!")
            loss.backward()
        for o in opts:
            o.step()
        m.zero_grad(set_to_none=True)

        st_1 = step + 1
        if st_1 in eval_steps:
            tok_m = st_1 * batch_seqs * data["seq_len"] / 1e6
            sc_dict = evaluate_upg(m, data, dev)
            sc_dict["tokens_M"] = round(tok_m, 2)
            sc_dict["elapsed_s"] = round(time.time() - t0, 1)
            ckpts[f"{tok_m:.1f}M"] = sc_dict
            print(
                f"    [{name} @ {tok_m:5.1f}M tok, {sc_dict['elapsed_s']:6.1f}s] val={sc_dict['val']:.4f} | "
                f"pc0={sc_dict.get('pc_0')} pc1={sc_dict.get('pc_1')} pc10={sc_dict.get('pc_10')} "
                f"pc100={sc_dict.get('pc_100')} pc1000={sc_dict.get('pc_1000')}"
            )

    tot_s = round(time.time() - t0, 1)
    b1_lat = measure_b1_latency(m, data, upgraded=True, dev=dev)
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    nparam = sum(w.numel() for w in m.parameters())
    tab_param = sum(w.numel() for w in word_tabs + pair_tabs)
    dense_param = nparam - tab_param
    tok_s = int((steps * batch_seqs * data["seq_len"]) / max(1e-3, tot_s))

    del m, opts
    torch.cuda.empty_cache()

    return dict(
        name=name, params=nparam, dense_params=dense_param, tab_params=tab_param,
        peak_vram_gb=peak_gb, b1_lat_ms=b1_lat, total_s=tot_s, tok_per_s=tok_s, checkpoints=ckpts,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--tokens_m", type=float, default=30.0)
    args = p.parse_args()

    tot_tok = int(args.tokens_m * 1e6) if not args.probe else int(1.0 * 1e6)
    data = prepare_data_with_3g(tot_tok, seq_len=256, prefix_frac=0.10, c3_prefix_thresh=10)
    eval_fr = (1.0,) if args.probe else (0.25, 0.5, 0.75, 1.0)

    # Fresh-init b1 latency reference for plain baselines (zero training)
    b1_ref = {}
    for b_nm, d_ref in [("plain_d256_eqFLOP", 256), ("plain_d384_eqWeights", 384)]:
        m_ref = bridge.BridgeModel(V_GPT2, d_ref, 6, 8, set(), "none", 0, 1).to("cuda")
        b1_ref[b_nm] = measure_b1_latency(m_ref, data, upgraded=False, dev="cuda")
        del m_ref
        torch.cuda.empty_cache()
    print(f"Plain batch-1 latency ref (seq_len=256, bf16): {b1_ref}")

    arms = [
        ("U1_shared_d512_all6",   dict(d_mem=512, mid_hash_rows=200000, use_3g=False, bank_mode="mp_ve",      ctx_gate=False, inject_res=False, pre_head_mem=False)),
        ("U123_d256_prefix10",    dict(d_mem=512, mid_hash_rows=200000, use_3g=True,  bank_mode="mp_ve",      ctx_gate=True,  inject_res=True,  pre_head_mem=True)),
        ("CTRL_hs_d256_prefix10", dict(d_mem=512, hs_pair_rows=100000,  use_3g=False, bank_mode="hard_stack", ctx_gate=True,  inject_res=True,  pre_head_mem=True)),
    ]

    if not args.probe and os.path.exists(OUT_JSON):
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {"tokens_M": tot_tok / 1e6, "n_hi_prefix10": data["n_hi"], "n_mid_prefix10": data["n_mid"], "arms": {}}
    results["n_3g_prefix10"] = data["n_3g"]
    results["b1_latency_ref"] = b1_ref

    for nm, cfg in arms:
        print(f"\n--- Running {nm} (V=50,257, tokens={tot_tok / 1e6:.2f}M) ---")
        res_arm = run_upg_arm(nm, data, eval_fracs=eval_fr, **cfg)
        results["arms"][nm] = res_arm
        print(
            f"  DONE {nm}: params={res_arm['params']:,} (dense={res_arm['dense_params']:,}, tab={res_arm['tab_params']:,}), "
            f"peak_VRAM={res_arm['peak_vram_gb']} GB, b1_lat={res_arm['b1_lat_ms']} ms, "
            f"time={res_arm['total_s']}s, throughput={res_arm['tok_per_s']:,} tok/s"
        )
        if not args.probe:
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    u_dense = results["arms"]["U123_d256_prefix10"]["dense_params"]
    c_dense = results["arms"]["CTRL_hs_d256_prefix10"]["dense_params"]
    assert u_dense == c_dense, f"Dense param mismatch: U123={u_dense} vs CTRL_hs={c_dense}"
    print(f"\nVerified dense_params equality: U123={u_dense:,} == CTRL_hs={c_dense:,}")

    if not args.probe:
        print("\n=== 4-WAY SUMMARY @ 30.0M TOKENS ===")
        arms_d = results["arms"]
        for k in ["plain_d256_eqFLOP", "word_d256_last2", "mp_ve_d256_prefix10", "U1_shared_d512_all6",
                  "CTRL_hs_d256_prefix10", "U123_d256_prefix10", "plain_d384_eqWeights"]:
            if k in arms_d:
                a = arms_d[k]
                fin = a["checkpoints"].get("30.0M", {})
                dense_p = a.get("dense_params", a["params"] - a["tab_params"])
                lat = a.get("b1_lat_ms", b1_ref.get(k, "—"))
                print(
                    f"  {k:23s} | val={fin.get('val'):.4f} | pc10={fin.get('pc_10')} pc100={fin.get('pc_100')} | "
                    f"dense={dense_p:,} tab={a['tab_params']:,} | vram={a['peak_vram_gb']}GB | b1={lat}ms | {a['tok_per_s']:,} tok/s"
                )


if __name__ == "__main__":
    main()
