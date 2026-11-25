"""Sequential diagnostic & improvement queue for U123 at V=50,257 (30M tokens).

Phase 1 (Seed 1 — the seed where baseline U123 trailed CTRL_hs_pruned most):
  1. U123_v2_initfix      : Fix the 128x variance-shrink init bug on ve_mid_p (d_mem**-0.5)
  2. U123_v2_fullmid      : Replace low-rank mid bottleneck with 50k x 512 direct full-rank table
  3. U123_v2_fullmid_mid2 : Full-rank mid + stricter prefix cutoff (mid >= 2)

Phase 2 (Auto-complete 3 seeds: Seeds 0 & 2 on the best Phase-1 architecture):
  Saves incrementally after every run to results/u123_v2_results.json.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_v50257_bridge as bridge
import run_v50257_upgrades as upg

OUT_JSON = os.path.join(ROOT, "results", "u123_v2_results.json")
BRIDGE_JSON = os.path.join(ROOT, "results", "v50257_bridge_results.json")
V_GPT2 = bridge.V_GPT2


class SharedBridgeBankV2(nn.Module):
    """Improved SharedBridgeBank supporting initfix and direct full-rank mid."""
    def __init__(self, V, d_mem, n_hi, mid_hash_rows=200000, n_3g=0, use_3g=True,
                 bank_mode="initfix"):
        super().__init__()
        self.V = V
        self.d_mem = d_mem
        self.mid_hash_rows = mid_hash_rows
        self.bank_mode = bank_mode

        self.ve_word = nn.Parameter(torch.randn(V, d_mem) * d_mem ** -0.5)
        self.ve_hi = nn.Parameter(torch.randn(n_hi + 1, d_mem) * d_mem ** -0.5)
        with torch.no_grad():
            self.ve_hi[0].zero_()

        self.ve_mid_a = self.ve_mid_p = self.ve_mid_full = self.ve_3g = self.ve_pair = None
        if bank_mode == "initfix":
            r = max(4, d_mem // 4)
            # Proper variance alignment: Var(a_k)=1/r, Var(p_{kj})=1/d_mem -> Var((a@p)_j)=1/d_mem
            self.ve_mid_a = nn.Parameter(torch.randn(mid_hash_rows, r) * r ** -0.5)
            self.ve_mid_p = nn.Parameter(torch.randn(r, d_mem) * d_mem ** -0.5)
        elif bank_mode == "fullmid":
            # Parameter-matched direct full-rank mid hash table (eliminates r=128 bottleneck)
            self.ve_mid_full = nn.Parameter(torch.randn(mid_hash_rows, d_mem) * d_mem ** -0.5)
        else:
            raise ValueError(f"Unknown bank_mode: {bank_mode}")

        if use_3g and n_3g > 0:
            self.ve_3g = nn.Parameter(torch.randn(n_3g + 1, d_mem) * d_mem ** -0.5)
            with torch.no_grad():
                self.ve_3g[0].zero_()

    def lookup(self, ids, hi_id, is_mid, g3_id, out_dtype):
        w_vec = self.ve_word[ids].to(out_dtype)
        p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
        hi_vec = self.ve_hi[hi_id].to(out_dtype) * (hi_id > 0).unsqueeze(-1).to(out_dtype)
        h_idx = (p1 * 1000003 + ids) % self.mid_hash_rows
        if self.bank_mode == "initfix":
            mid_vec = (self.ve_mid_a[h_idx].to(out_dtype) @ self.ve_mid_p.to(out_dtype)) * is_mid.unsqueeze(-1).to(out_dtype)
        else:
            mid_vec = self.ve_mid_full[h_idx].to(out_dtype) * is_mid.unsqueeze(-1).to(out_dtype)
        p_vec = hi_vec + mid_vec
        if self.ve_3g is not None and g3_id is not None:
            g3_vec = self.ve_3g[g3_id].to(out_dtype) * (g3_id > 0).unsqueeze(-1).to(out_dtype)
            p_vec = p_vec + g3_vec
        return w_vec, p_vec


def build_mid_mask(rows_or_val, hi_keys, mid_keys):
    p = rows_or_val[:, :-1] * V_GPT2 + rows_or_val[:, 1:]
    out = np.zeros_like(rows_or_val, dtype=bool)
    if len(mid_keys):
        ix = np.clip(np.searchsorted(mid_keys, p), 0, len(mid_keys) - 1)
        out[:, 1:] = (mid_keys[ix] == p)
    return out


def prepare_all_data(total_tokens=30_000_000, seq_len=256, prefix_frac=0.10):
    data = upg.prepare_data_with_3g(total_tokens, seq_len=seq_len, prefix_frac=prefix_frac, c3_prefix_thresh=10)
    # Build stricter mid>=2 masks from 10% prefix ONLY (zero future leak)
    n_pref = max(1, int(len(data["rows"]) * prefix_frac))
    pref = data["rows"][:n_pref]
    tp_pref = pref[:, :-1].reshape(-1) * V_GPT2 + pref[:, 1:].reshape(-1)
    u_pref, c_pref = np.unique(tp_pref, return_counts=True)
    hi_th = max(2, int(round(100 * prefix_frac)))
    mid2_keys = u_pref[(c_pref >= 2) & (c_pref < hi_th)]
    data["tr_mid2_mask"] = build_mid_mask(data["rows"], None, mid2_keys)
    data["val_mid2_mask"] = build_mid_mask(data["val"], None, mid2_keys)
    data["n_mid2"] = int(len(mid2_keys))
    print(f"Prepared strict mid>=2 prefix tier: {data['n_mid2']:,} pairs (down from {data['n_mid']:,} @ mid>=1)")
    return data


def run_v2_arm(name, data, bank_mode, mid_hash_rows, strict_mid2=False, seed=0):
    orig_bank_cls = upg.SharedBridgeBank

    class _BankShim(SharedBridgeBankV2):
        def __init__(self, V, d_mem, n_hi, mid_hash_rows=mid_hash_rows, n_3g=0, use_3g=True,
                     bank_mode=bank_mode, hs_pair_rows=100000):
            super().__init__(V, d_mem, n_hi, mid_hash_rows=mid_hash_rows, n_3g=n_3g, use_3g=use_3g,
                             bank_mode=bank_mode)

    upg.SharedBridgeBank = _BankShim
    orig_init = upg.UpgradedBridgeModel.__init__

    # Ensure Optimizer collects ve_mid_full alongside ve_mid_a/ve_mid_p
    def _patched_run():
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.reset_peak_memory_stats()

        run_data = data
        if strict_mid2:
            run_data = dict(data)
            run_data["tr_mid_mask"] = data["tr_mid2_mask"]
            run_data["val_mid_mask"] = data["val_mid2_mask"]

        d, L, H, d_mem = 256, 6, 8, 512
        m = upg.UpgradedBridgeModel(
            V_GPT2, d, L, H, d_mem, run_data["n_hi"], mid_hash_rows, run_data["n_3g"], use_3g=True,
            bank_mode=bank_mode, hs_pair_rows=100000, ctx_gate=True, inject_res=True, pre_head_mem=True,
        ).to("cuda")

        word_tabs = [m.bank.ve_word]
        pair_tabs = [z for z in (m.bank.ve_hi, m.bank.ve_mid_a, m.bank.ve_mid_p, m.bank.ve_mid_full, m.bank.ve_3g) if z is not None]
        all_tabs = set(id(w) for w in word_tabs + pair_tabs)
        mats = [
            w for n, w in m.named_parameters()
            if w.ndim == 2 and (n.startswith("blocks") or n.startswith("head_proj") or n.startswith("head_gate"))
            and id(w) not in all_tabs
        ]
        mat_ids = set(id(w) for w in mats)
        rest = [w for w in m.parameters() if w.requires_grad and id(w) not in all_tabs and id(w) not in mat_ids]

        opts = [
            upg.Muon(mats, 0.015),
            torch.optim.AdamW([dict(params=rest, lr=0.05, weight_decay=0.0)], betas=(0.9, 0.95)),
            torch.optim.Adam([dict(params=word_tabs, lr=0.05), dict(params=pair_tabs, lr=0.03)], betas=(0.9, 0.95)),
        ]
        base = [[g["lr"] for g in o.param_groups] for o in opts]

        batch_seqs, accum = 32, 2
        n_seqs = len(run_data["rows"])
        steps = n_seqs // batch_seqs
        mb = batch_seqs // accum
        eval_steps = sorted(set(max(1, min(steps, int(round(steps * fr)))) for fr in (0.25, 0.5, 0.75, 1.0)))
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
                xb = torch.from_numpy(run_data["rows"][s0:s1]).to("cuda", dtype=torch.long)
                hi = torch.from_numpy(run_data["tr_hi"][s0:s1, :-1]).to("cuda", dtype=torch.long)
                md = torch.from_numpy(run_data["tr_mid_mask"][s0:s1, :-1]).to("cuda")
                g3 = torch.from_numpy(run_data["tr_3g"][s0:s1, :-1]).to("cuda", dtype=torch.long)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = m(xb[:, :-1], hi, md, g3, targets=xb[:, 1:]) / accum
                if torch.isnan(loss):
                    raise RuntimeError(f"NaN loss in {name} seed={seed} step={step}")
                loss.backward()
            for o in opts:
                o.step()
            m.zero_grad(set_to_none=True)

            st_1 = step + 1
            if st_1 in eval_steps:
                tok_m = st_1 * batch_seqs * run_data["seq_len"] / 1e6
                sc_dict = upg.evaluate_upg(m, run_data, "cuda")
                sc_dict["tokens_M"] = round(tok_m, 2)
                sc_dict["elapsed_s"] = round(time.time() - t0, 1)
                ckpts[f"{tok_m:.1f}M"] = sc_dict
                print(
                    f"    [{name} s={seed} @ {tok_m:5.1f}M tok, {sc_dict['elapsed_s']:6.1f}s] val={sc_dict['val']:.4f} | "
                    f"pc0={sc_dict.get('pc_0')} pc1={sc_dict.get('pc_1')} pc10={sc_dict.get('pc_10')} "
                    f"pc100={sc_dict.get('pc_100')} pc1000={sc_dict.get('pc_1000')}"
                )

        tot_s = round(time.time() - t0, 1)
        b1_lat = upg.measure_b1_latency(m, run_data, upgraded=True, dev="cuda")
        peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        nparam = sum(w.numel() for w in m.parameters())
        tab_param = sum(w.numel() for w in word_tabs + pair_tabs)
        dense_param = nparam - tab_param
        tok_s = int((steps * batch_seqs * run_data["seq_len"]) / max(1e-3, tot_s))

        del m, opts
        torch.cuda.empty_cache()

        return dict(
            name=name, seed=seed, params=nparam, dense_params=dense_param, tab_params=tab_param,
            peak_vram_gb=peak_gb, b1_lat_ms=b1_lat, total_s=tot_s, tok_per_s=tok_s, checkpoints=ckpts,
        )

    try:
        return _patched_run()
    finally:
        upg.SharedBridgeBank = orig_bank_cls


def update_summaries(results):
    ck_names = ["7.5M", "15.0M", "22.5M", "30.0M"]
    bk_names = ["val", "pc_0", "pc_1", "pc_10", "pc_100", "pc_1000"]
    for nm, a in results["arms"].items():
        seeds_d = a.get("seeds", {})
        s_keys = sorted(seeds_d.keys(), key=int)
        ms = {"n_seeds": len(s_keys), "seed_ids": [int(k) for k in s_keys], "checkpoints": {}}
        for ck in ck_names:
            runs = [seeds_d[k]["checkpoints"][ck] for k in s_keys if ck in seeds_d[k].get("checkpoints", {})]
            if not runs:
                continue
            ms["checkpoints"][ck] = {}
            for bk in bk_names:
                vals = [r[bk] for r in runs if bk in r]
                if vals:
                    ms["checkpoints"][ck][f"{bk}_mean"] = round(float(np.mean(vals)), 4)
                    ms["checkpoints"][ck][f"{bk}_std"] = round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0
        a["multiseed_summary"] = ms

    if os.path.exists(BRIDGE_JSON):
        with open(BRIDGE_JSON, encoding="utf-8") as f:
            br = json.load(f)["arms"]
        paired = {}
        for nm, a in results["arms"].items():
            for ref in ("CTRL_hs_pruned_d256_prefix10", "U123_d256_prefix10", "plain_d384_eqWeights"):
                if ref not in br or "seeds" not in br[ref]:
                    continue
                common = sorted(set(a["seeds"].keys()) & set(br[ref]["seeds"].keys()), key=int)
                if not common:
                    continue
                tag = f"{nm}_vs_{ref}"
                paired[tag] = {"common_seeds": [int(s) for s in common], "checkpoints": {}}
                for bk in ("val", "pc_0", "pc_10", "pc_100", "pc_1000"):
                    diffs = [
                        br[ref]["seeds"][s]["checkpoints"]["30.0M"][bk]
                        - a["seeds"][s]["checkpoints"]["30.0M"][bk]
                        for s in common
                    ]
                    m = float(np.mean(diffs))
                    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
                    t = m / (sd / math.sqrt(len(diffs))) if sd > 1e-9 else None
                    paired[tag]["checkpoints"][bk] = {
                        "diff_mean": round(m, 4), "diff_std": round(sd, 4),
                        "t_stat": round(t, 2) if t is not None else None,
                        "per_seed": [round(x, 4) for x in diffs],
                    }
        results["paired_vs_bridge"] = paired


def save_json(results):
    update_summaries(results)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_if_needed(results, data, arm, cfg, seed):
    s_key = str(seed)
    a = results["arms"].setdefault(arm, {"seeds": {}})
    if s_key in a["seeds"] and "30.0M" in a["seeds"][s_key].get("checkpoints", {}):
        fin = a["seeds"][s_key]["checkpoints"]["30.0M"]
        print(f"[SKIP] {arm} seed={seed} already done (30M val={fin['val']:.4f})")
        return fin["val"]
    print(f"\n================ Running {arm} | seed={seed} ================")
    res = run_v2_arm(arm, data, seed=seed, **cfg)
    a.setdefault("params", res["params"])
    a.setdefault("dense_params", res["dense_params"])
    a.setdefault("tab_params", res["tab_params"])
    a["seeds"][s_key] = {k: res[k] for k in ("seed", "peak_vram_gb", "b1_lat_ms", "total_s", "tok_per_s", "checkpoints")}
    save_json(results)
    fin = res["checkpoints"]["30.0M"]["val"]
    print(f"  --> Saved {arm} seed={seed}: 30M val={fin:.4f} (tab={res['tab_params']:,}, b1={res['b1_lat_ms']}ms, {res['tok_per_s']:,} tok/s)")
    return fin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase2_seeds", default="0,2")
    args = p.parse_args()

    data = prepare_all_data(30_000_000, seq_len=256, prefix_frac=0.10)
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {"tokens_M": 30.0, "n_hi": data["n_hi"], "n_mid1": data["n_mid"], "n_mid2": data["n_mid2"], "n_3g": data["n_3g"], "arms": {}}

    cfgs = {
        "U123_v2_initfix":      dict(bank_mode="initfix", mid_hash_rows=200000, strict_mid2=False),
        "U123_v2_fullmid":      dict(bank_mode="fullmid", mid_hash_rows=50000,  strict_mid2=False),
        "U123_v2_fullmid_mid2": dict(bank_mode="fullmid", mid_hash_rows=50000,  strict_mid2=True),
    }

    # Phase 1: Diagnose all 3 U123_v2 improvements cleanly on Seed 1
    p1_scores = {}
    for arm in ["U123_v2_initfix", "U123_v2_fullmid", "U123_v2_fullmid_mid2"]:
        p1_scores[arm] = run_if_needed(results, data, arm, cfgs[arm], seed=1)

    best_arm = min(p1_scores, key=p1_scores.get)
    print(f"\n=== Phase-1 Winner on Seed 1: {best_arm} ({p1_scores[best_arm]:.4f}) ===")

    # Phase 2: Finish 3-seed replication on the best Phase-1 architecture
    extra_seeds = [int(s) for s in args.phase2_seeds.split(",") if s.strip()]
    for seed in extra_seeds:
        run_if_needed(results, data, best_arm, cfgs[best_arm], seed=seed)

    print("\n================ FINAL U123_v2 PAIRED SUMMARY ================")
    for k, v in results.get("paired_vs_bridge", {}).items():
        if best_arm in k:
            print(f"  {k} {v['common_seeds']}: val={v['checkpoints'].get('val')} | pc10={v['checkpoints'].get('pc_10')}")


if __name__ == "__main__":
    main()
