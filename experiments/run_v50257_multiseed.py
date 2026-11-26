r"""Multi-seed (seeds 0, 1, 2) + Disentangled Control ('CTRL_hs_pruned_d256_prefix10') runner for V=50,257 (30M tokens).
Idempotent & incremental:
  - Preserves existing seed=0 results in v50257_bridge_results.json under arms[name]["seeds"]["0"]
  - Adds CTRL_hs_pruned_d256_prefix10 (hashed 100K x 512 pair table + maturity pruning of <10 projected sightings)
    to cleanly separate:
      (a) Maturity Pruning effect: CTRL_hs_pruned vs CTRL_hs
      (b) Exact 2g/3g De-Collision effect: U123 vs CTRL_hs_pruned
  - Runs seed=0 for CTRL_hs_pruned first (~20m), then seed=1 and seed=2 across
    [U123_d256_prefix10, CTRL_hs_d256_prefix10, CTRL_hs_pruned_d256_prefix10, plain_d384_eqWeights],
    saving and updating 3-seed paired statistics at every checkpoint (7.5M, 15.0M, 22.5M, 30.0M) after EACH run!
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_v50257_bridge as bridge
import run_v50257_upgrades as upg

OUT_JSON = bridge.OUT_JSON
V_GPT2 = bridge.V_GPT2


# Monkey-patch SharedBridgeBank.lookup to support bank_mode="hard_stack_pruned"
_orig_init = upg.SharedBridgeBank.__init__


def _patched_init(self, V, d_mem, n_hi, mid_hash_rows=200000, n_3g=0, use_3g=True,
                  bank_mode="mp_ve", hs_pair_rows=100000):
    if bank_mode in ("hard_stack_pruned", "uniform_mgram"):
        _orig_init(self, V, d_mem, n_hi, mid_hash_rows=mid_hash_rows, n_3g=n_3g,
                   use_3g=False, bank_mode="hard_stack", hs_pair_rows=hs_pair_rows)
        self.bank_mode = bank_mode
    else:
        _orig_init(self, V, d_mem, n_hi, mid_hash_rows=mid_hash_rows, n_3g=n_3g,
                   use_3g=use_3g, bank_mode=bank_mode, hs_pair_rows=hs_pair_rows)


def _patched_lookup(self, ids, hi_id, is_mid, g3_id, out_dtype):
    w_vec = self.ve_word[ids].to(out_dtype)
    p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
    if self.bank_mode == "hard_stack":
        h_idx = (p1 * 1000003 + ids) % self.hs_pair_rows
        return w_vec, self.ve_pair[h_idx].to(out_dtype)
    if self.bank_mode == "uniform_mgram":
        p2 = F.pad(ids, (2, 0), value=self.V)[:, :-2]
        h2 = (p1 * 1000003 + ids) % self.hs_pair_rows
        h3 = (p2 * 998244353 + p1 * 1000003 + ids + 17) % self.hs_pair_rows
        return w_vec, self.ve_pair[h2].to(out_dtype) + self.ve_pair[h3].to(out_dtype)
    if self.bank_mode == "hard_stack_pruned":
        h_idx = (p1 * 1000003 + ids) % self.hs_pair_rows
        # Keep ONLY pairs with prefix count >= 1 (projected >= 10 in 30M tokens: hi or mid)!
        mature_mask = ((hi_id > 0) | is_mid).unsqueeze(-1).to(out_dtype)
        return w_vec, self.ve_pair[h_idx].to(out_dtype) * mature_mask

    hi_vec = self.ve_hi[hi_id].to(out_dtype) * (hi_id > 0).unsqueeze(-1).to(out_dtype)
    h_idx = (p1 * 1000003 + ids) % self.mid_hash_rows
    mid_vec = (self.ve_mid_a[h_idx].to(out_dtype) @ self.ve_mid_p.to(out_dtype)) * is_mid.unsqueeze(-1).to(out_dtype)
    p_vec = hi_vec + mid_vec
    if self.ve_3g is not None and g3_id is not None:
        g3_vec = self.ve_3g[g3_id].to(out_dtype) * (g3_id > 0).unsqueeze(-1).to(out_dtype)
        p_vec = p_vec + g3_vec
    return w_vec, p_vec


upg.SharedBridgeBank.__init__ = _patched_init
upg.SharedBridgeBank.lookup = _patched_lookup


def update_summaries(results):
    arms = results["arms"]
    ck_names = ["7.5M", "15.0M", "22.5M", "30.0M"]
    bk_names = ["val", "pc_0", "pc_1", "pc_10", "pc_100", "pc_1000"]

    for name, a in arms.items():
        if "seeds" not in a:
            # Migrate legacy single-seed entry into seeds["0"]
            a["seeds"] = {
                "0": {
                    "seed": 0,
                    "total_s": a.get("total_s"),
                    "tok_per_s": a.get("tok_per_s"),
                    "peak_vram_gb": a.get("peak_vram_gb"),
                    "checkpoints": a.get("checkpoints", {}),
                }
            }
        seeds_d = a["seeds"]
        s_keys = sorted(seeds_d.keys(), key=int)
        ms = {"n_seeds": len(s_keys), "seed_ids": [int(k) for k in s_keys], "checkpoints": {}}
        for ck in ck_names:
            ck_runs = [seeds_d[k]["checkpoints"][ck] for k in s_keys if ck in seeds_d[k].get("checkpoints", {})]
            if not ck_runs:
                continue
            ms["checkpoints"][ck] = {}
            for bk in bk_names:
                vals = [r[bk] for r in ck_runs if bk in r]
                if vals:
                    ms["checkpoints"][ck][f"{bk}_mean"] = round(float(np.mean(vals)), 4)
                    ms["checkpoints"][ck][f"{bk}_std"] = round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0
        a["multiseed_summary"] = ms

    # Compute paired trajectories across common seeds
    pairs_to_track = [
        ("U123_vs_CTRL_hs_total",          "CTRL_hs_d256_prefix10",        "U123_d256_prefix10"),
        ("Pruning_only_CTRL_pruned_vs_hs", "CTRL_hs_d256_prefix10",        "CTRL_hs_pruned_d256_prefix10"),
        ("Exact_only_U123_vs_CTRL_pruned", "CTRL_hs_pruned_d256_prefix10", "U123_d256_prefix10"),
        ("U123_vs_plain_d384",             "plain_d384_eqWeights",         "U123_d256_prefix10"),
        ("U123_no_ctxgate_vs_CTRL_hs",     "CTRL_hs_d256_prefix10",        "U123_no_ctxgate_d256_prefix10"),
        ("U123_no_ctxgate_vs_U123",        "U123_d256_prefix10",           "U123_no_ctxgate_d256_prefix10"),
    ]
    paired_out = {}
    for tag, ref_nm, arm_nm in pairs_to_track:
        if ref_nm not in arms or arm_nm not in arms:
            continue
        ref_seeds = arms[ref_nm].get("seeds", {})
        arm_seeds = arms[arm_nm].get("seeds", {})
        common = sorted(set(ref_seeds.keys()) & set(arm_seeds.keys()), key=int)
        if not common:
            continue
        paired_out[tag] = {"common_seeds": [int(s) for s in common], "checkpoints": {}}
        for ck in ck_names:
            paired_out[tag]["checkpoints"][ck] = {}
            for bk in bk_names:
                diffs = [
                    ref_seeds[s]["checkpoints"][ck][bk] - arm_seeds[s]["checkpoints"][ck][bk]
                    for s in common
                    if ck in ref_seeds[s]["checkpoints"] and ck in arm_seeds[s]["checkpoints"]
                ]
                if diffs:
                    m = float(np.mean(diffs))
                    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
                    t = m / (sd / math.sqrt(len(diffs))) if sd > 1e-9 else None
                    paired_out[tag]["checkpoints"][ck][bk] = {
                        "diff_mean": round(m, 4),
                        "diff_std": round(sd, 4),
                        "t_stat": round(t, 2) if t is not None else None,
                        "per_seed": [round(x, 4) for x in diffs],
                    }
    results["paired_multiseed_comparisons"] = paired_out


def save_results(results):
    update_summaries(results)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_single(arm_name, seed, data):
    if arm_name == "plain_d384_eqWeights":
        return bridge.run_arm(
            arm_name, data, d=384, L=6, H=8, ve_layers=[], mode="none",
            eval_fracs=(0.25, 0.5, 0.75, 1.0), seed=seed, dev="cuda",
        )
    if arm_name == "U123_d256_prefix10":
        return upg.run_upg_arm(
            arm_name, data, d_mem=512, mid_hash_rows=200000, use_3g=True,
            bank_mode="mp_ve", ctx_gate=True, inject_res=True, pre_head_mem=True,
            eval_fracs=(0.25, 0.5, 0.75, 1.0), seed=seed, dev="cuda",
        )
    if arm_name == "U123_no_ctxgate_d256_prefix10":
        return upg.run_upg_arm(
            arm_name, data, d_mem=512, mid_hash_rows=200000, use_3g=True,
            bank_mode="mp_ve", ctx_gate=False, inject_res=True, pre_head_mem=True,
            eval_fracs=(0.25, 0.5, 0.75, 1.0), seed=seed, dev="cuda",
        )
    if arm_name == "CTRL_hs_d256_prefix10":
        return upg.run_upg_arm(
            arm_name, data, d_mem=512, hs_pair_rows=100000, use_3g=False,
            bank_mode="hard_stack", ctx_gate=True, inject_res=True, pre_head_mem=True,
            eval_fracs=(0.25, 0.5, 0.75, 1.0), seed=seed, dev="cuda",
        )
    if arm_name == "CTRL_hs_pruned_d256_prefix10":
        return upg.run_upg_arm(
            arm_name, data, d_mem=512, hs_pair_rows=100000, use_3g=False,
            bank_mode="hard_stack_pruned", ctx_gate=True, inject_res=True, pre_head_mem=True,
            eval_fracs=(0.25, 0.5, 0.75, 1.0), seed=seed, dev="cuda",
        )
    if arm_name == "uniform_mgram_d256_prefix10":
        return upg.run_upg_arm(
            arm_name, data, d_mem=512, hs_pair_rows=100000, use_3g=False,
            bank_mode="uniform_mgram", ctx_gate=True, inject_res=True, pre_head_mem=True,
            eval_fracs=(0.25, 0.5, 0.75, 1.0), seed=seed, dev="cuda",
        )
    raise ValueError(arm_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--arms", default="CTRL_hs_pruned_d256_prefix10,U123_d256_prefix10,CTRL_hs_d256_prefix10,plain_d384_eqWeights")
    args = p.parse_args()

    seed_list = [int(x) for x in args.seeds.split(",") if x.strip()]
    arm_list = [x.strip() for x in args.arms.split(",") if x.strip()]

    with open(OUT_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)
    save_results(results)

    data = upg.prepare_data_with_3g(30_000_000, seq_len=256, prefix_frac=0.10, c3_prefix_thresh=10)

    for sd in seed_list:
        for arm_name in arm_list:
            existing_arm = results["arms"].get(arm_name, {})
            if str(sd) in existing_arm.get("seeds", {}):
                print(f"[Skip] {arm_name} seed={sd} already exists in {OUT_JSON}")
                continue
            print(f"\n================ Running {arm_name} | seed={sd} (30.0M tokens, V=50,257) ================")
            res_run = run_single(arm_name, sd, data)
            if arm_name not in results["arms"]:
                results["arms"][arm_name] = dict(res_run)
                results["arms"][arm_name]["seeds"] = {}
            results["arms"][arm_name]["seeds"][str(sd)] = {
                "seed": sd,
                "total_s": res_run["total_s"],
                "tok_per_s": res_run["tok_per_s"],
                "peak_vram_gb": res_run["peak_vram_gb"],
                "checkpoints": res_run["checkpoints"],
            }
            # Keep top-level checkpoints synced to mean across available seeds
            save_results(results)
            ms = results["arms"][arm_name]["multiseed_summary"]
            fin = ms["checkpoints"]["30.0M"]
            print(f"  --> Saved {arm_name} seed={sd} | {ms['n_seeds']}-seed 30M val = {fin['val_mean']:.4f}±{fin['val_std']:.4f}")

    print("\n================ FINAL 3-SEED PAIRED SUMMARY @ 30.0M TOKENS ================")
    for tag, comp in results.get("paired_multiseed_comparisons", {}).items():
        c30 = comp["checkpoints"].get("30.0M", {})
        print(f"  {tag:32s} (seeds={comp['common_seeds']}): val={c30.get('val')} | pc0={c30.get('pc_0')} | pc100={c30.get('pc_100')}")


if __name__ == "__main__":
    main()
