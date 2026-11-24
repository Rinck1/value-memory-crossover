"""Focused 3-seed comparison & ablation suite (L=4, w=32, V=6000, 300s) extending upgrade_search_results.json:
  1. U12_plus_prehead_no3g_d64      : Pure 2-gram U12 + pre-head readout (no 3-gram), ~49% fewer table params than CTRL_hs.
  2. CTRL_hs_pruned_d64_dual_prehead: Hashed CTRL_hs with <10 sighting pruning (hi_id>0 | mid_id>0), isolates pruning gain.
  3. CTRL_hs_plus_3g_d64_dual_prehead: Hashed CTRL_hs PLUS matched hashed 3-gram table, removes 3-gram presence confound.
  4. U123_random_tier_d64           : Identical U123 architecture & param budget, random keys instead of frequency routing.
  5. U123_no_ctxgate_d64            : U123 without attention V context gate (ctx_gate=False), isolates gating gain.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)
import run_local_suite as rls
import run_upgrade_search as rus


_orig_init = rus.SharedMemoryBank.__init__
_orig_lookup = rus.SharedMemoryBank.lookup


def _patched_init(self, V, d_mem, n_hi, n_mid, n_3g=0, use_3g=False, bank_mode="mp_ve", pair_rows=24000):
    if bank_mode in ("hard_stack_pruned", "hard_stack_3g"):
        _orig_init(self, V, d_mem, n_hi, n_mid, n_3g=n_3g, use_3g=False, bank_mode="hard_stack", pair_rows=pair_rows)
        self.bank_mode = bank_mode
        if bank_mode == "hard_stack_3g" and n_3g > 0:
            self.n_3g_hash = n_3g + 1
            self.ve_3g = nn.Parameter(torch.randn(self.n_3g_hash, d_mem) * d_mem ** -0.5)
    elif bank_mode == "mp_ve_no_mid":
        _orig_init(self, V, d_mem, n_hi, 0, n_3g=n_3g, use_3g=use_3g, bank_mode="mp_ve", pair_rows=pair_rows)
        self.ve_mid_a = self.ve_mid_p = None
        self.bank_mode = bank_mode
    elif bank_mode == "mp_ve_full_mid":
        _orig_init(self, V, d_mem, n_hi, 0, n_3g=n_3g, use_3g=use_3g, bank_mode="mp_ve", pair_rows=pair_rows)
        self.ve_mid_p = None
        self.ve_mid_a = nn.Parameter(torch.randn(n_mid + 1, d_mem) * d_mem ** -0.5)
        with torch.no_grad():
            self.ve_mid_a[0].zero_()
        self.bank_mode = bank_mode
    else:
        _orig_init(self, V, d_mem, n_hi, n_mid, n_3g=n_3g, use_3g=use_3g, bank_mode=bank_mode, pair_rows=pair_rows)


def _patched_lookup(self, ids, hi_id, mid_id, g3_id=None):
    if self.bank_mode == "hard_stack_pruned":
        w_vec = self.ve_word[ids]
        p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
        h_idx = (p1 * 1000003 + ids) % self.pair_rows
        mature = ((hi_id > 0) | (mid_id > 0)).unsqueeze(-1).float()
        return w_vec, self.ve_pair[h_idx] * mature
    if self.bank_mode == "hard_stack_3g":
        w_vec = self.ve_word[ids]
        p1 = F.pad(ids, (1, 0), value=self.V)[:, :-1]
        p2 = F.pad(ids, (2, 0), value=self.V)[:, :-2]
        h2 = (p1 * 1000003 + ids) % self.pair_rows
        p_vec = self.ve_pair[h2]
        if self.ve_3g is not None:
            h3 = ((p2 * 1000003 + p1) * 1000003 + ids) % self.n_3g_hash
            p_vec = p_vec + self.ve_3g[h3]
        return w_vec, p_vec
    if self.bank_mode in ("mp_ve_no_mid", "mp_ve_full_mid"):
        w_vec = self.ve_word[ids]
        p_vec = self.ve_hi[hi_id] * (hi_id > 0).unsqueeze(-1).float()
        if self.bank_mode == "mp_ve_full_mid" and self.ve_mid_a is not None:
            p_vec = p_vec + self.ve_mid_a[mid_id] * (mid_id > 0).unsqueeze(-1).float()
        if self.ve_3g is not None and g3_id is not None:
            p_vec = p_vec + self.ve_3g[g3_id] * (g3_id > 0).unsqueeze(-1).float()
        return w_vec, p_vec
    return _orig_lookup(self, ids, hi_id, mid_id, g3_id)


rus.SharedMemoryBank.__init__ = _patched_init
rus.SharedMemoryBank.lookup = _patched_lookup


def make_random_tier_data(base_data, c3_thresh=30, seed=42):
    rng = np.random.RandomState(seed)
    rows, val, V = base_data["rows"], base_data["val"], base_data["V"]
    B = V + 1
    p12 = (rows[:, :-1].astype(np.int64) * B + rows[:, 1:].astype(np.int64)).reshape(-1)
    u12 = np.unique(p12)
    perm12 = rng.permutation(len(u12))
    n_hi, n_mid = base_data["n_hi"], base_data["n_mid"]
    rand_hi = np.sort(u12[perm12[:n_hi]])
    rand_mid = np.sort(u12[perm12[n_hi:n_hi + n_mid]])

    tr3 = (rows[:, :-2].astype(np.int64) * B + rows[:, 1:-1].astype(np.int64)) * B + rows[:, 2:].astype(np.int64)
    u3 = np.unique(tr3.reshape(-1))
    n_3g = base_data["n_3g"]
    rand_3g = np.sort(u3[rng.permutation(len(u3))[:n_3g]])

    def get_ids_2g(arr, keys):
        p = arr[:, :-1].astype(np.int64) * B + arr[:, 1:].astype(np.int64)
        ix = np.clip(np.searchsorted(keys, p), 0, len(keys) - 1)
        out = np.zeros_like(arr, dtype=np.int64)
        out[:, 1:] = np.where(keys[ix] == p, ix + 1, 0)
        return out

    def get_ids_3g(arr, keys):
        k3 = (arr[:, :-2].astype(np.int64) * B + arr[:, 1:-1].astype(np.int64)) * B + arr[:, 2:].astype(np.int64)
        ix = np.clip(np.searchsorted(keys, k3), 0, len(keys) - 1)
        out = np.zeros_like(arr, dtype=np.int64)
        out[:, 2:] = np.where(keys[ix] == k3, ix + 1, 0)
        return out

    d = dict(base_data)
    d["tr_hi"], d["val_hi"] = get_ids_2g(rows, rand_hi), get_ids_2g(val, rand_hi)
    d["tr_mid"], d["val_mid"] = get_ids_2g(rows, rand_mid), get_ids_2g(val, rand_mid)
    d["tr_3g"], d["val_3g"] = get_ids_3g(rows, rand_3g), get_ids_3g(val, rand_3g)
    return d


def main():
    data = rus.prepare_rich_data(6000, 300, 128, c3_thresh=30)
    data_rand = make_random_tier_data(data, c3_thresh=30, seed=42)

    with open(os.path.join(RESULTS, "upgrade_search_results.json"), "r", encoding="utf-8") as f:
        upg = json.load(f)
    ref_ctrl_hs = upg["CTRL_hs_d64_dual_prehead"]
    ref_ctrl_w = upg["CTRL_word_d64_dual_prehead"]
    ref_u123 = upg["U123_plus_prehead_d64"]

    out_path = os.path.join(RESULTS, "extra_ablations_results.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    cfgs = [
        ("U12_plus_prehead_no3g_d64",        data,      dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True, ctx_gate=True,  inject_res=True, use_3g=False, bank_mode="mp_ve",             pre_head_mem=True)),
        ("CTRL_hs_pruned_d64_dual_prehead",  data,      dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True, ctx_gate=True,  inject_res=True, use_3g=False, bank_mode="hard_stack_pruned", pre_head_mem=True, pair_rows=24000)),
        ("CTRL_hs_plus_3g_d64_dual_prehead", data,      dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True, ctx_gate=True,  inject_res=True, use_3g=True,  bank_mode="hard_stack_3g",     pre_head_mem=True, pair_rows=24000)),
        ("U123_random_tier_d64",             data_rand, dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True, ctx_gate=True,  inject_res=True,  use_3g=True,  bank_mode="mp_ve",             pre_head_mem=True)),
        ("U123_no_ctxgate_d64",              data,      dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True, ctx_gate=False, inject_res=True,  use_3g=True,  bank_mode="mp_ve",             pre_head_mem=True)),
        ("U123_no_mid_tier_d64",             data,      dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True, ctx_gate=True,  inject_res=True,  use_3g=True,  bank_mode="mp_ve_no_mid",      pre_head_mem=True)),
        ("U123_mid_fullrank_d64",            data,      dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True, ctx_gate=True,  inject_res=True,  use_3g=True,  bank_mode="mp_ve_full_mid",    pre_head_mem=True)),
        ("U123_no_res_d64",                  data,      dict(ve_layers=[0,1,2,3], d_mem=64, shared_bank=True, ctx_gate=True,  inject_res=False, use_3g=True,  bank_mode="mp_ve",             pre_head_mem=True)),
    ]

    print(f"Refs | CTRL_word={ref_ctrl_w['mean']:.4f} | CTRL_hs={ref_ctrl_hs['mean']:.4f} | U123={ref_u123['mean']:.4f}\n")
    for name, d_use, cfg in cfgs:
        if name in results:
            s = results[name]
            print(f"[Cached] {name:33s} | known={s['mean']:.4f}±{s['std']:.4f}")
            continue
        runs = [rus.train_eval_upg(d_use, seed=sd, measure_latency=(sd == 0), **cfg) for sd in range(3)]
        s = rls.summarize(runs)
        s["dense_params"] = runs[0]["dense_params"]
        results[name] = s
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(
            f"{name:33s} | known={s['mean']:.4f}±{s['std']:.4f} | tab={s['tab_params']:9,d} | b1={s['lat_ms']:5.2f}ms | "
            f"vs CTRL_hs:{rls.paired_str(ref_ctrl_hs, s)} | vs U123:{rls.paired_str(ref_u123, s)}"
        )

    print("\n=== Disentangled 3-Seed Paired Summary ===")
    pruned = results["CTRL_hs_pruned_d64_dual_prehead"]
    no3g = results["U12_plus_prehead_no3g_d64"]
    hs3g = results["CTRL_hs_plus_3g_d64_dual_prehead"]
    rand_t = results["U123_random_tier_d64"]
    no_cg = results["U123_no_ctxgate_d64"]
    print(f"  1) Pruning only (CTRL_hs -> CTRL_hs_pruned)  : {rls.paired_str(ref_ctrl_hs, pruned)}")
    print(f"  2) Exact 2g de-coll (CTRL_hs_pruned -> U12)  : {rls.paired_str(pruned, no3g)}  [U12 vs CTRL_hs: {rls.paired_str(ref_ctrl_hs, no3g)}]")
    print(f"  3) Exact 3g boost (U12 -> U123)              : {rls.paired_str(no3g, ref_u123)}")
    print(f"  4) Fair 3g control (CTRL_hs+3g -> U123)      : {rls.paired_str(hs3g, ref_u123)}")
    print(f"  5) Frequency vs Random keys (Rand -> U123)   : {rls.paired_str(rand_t, ref_u123)}")
    print(f"  6) Context V-gate effect (NoGate -> U123)    : {rls.paired_str(no_cg, ref_u123)}")
    no_mid = results["U123_no_mid_tier_d64"]
    full_mid = results["U123_mid_fullrank_d64"]
    no_res = results["U123_no_res_d64"]
    print(f"  7) Mid low-rank tier gain (NoMid -> U123)    : {rls.paired_str(no_mid, ref_u123)}")
    print(f"  8) Mid full-rank vs d/4 (U123 -> FullMid)    : {rls.paired_str(ref_u123, full_mid)}")
    print(f"  9) Residual injection gain (NoRes -> U123)   : {rls.paired_str(no_res, ref_u123)}")
    print("Saved to", out_path)


if __name__ == "__main__":
    main()
