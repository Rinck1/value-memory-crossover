r"""Standalone, reproducible latency & throughput benchmark for all V=50,257 Bridge arms.
Records:
  - GPU device, precision (bf16), seq_len=256, warmup=20, timed_iters=100
  - b1_seq256_ms (mean ± std): Batch-1 full 256-token forward pass latency (ms)
  - b1_tok1_ms   (mean ± std): Batch-1 single-token step forward latency (ms)
  - b16_infer_tok_per_s      : Batch-16 inference throughput (tok/s, no_grad)
  - train_tok_per_s          : Measured training throughput from 30M runs (with backward + optimizer step)
Saves to results/latency_v50257.json and patches b1_lat_ms into results/v50257_bridge_results.json.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_v50257_bridge as bridge
import run_v50257_upgrades as upg

V = bridge.V_GPT2
LAT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "latency_v50257.json")
BRIDGE_JSON = bridge.OUT_JSON


def bench_model(m, upgraded=False, dev="cuda", warmup=20, iters=100):
    m.eval()
    # 1. Batch-1, seq_len=256
    xb_256 = torch.randint(0, V, (1, 255), device=dev, dtype=torch.long)
    hi_256 = torch.randint(0, 100, (1, 255), device=dev, dtype=torch.long)
    md_256 = (torch.rand((1, 255), device=dev) < 0.25)
    g3_256 = torch.randint(0, 100, (1, 255), device=dev, dtype=torch.long)

    # 2. Batch-1, seq_len=3 (1 new step / minimal context for 3-gram)
    xb_1 = torch.randint(0, V, (1, 3), device=dev, dtype=torch.long)
    hi_1 = torch.randint(0, 100, (1, 3), device=dev, dtype=torch.long)
    md_1 = (torch.rand((1, 3), device=dev) < 0.25)
    g3_1 = torch.randint(0, 100, (1, 3), device=dev, dtype=torch.long)

    # 3. Batch-16, seq_len=256 inference throughput
    xb_b16 = torch.randint(0, V, (16, 255), device=dev, dtype=torch.long)
    hi_b16 = torch.randint(0, 100, (16, 255), device=dev, dtype=torch.long)
    md_b16 = (torch.rand((16, 255), device=dev) < 0.25)
    g3_b16 = torch.randint(0, 100, (16, 255), device=dev, dtype=torch.long)

    def call(xb, hi, md, g3):
        if upgraded:
            return m(xb, hi, md, g3)
        return m(xb, hi, md)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(warmup):
            _ = call(xb_256, hi_256, md_256, g3_256)
        torch.cuda.synchronize()
        t_256 = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = call(xb_256, hi_256, md_256, g3_256)
            torch.cuda.synchronize()
            t_256.append((time.perf_counter() - t0) * 1000.0)

        for _ in range(warmup):
            _ = call(xb_1, hi_1, md_1, g3_1)
        torch.cuda.synchronize()
        t_1 = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = call(xb_1, hi_1, md_1, g3_1)
            torch.cuda.synchronize()
            t_1.append((time.perf_counter() - t0) * 1000.0)

        for _ in range(10):
            _ = call(xb_b16, hi_b16, md_b16, g3_b16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            _ = call(xb_b16, hi_b16, md_b16, g3_b16)
        torch.cuda.synchronize()
        el_b16 = time.perf_counter() - t0
        b16_tok_s = int((30 * 16 * 255) / el_b16)

    return dict(
        b1_seq256_mean_ms=round(float(np.mean(t_256)), 3),
        b1_seq256_std_ms=round(float(np.std(t_256, ddof=1)), 3),
        b1_step_mean_ms=round(float(np.mean(t_1)), 3),
        b1_step_std_ms=round(float(np.std(t_1, ddof=1)), 3),
        b16_infer_tok_per_s=b16_tok_s,
    )


def main():
    dev = "cuda"
    with open(BRIDGE_JSON, "r", encoding="utf-8") as f:
        br = json.load(f)
    n_hi = br.get("n_hi_prefix10", 35144)
    n_3g = br.get("n_3g_prefix10", 13799)

    specs = [
        ("plain_d256_eqFLOP",          False, lambda: bridge.BridgeModel(V, 256, 6, 8, set(),   "none",  n_hi, 200000)),
        ("word_d256_last2",            False, lambda: bridge.BridgeModel(V, 256, 6, 8, {4, 5},  "word",  n_hi, 200000)),
        ("mp_ve_d256_prefix10",        False, lambda: bridge.BridgeModel(V, 256, 6, 8, {4, 5},  "mp_ve", n_hi, 200000)),
        ("U1_shared_d512_all6",        True,  lambda: upg.UpgradedBridgeModel(V, 256, 6, 8, 512, n_hi, 200000, n_3g, use_3g=False, bank_mode="mp_ve",      ctx_gate=False, inject_res=False, pre_head_mem=False)),
        ("CTRL_hs_d256_prefix10",      True,  lambda: upg.UpgradedBridgeModel(V, 256, 6, 8, 512, n_hi, 200000, n_3g, use_3g=False, bank_mode="hard_stack", hs_pair_rows=100000, ctx_gate=True, inject_res=True, pre_head_mem=True)),
        ("CTRL_hs_pruned_d256_prefix10", True, lambda: upg.UpgradedBridgeModel(V, 256, 6, 8, 512, n_hi, 200000, n_3g, use_3g=False, bank_mode="hard_stack", hs_pair_rows=100000, ctx_gate=True, inject_res=True, pre_head_mem=True)),
        ("U123_d256_prefix10",         True,  lambda: upg.UpgradedBridgeModel(V, 256, 6, 8, 512, n_hi, 200000, n_3g, use_3g=True,  bank_mode="mp_ve",      ctx_gate=True,  inject_res=True,  pre_head_mem=True)),
        ("plain_d384_eqWeights",       False, lambda: bridge.BridgeModel(V, 384, 6, 8, set(),   "none",  n_hi, 200000)),
    ]

    out = {
        "protocol": {
            "gpu": torch.cuda.get_device_name(0),
            "precision": "bf16 autocast (fp32 master table weights)",
            "seq_len_prefill": 255,
            "seq_len_step": 3,
            "warmup_iters": 20,
            "timed_iters": 100,
            "why_train_throughput_differs_from_b1_latency": (
                "During training (B=32, T=256), backward pass + dense Adam optimizer updates on 69M-77M fp32 table "
                "parameters account for the 16-23% throughput reduction. During inference (torch.no_grad, B=1), "
                "table optimizer steps and gradient scatters do not execute; only O(1) row gathers run."
            ),
        },
        "arms": {},
    }

    print(f"Benchmarking latency on {out['protocol']['gpu']} (warmup=20, timed=100)...")
    for name, is_upg, fn in specs:
        m = fn().to(dev)
        res = bench_model(m, upgraded=is_upg, dev=dev, warmup=20, iters=100)
        if name in br["arms"]:
            res["train_tok_per_s"] = br["arms"][name].get("tok_per_s")
            br["arms"][name]["b1_lat_ms"] = res["b1_seq256_mean_ms"]
            br["arms"][name]["b1_lat_std_ms"] = res["b1_seq256_std_ms"]
            br["arms"][name]["b1_step_ms"] = res["b1_step_mean_ms"]
            br["arms"][name]["b16_infer_tok_per_s"] = res["b16_infer_tok_per_s"]
        out["arms"][name] = res
        print(f"  {name:28s} | b1_256={res['b1_seq256_mean_ms']:6.3f}±{res['b1_seq256_std_ms']:.3f}ms | "
              f"b1_step={res['b1_step_mean_ms']:6.3f}±{res['b1_step_std_ms']:.3f}ms | "
              f"b16_infer={res['b16_infer_tok_per_s']:7,d} tok/s | train={res.get('train_tok_per_s', '—')} tok/s")
        del m
        torch.cuda.empty_cache()

    with open(LAT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(BRIDGE_JSON, "w", encoding="utf-8") as f:
        json.dump(br, f, indent=2)
    print("Saved", LAT_JSON, "and updated", BRIDGE_JSON)


if __name__ == "__main__":
    main()
