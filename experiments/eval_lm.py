#!/usr/bin/env python3
"""Zero-shot lm-evaluation-harness wrapper for the 125M value-memory models.

DRAFT: the lm_eval API is version-sensitive; run `--dry-run` first, then a
single task with `--limit 10` on the cloud before the full suite.

Requirements:
    pip install lm-eval transformers
    # GPT-2 tokenizer cache (or network access)

Example:
    python experiments/eval_lm.py \
        --checkpoint runs_125m/u123_s0/ckpt_2500M.pt \
        --countdata results/countdata_125m.npz \
        --tasks lambada_openai,wikitext,piqa,arc_easy,hellaswag \
        --num_fewshot 0 --batch_size 1 --output results/lm_eval/u123_s0.json

Dry run (loads the checkpoint and evaluates one tiny batch, no lm_eval needed):
    python experiments/eval_lm.py --checkpoint ... --countdata ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train_125m as t  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _strip_prefixes(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        changed = True
        while changed:
            changed = False
            for pref in ("module.", "_orig_mod."):
                if k.startswith(pref):
                    k = k[len(pref):]
                    changed = True
        out[k] = v
    return out


def _find_suffix(state: dict, suffix: str):
    for k, v in state.items():
        if k == suffix or k.endswith("." + suffix):
            return v
    return None


def build_model_from_checkpoint(checkpoint: str, countdata: str, max_seq: int, device: str):
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg_args = ckpt["config"]
    state = _strip_prefixes(ckpt["model"])

    arm = cfg_args["arm"]
    layers = int(cfg_args["layers"])
    d = int(cfg_args["d"])
    heads = int(cfg_args["heads"])
    d_mem = int(cfg_args.get("d_mem", 256))
    pair_rows = int(cfg_args.get("pair_rows", 1_000_000))
    V = int(cfg_args.get("V", 50257))

    try:
        cfg = t.arm_config(arm, layers, d, d_mem)
    except ValueError as e:
        raise SystemExit(
            f"[eval] arm '{arm}' is not supported by train_125m.arm_config: {e}\n"
            "Implement/port the arm in train_125m.py first."
        ) from e

    ve_hi = _find_suffix(state, "ve_hi")
    ve_mid_a = _find_suffix(state, "ve_mid_a")
    ve_mid_full = _find_suffix(state, "ve_mid_full")
    ve_3g = _find_suffix(state, "ve_3g")
    n_hi = 0 if ve_hi is None else int(ve_hi.shape[0]) - 1
    n_mid = 0
    if ve_mid_a is not None:
        n_mid = int(ve_mid_a.shape[0]) - 1
    elif ve_mid_full is not None:
        n_mid = int(ve_mid_full.shape[0])
    n_3g = 0 if ve_3g is None else int(ve_3g.shape[0]) - 1
    use_3g = bool(cfg["use_3g"] and n_3g > 0)

    model = t.Model(
        V, d, layers, heads, max_seq,
        arm=arm, ve_layers=cfg["ve_layers"], d_mem=cfg["d_mem"],
        memory_mode=cfg["memory_mode"], n_hi=n_hi, n_mid=n_mid, n_3g=n_3g,
        pair_rows=pair_rows, gate_mode=cfg["gate_mode"],
        use_ctx_gate=cfg["ctx_gate"], inject_res=cfg["inject_res"],
        pre_head_mem=cfg["prehead"], share_bank_across_layers=cfg["shared"],
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"[eval] state_dict mismatch for arm={arm}\n"
            f"  missing={missing[:8]} (n={len(missing)})\n"
            f"  unexpected={unexpected[:8]} (n={len(unexpected)})"
        )
    model.eval().to(device)

    z = np.load(countdata)
    cd = t.CountData(
        V=int(z["V"]),
        hi_keys=z["hi_keys"], mid_keys=z["mid_keys"], g3_keys=z["g3_keys"],
        mature_keys=z["mature_keys"], all_keys=z["all_keys"],
        all_counts=z["all_counts"],
        eff_hi=int(z["eff_hi"]), eff_mid=int(z["eff_mid"]), eff_g3=int(z["eff_g3"]),
    )
    if len(cd.hi_keys) != n_hi or len(cd.mid_keys) != n_mid or (use_3g and len(cd.g3_keys) != n_3g):
        raise SystemExit(
            "[eval] CountData does not match checkpoint table sizes:\n"
            f"  hi {len(cd.hi_keys)} vs {n_hi}; mid {len(cd.mid_keys)} vs {n_mid}; "
            f"3g {len(cd.g3_keys)} vs {n_3g}\n"
            "Use the countdata file saved from the same 250M-token prefix as training."
        )
    return model, cd, cfg, V, d, layers, heads


# ---------------------------------------------------------------------------
# lm_eval LM wrapper
# ---------------------------------------------------------------------------


def make_lm_class():
    try:
        from lm_eval.api.model import LM
    except Exception as e:  # pragma: no cover - only on machines without lm_eval
        raise SystemExit(
            "lm_eval is not installed. On the cloud: pip install lm-eval\n"
            f"Original import error: {e}"
        )

    class ValueMemoryLM(LM):
        def __init__(self, checkpoint, countdata, device="cuda", max_length=1024, batch_size=1):
            super().__init__()
            self._device = device
            self._max_length = max_length
            self._batch_size = batch_size
            self.model, self.cd, self.cfg, self.V, self.d, self.layers, self.heads = \
                build_model_from_checkpoint(checkpoint, countdata, max_length, device)
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained("gpt2")
            self.eot = int(self.tok.eos_token_id)

        # ---- properties required by lm_eval ----
        @property
        def eot_token_id(self):
            return self.eot

        @property
        def max_length(self):
            return self._max_length

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def world_size(self):
            return 1

        @property
        def rank(self):
            return 0

        def tok_encode(self, string, **kwargs):
            return self.tok.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens, **kwargs):
            return self.tok.decode(tokens, skip_special_tokens=True)

        # ---- core forward ----
        def _logits_from_tokens(self, tokens: list[int]):
            if len(tokens) < 2:
                return None
            toks = np.asarray(tokens, dtype=np.int64)[None, :]
            md = t.make_train_ids(toks, self.cd, self.V, self.cfg)
            ids = torch.from_numpy(toks[:, :-1]).to(self._device)
            mem = {k: torch.from_numpy(v).to(self._device) for k, v in md.items()}
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=(self._device.startswith("cuda")),
            ):
                logits = self.model(ids, **mem)
            return logits[0].float()

        def _logprob(self, context: str, continuation: str):
            ctx = self.tok_encode(context)
            cont = self.tok_encode(continuation)
            if len(cont) == 0:
                return 0.0, True
            max_ctx = max(0, self._max_length - len(cont))
            if len(ctx) > max_ctx:
                ctx = ctx[-max_ctx:] if max_ctx > 0 else []
            tokens = ctx + cont
            if len(tokens) < 2:
                return 0.0, True
            logits = self._logits_from_tokens(tokens)
            if logits is None:
                return 0.0, True
            start = max(0, len(ctx) - 1)
            end = len(tokens) - 1
            total = 0.0
            greedy = True
            for j in range(start, end):
                target = tokens[j + 1]
                lp = F.log_softmax(logits[j], dim=-1)[target].item()
                total += lp
                if int(logits[j].argmax().item()) != target:
                    greedy = False
            return float(total), bool(greedy)

        # ---- lm_eval API ----
        def loglikelihood(self, requests):
            out = []
            for req in requests:
                context, continuation = req.args
                out.append(self._logprob(context, continuation))
            return out

        def loglikelihood_rolling(self, requests):
            out = []
            for req in requests:
                text = req.args[0]
                toks = self.tok_encode(text)
                if len(toks) < 2:
                    out.append(0.0)
                    continue
                total = 0.0
                start = 0
                while start < len(toks) - 1:
                    end = min(len(toks), start + self._max_length)
                    window = toks[start:end]
                    logits = self._logits_from_tokens(window)
                    if logits is not None:
                        first = 1 if start == 0 else 0
                        for j in range(first, len(window) - 1):
                            target = window[j + 1]
                            total += float(F.log_softmax(logits[j], dim=-1)[target].item())
                    start = end - 1
                out.append(float(total))
            return out

        def generate_until(self, requests):
            out = []
            for req in requests:
                context, gen_kwargs = req.args
                until = gen_kwargs.get("until", [])
                max_new = int(gen_kwargs.get("max_gen_toks", 32))
                do_sample = bool(gen_kwargs.get("do_sample", False))
                temperature = float(gen_kwargs.get("temperature", 1.0))
                toks = self.tok_encode(context)
                for _ in range(max_new):
                    if len(toks) >= self._max_length:
                        toks = toks[-(self._max_length - 1):]
                    logits = self._logits_from_tokens(toks)
                    if logits is None:
                        break
                    nxt_logits = logits[-1] / max(1e-6, temperature)
                    if do_sample:
                        probs = F.softmax(nxt_logits, dim=-1)
                        nxt = int(torch.multinomial(probs, 1).item())
                    else:
                        nxt = int(nxt_logits.argmax().item())
                    toks.append(nxt)
                    text = self.tok_decode(toks)
                    if any(text.endswith(u) for u in until):
                        break
                text = self.tok_decode(toks)
                for u in until:
                    if u in text:
                        text = text.split(u)[0]
                out.append(text)
            return out

    return ValueMemoryLM


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--countdata", required=True)
    p.add_argument("--tasks", default="lambada_openai,wikitext,piqa,arc_easy,hellaswag")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=float, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = os.path.join(ROOT, "results", "lm_eval", stem + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.dry_run:
        # Load and run one tiny forward without lm_eval.
        model, cd, cfg, V, d, layers, heads = build_model_from_checkpoint(
            args.checkpoint, args.countdata, args.max_length, args.device)
        toks = np.random.default_rng(0).integers(0, V, size=(1, 65), dtype=np.int64)
        md = t.make_train_ids(toks, cd, V, cfg)
        ids = torch.from_numpy(toks[:, :-1]).to(args.device)
        mem = {k: torch.from_numpy(v).to(args.device) for k, v in md.items()}
        with torch.no_grad():
            logits = model(ids, **mem)
        print(f"[dry-run] logits {tuple(logits.shape)} finite={bool(torch.isfinite(logits).all())}")
        return

    LM = make_lm_class()
    lm = LM(args.checkpoint, args.countdata, device=args.device,
            max_length=args.max_length, batch_size=args.batch_size)

    from lm_eval import simple_evaluate
    kwargs = dict(
        model=lm,
        tasks=[x for x in args.tasks.split(",") if x],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )
    if args.limit is not None:
        kwargs["limit"] = args.limit
    try:
        results = simple_evaluate(**kwargs)
    except TypeError:
        # Older/newer lm_eval signatures may not accept a batch_size for LM objects.
        kwargs.pop("batch_size", None)
        results = simple_evaluate(**kwargs)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[ok] wrote {args.output}")
    print(json.dumps(results.get("results", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
