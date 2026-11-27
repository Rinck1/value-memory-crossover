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


def attach_memory_ablation_and_gate_hooks(model, zero_memory: str = "none"):
    """Multiply memory contributions by 0 inside forward without altering checkpoint/architecture,
    verify projection layers are bias-free (no gate-bias leakage), and record actual gate openings.

    zero_memory:
      - "none" : normal forward (records gate openings only)
      - "ngram": zeros 2g/3g pair memory p_raw (keeps 1g word table ve_word); zeros Engram output
      - "all"  : zeros ALL memory (w_raw and p_raw, plus Engram output) for pure-backbone vs plain184
    """
    if zero_memory not in ("none", "ngram", "all"):
        raise ValueError(f"Unsupported zero_memory mode: {zero_memory}")

    # Assert all memory projections have bias=False so zeroing w_raw/p_raw yields exact zero contribution
    for blk in model.blocks:
        for attr in ("proj_w_v", "proj_p_v", "proj_res"):
            mod = getattr(blk, attr, None)
            if mod is not None and mod.bias is not None:
                raise RuntimeError(f"[zero-memory] unexpected bias in block.{attr}; cannot guarantee zero leak")
    if getattr(model, "pre_head_mem", False) and model.head_proj.bias is not None:
        raise RuntimeError("[zero-memory] unexpected bias in model.head_proj; cannot guarantee zero leak")

    def _wrap_bank(bank):
        orig_lookup = bank.lookup

        def _wrapped_lookup(*args, **kwargs):
            w_raw, p_raw = orig_lookup(*args, **kwargs)
            if zero_memory == "all" and w_raw is not None:
                w_raw = w_raw * 0.0
            if zero_memory in ("ngram", "all") and p_raw is not None:
                p_raw = p_raw * 0.0
            return w_raw, p_raw

        bank.lookup = _wrapped_lookup

    if zero_memory != "none":
        if getattr(model, "bank", None) is not None:
            _wrap_bank(model.bank)
        for blk in model.blocks:
            if getattr(blk, "memory", None) is not None:
                _wrap_bank(blk.memory)
        if getattr(model, "engram", None) is not None:
            model.engram.forward = lambda x, ids, escale: x

    # Forward hooks recording per-token mean gate openings on the last forward pass
    gate_buffers: dict[str, list[torch.Tensor]] = {"gate_v": [], "gate_res": [], "head_gate": []}

    def _make_hook(kind: str, scale: float):
        def _hook(_mod, _inp, out):
            # out shape: (B, T, d) -> mean across d gives per-token opening (B, T)
            val = (scale * torch.sigmoid(out.detach().float())).mean(dim=-1).cpu()
            gate_buffers[kind].append(val)
        return _hook

    handles = []
    for blk in model.blocks:
        if getattr(blk, "gate_v", None) is not None:
            handles.append(blk.gate_v.register_forward_hook(_make_hook("gate_v", 2.0)))
        if getattr(blk, "gate_res", None) is not None:
            handles.append(blk.gate_res.register_forward_hook(_make_hook("gate_res", 1.0)))
    if getattr(model, "pre_head_mem", False) and getattr(model, "head_gate", None) is not None:
        handles.append(model.head_gate.register_forward_hook(_make_hook("head_gate", 1.0)))

    return gate_buffers, handles


def build_model_from_checkpoint(checkpoint: str, countdata: str, max_seq: int | None,
                                device: str, dtype: str = "auto", zero_memory: str = "none"):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_args = ckpt["config"]
    state = _strip_prefixes(ckpt["model"])

    arm = cfg_args["arm"]
    layers = int(cfg_args["layers"])
    d = int(cfg_args["d"])
    heads = int(cfg_args["heads"])
    d_mem = int(cfg_args.get("d_mem", 256))
    pair_rows = int(cfg_args.get("pair_rows", 1_000_000))
    V = int(cfg_args.get("V", 50257))
    seq_len = int(max_seq if max_seq is not None else cfg_args.get("seq_len", 512))

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
    ve_word_hi = _find_suffix(state, "ve_word_hi")
    n_hi = 0 if ve_hi is None else int(ve_hi.shape[0]) - 1
    n_mid = 0
    if ve_mid_a is not None:
        n_mid = int(ve_mid_a.shape[0]) - 1
    elif ve_mid_full is not None:
        n_mid = int(ve_mid_full.shape[0])
    n_3g = 0 if ve_3g is None else int(ve_3g.shape[0]) - 1
    n_w_hi = None if ve_word_hi is None else int(ve_word_hi.shape[0])
    use_3g = bool(cfg["use_3g"] and n_3g > 0)

    model = t.Model(
        V, d, layers, heads, seq_len,
        arm=arm, ve_layers=cfg["ve_layers"], d_mem=cfg["d_mem"],
        memory_mode=cfg["memory_mode"], n_hi=n_hi, n_mid=n_mid, n_3g=n_3g,
        pair_rows=pair_rows, gate_mode=cfg["gate_mode"],
        use_ctx_gate=cfg["ctx_gate"], inject_res=cfg["inject_res"],
        pre_head_mem=cfg["prehead"], share_bank_across_layers=cfg["shared"],
        use_engram=cfg.get("use_engram", False), n_w_hi=n_w_hi,
        use_hash_3g=cfg.get("use_hash_3g", False),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"[eval] state_dict mismatch for arm={arm}\n"
            f"  missing={missing[:8]} (n={len(missing)})\n"
            f"  unexpected={unexpected[:8]} (n={len(unexpected)})"
        )
    dev = torch.device(device)
    target_dtype = torch.bfloat16 if (dtype == "bf16" or (dtype == "auto" and dev.type == "cuda")) else torch.float32
    model.eval().to(device=dev, dtype=target_dtype)

    z = np.load(countdata)
    cd = t.CountData(
        V=int(z["V"]),
        hi_keys=z["hi_keys"], mid_keys=z["mid_keys"], g3_keys=z["g3_keys"],
        mature_keys=z["mature_keys"], all_keys=z["all_keys"],
        all_counts=z["all_counts"],
        eff_hi=int(z["eff_hi"]), eff_mid=int(z["eff_mid"]), eff_g3=int(z["eff_g3"]),
        xgram_hi_words=z["xgram_hi_words"] if "xgram_hi_words" in z else None,
    )
    if cd.V != V:
        raise SystemExit(f"[eval] CountData V={cd.V} != checkpoint V={V}")
    if cfg["memory_mode"] == "tiered":
        if len(cd.hi_keys) != n_hi or len(cd.mid_keys) != n_mid:
            raise SystemExit(
                "[eval] CountData does not match tiered checkpoint table sizes:\n"
                f"  hi {len(cd.hi_keys)} vs {n_hi}; mid {len(cd.mid_keys)} vs {n_mid}\n"
                "Use the countdata file saved from the same 250M-token prefix as training."
            )
    if use_3g and len(cd.g3_keys) != n_3g:
        raise SystemExit(
            f"[eval] CountData 3g {len(cd.g3_keys)} != checkpoint {n_3g}"
        )
    if cfg["memory_mode"] == "hash" and cfg["gate_mode"] == "tier" and len(cd.mature_keys) == 0:
        raise SystemExit("[eval] CountData has empty mature_keys for pruned hash arm")
    gate_buffers, _handles = attach_memory_ablation_and_gate_hooks(model, zero_memory=zero_memory)
    model._gate_buffers = gate_buffers
    model._zero_memory = zero_memory
    return model, cd, cfg, cfg_args, V, d, layers, heads, seq_len


@torch.no_grad()
def evaluate_val_shard(model, cd, cfg, cfg_args, val_shard: str, device: str, seq_len: int) -> dict:
    """Verify exact checkpoint+CountData alignment and compute bucketed FineWeb loss, top-1 accuracy,
    and per-bucket actual gate openings (supports --zero-memory diagnostic)."""
    val_stream = t.ShardStream([val_shard])
    val_seqs = int(cfg_args.get("val_seqs", 256))
    starts = np.linspace(0, val_stream.total - seq_len - 2, val_seqs).astype(np.int64)
    val_tokens = np.stack([val_stream.read(int(s), seq_len + 1) for s in starts]).astype(np.int64)

    V = int(cfg_args.get("V", 50257))
    bs = int(cfg_args.get("eval_batch", 4))
    count_tokens = int(cfg_args.get("count_tokens", 250_000_000))
    train_tokens = float(cfg_args.get("train_tokens", 2_500_000_000))
    auto_scale = int(cfg_args.get("auto_scale_thresh", 1))
    scale = (count_tokens / max(1.0, train_tokens)) if auto_scale else 1.0
    dev = torch.device(device)
    gate_buffers = getattr(model, "_gate_buffers", {"gate_v": [], "gate_res": [], "head_gate": []})

    losses, accs, buckets = [], [], []
    gate_tok_chunks: dict[str, list[torch.Tensor]] = {"gate_v": [], "gate_res": [], "head_gate": []}
    for i in range(0, len(val_tokens), bs):
        toks = val_tokens[i:i + bs]
        t_md = t.make_train_ids(toks, cd, V, cfg)
        counts = t._lookup_pair_counts(toks, cd.all_keys, cd.all_counts, V)
        b_id = np.digitize(counts.astype(np.float64) / max(1e-9, scale), [1, 10, 100, 1000]).astype(np.int64)
        ids = torch.from_numpy(toks[:, :-1]).to(dev)
        tgt = torch.from_numpy(toks[:, 1:]).to(dev)
        mem = {k: torch.from_numpy(v).to(dev) for k, v in t_md.items()}
        for k_g in gate_buffers:
            gate_buffers[k_g].clear()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            logits = model(ids, **mem)
            lg = logits.float().reshape(-1, V)
            tf = tgt.reshape(-1)
            loss = F.cross_entropy(lg, tf, reduction="none")
            acc = (lg.argmax(dim=-1) == tf).float()
        losses.append(loss.cpu())
        accs.append(acc.cpu())
        buckets.append(torch.from_numpy(b_id).reshape(-1))
        for k_g in gate_buffers:
            if gate_buffers[k_g]:
                # Average across active memory layers for each token -> (B*T,)
                stacked = torch.stack(gate_buffers[k_g], dim=0).mean(dim=0).reshape(-1)
                gate_tok_chunks[k_g].append(stacked)
    loss = torch.cat(losses)
    acc = torch.cat(accs)
    bucket = torch.cat(buckets)
    out: dict = {
        "zero_memory": getattr(model, "_zero_memory", "none"),
        "val": round(float(loss.mean()), 5),
        "acc": round(float(acc.mean()), 5),
    }
    for b in range(5):
        sel = bucket == b
        if int(sel.sum()) > 0:
            out[f"pc_{b}"] = round(float(loss[sel].mean()), 5)
            out[f"acc_pc_{b}"] = round(float(acc[sel].mean()), 5)
    gate_summary = {}
    for k_g, chunks in gate_tok_chunks.items():
        if chunks:
            gv = torch.cat(chunks)
            gs = {"overall": round(float(gv.mean()), 5)}
            for b in range(5):
                sel = bucket == b
                if int(sel.sum()) > 0:
                    gs[f"pc_{b}"] = round(float(gv[sel].mean()), 5)
            gate_summary[k_g] = gs
    if gate_summary:
        out["gate_openings"] = gate_summary
    return out


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
        def __init__(self, checkpoint, countdata, device="cuda", max_length=None, batch_size=1,
                     dtype="auto", zero_memory="none"):
            super().__init__()
            self._device = device
            self._batch_size = max(1, int(batch_size))
            (self.model, self.cd, self.cfg, self.cfg_args,
             self.V, self.d, self.layers, self.heads, self._max_length) = \
                build_model_from_checkpoint(checkpoint, countdata, max_length, device,
                                            dtype=dtype, zero_memory=zero_memory)
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained("gpt2")
            self.eot = int(self.tok.eos_token_id)

        # ---- properties required by lm_eval ----
        @property
        def eot_token_id(self):
            return self.eot

        @property
        def prefix_token_id(self):
            return self.eot

        @property
        def max_length(self):
            return self._max_length

        @property
        def max_gen_toks(self):
            return 256

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
        def _forward_batch(self, seqs: list[list[int]]):
            max_len = max(len(s) for s in seqs)
            toks = np.full((len(seqs), max_len), self.eot, dtype=np.int64)
            for i, s in enumerate(seqs):
                toks[i, :len(s)] = s
            md = t.make_train_ids(toks, self.cd, self.V, self.cfg)
            ids = torch.from_numpy(toks[:, :-1]).to(self._device)
            mem = {k: torch.from_numpy(v).to(self._device) for k, v in md.items()}
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=(self._device.startswith("cuda")),
            ):
                logits = self.model(ids, **mem)
            return logits

        def _logits_teacher_forced(self, tokens: list[int]):
            if len(tokens) < 2:
                return None
            return self._forward_batch([tokens])[0].float()

        def _logits_next_token(self, prefix_tokens: list[int]):
            pref = prefix_tokens if prefix_tokens else [self.eot]
            # Appending a dummy target shifts inputs to toks[:, :-1] == pref without touching targets in make_train_ids.
            return self._forward_batch([pref + [0]])[0, -1].float()

        def _prepare_logprob_item(self, context: str, continuation: str):
            ctx = self.tok_encode(context)
            cont = self.tok_encode(continuation)
            if not cont:
                return None
            if not ctx:
                ctx = [self.eot]
            seq = (ctx + cont)[-(self._max_length + 1):]
            n_scored = min(len(cont), len(seq) - 1)
            start = (len(seq) - 1) - n_scored
            return seq, start, n_scored

        # ---- lm_eval API ----
        def loglikelihood(self, requests):
            results: list[tuple[float, bool] | None] = [None] * len(requests)
            pending = []
            for idx, req in enumerate(requests):
                context, continuation = req.args
                prep = self._prepare_logprob_item(context, continuation)
                if prep is None:
                    results[idx] = (0.0, True)
                else:
                    pending.append((idx, *prep))

            pending.sort(key=lambda item: len(item[1]))
            for b0 in range(0, len(pending), self._batch_size):
                chunk = pending[b0:b0 + self._batch_size]
                logits = self._forward_batch([item[1] for item in chunk])
                for b_i, (req_idx, seq, start, n_scored) in enumerate(chunk):
                    sub = logits[b_i, start:start + n_scored].float()
                    tgt = torch.tensor(seq[start + 1:start + 1 + n_scored],
                                       dtype=torch.long, device=sub.device)
                    lp = F.log_softmax(sub, dim=-1).gather(-1, tgt[:, None]).sum().item()
                    greedy = bool((sub.argmax(dim=-1) == tgt).all().item())
                    results[req_idx] = (float(lp), greedy)
            return results

        def loglikelihood_rolling(self, requests):
            out = []
            ctx_overlap = self._max_length // 2
            for req in requests:
                text = req.args[0]
                toks = self.tok_encode(text)
                if not toks:
                    out.append(0.0)
                    continue
                seq = [self.eot] + toks
                total = 0.0
                pos = 1
                while pos < len(seq):
                    win_start = max(0, pos - ctx_overlap)
                    win_end = min(len(seq), win_start + self._max_length + 1)
                    window = seq[win_start:win_end]
                    logits = self._logits_teacher_forced(window)
                    if logits is not None:
                        offset = (pos - 1) - win_start
                        sub = logits[offset:len(window) - 1]
                        tgt = torch.tensor(seq[pos:win_end], dtype=torch.long, device=sub.device)
                        total += float(F.log_softmax(sub, dim=-1).gather(-1, tgt[:, None]).sum().item())
                    pos = win_end
                out.append(float(total))
            return out

        def generate_until(self, requests):
            out = []
            for req in requests:
                context, gen_kwargs = req.args
                until = list(gen_kwargs.get("until", []))
                max_new = int(gen_kwargs.get("max_gen_toks", 32))
                do_sample = bool(gen_kwargs.get("do_sample", False))
                temperature = float(gen_kwargs.get("temperature", 1.0))
                prompt_toks = self.tok_encode(context) or [self.eot]
                generated = []
                for _ in range(max_new):
                    prefix = (prompt_toks + generated)[-self._max_length:]
                    nxt_logits = self._logits_next_token(prefix) / max(1e-6, temperature)
                    if do_sample:
                        probs = F.softmax(nxt_logits, dim=-1)
                        nxt = int(torch.multinomial(probs, 1).item())
                    else:
                        nxt = int(nxt_logits.argmax().item())
                    generated.append(nxt)
                    gen_text = self.tok_decode(generated)
                    if any(u in gen_text for u in until):
                        break
                gen_text = self.tok_decode(generated)
                for u in until:
                    if u in gen_text:
                        gen_text = gen_text.split(u, 1)[0]
                out.append(gen_text)
            return out

    return ValueMemoryLM


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--countdata", required=True)
    p.add_argument("--tasks", default="lambada_openai,wikitext,piqa,arc_easy,sciq,hellaswag")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=None,
                   help="Defaults to checkpoint config seq_len (512) to stay inside trained RoPE range")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"],
                   help="auto uses bfloat16 on CUDA (halves 350M VRAM to ~2GB) and float32 on CPU")
    p.add_argument("--limit", type=float, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify-val-shard", default=None,
                   help="Optional path to fineweb_val_000000.bin to verify exact training val-loss parity")
    p.add_argument("--zero-memory", default="none", choices=["none", "ngram", "all"],
                   help="Inference-time memory knockout diagnostic: none | ngram (zero 2g/3g) | all (zero word+2g+3g)")
    args = p.parse_args()

    user_output = args.output
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = os.path.join(ROOT, "results", "lm_eval", stem + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.verify_val_shard:
        model, cd, cfg, cfg_args, V, d, layers, heads, seq_len = build_model_from_checkpoint(
            args.checkpoint, args.countdata, args.max_length, args.device,
            dtype=args.dtype, zero_memory=args.zero_memory)
        res = evaluate_val_shard(model, cd, cfg, cfg_args, args.verify_val_shard, args.device, seq_len)
        print(f"[verify-val] arm={cfg_args['arm']} zero_memory={args.zero_memory} seq_len={seq_len} {res}")
        if user_output is not None:
            with open(user_output, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"[ok] wrote {user_output}")
        return

    if args.dry_run:
        # Load and run one tiny forward without lm_eval.
        model, cd, cfg, cfg_args, V, d, layers, heads, seq_len = build_model_from_checkpoint(
            args.checkpoint, args.countdata, args.max_length, args.device,
            dtype=args.dtype, zero_memory=args.zero_memory)
        n_tok = min(64, seq_len)
        toks = np.random.default_rng(0).integers(0, V, size=(1, n_tok + 1), dtype=np.int64)
        md = t.make_train_ids(toks, cd, V, cfg)
        ids = torch.from_numpy(toks[:, :-1]).to(args.device)
        mem = {k: torch.from_numpy(v).to(args.device) for k, v in md.items()}
        with torch.no_grad():
            logits = model(ids, **mem)
        g_info = {k: round(float(torch.stack(v).mean()), 4) for k, v in model._gate_buffers.items() if v}
        print(f"[dry-run] arm={cfg_args['arm']} zero_memory={args.zero_memory} dtype={args.dtype} "
              f"seq_len={seq_len} logits {tuple(logits.shape)} finite={bool(torch.isfinite(logits).all())} gates={g_info}")
        return

    LM = make_lm_class()
    lm = LM(args.checkpoint, args.countdata, device=args.device,
            max_length=args.max_length, batch_size=args.batch_size,
            dtype=args.dtype, zero_memory=args.zero_memory)

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
