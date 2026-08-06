"""Trainer-step throughput benchmark — settles P7's topology question (§4).

The packet assigns spark the trainer and turing the rollout servers, and makes
the pilot's *first* deliverable the pipeline balance: "If spark's trainer step
exceeds ~2× turing's batch generation time, the topology inverts and turing
idles — the documented fallback is full time-multiplex on turing. Do not push a
losing topology uphill for aesthetics."

Building the whole two-box pipeline before checking that spark can *carry* the
trainer would be exactly that. This script measures one number on each box:
optimizer-step throughput in tokens/s for the real model under the real
numerics (**fp32 master weights + fp32 AdamW**, per activity 007 — at LR 1e-6 a
bf16 update is far below the format's quantum and rounds to zero silently; and
per P7 §11, no bitsandbytes on spark).

Usage (identical on both boxes)::

    python scripts/bench_trainer_step.py --model Qwen/Qwen3-1.7B --steps 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--seq_len", type=int, default=1024,
                    help="tokens per sequence; DeepMath rollouts run ~3k total, "
                         "GSM8K ~500, so 1024 is a fair mid-point")
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8, help="v1 §7.3 grad-accum")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM

    dev = "cuda"
    torch.manual_seed(0)
    name = torch.cuda.get_device_name(0)
    print(f"[bench] device={name}  torch={torch.__version__}", flush=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, trust_remote_code=True
    ).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    load_s = time.time() - t0
    print(f"[bench] loaded fp32 in {load_s:.1f}s", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    V = int(model.config.vocab_size)
    B, T = args.micro_batch, args.seq_len
    ids = torch.randint(0, V - 1000, (B, T), device=dev)
    mask = torch.ones_like(ids)

    def one_step() -> float:
        s = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            # bf16 autocast on the forward (activity 007: fp32 SDPA falls back
            # to the math backend and materializes a full (T,T) matrix).
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=mask, labels=ids)
                loss = out.loss / args.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        return time.time() - s

    for i in range(args.warmup):
        one_step()
        print(f"[bench] warmup {i+1}/{args.warmup}", flush=True)

    times = []
    for i in range(args.steps):
        dt = one_step()
        times.append(dt)
        print(f"[bench] step {i+1}/{args.steps}: {dt:.2f}s", flush=True)

    tok_per_step = B * T * args.accum
    mean = sum(times) / len(times)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    res = {
        "device": name,
        "torch": torch.__version__,
        "model": args.model,
        "seq_len": T, "micro_batch": B, "accum": args.accum,
        "tokens_per_optimizer_step": tok_per_step,
        "seconds_per_step_mean": mean,
        "seconds_per_step_min": min(times),
        "tokens_per_second": tok_per_step / mean,
        "peak_mem_gb": peak_gb,
        "load_seconds": load_s,
    }
    print("\n[bench] " + json.dumps(res, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
