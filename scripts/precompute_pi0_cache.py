"""Precompute every frozen-pi_0 reference value Round 0 needs (packet P4 §7).

Run once, before the inoculation trainer, on the same box.

The trainer's 32 GB budget holds fp32 theta + grads + Adam moments + the SED
shadow at ~31 GB. A resident frozen pi_0 copy does not fit, and the packet's
stated fallback (move the pi_0-side metrics to the spark server) needs a server
launched with ``--max-logprobs 512`` shipping ~60k x 512 logprob payloads over
HTTP. Since pi_0 is *frozen*, the third option is strictly better: compute its
contribution once and store it.

Cached at a fixed seeded sample of control-set think positions:

* ``top_ids`` / ``top_lp`` — the top-512 support and logprobs, for S2's
  ``KL(pi_theta || pi_0)``
* ``actual_lp`` — actual-token logprob, for meter test (b)
* ``entropy`` — top-512 entropy, for S3's drop-vs-pi_0

Positions come from :func:`whetstone.round0_eval.build_eval_sets` with the same
seed the trainer uses, so cache row ``i`` is control trace ``i``'s sampled
positions in order. The uid list is stored and re-checked at load time.

Usage (turing, GPU free):
    python scripts/precompute_pi0_cache.py \
        --corpus /data/whetstone/corpora/seed_register_qwen \
        --out /data/whetstone/runs/round0/pi0_cache.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whetstone.round0_eval import TOPK, build_eval_sets, score_positions  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-control", type=int, default=0, help="0 = all 200")
    ap.add_argument("--n-control-positions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=TOPK)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    print("[build] eval sets ...", flush=True)
    sets = build_eval_sets(
        tok,
        args.corpus,
        n_control=args.n_control,
        n_control_positions=args.n_control_positions,
        seed=args.seed,
    )
    n_pos = sum(len(p) for p in sets.control_positions)
    print(f"[build] control {len(sets.control)} traces, {n_pos:,} sampled think positions; "
          f"heldout {len(sets.heldout)} traces")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()

    top_ids, top_lp, actual_lp, entropy, offsets, uids = [], [], [], [], [0], []
    t0 = time.time()
    for i, (seq, pos) in enumerate(zip(sets.control, sets.control_positions), 1):
        s = score_positions(model, seq, pos, topk=args.topk, want_top=True)
        top_ids.append(s["top_ids"])
        top_lp.append(s["top_lp"])
        actual_lp.append(s["actual_lp"].astype(np.float32))
        entropy.append(s["entropy"].astype(np.float32))
        offsets.append(offsets[-1] + len(pos))
        uids.append(seq.uid)
        if i % 25 == 0:
            print(f"  {i}/{len(sets.control)}  ({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        top_ids=np.concatenate(top_ids),
        top_lp=np.concatenate(top_lp),
        actual_lp=np.concatenate(actual_lp),
        entropy=np.concatenate(entropy),
        offsets=np.array(offsets, dtype=np.int64),
        uids=np.array(uids),
        positions=np.concatenate(sets.control_positions),
        meta=np.array([args.model, str(args.seed), str(args.n_control_positions),
                       str(args.topk)]),
    )
    size = out.stat().st_size / 1e6
    ent = np.concatenate(entropy)
    print(f"[write] {out}  {size:.0f} MB  |  {offsets[-1]:,} positions")
    print(f"[pi_0 control think entropy] median={np.median(ent):.4f} "
          f"mean={np.mean(ent):.4f} p80={np.percentile(ent, 80):.4f} nats")
    print(f"[pi_0 control actual logprob] mean={np.mean(np.concatenate(actual_lp)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
