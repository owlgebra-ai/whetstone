"""Stage-C rollout worker — runs on **turing** (packet P7 §4).

Serves the trainer's rollout requests off the shared `/data` bus and reloads the
policy whenever the trainer publishes a new export. Generation is offline vLLM
(``LLM.generate``), not the HTTP server, so the response carries exact
**token ids** — every segment mask in this project is computed on ids, and
re-tokenizing a decoded string does not round-trip.

Weight swap: a **full process-internal engine rebuild**. The packet leaves the
mechanism to the pilot ("vLLM sleep/wake reload vs full restart ~75 s") and asks
for the measured staleness to be recorded rather than hidden — this worker logs
``swap_seconds`` and the trainer logs how many steps ran against a stale policy.

Usage (turing)::

    python scripts/stagec_rollout_worker.py \\
        --run_dir /data/whetstone/runs/stagec/pilot \\
        --init_model /data/whetstone/ckpt/stageb/golden/round1/final
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.rollout_bus import RolloutBus, RolloutRequest
from whetstone.segments import blank_token_ids_for, parse_segments


def _seed_for(uid: str, step: int, base: int) -> int:
    """``sha1(uid:step:seed)`` — packet §11.

    Per-problem-per-step, so a group is never byte-identical to the same
    problem's group at another step (which would make the two steps' gradients
    redundant), and the whole run is reproducible from ``base``.
    """
    h = hashlib.sha1(f"{uid}:{step}:{base}".encode()).hexdigest()
    return int(h[:8], 16)


class Engine:
    """Owns the vLLM engine and knows how to swap the policy underneath it."""

    def __init__(self, model: str, args) -> None:
        self.args = args
        self.model_path = model
        self.version = 0
        self.llm = None
        self.tok = None
        self.blank_ids = frozenset()
        self._build()

    def _build(self) -> None:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.blank_ids = blank_token_ids_for(self.tok)
        self.llm = LLM(model=self.model_path, dtype="bfloat16",
                       gpu_memory_utilization=self.args.gpu_mem,
                       max_model_len=self.args.max_model_len,
                       trust_remote_code=True)
        self.build_seconds = time.time() - t0
        print(f"[worker] engine up on {self.model_path} in "
              f"{self.build_seconds:.1f}s", flush=True)
        del torch

    def swap(self, new_path: str, version: int) -> float:
        import torch

        t0 = time.time()
        print(f"[worker] swapping to v{version}: {new_path}", flush=True)
        del self.llm
        self.llm = None
        gc.collect()
        torch.cuda.empty_cache()
        self.model_path = new_path
        self._build()
        self.version = version
        dt = time.time() - t0
        print(f"[worker] swap complete in {dt:.1f}s", flush=True)
        return dt

    def generate(self, req: RolloutRequest) -> list:
        from vllm import SamplingParams

        p = req.params
        prompts, sps = [], []
        for it in req.items:
            prompts.append(self.tok.apply_chat_template(
                [{"role": "user", "content": it["prompt"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True))
            sps.append(SamplingParams(
                temperature=p["temperature"], top_p=p["top_p"],
                max_tokens=p["max_tokens"], n=p["K"],
                logprobs=0,     # logprob of the sampled token -> logp_old
                seed=_seed_for(it["uid"], req.step, p.get("seed", 0)),
            ))
        outs = self.llm.generate(prompts, sps)

        rows = []
        for it, out in zip(req.items, outs):
            cands = []
            for c in out.outputs:
                ids = list(c.token_ids)
                m = parse_segments(ids, prompt_len=0, blank_token_ids=self.blank_ids)
                # logprobs=0 gives {token_id: Logprob} per position for the
                # sampled token only; this is the behaviour policy's logp, which
                # DAPO's ratio needs and which the trainer must NOT recompute
                # (its weights may already be one sync ahead).
                lp = []
                for pos, d in enumerate(c.logprobs or []):
                    tid = ids[pos]
                    e = d.get(tid) if d else None
                    lp.append(float(e.logprob) if e is not None else 0.0)
                cands.append({
                    "token_ids": ids,
                    "text": c.text,
                    "logp_old": lp,
                    "think_len": m.think_len, "answer_len": m.answer_len,
                    "think_start": m.think_start, "think_end": m.think_end,
                    "answer_start": m.answer_start, "answer_end": m.answer_end,
                    "g": m.g, "gate_reason": m.reason,
                    "finish_reason": c.finish_reason,
                })
            rows.append({
                "uid": it["uid"], "prompt": it["prompt"],
                "ground_truth": it.get("ground_truth", ""),
                "level": it.get("level"), "p_hat": it.get("p_hat", 0.5),
                "seen": it.get("seen", False),
                "prompt_token_ids": list(out.prompt_token_ids or []),
                "candidates": cands,
            })
        return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--init_model", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=13312)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--idle_exit_seconds", type=float, default=0.0,
                    help="exit after this long with no request (0 = never)")
    args = ap.parse_args(argv)

    bus = RolloutBus(args.run_dir)
    eng = Engine(args.init_model, args)
    handled: set = set()
    bus.write_status({"state": "ready", "version": 0, "model": args.init_model})
    print(f"[worker] polling {bus.req_dir}", flush=True)

    last_seen = time.time()
    while True:
        cur = bus.current_weights()
        if cur and cur[0] > eng.version:
            v, path = cur
            dt = eng.swap(path, v)
            bus.write_status({"state": "ready", "version": v, "model": path,
                              "last_swap_seconds": dt})

        req = bus.poll_request(handled)
        if req is None:
            if args.idle_exit_seconds and time.time() - last_seen > args.idle_exit_seconds:
                print("[worker] idle timeout; exiting", flush=True)
                return 0
            time.sleep(args.poll)
            continue

        last_seen = time.time()
        bus.write_status({"state": "generating", "step": req.step,
                          "version": eng.version, "n_items": len(req.items)})
        t0 = time.time()
        rows = eng.generate(req)
        dt = time.time() - t0
        n_cand = sum(len(r["candidates"]) for r in rows)
        meta = {
            "generate_seconds": dt,
            "n_items": len(rows), "n_candidates": n_cand,
            "engine_version": eng.version,
            "requested_version": req.weights_version,
            # The staleness the packet asks to be recorded, not hidden.
            "staleness_versions": req.weights_version - eng.version,
        }
        bus.post_response(req.step, rows, meta)
        handled.add(req.step)
        print(f"[worker] step {req.step}: {n_cand} rollouts in {dt:.1f}s "
              f"(v{eng.version}, staleness {meta['staleness_versions']})", flush=True)
        bus.write_status({"state": "ready", "version": eng.version,
                          "last_step": req.step, "last_generate_seconds": dt})


if __name__ == "__main__":
    raise SystemExit(main())
