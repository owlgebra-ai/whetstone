"""Prompt calibration probe — v1 Step 2.1, run unchanged against Qwen3-1.7B.

Design §1 precondition 3 keeps this probe from v1 verbatim: before spending
GPU-days on a bulk harvest, run the *exact* planned sampling config on a small
stratified slice and check five metrics. A 30-minute probe catches prompt and
template faults that would otherwise waste the whole run.

The five metrics (v1 `trashed/WHETSTONE_PROCEDURE.md` Step 2.1):

  1. **Format compliance** >= 80% — parseable ``<think>`` block followed by a
     final answer. Lower means the system prompt or chat template is wrong.
  2. **Verifier acceptance shape** — 10 decisions dumped for hand inspection.
     A verifier rejecting ``"10,000,000"`` against ``"10000000"`` silently
     halves yield.
  3. **Per-level pass rate** — roughly U-shaped (easy ~90%, hard ~10%).
     Uniformly low across levels means a broken config, not a hard pool.
  4. **Median think tokens per level** — must grow with level. Flat (~500
     everywhere) means truncation, not reasoning.
  5. **Token-cap-hit rate** < 10% of rollouts hitting ``max_tokens`` without
     ``</think>``.

Two v2-specific additions, both inside the probe's stated purpose ("catch
prompt/template faults before bulk generation"):

  * Format compliance is measured with the **token-level** parser
    (:mod:`whetstone.segments`), not a string search, so the probe validates the
    same masks Stages A–C will route on.
  * Two prompt variants are compared in one run — ``sys`` (v1's "put reasoning
    between <think> tags" system prompt) and ``nosys`` (no system message at
    all). Qwen3 thinks natively, so the v1 instruction is at best redundant;
    this is the cheapest place to find out whether it hurts.

Prompts are built by importing ``harvest._build_prompt`` — the probe must
exercise the real harvest prompt path, not a copy of it.

Usage::

    python scripts/calibration_probe.py \\
        --pool /data/whetstone/data/pool/train_30k.jsonl \\
        --out_dir /data/whetstone/runs/calibration_probe \\
        --n 50 --K 2 --temperature 0.9 --max_tokens 32768
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np

from harvest import SYS_PROMPT_V1, _build_prompt  # the REAL harvest path
from whetstone.poolutil import read_jsonl, stratified_sample, write_jsonl
from whetstone.segments import blank_token_ids_for, parse_segments
from whetstone.verify import extract_answer, verify_response

VARIANTS = {
    "sys": True,     # v1 system prompt
    "nosys": False,  # no system message — Qwen3 thinks natively
}


def build_slice(pool: str, n: int, seed: int, out_path: str) -> list[dict]:
    """Stratified slice, v1 Step 2.1 shape but *proportional* per level.

    v1 sampled ``50 // len(buckets)`` per bucket. That is unfillable on this
    pool: DeepMath's difficulty histogram is peaked at 5-8 with 38 rows at
    level 2 and 13 at level 10 (activity 002 note 1). ``stratified_sample``
    preserves the pool's own level distribution instead.
    """
    rows = read_jsonl(pool)
    sample = stratified_sample(rows, lambda r: str(r.get("level", "_")), n,
                               random.Random(seed))
    write_jsonl(out_path, sample)
    return sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--pool", default="/data/whetstone/data/pool/train_30k.jsonl")
    ap.add_argument("--out_dir", default="/data/whetstone/runs/calibration_probe")
    ap.add_argument("--n", type=int, default=50)
    # Defaults MUST mirror the planned P3 seed harvest exactly (P3 Part 1):
    # K=2, T=0.9, top_p=0.95, max_tokens=32768, max_model_len=34816.
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--max_model_len", type=int, default=34816)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="sys,nosys")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    slice_path = os.path.join(args.out_dir, f"calib{args.n}.jsonl")
    probs = (read_jsonl(slice_path) if os.path.exists(slice_path)
             else build_slice(args.pool, args.n, args.seed, slice_path))
    print(f"[slice] {len(probs)} problems -> {slice_path}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    blank = blank_token_ids_for(tok)
    sys_prompt_text = SYS_PROMPT_V1

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, dtype="bfloat16")
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, n=args.K, seed=args.seed)

    report = {"config": vars(args), "variants": {}}
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]

    for variant in wanted:
        use_sys = VARIANTS[variant]
        sysp = sys_prompt_text if use_sys else ""
        prompts = [
            _build_prompt(tok, sysp, r["prompt"],
                          prefill_think=False, enable_thinking=True)
            for r in probs
        ]
        print(f"\n[{variant}] system_prompt={'v1' if use_sys else 'NONE'} "
              f"— generating {len(prompts)}x{args.K}", flush=True)
        outs = llm.generate(prompts, sp)

        recs = []
        for r, o in zip(probs, outs):
            for k, cand in enumerate(o.outputs):
                ids = list(cand.token_ids)
                m = parse_segments(ids, blank_token_ids=blank)
                ans = extract_answer(cand.text)
                ok = verify_response(cand.text, r.get("ground_truth", ""))
                recs.append({
                    "_uid": r["_uid"],
                    "candidate_idx": k,
                    "level": r.get("level"),
                    "g": m.g,
                    "reason": m.reason,
                    "warnings": list(m.warnings),
                    "think_tokens": m.think_len,
                    "answer_tokens": m.answer_len,
                    "finish_reason": cand.finish_reason,
                    "cap_hit_no_close": cand.finish_reason == "length" and m.close_idx < 0,
                    "extracted": ans,
                    "ground_truth": r.get("ground_truth"),
                    "correct": bool(ok),
                    "text_tail": cand.text[-300:],
                })
        write_jsonl(os.path.join(args.out_dir, f"rollouts_{variant}.jsonl"), recs)
        report["variants"][variant] = _metrics(recs)
        _print_metrics(variant, report["variants"][variant])

    with open(os.path.join(args.out_dir, "probe.json"), "w") as f:
        json.dump(report, f, indent=1)

    # --- metric 2: 10 verifier decisions for hand inspection ---------------
    v0 = wanted[0]
    sample_recs = read_jsonl(os.path.join(args.out_dir, f"rollouts_{v0}.jsonl"))
    rng = random.Random(0)
    print("\n" + "=" * 72)
    print(f"METRIC 2 — 10 verifier decisions to hand-check  [{v0}]")
    print("=" * 72)
    for rec in rng.sample(sample_recs, min(10, len(sample_recs))):
        print(f"  correct={str(rec['correct']):<5} extracted={str(rec['extracted'])[:40]!r:<42} "
              f"gold={str(rec['ground_truth'])[:40]!r}")

    print("\n" + "=" * 72)
    print("DECISION RULE (v1 Step 2.1): if any metric fails, fix the prompt or "
          "config and re-run. Do NOT proceed to the P3 seed harvest until all "
          "five are healthy.")
    print("=" * 72)
    print(f"artifacts: {os.path.join(args.out_dir, 'probe.json')}")
    return 0


def _metrics(recs: list[dict]) -> dict:
    n = len(recs)
    levels = sorted({r["level"] for r in recs if r["level"] is not None})

    # 1. format compliance: gate passes AND an answer is extractable
    compliant = [r for r in recs if r["g"] == 1 and r["extracted"] is not None]
    # 5. cap-hit without </think>
    cap_hits = [r for r in recs if r["cap_hit_no_close"]]

    per_level = {}
    for L in levels:
        sel = [r for r in recs if r["level"] == L]
        ok = [r for r in sel if r["correct"]]
        think = [r["think_tokens"] for r in sel if r["g"] == 1]
        per_level[int(L)] = {
            "n": len(sel),
            "pass_rate": len(ok) / len(sel) if sel else None,
            "median_think_tokens": float(np.median(think)) if think else None,
            "median_answer_tokens": float(np.median(
                [r["answer_tokens"] for r in sel if r["g"] == 1])) if think else None,
        }

    gate_fail = {}
    for r in recs:
        if r["g"] == 0:
            gate_fail[r["reason"]] = gate_fail.get(r["reason"], 0) + 1

    med_think = [per_level[L]["median_think_tokens"] for L in sorted(per_level)
                 if per_level[L]["median_think_tokens"] is not None]
    grows = (len(med_think) >= 2 and
             np.corrcoef(np.arange(len(med_think)), med_think)[0, 1] > 0.3)

    return {
        "n_rollouts": n,
        "m1_format_compliance": len(compliant) / n if n else 0.0,
        "m1_pass": (len(compliant) / n if n else 0.0) >= 0.80,
        "m3_pass_rate_overall": sum(1 for r in recs if r["correct"]) / n if n else 0.0,
        "m4_median_think_grows_with_level": bool(grows),
        "m5_cap_hit_rate": len(cap_hits) / n if n else 0.0,
        "m5_pass": (len(cap_hits) / n if n else 0.0) < 0.10,
        "per_level": per_level,
        "gate_fail_reasons": gate_fail,
        "median_think_tokens": float(np.median([r["think_tokens"] for r in recs
                                                if r["g"] == 1])) if compliant else None,
        "median_answer_tokens": float(np.median([r["answer_tokens"] for r in recs
                                                 if r["g"] == 1])) if compliant else None,
    }


def _print_metrics(variant: str, m: dict) -> None:
    print("=" * 72)
    print(f"CALIBRATION PROBE — variant '{variant}'  (n={m['n_rollouts']} rollouts)")
    print("=" * 72)
    print(f"  M1 format compliance   {m['m1_format_compliance']:6.1%}   "
          f"[{'PASS' if m['m1_pass'] else 'FAIL'}]  (need >= 80%)")
    print(f"  M3 pass rate overall   {m['m3_pass_rate_overall']:6.1%}")
    print(f"  M4 think grows w/level {str(m['m4_median_think_grows_with_level']):>6}   "
          f"[{'PASS' if m['m4_median_think_grows_with_level'] else 'CHECK'}]")
    print(f"  M5 cap-hit rate        {m['m5_cap_hit_rate']:6.1%}   "
          f"[{'PASS' if m['m5_pass'] else 'FAIL'}]  (need < 10%)")
    print(f"  median think {m['median_think_tokens']} tok / "
          f"answer {m['median_answer_tokens']} tok   (separate, always)")
    if m["gate_fail_reasons"]:
        print(f"  gate failures: {m['gate_fail_reasons']}")
    print("  M3/M4 per level (want U-shaped pass rate, growing think length):")
    print(f"    {'lvl':>4} {'n':>5} {'pass':>7} {'think':>8} {'answer':>8}")
    for L in sorted(m["per_level"]):
        d = m["per_level"][L]
        pr = f"{d['pass_rate']:.1%}" if d["pass_rate"] is not None else "-"
        print(f"    {L:>4} {d['n']:>5} {pr:>7} "
              f"{str(d['median_think_tokens']):>8} {str(d['median_answer_tokens']):>8}")


if __name__ == "__main__":
    raise SystemExit(main())
