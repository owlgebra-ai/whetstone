"""§3.6 — Sufficiency gate (Δlogp filter).

Runs *after* Stage 2 compression and *before* Stage 3 SFT. Rejects
compressions that drop too much information by checking whether the
compressed trace improves the base model's likelihood of the gold answer:

    delta = log P_base(a* | q, compact_trace) − log P_base(a* | q)
    keep iff delta > 0

Expected pass rate ≈ 70%. The ~30% that fail concentrate on problems with
critical case-splits or verification steps that the compact register elides.
Reject rather than keep — installing them in SFT data installs sloppy reasoning.

Gates Stage 3 entry, not Stage 2 internally, so the raw compression artifacts
remain available for diagnosis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _score(model, tokenizer, prompt: str, completion: str,
           device: str, max_length: int) -> float:
    """Return sum log P(completion | prompt) under the model."""
    pp_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    pc_ids = tokenizer(prompt + completion, return_tensors="pt",
                       add_special_tokens=False).input_ids
    if pc_ids.size(1) > max_length:
        return float("-inf")
    pc_ids = pc_ids.to(device)
    prompt_len = pp_ids.size(1)

    with torch.no_grad():
        logits = model(pc_ids).logits[:, :-1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        targets = pc_ids[:, 1:]
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    completion_log_probs = token_log_probs[0, prompt_len - 1 :]
    return completion_log_probs.sum().item()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="WHETSTONE §3.6 Δlogp sufficiency gate")
    ap.add_argument("--input", required=True, help="Stage 2 compactB.jsonl")
    ap.add_argument("--output", required=True, help="Gated output (keep iff delta > 0)")
    ap.add_argument("--model", required=True, help="BASE model id or path")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep-only", action="store_true",
                    help="If set, write only passing rows; else annotate all rows")
    return ap.parse_args(argv)


def _scan_seen(output: str) -> set[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    if not os.path.exists(output):
        return seen
    with open(output) as f:
        for line in f:
            try:
                r = json.loads(line)
                seen.add((r["_uid"], r.get("src_candidate_idx", 0)))
            except json.JSONDecodeError:
                continue
    return seen


def main(argv=None):
    args = parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(args.device).eval()

    seen = _scan_seen(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "a", buffering=1)

    n_total = n_pass = 0
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            key = (r["_uid"], r.get("src_candidate_idx", 0))
            if key in seen:
                continue

            q = r.get("prompt", "")
            compact = r.get("compact", "")
            gold = r.get("ground_truth", "")
            if not compact or not gold:
                continue

            lp_with = _score(model, tokenizer,
                             prompt=f"{q}\n\nReasoning:\n{compact}\n\nFinal answer: ",
                             completion=gold, device=args.device, max_length=args.max_length)
            lp_without = _score(model, tokenizer,
                                prompt=f"{q}\n\nFinal answer: ",
                                completion=gold, device=args.device,
                                max_length=args.max_length)

            if lp_with == float("-inf") or lp_without == float("-inf"):
                delta, verdict = float("-inf"), "fail"
            else:
                delta = lp_with - lp_without
                verdict = "pass" if delta > 0 else "fail"

            rec = dict(r)
            rec["delta_logp"] = delta
            rec["logp_with_compact"] = lp_with
            rec["logp_without_compact"] = lp_without
            rec["sufficiency_verdict"] = verdict
            if verdict == "pass" or not args.keep_only:
                out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

            n_total += 1
            n_pass += int(verdict == "pass")
            if n_total % 50 == 0:
                rate = n_pass / n_total
                print(f"[Δlogp] {n_total} scored, pass rate so far = {rate:.2%}",
                      flush=True)

    out_f.close()
    rate = n_pass / max(1, n_total)
    print(f"[Δlogp] DONE. {n_pass}/{n_total} = {rate:.2%} pass", flush=True)
    if rate < 0.5:
        print("[Δlogp] WARN: pass rate < 50%. Re-inspect Stage 2 compression prompt.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
