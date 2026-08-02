"""M3 — style-tax size and concentration under the frozen scorer (packet P3a).

The decisive bake-off metric. A compact register costs the frozen checkpoint
π_0 some extra surprisal on the think tokens; Round 0's inoculation loss exists
to mask exactly that cost (design §2). Two properties decide whether Round 0
has anything to work with:

  * **size** — mean and p95 surprisal elevation of a compact corpus's think
    tokens over the same model's surprisal on the *verbose* originals. If the
    clean-register p95 gap approaches τ_leap scale (≈4 nats) the register is
    indistinguishable from a genuine reasoning leap, which resurrects Risk 1.
  * **concentration** — how much of the total excess surprisal a handful of
    token types carry. Concentrated → the inoculation mask is a clean handle.
    Diffuse → there is nothing to mask and Round 0 degenerates toward full SFT,
    with its overtraining/infection risk.

Scoring is design §12.2's single teacher-forced prefill pass: `prompt_logprobs
= 2` returns both the actual token's logprob and the rank-1 token's logprob at
every position, giving surprisal `-logp(actual)` and the gap
`d_t = logp(top1) - logp(actual)` (>= 0 by construction) in one pass.

Type aggregation follows design §12.3 verbatim: per token-id collect surprisal
across occurrences; R = {types: mean surprisal > 75th percentile AND
across-occurrence std < median} ∪ structural whitelist, with a minimum
occurrence count so a type seen twice cannot enter on noise.

Run on **spark** (prefill-only — what the GB10 is for)::

    VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/style_tax.py \\
        --control /data/whetstone/runs/register_bakeoff/subset.jsonl \\
        --corpora A=/…/bakeoff_A.jsonl B=/…/bakeoff_B.jsonl \\
        --out /data/whetstone/runs/register_bakeoff/style_tax.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from whetstone.poolutil import read_jsonl
from whetstone.segments import blank_token_ids_for, parse_segments

MIN_OCCURRENCES = 10       # packet P3a gotcha 3: at n=50 traces this matters
TOP_N = 20


def _completion_for_control(r: dict) -> str:
    """Rebuild a full rollout from the verbose original, same shape as a compact
    record's `completion`, so both are scored through an identical code path."""
    return f"<think>\n{r['verbose_think']}\n</think>\n\n{r['answer']}"


def score_corpus(llm, tok, rows, blank, label, max_len):
    """Teacher-force each (prompt, completion); return think-token surprisal + d_t."""
    from vllm import SamplingParams

    prompts, metas = [], []
    for r in rows:
        p_text = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        p_ids = tok(p_text, add_special_tokens=False).input_ids
        c_ids = tok(r["completion"], add_special_tokens=False).input_ids
        full = list(p_ids) + list(c_ids)
        if len(full) > max_len:
            continue
        prompts.append({"prompt_token_ids": full})
        metas.append((r["_uid"], full, len(p_ids)))

    outs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=2,
                                                temperature=0.0))

    surp, gaps, types, n_gate_fail = [], [], [], 0
    for (uid, full, p_len), o in zip(metas, outs):
        m = parse_segments(full, prompt_len=p_len, blank_token_ids=blank)
        if m.g != 1:
            n_gate_fail += 1
            continue
        pl = o.prompt_logprobs
        for t in range(p_len, len(full)):
            if not m.think_mask[t]:
                continue
            entry = pl[t]
            if entry is None or full[t] not in entry:
                continue
            actual = entry[full[t]]
            top1 = next(e for e in entry.values() if e.rank == 1)
            surp.append(-actual.logprob)
            gaps.append(top1.logprob - actual.logprob)
            types.append(full[t])
    print(f"  [{label}] {len(surp):,} think tokens scored "
          f"({n_gate_fail} traces gate-failed)", flush=True)
    return (np.asarray(surp, dtype=np.float64),
            np.asarray(gaps, dtype=np.float64),
            np.asarray(types, dtype=np.int64))


def type_table(surp, types, tok):
    """Per-token-type surprisal stats (design §12.3 R-recipe inputs)."""
    order = np.argsort(types, kind="stable")
    ts, ss = types[order], surp[order]
    uniq, starts = np.unique(ts, return_index=True)
    bounds = list(starts) + [len(ts)]
    out = []
    for tid, a, b in zip(uniq, bounds[:-1], bounds[1:]):
        block = ss[a:b]
        if block.size < MIN_OCCURRENCES:
            continue
        out.append({
            "token_id": int(tid),
            "token": tok.decode([int(tid)]),
            "count": int(block.size),
            "mean_surprisal": float(block.mean()),
            "std_surprisal": float(block.std()),
        })
    return out


def analyse(name, surp, gaps, types, tok, baseline_mean):
    tt = type_table(surp, types, tok)
    means = np.array([t["mean_surprisal"] for t in tt])
    stds = np.array([t["std_surprisal"] for t in tt])
    proto_r = []
    if tt.__len__():
        hi, lo = np.percentile(means, 75), np.median(stds)
        proto_r = [t for t in tt if t["mean_surprisal"] > hi and t["std_surprisal"] < lo]

    # Excess surprisal relative to the verbose control's mean think token.
    excess = np.clip(surp - baseline_mean, 0.0, None)
    total_excess = float(excess.sum())
    per_type: dict[int, float] = {}
    for tid, e in zip(types, excess):
        per_type[int(tid)] = per_type.get(int(tid), 0.0) + float(e)
    ranked = sorted(per_type.items(), key=lambda kv: -kv[1])
    top = [{"token": tok.decode([tid]), "token_id": tid,
            "excess_nats": e, "share": e / total_excess if total_excess else 0.0}
           for tid, e in ranked[:TOP_N]]
    top20_share = sum(t["share"] for t in top)

    return {
        "name": name,
        "n_think_tokens": int(surp.size),
        "surprisal": {
            "mean": float(surp.mean()), "median": float(np.median(surp)),
            "p95": float(np.percentile(surp, 95)), "p99": float(np.percentile(surp, 99)),
        },
        "d_t_gap": {
            "mean": float(gaps.mean()), "median": float(np.median(gaps)),
            "p95": float(np.percentile(gaps, 95)), "max": float(gaps.max()),
        },
        "n_types_over_min_occ": len(tt),
        "proto_R_size": len(proto_r),
        "proto_R_sample": [t["token"] for t in proto_r[:40]],
        "total_excess_nats": total_excess,
        "excess_per_think_token": total_excess / max(1, surp.size),
        "top20_excess_share": top20_share,
        "top_types": top,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, help="subset.jsonl (verbose originals)")
    ap.add_argument("--corpora", nargs="+", required=True, help="LABEL=path …")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--max_len", type=int, default=24576)
    ap.add_argument("--gpu_mem", type=float, default=0.60)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(args.model)
    blank = blank_token_ids_for(tok)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_len,
              gpu_memory_utilization=args.gpu_mem, enforce_eager=True)

    control_rows = read_jsonl(args.control)
    for r in control_rows:
        r["completion"] = _completion_for_control(r)
    print("[score] control (verbose originals)", flush=True)
    c_s, c_g, c_t = score_corpus(llm, tok, control_rows, blank, "control", args.max_len)
    baseline = float(c_s.mean())

    report = {"model": args.model, "min_occurrences": MIN_OCCURRENCES,
              "baseline_mean_surprisal": baseline,
              "control": analyse("control(verbose)", c_s, c_g, c_t, tok, baseline),
              "arms": {}}

    for spec in args.corpora:
        label, path = spec.split("=", 1)
        print(f"[score] arm {label}: {path}", flush=True)
        s, g, t = score_corpus(llm, tok, read_jsonl(path), blank, label, args.max_len)
        report["arms"][label] = analyse(label, s, g, t, tok, baseline)

    # ---- print -------------------------------------------------------------
    def line(name, fn):
        print(f"{name:<30}{fn(report['control']):>14}", end="")
        for a in report["arms"].values():
            print(f"{fn(a):>14}", end="")
        print()

    print()
    print(f"{'':<30}{'verbose ctrl':>14}", end="")
    for label in report["arms"]:
        print(f"{('arm ' + label):>14}", end="")
    print()
    line("think tokens scored", lambda a: f"{a['n_think_tokens']:,}")
    line("mean surprisal (nats)", lambda a: f"{a['surprisal']['mean']:.3f}")
    line("p95 surprisal", lambda a: f"{a['surprisal']['p95']:.3f}")
    line("mean d_t gap", lambda a: f"{a['d_t_gap']['mean']:.3f}")
    line("p95 d_t gap", lambda a: f"{a['d_t_gap']['p95']:.3f}")
    line("excess nats / think token", lambda a: f"{a['excess_per_think_token']:.3f}")
    line("types >= 10 occurrences", lambda a: f"{a['n_types_over_min_occ']:,}")
    line("proto-R size", lambda a: f"{a['proto_R_size']}")
    line("top-20 excess share", lambda a: f"{a['top20_excess_share']:.1%}")

    for label, a in report["arms"].items():
        print(f"\ntop-10 excess-surprisal types, arm {label}:")
        for t in a["top_types"][:10]:
            print(f"   {t['token']!r:<16} {t['excess_nats']:>10.0f} nats  "
                  f"{t['share']:>6.1%}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
