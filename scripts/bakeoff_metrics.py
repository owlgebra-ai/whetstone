"""M1 + register-adoption diagnostics for the register bake-off (packet P3a).

M1 is compression measured in **tokens, not chars** (packet P3a step 2): the
think-segment token count per trace via `whetstone.segments` masks over the
compact completion, reported as median + IQR against the verbose originals,
plus the fraction of traces under `B_target = 600` (design §12.6).

Two diagnostics are added because the chunk-alignment check found them, and
neither is visible in a compression ratio:

  * **register adoption** — occurrences per 100 think tokens of each card's own
    connective set. A card that the model does not actually follow is not being
    measured by the other metrics; it is being measured by this one.
  * **stalled-chunk rate** — fraction of compact chunks that are a
    near-duplicate of the preceding compact chunk. Cumulative-context chunkwise
    compression has a repetition attractor: once two consecutive compacts
    agree, the model copies forever. A trace that stalls has a flattering
    compression ratio and worthless content, so this is read *before* M1.

Usage::

    python scripts/bakeoff_metrics.py \\
        --subset /data/whetstone/runs/register_bakeoff/subset.jsonl \\
        --arms A=/…/bakeoff_A.jsonl B=/…/bakeoff_B.jsonl \\
        --out /data/whetstone/runs/register_bakeoff/m1_metrics.json
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

B_TARGET = 600

# Card §2 structural whitelists, verbatim. Counted as whole tokens/words so
# that e.g. "no" inside "not" or a stray semicolon in prose is not scored.
MARKERS = {
    "A": ["⇒", "→", "✓", "✗", "!", "?", ";", "chk:", "case ", "goal:", "let ", "sub "],
    "B": ["so ", "ok", "no.", "note ", "find ", "check:", "case ", "goal:", "let ", "sub "],
}


def _q(x: list[float]) -> dict:
    a = np.asarray(x, dtype=float)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _stalled(compacts: list[str]) -> float:
    """Fraction of chunks that duplicate the previous chunk's first 120 chars."""
    if len(compacts) < 2:
        return 0.0
    dup = sum(1 for a, b in zip(compacts, compacts[1:])
              if a[:120].strip() and a[:120].strip() == b[:120].strip())
    return dup / (len(compacts) - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="LABEL=path …")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    blank = blank_token_ids_for(tok)

    subset = read_jsonl(args.subset)
    report = {"verbose": _q([r["verbose_think_tokens"] for r in subset]), "arms": {}}
    report["verbose"]["source"] = args.subset

    for spec in args.arms:
        label, path = spec.split("=", 1)
        rows = read_jsonl(path)
        think_tokens, ratios, adoption, stalls, answer_tokens = [], [], [], [], []
        gate_fail = []
        for r in rows:
            ids = tok(r["completion"], add_special_tokens=False).input_ids
            m = parse_segments(ids, blank_token_ids=blank)
            if m.g != 1:
                gate_fail.append((r["_uid"], m.reason))
                continue
            think_tokens.append(m.think_len)
            answer_tokens.append(m.answer_len)
            ratios.append(m.think_len / max(1, r["verbose_think_tokens"]))
            body = r["compact_think"]
            hits = sum(body.count(s) for s in MARKERS[label]) if label in MARKERS else 0
            adoption.append(100.0 * hits / max(1, m.think_len))
            stalls.append(_stalled(r["compacts_per_chunk"]))

        meta_path = path + ".meta.json"
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        report["arms"][label] = {
            "path": path,
            "n": len(rows),
            "card_path": meta.get("card_path"),
            "card_git_sha": meta.get("card_git_sha"),
            "rendered_prompt_sha1": meta.get("rendered_prompt_sha1"),
            "rendered_prompt_chars": meta.get("rendered_prompt_chars"),
            "verify_pass": sum(1 for r in rows if r["verify_ok"]),
            "gate_fail": gate_fail,
            "think_tokens": _q(think_tokens),
            "answer_tokens": _q(answer_tokens),
            "compression_ratio": _q(ratios),
            "frac_under_B_target": float(np.mean([t < B_TARGET for t in think_tokens])),
            "markers_per_100_think_tokens": _q(adoption),
            "stalled_chunk_rate": _q(stalls),
            "frac_traces_mostly_stalled": float(np.mean([s >= 0.5 for s in stalls])),
        }

    v = report["verbose"]
    print(f"{'':<26}{'verbose':>12}", end="")
    for label in report["arms"]:
        print(f"{('arm ' + label):>12}", end="")
    print()

    def row(name, fn_v, fn_a):
        print(f"{name:<26}{fn_v(v):>12}", end="")
        for a in report["arms"].values():
            print(f"{fn_a(a):>12}", end="")
        print()

    row("think tokens median", lambda v: f"{v['median']:.0f}",
        lambda a: f"{a['think_tokens']['median']:.0f}")
    row("  IQR", lambda v: f"{v['p25']:.0f}-{v['p75']:.0f}",
        lambda a: f"{a['think_tokens']['p25']:.0f}-{a['think_tokens']['p75']:.0f}")
    row("compression ratio med", lambda v: "1.000",
        lambda a: f"{a['compression_ratio']['median']:.3f}")
    row("% under B_target=600", lambda v: f"{np.mean([r['verbose_think_tokens'] < B_TARGET for r in subset]):.0%}",
        lambda a: f"{a['frac_under_B_target']:.0%}")
    row("markers /100 think tok", lambda v: "-",
        lambda a: f"{a['markers_per_100_think_tokens']['median']:.2f}")
    row("stalled-chunk rate med", lambda v: "-",
        lambda a: f"{a['stalled_chunk_rate']['median']:.2f}")
    row("% traces >=50% stalled", lambda v: "-",
        lambda a: f"{a['frac_traces_mostly_stalled']:.0%}")
    row("verify_response pass", lambda v: f"{len(subset)}", lambda a: f"{a['verify_pass']}/{a['n']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
