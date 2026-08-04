"""Stage-A problem subset: 2,000 GSM8K + 2,000 DeepMath (packet P5 Part 0).

The 4,000 problems the frozen 32B teacher will ghostwrite for. Two shaping
rules, both deliberate:

**Level floors, not proportional stratification.** The pool's level histogram
is peaked at 5–8 and nearly empty at 2–3 and 10 (activity 002). A purely
proportional 2,000-problem draw leaves level 9 at ~105 — too thin to support any
hard-tier claim the write-up wants to make. So levels 2 and 10 are taken whole,
level 3 gets a floor of 100, level 9 a floor of 250, and levels 4–8 split the
remainder proportionally. The shape is documented rather than discovered: a
reader of the corpus stats must be able to see that the hard band was
oversampled on purpose.

**Trace preference inside every stratum.** A problem with a verified verbose
trace can be generated under ``gold+trace`` conditioning — the teacher gets the
student's own reasoning to compress, which is what activity 005's compression
setup measured. A problem without one falls back to ``gold``. Neither is wrong,
but the first is strictly more informative, so traced problems are drawn first
and the achieved fraction is reported. It is *not* a filter: filtering on trace
availability would bias the subset toward whatever the K=2 seed harvest happened
to solve, which correlates with difficulty.

**Trace selection is deterministic: the SHORTEST verified candidate.** Up to two
verified traces exist per uid. The short one is preferred because the 12,288-token
conditioning cap (packet §5) is what decides ``gold+trace`` vs ``gold``, and a
longer trace both throttles concurrency and is likelier to fall back. Whichever
one is chosen is pinned here by ``trace_candidate_idx`` and carried in the record,
because the structural gate scores the compact rewrite *against its own source*
— generation and annotation must see the same trace or the annotation is
meaningless.

Think lengths come from :func:`whetstone.segments.parse_segments` over the
stored ``completion_token_ids``, never from re-tokenizing decoded text (design
§12.1 — a decoded split does not round-trip at the ``<think>`` boundary).

Usage::

    python scripts/select_stagea_subset.py \\
        --pool  /data/whetstone/data/pool/train_30k.jsonl \\
        --seed_verified /data/whetstone/corpora/seed/seed_verified.jsonl \\
        --outdir /data/whetstone/corpora/stagea
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl, write_meta
from whetstone.segments import parse_segments

#: DeepMath level floors (packet §4 step 2). Levels not listed here split the
#: remainder proportionally to their pool counts.
FLOORS = {2: None, 10: None, 3: 100, 9: 250}   # None = "take the whole stratum"
PROPORTIONAL_LEVELS = (4, 5, 6, 7, 8)


def load_traces(path: str) -> dict[str, dict]:
    """``{uid: {verbose_think, verbose_think_tokens, trace_candidate_idx}}``.

    Keeps the shortest verified candidate per uid (see module docstring). Think
    length is the segment parser's ``think_len`` over the stored token ids.
    """
    best: dict[str, dict] = {}
    n_read = n_gate = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_read += 1
            ids = r.get("completion_token_ids") or []
            if not ids:
                continue
            masks = parse_segments(ids, prompt_len=0)
            if masks.g != 1:
                n_gate += 1
                continue
            text = r.get("completion", "")
            if "<think>" not in text or "</think>" not in text:
                n_gate += 1
                continue
            think = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
            if not think:
                n_gate += 1
                continue
            cand = {
                "verbose_think": think,
                "verbose_think_tokens": masks.think_len,
                "trace_candidate_idx": r.get("candidate_idx"),
            }
            prev = best.get(r["_uid"])
            if prev is None or cand["verbose_think_tokens"] < prev["verbose_think_tokens"]:
                best[r["_uid"]] = cand
    print(f"[traces] {n_read} verified records -> {len(best)} uids "
          f"({n_gate} skipped: gate or missing think)")
    return best


def largest_remainder(counts: dict[int, int], total: int) -> dict[int, int]:
    """Apportion ``total`` across ``counts`` proportionally, largest remainder."""
    pool_total = sum(counts.values())
    if pool_total == 0:
        return {k: 0 for k in counts}
    exact = {k: total * v / pool_total for k, v in counts.items()}
    out = {k: int(v) for k, v in exact.items()}
    short = total - sum(out.values())
    for k in sorted(counts, key=lambda k: (-(exact[k] - out[k]), k))[:short]:
        out[k] += 1
    return out


def draw(rows: list[dict], n: int, traced: set[str], rng: random.Random) -> list[dict]:
    """Take ``n`` rows, traced ones first, shuffled within each group."""
    have = [r for r in rows if r["_uid"] in traced]
    lack = [r for r in rows if r["_uid"] not in traced]
    rng.shuffle(have)
    rng.shuffle(lack)
    return (have + lack)[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="/data/whetstone/data/pool/train_30k.jsonl")
    ap.add_argument("--seed_verified",
                    default="/data/whetstone/corpora/seed/seed_verified.jsonl")
    ap.add_argument("--outdir", default="/data/whetstone/corpora/stagea")
    ap.add_argument("--n_gsm8k", type=int, default=2000)
    ap.add_argument("--n_deepmath", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_trace_tokens", type=int, default=12288,
                    help="Traces longer than this are carried but flagged: the "
                         "generator falls back to gold-only conditioning "
                         "(packet §5). Reported here so the fallback rate is "
                         "known before the run, not discovered during it.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = read_jsonl(args.pool)
    traces = load_traces(args.seed_verified)
    traced = set(traces)

    gsm = [r for r in pool if r["source"] == "gsm8k"]
    deep = [r for r in pool if r["source"] == "deepmath"]
    print(f"[pool] {len(pool)} rows: {len(gsm)} gsm8k, {len(deep)} deepmath")

    # --- GSM8K: all level 1, traced first -------------------------------
    picked = draw(gsm, args.n_gsm8k, traced, rng)
    if len(picked) < args.n_gsm8k:
        raise SystemExit(f"only {len(picked)} gsm8k rows for n={args.n_gsm8k}")

    # --- DeepMath: floors, then proportional remainder ------------------
    by_level: dict[int, list[dict]] = defaultdict(list)
    for r in deep:
        by_level[r["level"]].append(r)

    targets: dict[int, int] = {}
    for lv, floor in FLOORS.items():
        avail = len(by_level.get(lv, []))
        targets[lv] = avail if floor is None else min(floor, avail)
    remainder = args.n_deepmath - sum(targets.values())
    if remainder < 0:
        raise SystemExit(f"floors ({sum(targets.values())}) exceed n_deepmath")
    prop_counts = {lv: len(by_level.get(lv, [])) for lv in PROPORTIONAL_LEVELS}
    targets.update(largest_remainder(prop_counts, remainder))

    for lv in sorted(targets):
        avail = len(by_level.get(lv, []))
        if targets[lv] > avail:
            raise SystemExit(f"level {lv}: want {targets[lv]}, pool has {avail}")
        picked += draw(by_level[lv], targets[lv], traced, rng)

    # --- emit -----------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for r in picked:
        t = traces.get(r["_uid"])
        long_trace = bool(t) and t["verbose_think_tokens"] > args.max_trace_tokens
        rec = {
            "_uid": r["_uid"],
            "prompt": r["prompt"],
            "ground_truth": r["ground_truth"],
            "level": r["level"],
            "source": r["source"],
            "has_trace": bool(t),
            "conditioned_on": "gold+trace" if (t and not long_trace) else "gold",
            "trace_fallback_reason": (
                None if (t and not long_trace)
                else ("trace_too_long" if long_trace else "no_trace")),
        }
        if t:
            rec.update(t)
        rows.append(rec)
    rng.shuffle(rows)          # packet §11: level-clustered order makes an
                               # interrupted run unrepresentative (005 f2)

    jsonl = os.path.join(args.outdir, "subset_stagea.jsonl")
    with open(jsonl, "w") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    uids_path = os.path.join(args.outdir, "subset_stagea_uids.json")
    with open(uids_path, "w") as fh:
        json.dump([r["_uid"] for r in rows], fh, indent=1)

    # --- report ---------------------------------------------------------
    cond = Counter(r["conditioned_on"] for r in rows)
    n = len(rows)
    print(f"\n[subset] {n} problems -> {jsonl}")
    print(f"[subset] uids (resume contract) -> {uids_path}")
    print(f"\n  {'level':>5} {'source':>9} {'n':>5} {'gold+trace':>11} {'frac':>6}")
    table = []
    for (src, lv), grp in sorted(
            ((k, list(g)) for k, g in
             _groupby(rows, lambda r: (r["source"], r["level"]))),
            key=lambda kv: (kv[0][0], kv[0][1])):
        nt = sum(r["conditioned_on"] == "gold+trace" for r in grp)
        print(f"  {lv:>5} {src:>9} {len(grp):>5} {nt:>11} {nt/len(grp):>6.1%}")
        table.append({"source": src, "level": lv, "n": len(grp), "gold_trace": nt})
    print(f"\n  conditioning: {dict(cond)}  "
          f"gold+trace = {cond['gold+trace']/n:.1%}")
    fb = Counter(r["trace_fallback_reason"] for r in rows if r["trace_fallback_reason"])
    print(f"  fallback reasons: {dict(fb)}")
    tt = [r["verbose_think_tokens"] for r in rows if r.get("verbose_think_tokens")]
    if tt:
        tt.sort()
        print(f"  verbose think tokens (traced): median {tt[len(tt)//2]}, "
              f"p95 {tt[int(.95*len(tt))]}, max {tt[-1]}")
    print(f"\n  drafts to generate: {n} x K  ({n*8} at K=8)")

    write_meta(jsonl, {
        "builder": "scripts/select_stagea_subset.py",
        "packet": "P5 Part 0",
        "seed": args.seed,
        "pool": args.pool,
        "seed_verified": args.seed_verified,
        "n": n,
        "n_gsm8k": args.n_gsm8k,
        "n_deepmath": args.n_deepmath,
        "deepmath_level_targets": {str(k): v for k, v in sorted(targets.items())},
        "floors": {str(k): v for k, v in FLOORS.items()},
        "proportional_levels": list(PROPORTIONAL_LEVELS),
        "max_trace_tokens": args.max_trace_tokens,
        "conditioning": dict(cond),
        "gold_trace_frac": round(cond["gold+trace"] / n, 4),
        "fallback_reasons": dict(fb),
        "by_source_level": table,
        "trace_rule": "shortest verified candidate per uid, pinned by trace_candidate_idx",
        "note": ("Row order is shuffled: an interrupted run must still be a "
                 "representative slice (activity 005 finding 2). "
                 "subset_stagea_uids.json is the resume contract."),
    })
    return 0


def _groupby(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out.items()


if __name__ == "__main__":
    raise SystemExit(main())
