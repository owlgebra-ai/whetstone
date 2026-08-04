"""Rolling GLM faithfulness audit over the Stage-A corpus (packet P5 Part 4).

Every 50 problems that finish selection, 10 kept drafts go to the external judge.
Its verdicts land in ``audit_rolling.jsonl`` and the dashboard and **nowhere
else** — no judge output ever enters a corpus record. That boundary is the
central-model principle (v1 §3): the judge reads and scores, it does not shape
training data. A judge verdict that became a filter would make an external model
a silent co-author of the corpus.

**Why a rolling audit rather than one pass at the end.** The failure this catches
— a prompt that makes the teacher assert the gold instead of deriving it — is
invisible to every automatic check in the pipeline. ``verify_response`` grades
the answer segment, which is correct by construction when the answer is in
context; G_spike measures followability, and an asserted answer is *highly*
followable. So a corpus can be 100% verified, low-spike, on-register and still
worthless. Catching that on day 1 instead of day 2 is the entire value.

**Pause rule** (provisional until the Part-5 checkpoint pins it): over the
trailing 200 judgments, ``faithful < 55%`` or ``wrong > 15%`` writes a ``PAUSE``
flag next to the log and shouts. It does not kill the generator — stopping a
24-hour run is a decision with a person attached to it, and a judge endpoint
having a bad afternoon is a likelier cause of one bad window than the teacher
suddenly breaking.

Calibration context: the 1.7B self-compressions judged **40% faithful / 21%
wrong** (activity 005 f13). The 32B *with gold in hand* must be far better; if it
is not, the prompt is broken.

Only drafts with a verbose source are eligible — with nothing to compare
against, the rubric has no question to answer.

Usage::

    export FAITHFULNESS_BASE_URL=https://api.z.ai/api/anthropic
    export FAITHFULNESS_AUTH_TOKEN=...
    python scripts/stagea_audit_loop.py --follow
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_selected(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def judged_uids(path: str) -> set:
    out = set()
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            try:
                out.add(json.loads(line)["_uid"])
            except Exception:                                  # noqa: BLE001
                continue
    return out


def sample_round(selected: list[dict], done: set, n: int,
                 rng: random.Random) -> list[dict]:
    """``n`` kept drafts, stratified by level, from problems not yet judged.

    Stratified rather than uniform because the level histogram is peaked at 5–8:
    a uniform draw would put almost no judgments on the hard band, which is
    exactly where an asserted-answer failure is most likely and most costly.
    """
    pool = [r for r in selected
            if r["_uid"] not in done and (r.get("verbose_think") or "").strip()]
    if not pool:
        return []
    by_level: dict = defaultdict(list)
    for r in pool:
        by_level[r.get("level")].append(r)
    # One per level, round-robin, until n — keeps thin strata represented.
    for v in by_level.values():
        rng.shuffle(v)
    out, levels = [], sorted(by_level, key=lambda x: (x is None, x))
    while len(out) < n and any(by_level.values()):
        for lv in levels:
            if by_level[lv] and len(out) < n:
                out.append(by_level[lv].pop())
    return out


def _load_verdicts(path: str) -> list[dict]:
    rows = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:                              # noqa: BLE001
                    continue
                if r.get("parsed") and r.get("verdict"):
                    rows.append(r)
    return rows


def _rates(rows: list[dict]) -> tuple[int, float, float]:
    if not rows:
        return 0, 0.0, 0.0
    c = Counter(r["verdict"] for r in rows)
    return len(rows), c["faithful"] / len(rows), c["wrong"] / len(rows)


def trailing_rates(path: str, window: int) -> tuple[int, float, float]:
    """(n, faithful_frac, wrong_frac) over the trailing ``window`` judgments."""
    return _rates(_load_verdicts(path)[-window:])


def band_rates(path: str, window: int, hard_from: int = 6) -> dict:
    """Aggregate **and** hard-band rates over the trailing window.

    The aggregate alone is not a stable quantity here and must not be the alarm
    (activity 008). Level-1 GSM8K audits at ~95% faithful and levels 8–9 at
    ~40%, so the aggregate moves with whatever mix the sampler happened to draw:
    measured over this run, the trailing-200 composition went from 58% level-1
    to 17% level-1 as the corpus grew, and the aggregate fell 88% → 66% with no
    established per-band change behind it. An alarm on that number reports the
    sampler, not the teacher.
    """
    rows = _load_verdicts(path)[-window:]
    hard = [r for r in rows if (r.get("level") or 0) >= hard_from]
    n, f, w = _rates(rows)
    hn, hf, hw = _rates(hard)
    return {"n": n, "faithful": f, "wrong": w,
            "hard_n": hn, "hard_faithful": hf, "hard_wrong": hw}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected",
                    default="/data/whetstone/corpora/stagea_selected/selected.jsonl")
    ap.add_argument("--output",
                    default="/data/whetstone/runs/stagea/audit_rolling.jsonl")
    ap.add_argument("--scratch", default="/data/whetstone/runs/stagea/_audit_round.jsonl")
    ap.add_argument("--every", type=int, default=50,
                    help="problems per audit round")
    ap.add_argument("--n", type=int, default=10,
                    help="judgments per --every problems (a cumulative rate)")
    ap.add_argument("--max_per_cycle", type=int, default=40,
                    help="ceiling on judgments in one invocation, so a long "
                         "gap cannot turn into an unbounded API burst")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=200,
                    help="trailing judgments the pause rule looks at")
    # PINNED at the Part-5 calibration checkpoint (activity 008). The alarm is
    # per band: the aggregate pair alone fires on sample composition.
    ap.add_argument("--min_faithful", type=float, default=0.70)
    ap.add_argument("--max_wrong", type=float, default=0.15)
    ap.add_argument("--hard_from", type=int, default=6,
                    help="levels >= this form the hard band")
    ap.add_argument("--hard_min_faithful", type=float, default=0.45)
    ap.add_argument("--hard_max_wrong", type=float, default=0.35)
    ap.add_argument("--require_hard_breach", action="store_true", default=True,
                    help="only raise PAUSE when the HARD BAND breaches. The "
                         "aggregate is still reported and still logged, but it "
                         "moves with sampler composition, so on its own it is a "
                         "false alarm generator (activity 008).")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--poll_s", type=int, default=300)
    args = ap.parse_args()

    if not os.environ.get("FAITHFULNESS_AUTH_TOKEN"):
        raise SystemExit(
            "[audit] FAITHFULNESS_AUTH_TOKEN is not set. Export it (and "
            "FAITHFULNESS_BASE_URL) in the shell that runs this — the token is "
            "never read from a file or written to one.")

    rng = random.Random(args.seed)
    pause_flag = args.output + ".PAUSE"
    last_mark = 0

    while True:
        selected = load_selected(args.selected)
        n_problems = len({r["_uid"] for r in selected})
        mark = n_problems // args.every
        if mark <= last_mark:
            if not args.follow:
                break
            time.sleep(args.poll_s)
            continue

        done = judged_uids(args.output)
        # "10 judgments per 50 problems" is a CUMULATIVE rate, not a fixed batch
        # per invocation. A fixed batch silently under-audits whenever the poll
        # interval is slower than generation: at 48 drafts/min the corpus grows
        # ~90 problems per 15-minute cycle, which the packet's cadence says is
        # 18 judgments, so a flat 10 runs at 55% of the mandated rate and lands
        # near 570 of the 800 the packet asks for. Judge up to the shortfall
        # instead, capped so one cycle cannot become an unbounded API burst.
        target = mark * args.n
        want = max(0, min(target - len(done), args.max_per_cycle))
        if want == 0:
            last_mark = mark
            if not args.follow:
                break
            time.sleep(args.poll_s)
            continue
        batch = sample_round(selected, done, want, rng)
        if not batch:
            last_mark = mark
            if not args.follow:
                break
            time.sleep(args.poll_s)
            continue

        os.makedirs(os.path.dirname(os.path.abspath(args.scratch)), exist_ok=True)
        with open(args.scratch, "w") as fh:
            for r in batch:
                fh.write(json.dumps({
                    "_uid": r["_uid"], "level": r.get("level"),
                    "prompt": r.get("prompt", ""),
                    "verbose_think": r.get("verbose_think", ""),
                    "compact_think": r.get("compact_think", ""),
                }, ensure_ascii=False) + "\n")

        print(f"[round {mark}] {n_problems} problems selected, "
              f"{len(done)}/{target} judged so far, judging "
              f"{len(batch)} (levels "
              f"{sorted(Counter(r.get('level') for r in batch).items(), key=str)})",
              flush=True)
        rc = subprocess.run(
            [sys.executable, os.path.join(REPO, "scripts", "faithfulness_audit.py"),
             "--corpus", args.scratch, "--output", args.output,
             "--n", str(len(batch)), "--model", args.model,
             "--concurrency", str(args.concurrency)],
            cwd=REPO).returncode
        if rc != 0:
            print(f"[round {mark}] judge exited {rc} — not advancing the mark, "
                  "will retry next poll", flush=True)
            if not args.follow:
                return rc
            time.sleep(args.poll_s)
            continue

        last_mark = mark
        br = band_rates(args.output, args.window, args.hard_from)
        print(f"[rates] trailing {br['n']}: faithful {br['faithful']:.1%} "
              f"wrong {br['wrong']:.1%}  |  hard band (L>={args.hard_from}) "
              f"n={br['hard_n']} faithful {br['hard_faithful']:.1%} "
              f"wrong {br['hard_wrong']:.1%}  "
              f"(1.7B reference: 40% / 21%)", flush=True)
        agg_bad = br["n"] >= args.window and (
            br["faithful"] < args.min_faithful or br["wrong"] > args.max_wrong)
        hard_bad = br["hard_n"] >= 30 and (
            br["hard_faithful"] < args.hard_min_faithful
            or br["hard_wrong"] > args.hard_max_wrong)
        if agg_bad and not hard_bad:
            print(f"[rates] aggregate outside bounds but the hard band holds — "
                  f"this is a composition shift, not a quality drop. "
                  f"Not raising PAUSE.", flush=True)
        if hard_bad or (agg_bad and not args.require_hard_breach):
            with open(pause_flag, "w") as fh:
                json.dump({**br, "min_faithful": args.min_faithful,
                           "max_wrong": args.max_wrong,
                           "hard_min_faithful": args.hard_min_faithful,
                           "hard_max_wrong": args.hard_max_wrong,
                           "hard_band_breached": hard_bad,
                           "selected_problems": n_problems}, fh, indent=1)
            print("\n" + "!" * 72, flush=True)
            print(f"!! PAUSE RULE TRIPPED on the HARD BAND: n={br['hard_n']}, "
                  f"faithful {br['hard_faithful']:.1%} "
                  f"(floor {args.hard_min_faithful:.0%}), "
                  f"wrong {br['hard_wrong']:.1%} "
                  f"(ceiling {args.hard_max_wrong:.0%}).", flush=True)
            print(f"!! Flag written to {pause_flag}. Investigate and journal "
                  "before resuming generation.", flush=True)
            print("!" * 72 + "\n", flush=True)

        if not args.follow:
            break

    n, faithful, wrong = trailing_rates(args.output, args.window)
    print(f"[done] trailing {n} judgments: faithful {faithful:.1%}, "
          f"wrong {wrong:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
