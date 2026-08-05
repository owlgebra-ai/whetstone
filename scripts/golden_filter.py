"""Judge-filtered golden corpus: one certified trace per problem (user request,
2026-08-04).

**This makes the external judge a filter on training data, which the rest of
the pipeline forbids.** `faithfulness_audit.py`'s own docstring says: *"if a
judge verdict ever becomes a filter on training data, an external model is
shaping the corpus and the invariant is broken"*, and packet P5 §8 says judge
output goes to the dashboard, *"never into any corpus record"*. That rule is
deliberately overridden here, by user decision, and the deviation is attested
the way activity 005 attested the GLM bootstrap corpus:

* **Why it is defensible.** The central-model principle (v1 §3) protects the
  *compressor* from being an external teacher — and activity 006 already
  replaced the compressor with a frozen 32B. GLM choosing among 32B traces is a
  strictly smaller deviation than GLM writing them. Against that, activity 008
  measured 13% of selected traces judged **wrong**: an answer that verifies,
  in-register, low-spike, and a fabricated derivation underneath. Every
  automatic check in the pipeline passes those, and Stage B would train on them.
* **What it costs.** The corpus inherits one judge's notion of faithfulness as
  an unmeasured selection pressure, and any comparison against SCA /
  DeepCompress now carries a confound those arms do not have.
* **The mitigation, and it is not optional.** The unfiltered selected corpus is
  left completely intact. Stage B can be run on either, and the difference
  between them is itself a measurement. Never delete the unfiltered corpus.

**Why one draft at a time, best first.** The goal is maximum *unique problems*,
not maximum traces. Judging all 8 drafts of every problem would cost 8× the API
for no extra coverage. So each problem's candidates are walked in the **exact
selection ranking** (imported from `select_teacher_corpus`, never re-derived —
two orderings that drift apart is the same bug class as scoring under one
construction and thresholding under another) and judged one at a time until one
comes back faithful. A problem resolves on its first success; a problem whose
candidates are exhausted is recorded, not silently dropped.

**Two rubrics, because 38.5% of problems have no verbose source.** The standard
rubric compares compact against verbose and cannot run without one. Excluding
those problems would cap the golden set at ~2,460 of 4,000 — directly against
the goal of maximising unique problems. So source-less problems are judged by a
**self-contained** rubric instead: is every step justified, does the derivation
actually reach the answer, is anything asserted without support. Every record is
stamped with `rubric` so the two populations are never silently pooled, and they
are written to separate files. This is also the only instrument pointed at
activity 008 finding 13 — that the teacher confabulates when given the answer
without the reasoning (10.5% faithful / 73.7% wrong in the hard band).

Usage::

    export FAITHFULNESS_BASE_URL=... FAITHFULNESS_AUTH_TOKEN=...
    python scripts/golden_filter.py --concurrency 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from faithfulness_audit import RUBRIC, SYSTEM, _parse                # noqa: E402
from select_teacher_corpus import rank_candidates                    # noqa: E402
from whetstone.runio import repair_tail                              # noqa: E402

#: Escalating sleeps for a 429. Totals ~21 minutes, which spans a typical
#: per-minute/per-hour window reset without an operator having to babysit it.
RATE_LIMIT_BACKOFF_S = (30, 60, 120, 300, 600)

#: Rubric for problems with no verbose source. Deliberately *not* the
#: faithfulness rubric with an empty ORIGINAL: measured on this corpus, the
#: judge returns 94.7% "faithful" when handed a blank original, because there is
#: nothing to have dropped. It grades soundness instead, and the key question is
#: the one finding 13 raises — the teacher was given the answer, so "reaches the
#: right answer" is worthless evidence and only the *derivation* counts.
SELF_CONTAINED_SYSTEM = (
    "You are a strict grader auditing whether a compact mathematical derivation "
    "actually establishes its result. The author was GIVEN the correct answer in "
    "advance, so arriving at the right answer proves nothing — judge only whether "
    "each step is justified and whether the steps together derive the result. "
    "Reply with ONLY a JSON object. No prose, no code fence."
)

SELF_CONTAINED_RUBRIC = """PROBLEM:
{problem}

KNOWN-CORRECT ANSWER (the author was shown this in advance):
{gold}

COMPACT DERIVATION (what must be judged):
{compact}

The derivation is written in a compact notation; terse notation is fine and is
NOT a defect. Judge the reasoning, not the style.

Return JSON with exactly these keys:
  "unsupported_step"  : bool  — some step is asserted without justification and
                                does not follow from what precedes it
  "gap"               : bool  — the chain has a hole: a step the reader cannot
                                reconstruct from what is written
  "asserts_answer"    : bool  — the answer is essentially stated rather than
                                derived, or the derivation works backwards from it
  "invented_content"  : bool  — invokes a fact, theorem or value that is wrong or
                                not applicable here
  "verdict"           : "faithful" | "lossy" | "wrong"
  "note"              : string, <= 30 words, citing the specific step if not faithful

"faithful" = every step is justified and the chain establishes the result.
"lossy"    = the chain is followable but has a thin or under-justified step.
"wrong"    = a step is unsupported or incorrect, or the answer is asserted
             rather than derived."""


def load_candidates(drafts_path: str, scores_path: str) -> dict:
    """``{uid: [candidate, ...]}`` — kept, scored drafts, highest round only."""
    scores = {}
    with open(scores_path) as fh:
        for line in fh:
            try:
                s = json.loads(line)
            except Exception:                                      # noqa: BLE001
                continue
            scores[(s["_uid"], s["candidate_idx"], s.get("gen_round", 1))] = s

    by_uid: dict = defaultdict(list)
    with open(drafts_path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:                                      # noqa: BLE001
                continue
            if d.get("reject_reason") is not None:
                continue
            s = scores.get((d["_uid"], d["candidate_idx"], d.get("gen_round", 1)))
            if s is None or s.get("score_skip_reason"):
                continue
            d["_s"] = s
            by_uid[d["_uid"]].append(d)

    for uid, cs in list(by_uid.items()):
        rounds = {c.get("gen_round", 1) for c in cs}
        if len(rounds) > 1:
            top = max(rounds)
            by_uid[uid] = [c for c in cs if c.get("gen_round", 1) == top]
    return by_uid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafts", default="/data/whetstone/corpora/stagea_raw/drafts.jsonl")
    ap.add_argument("--scores", default="/data/whetstone/corpora/stagea_raw/scores.jsonl")
    ap.add_argument("--subset", default="/data/whetstone/corpora/stagea/subset_stagea.jsonl")
    ap.add_argument("--outdir", default="/data/whetstone/corpora/stagea_golden")
    ap.add_argument("--judgments",
                    default="/data/whetstone/runs/stagea/golden_judgments.jsonl",
                    help="append-only log of every judgment; the resume key")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--base_url", default=os.environ.get("FAITHFULNESS_BASE_URL"))
    ap.add_argument("--concurrency", type=int, default=6,
                    help="kept low on purpose: the judge endpoint returns 429 "
                         "well before the GPU side saturates, and a rate-limit "
                         "storm costs more wall-clock than it saves")
    ap.add_argument("--rebuild_only", action="store_true",
                    help="rebuild the golden corpus from the judgments log "
                         "with NO API calls. The log is the source of truth, "
                         "so this always works and always agrees with a live "
                         "run that got as far as the same judgments.")
    ap.add_argument("--fsync_every", type=int, default=25,
                    help="fsync the judgments log this often; the whole point "
                         "is that a quota stall or a box reboot loses at most "
                         "this many judgments")
    ap.add_argument("--max_consecutive_errors", type=int, default=25,
                    help="stop cleanly after this many consecutive API "
                         "failures rather than burning the remaining problems "
                         "into the 'exhausted' bucket")
    ap.add_argument("--max_attempts", type=int, default=8,
                    help="candidates to try per problem before giving up")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--max_verbose_chars", type=int, default=40000)
    ap.add_argument("--no_source_rubric", action="store_true", default=True,
                    help="judge source-less problems with the self-contained "
                         "rubric (default on). --no_source_rubric_off to skip "
                         "them entirely and keep the strict-faithfulness set.")
    ap.add_argument("--no_source_rubric_off", dest="no_source_rubric",
                    action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="0 = all problems")
    args = ap.parse_args()

    import anthropic
    token = os.environ.get("FAITHFULNESS_AUTH_TOKEN")
    if not token or not args.base_url:
        raise SystemExit("[golden] set FAITHFULNESS_BASE_URL and "
                         "FAITHFULNESS_AUTH_TOKEN")
    client = anthropic.Anthropic(base_url=args.base_url, auth_token=token,
                                 max_retries=4)

    sub = {}
    with open(args.subset) as fh:
        for line in fh:
            r = json.loads(line)
            sub[r["_uid"]] = r

    by_uid = load_candidates(args.drafts, args.scores)
    print(f"[in] {len(by_uid)} problems with at least one kept+scored draft")

    os.makedirs(args.outdir, exist_ok=True)
    repair_tail(args.judgments)
    judged: dict = {}
    resolved: set = set()
    if os.path.exists(args.judgments):
        with open(args.judgments) as fh:
            for line in fh:
                try:
                    j = json.loads(line)
                except Exception:                                  # noqa: BLE001
                    continue
                judged[(j["_uid"], j["candidate_idx"], j.get("gen_round", 1))] = j
                if j.get("verdict") == "faithful":
                    resolved.add(j["_uid"])
    print(f"[resume] {len(judged)} judgments on disk, {len(resolved)} problems "
          f"already resolved")

    # Order by FEWEST prior judgments, so untouched problems go first.
    # Under a constrained judge quota the objective is unique problems per
    # token, and the two populations differ sharply: a never-judged problem
    # resolves in ~1.8 judgments, while a partially-judged one has already had
    # its best candidates rejected and is disproportionately a hard case headed
    # for exhaustion. Corpus order mixes them and spends the window on the
    # expensive tail.
    prior = Counter()
    for (uid, _c, _r) in judged:
        prior[uid] += 1
    todo = sorted((u for u in by_uid if u not in resolved),
                  key=lambda u: (prior[u], u))
    if args.limit:
        todo = todo[:args.limit]
    fresh = sum(1 for u in todo if prior[u] == 0)
    print(f"[work] {len(todo)} problems to resolve — {fresh} never judged "
          f"(these go first), {len(todo) - fresh} partially judged")

    jf = open(args.judgments, "a", buffering=1)
    lock = threading.Lock()
    state = Counter()
    t0 = time.time()
    stop = threading.Event()

    def note_judgment(rec: dict) -> None:
        """Append + periodically fsync. Line buffering survives a process kill;
        only the fsync survives the box going down."""
        with lock:
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            state["judgments"] += 1
            if state["judgments"] % args.fsync_every == 0:
                jf.flush()
                os.fsync(jf.fileno())

    def judge_one(cand: dict, src: dict) -> dict:
        verbose = (sub.get(cand["_uid"], {}).get("verbose_think") or "").strip()
        if verbose:
            v = verbose[:args.max_verbose_chars]
            prompt = RUBRIC.format(problem=cand.get("prompt", ""), verbose=v,
                                   compact=cand["compact_think"])
            system, rubric = SYSTEM, "faithfulness"
        else:
            prompt = SELF_CONTAINED_RUBRIC.format(
                problem=cand.get("prompt", ""),
                gold=cand.get("ground_truth", ""),
                compact=cand["compact_think"])
            system, rubric = SELF_CONTAINED_SYSTEM, "self_contained"
        # Quota exhaustion is a *pause*, not a failure. The endpoint's limits
        # reset on a window, so a long sleep costs wall-clock and nothing else,
        # whereas giving up costs the caller a manual restart. The anthropic
        # client's own max_retries handles brief 429 bursts; this handles the
        # case where the daily/hourly window is genuinely spent.
        msg = None
        for attempt, backoff in enumerate(RATE_LIMIT_BACKOFF_S):
            try:
                msg = client.messages.create(
                    model=args.model, max_tokens=args.max_tokens, system=system,
                    messages=[{"role": "user", "content": prompt}])
                break
            except anthropic.RateLimitError:
                if stop.is_set():
                    raise
                with lock:
                    state["rate_limited"] += 1
                    if state["rate_limited"] % 20 == 1:
                        print(f"[quota] rate limited; sleeping {backoff}s "
                              f"(attempt {attempt + 1}/"
                              f"{len(RATE_LIMIT_BACKOFF_S)}). Progress is "
                              f"already on disk.", flush=True)
                time.sleep(backoff)
        if msg is None:
            raise RuntimeError("rate limited past the backoff schedule")
        text = "".join(b.text for b in msg.content if b.type == "text")
        parsed = _parse(text)
        rec = {"_uid": cand["_uid"], "candidate_idx": cand["candidate_idx"],
               "gen_round": cand.get("gen_round", 1),
               "level": cand.get("level"), "rubric": rubric,
               "conditioned_on": cand.get("conditioned_on"),
               "parsed": bool(parsed),
               "judge_model": args.model,
               "input_tokens": msg.usage.input_tokens,
               "output_tokens": msg.usage.output_tokens}
        if parsed:
            rec["verdict"] = parsed.get("verdict")
            rec["note"] = parsed.get("note")
            for k, v in parsed.items():
                if isinstance(v, bool):
                    rec[k] = v
        else:
            rec["verdict_raw"] = text[:400]
        return rec

    def resolve_problem(uid: str) -> tuple[str, dict | None, int, str]:
        """Walk this problem's candidates best-first until one is faithful."""
        cands = rank_candidates(by_uid[uid])[:args.max_attempts]
        tried = 0
        for c in cands:
            if stop.is_set():
                return uid, None, tried, "aborted"
            key = (uid, c["candidate_idx"], c.get("gen_round", 1))
            prev = judged.get(key)
            if prev is None:
                try:
                    prev = judge_one(c, sub.get(uid, {}))
                except Exception as exc:                           # noqa: BLE001
                    # An API failure is NOT evidence about the trace. Returning
                    # "exhausted" here would bake a transient 429 into
                    # unresolved_uids.json and the problem would never be
                    # retried; it is reported as an error so the next run picks
                    # it up.
                    with lock:
                        state["api_error"] += 1
                        state["consecutive_errors"] += 1
                        if (state["consecutive_errors"]
                                >= args.max_consecutive_errors
                                and not stop.is_set()):
                            stop.set()
                            print(f"\n[stop] {state['consecutive_errors']} "
                                  f"consecutive API errors — stopping cleanly. "
                                  f"Progress is in {args.judgments}; re-run the "
                                  f"same command to continue.\n"
                                  f"        last error: {type(exc).__name__}: "
                                  f"{str(exc)[:160]}", flush=True)
                    return uid, None, tried, "error"
                with lock:
                    state["consecutive_errors"] = 0
                note_judgment(prev)
                tried += 1
            if prev.get("verdict") == "faithful":
                return uid, {**c, "_judgment": prev}, tried, "resolved"
        return uid, None, tried, "exhausted"

    golden, exhausted, errored = [], [], []
    if not args.rebuild_only:
      with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(resolve_problem, u): u for u in todo}
        done = 0
        for fut in as_completed(futs):
            uid, winner, tried, why = fut.result()
            done += 1
            if winner is not None:
                golden.append(winner)
                state["resolved"] += 1
                state[f"tries_{min(tried, 4)}"] += 1
            elif why == "exhausted":
                exhausted.append(uid)
                state["exhausted"] += 1
            else:
                errored.append(uid)
                state[f"unresolved_{why}"] += 1
            if done % 100 == 0:
                el = (time.time() - t0) / 60
                print(f"[{done}/{len(todo)}] resolved {state['resolved']}  "
                      f"exhausted {state['exhausted']}  "
                      f"api_err {state['api_error']}  "
                      f"rate_limited {state['rate_limited']}  "
                      f"{state['judgments']} judgments  "
                      f"{state['judgments']/max(1e-9, el):.0f}/min", flush=True)
      jf.flush()
      os.fsync(jf.fileno())
    jf.close()

    # In rebuild mode nothing goes through resolve_problem, so the exhausted
    # list has to be derived from the log instead: a problem is exhausted when
    # every candidate it has was judged and none passed. Without this the
    # rebuild silently reports zero exhausted problems — and that list is a
    # deliverable (it is Stage-C rescue's clientele, the same role `unserved`
    # plays for selection).
    if args.rebuild_only:
        for uid, cs in by_uid.items():
            if uid in resolved:
                continue
            keys = [(uid, c["candidate_idx"], c.get("gen_round", 1))
                    for c in rank_candidates(cs)[:args.max_attempts]]
            if keys and all(k in judged for k in keys):
                exhausted.append(uid)

    # Re-attach problems that were already resolved on a previous run.
    for uid in resolved:
        if uid in by_uid and uid not in {g["_uid"] for g in golden}:
            for c in rank_candidates(by_uid[uid]):
                key = (uid, c["candidate_idx"], c.get("gen_round", 1))
                j = judged.get(key)
                if j and j.get("verdict") == "faithful":
                    golden.append({**c, "_judgment": j})
                    break

    by_rubric = Counter(g["_judgment"]["rubric"] for g in golden)
    out_paths = {}
    for rubric in ("faithfulness", "self_contained"):
        rows = [g for g in golden if g["_judgment"]["rubric"] == rubric]
        path = os.path.join(args.outdir, f"golden_{rubric}.jsonl")
        with open(path, "w") as fh:
            for g in rows:
                rec = {k: v for k, v in g.items()
                       if k not in ("_s", "_judgment")}
                rec.pop("completion_token_ids", None)
                rec.update({k: v for k, v in g["_s"].items()
                            if k not in ("_uid", "candidate_idx")})
                rec["judge_verdict"] = g["_judgment"].get("verdict")
                rec["judge_rubric"] = rubric
                rec["judge_note"] = g["_judgment"].get("note")
                rec["judge_model"] = g["_judgment"].get("judge_model")
                rec["verbose_think"] = sub.get(g["_uid"], {}).get("verbose_think", "")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_paths[rubric] = path

    # Two distinct buckets. `exhausted` is a finding about the teacher — every
    # candidate was judged and none passed. `errored` is a finding about the
    # API and must stay retriable; conflating them would silently retire
    # problems that were never actually judged.
    unresolved = exhausted + errored
    with open(os.path.join(args.outdir, "unresolved_uids.json"), "w") as fh:
        json.dump({"exhausted": sorted(exhausted),
                   "errored_retry_next_run": sorted(errored)}, fh, indent=1)

    lv = Counter(g.get("level") for g in golden)
    lv_un = Counter(sub.get(u, {}).get("level") for u in unresolved)
    mins = (time.time() - t0) / 60
    print(f"\n[done] {len(golden)} golden problems, {len(unresolved)} unresolved, "
          f"{state['judgments']} judgments in {mins:.1f} min")
    print(f"  by rubric: {dict(by_rubric)}")
    print(f"  {'lvl':>3} {'golden':>7} {'unresolved':>11} {'yield':>7}")
    for l in sorted(set(lv) | set(lv_un), key=lambda x: (x is None, x)):
        tot = lv[l] + lv_un[l]
        print(f"  {str(l):>3} {lv[l]:>7} {lv_un[l]:>11} "
              f"{100*lv[l]/max(1,tot):>6.1f}%")
    for path in out_paths.values():
        print(f"[out] {path}")
    print(f"[out] {os.path.join(args.outdir, 'unresolved_uids.json')}")
    print("\n  NOTE: the unfiltered selected corpus is untouched and must be "
          "kept — it is the control arm for whatever this filtering does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
