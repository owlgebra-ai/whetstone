"""Stage-A selection: 1–3 diverse survivors per problem (packet P5 Part 3).

Cheap, deterministic, and **re-runnable from scratch** over the append-only raw
corpus. That is the whole point of the raw/selected split: an F2 failure in the
selection rule costs one minute of CPU to fix, not a day of GPU. Nothing here
ever writes back to the raw corpus.

**Why the winner rule is lexicographic and not a weighted score.** Activity 007
finding 7 measured G_spike selecting *against* verification retention — r_pb
−0.113, p < 1e-4, driven by a 7.92-nat residual tax on ``chk`` that the Round-0
corpus could not calibrate because the student writes ``chk:`` as a trailing line
and the teacher writes it mid-trace. Folding ``verify_kept`` into a weighted sum
would let a large G_spike advantage buy out the verification; ordering it ahead
of G_spike cannot. The tax is in the p95 tail rather than the mean, so lowering λ
is the wrong knob (007, design §7) and is not done here.

Structural criteria fire **only when the source warrants them** — "kept the
verification its source had", never "contains a ``chk:``". Drafts with no verbose
source (``no_source``) are ranked on G_spike × G_budget alone; they cannot win a
structural tie-break they were never eligible for.

**Runners-up must be genuinely different**, or they teach the student the same
sentence twice: 8-gram Jaccard < 0.6 over think tokens, OR a differing structural
signature (a branch/verify presence flip, or a think length differing by ≥30%).
Priority goes to a runner-up that adds a structural property the winner lacks —
that is the diversity Stage B can actually learn from.

**Stage B must weight per problem, never per trace.** A problem with 3 keeps
would otherwise contribute 3× the gradient of a problem with 1, and the number of
keeps is a property of the teacher's sampling luck, not of the problem's value.
The handoff sidecar says so in writing.

Usage::

    python scripts/select_teacher_corpus.py \\
        --drafts  /data/whetstone/corpora/stagea_raw/drafts.jsonl \\
        --scores  /data/whetstone/corpora/stagea_raw/scores.jsonl \\
        --subset  /data/whetstone/corpora/stagea/subset_stagea.jsonl \\
        --outdir  /data/whetstone/corpora/stagea_selected
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import write_meta
from whetstone.round0 import MARKER_CLASSES, percentile

MAX_KEEP = 3
JACCARD_MAX = 0.6
LEN_DELTA = 0.30
#: In-memory scratch attached to a candidate during selection. Enumerated
#: explicitly rather than matched by underscore prefix — see the emit site.
SCRATCH_KEYS = frozenset({"_s", "_ng", "_sig", "_think", "_adds"})
ALL_MARKERS = tuple(m for cls in MARKER_CLASSES.values() for m in cls)


def ngrams8(text: str, n: int = 8) -> set:
    """Whitespace-token n-grams over the *whole* think body.

    Deliberately not :func:`whetstone.poolutil.ngrams`: that one lowercases,
    keeps only word characters and truncates at 400 chars — which erases the
    register's symbols, i.e. exactly the content that distinguishes two compact
    traces of the same problem.
    """
    toks = text.split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b) if inter else 0.0


def marker_density(think: str) -> float:
    """Register markers per 100 think chars — activity 006's ``mark/100ch``.

    Kept in the same units as that table (Qwen3-1.7B 1.22, 14B 0.82, 32B 2.10)
    so the selected corpus can be read against the raw-sample baseline directly.
    """
    if not think:
        return 0.0
    return 100.0 * sum(think.count(m) for m in ALL_MARKERS) / len(think)


#: Mean words per non-empty think line above which a trace reads as prose
#: rather than as register lines. Measured, not guessed: level-1 traces sit at
#: 6.5% over this line and level-9 traces at 54.4%.
PROSE_WORDS_PER_LINE = 14.0


def words_per_line(think: str) -> float:
    lines = [l for l in think.splitlines() if l.strip()]
    if not lines:
        return 0.0
    return sum(len(l.split()) for l in lines) / len(lines)


def in_register(think: str) -> bool:
    """Is this trace actually written in the register, or is it prose?

    Prefilling ``<think>\\ngoal:`` guarantees the *opener* — 100% of drafts at
    every level start with it — so "opens with goal:" measures the prefill, not
    the model. What decays with difficulty is the body: measured over the
    calibration slice, traces carrying a ``⇒`` fall 99.6% → 87.3% from level 1 to
    9, those carrying a ``chk``/``✓`` fall 97.2% → 54.4%, and those whose lines
    read as sentences rather than register steps rise 6.5% → **54.4%**. The
    teacher holds the register where it fits easily and reverts to prose exactly
    on the hard problems the register exists to compress.

    Two signals, both from the card: the result marker must be present (card
    §1.1) and lines must be steps rather than sentences (card §1.2 makes the
    newline the step boundary). Deliberately *not* a marker-density floor —
    density legitimately falls on symbolic problems whose lines are long LaTeX,
    so a density gate would penalise correct hard-problem style.

    Used as a selection *preference*, never a filter: 98.9% of problems have at
    least one in-register draft among their 8 (100% at level 9, where the
    per-draft rate is 42.9%), so unlike branch retention — which is clustered by
    problem and cannot be bought with K — this one best-of-K fixes for free.
    """
    return "⇒" in think and words_per_line(think) <= PROSE_WORDS_PER_LINE


def quality(s: dict) -> float:
    """G_spike(β=10) × G_budget — the packet's continuous criterion (c)."""
    gs, gb = s.get("g_spike_b10"), s.get("g_budget")
    if gs is None or gb is None:
        return -1.0
    return gs * gb


def signature(d: dict, s: dict) -> tuple:
    return (bool(s.get("compact_has_branch")), bool(s.get("compact_has_verify")))


def is_diverse(cand: dict, kept: list[dict]) -> tuple[bool, str]:
    """Different enough from *every* already-kept draft, with the reason."""
    for k in kept:
        j = jaccard(cand["_ng"], k["_ng"])
        sig_differs = cand["_sig"] != k["_sig"]
        lo = min(cand["_think"], k["_think"])
        hi = max(cand["_think"], k["_think"])
        len_differs = lo > 0 and (hi - lo) / lo >= LEN_DELTA
        if not (j < JACCARD_MAX or sig_differs or len_differs):
            return False, f"dup(j={j:.2f})"
    return True, ""


def source_properties(cands: list[dict]) -> tuple[bool, bool]:
    """``(src_has_verify, src_has_branch)`` for a problem's candidate list.

    Source-level properties are identical across a problem's drafts (they share
    one verbose source), so they are read off any candidate that has a source.
    """
    src = next((c for c in cands if not c["_s"].get("no_source", True)), None)
    return (bool(src and src["_s"].get("src_has_verify")),
            bool(src and src["_s"].get("src_has_branch")))


def rank_candidates(cands: list[dict]) -> list[dict]:
    """A problem's candidates, best first, under the selection ordering.

    Exported so the golden filter walks candidates in **exactly** the order
    selection would pick them, rather than re-deriving an ordering. Two
    rankings that drift apart is the same class of bug as a record scored under
    one construction and thresholded under another.
    """
    src_verify, src_branch = source_properties(cands)
    return sorted(cands, key=_rank_key(src_verify, src_branch))


def _rank_key(src_verify: bool, src_branch: bool):
    """The single candidate ordering, used by selection AND the golden filter.

    Register adherence ranks FIRST: a prose trace with a ``goal:`` header is not
    a compact-register example at all, and installing the register is the whole
    deliverable, so it should not win on a structural tie-break or on G_spike.
    It is affordable at the top precisely because it is the one property
    best-of-K can nearly always supply (98.9% of problems have an in-register
    candidate).

    Structural terms fire only when the source warrants them. ``None`` (no
    source) sorts with ``False``: it is not a claim, so it cannot win the
    tie-break — but it does not lose to a *failed* keep either, which is why the
    value is the same 0 rather than -1.
    """
    def rank(c):
        s = c["_s"]
        reg = 1 if in_register(c.get("compact_think", "")) else 0
        v = 1 if (src_verify and s.get("verify_kept") is True) else 0
        b = 1 if (src_branch and s.get("branch_kept") is True) else 0
        return (-reg, -v, -b, -quality(s), c["candidate_idx"])
    return rank


def select_one(cands: list[dict]) -> tuple[list[dict], dict]:
    """Winner + ≤2 diverse runners-up for one problem."""
    src_verify, src_branch = source_properties(cands)

    ordered = sorted(cands, key=_rank_key(src_verify, src_branch))
    winner = ordered[0]
    ws = winner["_s"]
    reasons = []
    if not in_register(winner.get("compact_think", "")):
        reasons.append("no_in_register_candidate")
    if src_verify:
        reasons.append("verify_kept" if ws.get("verify_kept") is True
                       else "verify_lost")
    if src_branch:
        reasons.append("branch_kept" if ws.get("branch_kept") is True
                       else "branch_lost")
    reasons.append("gspike_gbudget")
    winner["selection_rank"] = 0
    winner["selection_reason"] = "winner:" + "+".join(reasons)
    kept = [winner]

    # Runners-up in TWO passes, and the order is the point (measured: a single
    # rank-order pass put `branch_kept` at 26.7% where two passes reach the
    # packet's target). The winner rule ranks `verify_kept` above `branch_kept`,
    # and verification is ~5x commoner than branch retention, so the winner is
    # almost always a verify-keeper that dropped the branch. If runners-up are
    # then filled in rank order, the branch-keeping draft — typically ranked
    # 5th–8th, because branch-preserving traces are longer and score worse on
    # G_spike × G_budget — never gets looked at before the slots are gone.
    #
    # Pass 1 therefore takes candidates that ADD a structural property the
    # winner lacks; pass 2 fills what is left with plain diversity.
    def missing(key):
        return not any(k["_s"].get(key) for k in kept)

    for pass_no in (1, 2):
        for c in ordered[1:]:
            if len(kept) >= MAX_KEEP:
                break
            if c in kept:
                continue
            adds = [n for n, key in (("branch", "compact_has_branch"),
                                     ("verify", "compact_has_verify"))
                    if c["_s"].get(key) and missing(key)]
            if pass_no == 1 and not adds:
                continue
            ok, _ = is_diverse(c, kept)
            if not ok:
                continue
            c["_adds"] = adds
            c["selection_rank"] = len(kept)
            c["selection_reason"] = ("runner_up:adds_" + "+".join(adds) if adds
                                     else "runner_up:diverse")
            kept.append(c)
        if len(kept) >= MAX_KEEP:
            break

    stats = {"src_has_verify": src_verify, "src_has_branch": src_branch,
             "n_cands": len(cands)}
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafts", default="/data/whetstone/corpora/stagea_raw/drafts.jsonl")
    ap.add_argument("--scores", default="/data/whetstone/corpora/stagea_raw/scores.jsonl")
    ap.add_argument("--subset", default="/data/whetstone/corpora/stagea/subset_stagea.jsonl")
    ap.add_argument("--outdir", default="/data/whetstone/corpora/stagea_selected")
    ap.add_argument("--report", default=None,
                    help="also write the F2d dashboard numbers here as JSON")
    args = ap.parse_args()

    src_by_uid = {}
    for line in open(args.subset):
        if line.strip():
            r = json.loads(line)
            src_by_uid[r["_uid"]] = r

    scores: dict[tuple, dict] = {}
    for line in open(args.scores):
        line = line.strip()
        if not line:
            continue
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        scores[(s["_uid"], s["candidate_idx"], s.get("gen_round", 1))] = s

    by_uid: dict[str, list[dict]] = defaultdict(list)
    rejects: dict[str, Counter] = defaultdict(Counter)
    n_drafts = 0
    acc_all: list[bool] = []
    raw_struct = Counter()
    for line in open(args.drafts):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_drafts += 1
        acc_all.append(bool(d.get("verify_ok")))
        uid = d["_uid"]
        if d.get("reject_reason") is not None:
            rejects[uid][d["reject_reason"]] += 1
            continue
        s = scores.get((uid, d["candidate_idx"], d.get("gen_round", 1)))
        if s is None:
            rejects[uid]["unscored"] += 1
            continue
        if s.get("score_skip_reason"):
            rejects[uid][f"score_{s['score_skip_reason'].split(':')[0]}"] += 1
            continue
        # Raw per-draft structural baseline, over exactly the drafts selection
        # chooses among — this is what "selection must beat raw" is measured on.
        if s.get("src_has_verify"):
            raw_struct["verify_elig"] += 1
            raw_struct["verify_kept"] += bool(s.get("verify_kept"))
        if s.get("src_has_branch"):
            raw_struct["branch_elig"] += 1
            raw_struct["branch_kept"] += bool(s.get("branch_kept"))
        d["_s"] = s
        d["_ng"] = ngrams8(d.get("compact_think", ""))
        d["_sig"] = signature(d, s)
        d["_think"] = s.get("scored_think_tokens") or d.get("think_tokens") or 0
        by_uid[uid].append(d)

    # Keep only the highest generation round present for each problem. A
    # re-generation under changed conditioning (e.g. a problem promoted from
    # `gold` to `gold+trace` by a raised trace cap) appends round-2 drafts
    # rather than mutating the raw corpus, so the old ones are still on disk and
    # still auditable — they just must not compete. Without this the promoted
    # problems would effectively run at K=16 while everything else runs at K=8,
    # which biases selection in their favour on every criterion.
    n_superseded = 0
    for uid, cs in list(by_uid.items()):
        rounds = {c.get("gen_round", 1) for c in cs}
        if len(rounds) > 1:
            top = max(rounds)
            kept_cs = [c for c in cs if c.get("gen_round", 1) == top]
            n_superseded += len(cs) - len(kept_cs)
            by_uid[uid] = kept_cs
    if n_superseded:
        print(f"[rounds] {n_superseded} drafts superseded by a later "
              f"generation round")

    print(f"[in] {n_drafts} drafts, {len(scores)} score records, "
          f"{len(by_uid)} problems with ≥1 survivor")

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "selected.jsonl")
    selected_by_uid: dict[str, list] = defaultdict(list)
    sel_records: list = []
    sel_reasons, keeps_hist = Counter(), Counter()
    sel_struct = Counter()
    per_level = defaultdict(Counter)
    think_lens, ans_lens, dens = [], [], []
    n_sel = 0
    sel_acc = []

    with open(out_path, "w") as fh:
        for uid in sorted(by_uid):
            kept, st = select_one(by_uid[uid])
            selected_by_uid[uid] = kept
            sel_records.extend(kept)
            keeps_hist[len(kept)] += 1
            lv = kept[0].get("level")
            per_level[lv]["problems"] += 1
            per_level[lv]["kept"] += len(kept)
            for c in kept:
                s = c["_s"]
                sel_reasons[c["selection_reason"]] += 1
                sel_acc.append(bool(c.get("verify_ok")))
                if st["src_has_verify"]:
                    sel_struct["verify_elig"] += 1
                    sel_struct["verify_kept"] += bool(s.get("verify_kept"))
                if st["src_has_branch"]:
                    sel_struct["branch_elig"] += 1
                    sel_struct["branch_kept"] += bool(s.get("branch_kept"))
                think_lens.append(c["_think"])
                ans_lens.append(s.get("scored_answer_tokens")
                                or c.get("answer_tokens") or 0)
                dens.append(marker_density(c.get("compact_think", "")))
                # Drop only this script's own scratch keys. A blanket
                # "starts with _" rule also eats `_uid`, which is the join key
                # for Stage B, the audit and every later analysis — and it does
                # it silently, since a record missing its id is still valid
                # JSON. poolutil.write_jsonl whitelists `_uid` for the same
                # reason.
                rec = {k: v for k, v in c.items() if k not in SCRATCH_KEYS}
                rec.pop("completion_token_ids", None)   # 32k of ids per draft;
                                                        # Stage B re-tokenizes
                                                        # under the student
                rec.update({k: v for k, v in s.items()
                            if k not in ("_uid", "candidate_idx")})
                rec["n_kept"] = len(kept)
                if "_uid" not in rec:
                    raise SystemExit(
                        "[select] emitted a record with no _uid — the join key "
                        "for Stage B and the audit. Refusing to write a corpus "
                        "that cannot be joined back to its problems.")
                # Stage B and the audit both need the source next to the rewrite.
                srec = src_by_uid.get(uid, {})
                rec["verbose_think"] = srec.get("verbose_think", "")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_sel += 1

    unserved = {uid: dict(rejects[uid]) for uid in src_by_uid
                if uid not in by_uid and uid in rejects}
    unserved_path = os.path.join(args.outdir, "unserved_uids.json")
    with open(unserved_path, "w") as fh:
        json.dump(unserved, fh, indent=1)

    def pct(a, b):
        return round(100.0 * a / b, 2) if b else None

    # ---- register adherence, raw vs selected --------------------------
    raw_reg = [in_register(c.get("compact_think", ""))
               for cs in by_uid.values() for c in cs]
    sel_reg = [in_register(r.get("compact_think", "")) for r in sel_records]
    avail_reg = sum(1 for cs in by_uid.values()
                    if any(in_register(c.get("compact_think", "")) for c in cs))
    cap_reg = sum(1 for u, ks in selected_by_uid.items()
                  if any(in_register(k.get("compact_think", "")) for k in ks))
    register_stats = {
        "raw_per_draft_pct": pct(sum(raw_reg), len(raw_reg)),
        "selected_per_trace_pct": pct(sum(sel_reg), len(sel_reg)),
        "problems_with_in_register_candidate_pct": pct(avail_reg, len(by_uid)),
        "problems_captured_pct": pct(cap_reg, len(by_uid)),
        "prose_words_per_line_threshold": PROSE_WORDS_PER_LINE,
    }

    # ---- problem-level structural capture -----------------------------
    # The packet's targets ("verify >= 85%, branch >= 30% *on source-branching
    # problems*") are per-PROBLEM, not per-trace, and the difference is not
    # cosmetic: with 3 keeps per problem, one branch-keeping trace out of three
    # reads as 33% per-trace and 100% per-problem. `capture` is the number that
    # actually tests the selection RULE — of the problems where some candidate
    # kept the property, how many did selection keep it for. Anything below
    # 100% there is a bug in the rule; a low `captured` with a 100% `capture`
    # is a statement about the teacher, not about selection.
    problem_struct = {}
    for name, elig, kept_key in (("verify", "src_has_verify", "verify_kept"),
                                 ("branch", "src_has_branch", "branch_kept")):
        eligible = [u for u, cs in by_uid.items()
                    if cs and cs[0]["_s"].get(elig)]
        available = [u for u in eligible
                     if any(c["_s"].get(kept_key) for c in by_uid[u])]
        captured = [u for u in available
                    if any(c["_s"].get(kept_key) for c in selected_by_uid[u])]
        problem_struct[name] = {
            "eligible_problems": len(eligible),
            "available_pct": pct(len(available), len(eligible)),
            "captured_pct": pct(len(captured), len(eligible)),
            "capture_efficiency_pct": pct(len(captured), len(available)),
        }

    report = {
        "n_drafts": n_drafts,
        "n_problems_with_survivors": len(by_uid),
        "n_selected_traces": n_sel,
        "n_unserved": len(unserved),
        "acc_all_drafts_pct": pct(sum(acc_all), len(acc_all)),
        "acc_selected_pct": pct(sum(sel_acc), len(sel_acc)),
        "keeps_per_problem": {str(k): v for k, v in sorted(keeps_hist.items())},
        "selection_reasons": dict(sel_reasons.most_common()),
        "structural": {
            "raw_verify_kept_pct": pct(raw_struct["verify_kept"],
                                       raw_struct["verify_elig"]),
            "sel_verify_kept_pct": pct(sel_struct["verify_kept"],
                                       sel_struct["verify_elig"]),
            "raw_branch_kept_pct": pct(raw_struct["branch_kept"],
                                       raw_struct["branch_elig"]),
            "sel_branch_kept_pct": pct(sel_struct["branch_kept"],
                                       sel_struct["branch_elig"]),
            "verify_eligible_drafts": raw_struct["verify_elig"],
            "branch_eligible_drafts": raw_struct["branch_elig"],
        },
        "structural_per_problem": problem_struct,
        "register": register_stats,
        "think_tokens": {
            "median": percentile(think_lens, 50), "p25": percentile(think_lens, 25),
            "p75": percentile(think_lens, 75), "p95": percentile(think_lens, 95),
            "max": max(think_lens) if think_lens else None,
        },
        "answer_tokens": {
            "median": percentile(ans_lens, 50), "p95": percentile(ans_lens, 95),
        },
        "marker_density_per_100ch": {
            "median": round(percentile(dens, 50), 3),
            "mean": round(sum(dens) / len(dens), 3) if dens else None,
        },
        "per_level": {str(k): dict(v) for k, v in sorted(
            per_level.items(), key=lambda kv: (kv[0] is None, kv[0]))},
    }

    print(f"\n[out] {n_sel} traces over {len(by_uid)} problems -> {out_path}")
    print(f"[out] {len(unserved)} unserved -> {unserved_path}")
    print(f"\n  R_acc   all drafts {report['acc_all_drafts_pct']}%   "
          f"selected {report['acc_selected_pct']}%")
    st = report["structural"]
    print(f"  verify_kept  raw {st['raw_verify_kept_pct']}%  ->  "
          f"selected {st['sel_verify_kept_pct']}%   "
          f"(n_elig {st['verify_eligible_drafts']})")
    print(f"  branch_kept  raw {st['raw_branch_kept_pct']}%  ->  "
          f"selected {st['sel_branch_kept_pct']}%   "
          f"(n_elig {st['branch_eligible_drafts']})")
    rg = report["register"]
    print(f"\n  register adherence  raw/draft {rg['raw_per_draft_pct']}%  ->  "
          f"selected/trace {rg['selected_per_trace_pct']}%   "
          f"(problems with a candidate {rg['problems_with_in_register_candidate_pct']}%, "
          f"captured {rg['problems_captured_pct']}%)")
    print("\n  PER PROBLEM (the packet's F2b targets: verify >=85%, branch >=30%)")
    print(f"  {'':8} {'eligible':>9} {'available':>10} {'captured':>9} "
          f"{'capture eff':>12}")
    for name, ps in report["structural_per_problem"].items():
        print(f"  {name:<8} {ps['eligible_problems']:>9} "
              f"{str(ps['available_pct']) + '%':>10} "
              f"{str(ps['captured_pct']) + '%':>9} "
              f"{str(ps['capture_efficiency_pct']) + '%':>12}")
    print("  (capture efficiency < 100% means the RULE is losing available "
          "structure;\n   a low `available` is a statement about the teacher, "
          "not about selection)")
    print(f"\n  keeps/problem {report['keeps_per_problem']}")
    tl = report["think_tokens"]
    print(f"  think tokens  median {tl['median']}  IQR "
          f"[{tl['p25']}, {tl['p75']}]  p95 {tl['p95']}")
    print(f"  answer tokens median {report['answer_tokens']['median']}")
    print(f"  markers/100ch median "
          f"{report['marker_density_per_100ch']['median']}")
    print("  selection reasons:")
    for k, v in sel_reasons.most_common(8):
        print(f"    {k:<40} {v}")

    write_meta(out_path, {
        "builder": "scripts/select_teacher_corpus.py",
        "packet": "P5 Part 3",
        "drafts": args.drafts, "scores": args.scores,
        "max_keep": MAX_KEEP, "jaccard_max": JACCARD_MAX,
        "len_delta": LEN_DELTA,
        "report": report,
        "STAGE_B_WEIGHTING": (
            "Weight PER PROBLEM, never per trace: use 1/n_kept per trace, or "
            "sample one trace per problem per epoch. n_kept is a property of "
            "the teacher's sampling luck, not of the problem's value, so "
            "per-trace weighting would silently upweight whichever problems "
            "happened to yield diverse drafts."),
        "note": ("Selection never mutates the raw corpus. Re-run this script "
                 "from scratch after any rule change; the binding pass is the "
                 "final one over the complete raw corpus."),
    })
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"[out] dashboard numbers -> {args.report}")

    handoff = os.path.join(args.outdir, "STAGE_B_HANDOFF.md")
    with open(handoff, "w") as fh:
        fh.write(_handoff_text(out_path, unserved_path, args, report))
    print(f"[out] Stage-B handoff -> {handoff}")
    return 0


def _handoff_text(out_path, unserved_path, args, report) -> str:
    st = report["structural"]
    tl = report["think_tokens"]
    return f"""# Stage-A → Stage-B handoff

Written by `scripts/select_teacher_corpus.py`. Read this before P6 touches the
corpus.

## Paths

| what | path |
|---|---|
| selected corpus | `{out_path}` |
| raw drafts (append-only truth) | `{args.drafts}` |
| scores sidecar | `{args.scores}` |
| problem subset + verbose sources | `{args.subset}` |
| unserved problems | `{unserved_path}` |

## The one rule that will silently corrupt Stage B if ignored

**Weight per problem, never per trace.** Use `1/n_kept` per trace, or sample one
trace per problem per epoch. `n_kept` (1–{MAX_KEEP}) is a property of the
teacher's sampling luck, not of the problem's value — per-trace weighting
upweights whichever problems happened to yield diverse drafts, which correlates
with difficulty and with trace length. Every record carries `n_kept`.

Observed keeps per problem: `{report['keeps_per_problem']}`.

## Record fields Stage B needs

* `compact_think` / `answer` — the two segments. Rebuild the training sequence
  with `whetstone.round0.build_completion_text`; do **not** re-split
  `raw_text` on the decoded string.
* `verbose_think` — the source trace, present iff the problem had one. Absent
  for `no_source` records.
* `think_surprisal_hist` + `surprisal_bin_edges` — per-draft histogram of
  student-side surprisal over think tokens, under `scorer_v1`. **This is the
  ZPD sizing input** (activity 006 open item 2): the band-pass gate is
  σ(κ(log π_S(τ_t) − γ)), so the corpus-wide histogram gives the masked
  fraction for any γ without re-scoring.
* `g_spike_b5` / `g_spike_b10` / `g_budget` — selection inputs, kept for audit.
  They are **not** training weights.

## Caveats carried forward

* The student starts from the **original** checkpoint, never from `scorer_v1`
  (CLAUDE.md invariant). Round 0's EMA copy belongs to Round 0; Stage B builds
  a new one.
* Scorer gates must be **recomputed after every assimilation round** — stale
  gates are a named drift failure.
* Coverage is not uniform in difficulty: see the per-level table in
  `{os.path.basename(out_path)}.meta.json`. `unserved` problems are Stage-C
  rescue's clientele, not a silent loss.
* Structural retention in this corpus: `verify_kept` raw
  {st['raw_verify_kept_pct']}% → selected {st['sel_verify_kept_pct']}%;
  `branch_kept` raw {st['raw_branch_kept_pct']}% → selected
  {st['sel_branch_kept_pct']}%.
* Think length median {tl['median']} tokens, IQR [{tl['p25']}, {tl['p75']}] —
  report think and answer lengths separately, always.
"""


if __name__ == "__main__":
    raise SystemExit(main())
