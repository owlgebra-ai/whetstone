"""Build the SCA-matched WHETSTONE v2 eval suites (design §12.7, packet P1).

One JSONL per suite under `<out_dir>/<suite>.jsonl`, pure records (no header
line — pinned revisions and grading mode live in the sidecar `<suite>.meta.json`):

    {
        "_uid": "<suite>:<id>",
        "prompt": "<problem statement>",
        "ground_truth": "<gold answer>",
        "level": <int>,             # 0 where the suite has no difficulty notion
        "suite": "math500" | ... ,
        "subject": "...",
        "verifier": "whetstone.verify" | "code-exec",
        "choices": [...]            # GPQA only
    }

Suites (design §12.7): MATH-500, AMC23, MinervaMath, AIME24, AIME25 (math) plus
GPQA-Diamond and HumanEval (cross-domain), and **GSM8K test** — the validation
tier of the ratified eval plan (ROADMAP "Eval plan", 2026-08-02): the suite that
picks checkpoints and settles hyperparameters, kept off the headline tables so
repeated peeking cannot overfit them. Golds are stored **verbatim** — the
deterministic verifier normalizes at compare time; reformatting here would shift
measured accuracy.

Three suites are special:
  * **GPQA-Diamond** is multiple choice. Choices are deterministically shuffled
    (seed 0, per record), the prompt carries the lettered options and the
    "answer with the letter in \\boxed{}" instruction, and `ground_truth` is the
    letter. The repo is **gated** — accept the terms on huggingface.co and export
    `HF_TOKEN`, otherwise this suite is skipped with a warning.
  * **HumanEval** cannot be graded by `whetstone.verify` (needs code execution).
    Records carry `verifier: "code-exec"` and `grading: "code-exec-pending"`, and
    the sidecar repeats it, so nobody reports verifier numbers on it by accident.
  * **GSM8K** ships the gold *inside* the reference derivation, after a `####`
    marker on the last line. Only that tail is the ground truth — see
    `_norm_gsm8k`. Contamination against the train pool was pre-cleared by
    activity 002 Run 5 (0 hits vs GSM8K test).

`--standard_eval_from <val.jsonl>` builds Part 4's `standard_eval_300` (300 rows
sampled from the val pool, stratified by level, seed 0). That file is
append-only history: once written it must never be regenerated differently.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

from whetstone.poolutil import norm_text, read_jsonl, stratified_sample, write_jsonl, write_meta

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

MCQ_INSTRUCTION = (
    "Answer the following multiple-choice question. "
    "Give the letter of the correct option inside \\boxed{}."
)

# repo, config, split, pinned revision (resolved 2026-08-01; activity/002-data-pools.md)
SUITES = {
    "math500":      ("HuggingFaceH4/MATH-500",  None,                "test",  "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"),
    "amc23":        ("math-ai/amc23",           None,                "test",  "80815d37005feb82cd7f8fbc6901d5d3eff43057"),
    "minervamath":  ("math-ai/minervamath",     None,                "test",  "ee46ddc498933b1977577953250ca5c66be64f96"),
    "aime24":       ("math-ai/aime24",          None,                "test",  "83a7f387baaa524a8bda0022eac0541582297103"),
    "aime25":       ("math-ai/aime25",          None,                "test",  "563bb8404243c5f09de6ec262f2db674fe5bce9b"),
    "gpqa_diamond": ("Idavidrein/gpqa",         "gpqa_diamond",      "train", "633f5ee89ab8ad4522a9f850766b73f62147ffdd"),
    "humaneval":    ("openai/openai_humaneval", "openai_humaneval",  "test",  "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"),
    # validation tier — added by P6/activity 009 (resolved 2026-08-05); 1,319 rows
    "gsm8k_test":   ("openai/gsm8k",            "main",              "test",  "740312add88f781978c0658806c59bc2815b9866"),
}

# Row counts asserted at build time. A silent row-count change means the pinned
# revision moved (it cannot) or the loader changed its filtering (it can) — either
# way the suite is no longer the one the baseline card was measured on.
EXPECTED_ROWS = {"gsm8k_test": 1319}


def _first(rec: dict, *names: str):
    for n in names:
        if n in rec and rec[n] not in (None, ""):
            return rec[n]
    return None


def _as_level(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _norm_math(suite: str, rec: dict, idx: int) -> dict:
    prompt = _first(rec, "problem", "Problem", "question", "prompt")
    gold = _first(rec, "answer", "Answer", "ground_truth", "gold", "expected_answer")
    if gold is None:
        # math-ai/aime24 ships only `solution`, which is literally "\boxed{204}".
        sol = _first(rec, "solution", "Solution")
        if sol:
            m = BOXED_RE.search(str(sol))
            if m:
                gold = m.group(1).strip()
    if prompt is None or gold is None:
        return {}
    sid = _first(rec, "unique_id", "id", "ID", "problem_id", "name") or idx
    return {
        "_uid": f"{suite}:{sid}",
        "prompt": norm_text(prompt),
        "ground_truth": str(gold).strip(),
        "level": _as_level(_first(rec, "level", "difficulty")),
        "suite": suite,
        "subject": str(_first(rec, "subject", "topic", "category") or ""),
        "verifier": "whetstone.verify",
    }


def _norm_gpqa(suite: str, rec: dict, idx: int) -> dict:
    question = _first(rec, "Question", "question")
    correct = _first(rec, "Correct Answer", "correct_answer")
    wrong = [
        _first(rec, f"Incorrect Answer {i}", f"incorrect_answer_{i}") for i in (1, 2, 3)
    ]
    wrong = [w for w in wrong if w is not None]
    if question is None or correct is None or len(wrong) != 3:
        return {}
    options = [str(correct).strip()] + [str(w).strip() for w in wrong]
    rng = random.Random(1000 + idx)  # deterministic per-record option order
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    letter = "ABCD"[shuffled.index(options[0])]
    body = "\n".join(f"{'ABCD'[i]}) {opt}" for i, opt in enumerate(shuffled))
    sid = _first(rec, "Record ID", "id") or idx
    return {
        "_uid": f"{suite}:{sid}",
        "prompt": f"{MCQ_INSTRUCTION}\n\n{norm_text(question)}\n\n{body}",
        "ground_truth": letter,
        "level": 0,
        "suite": suite,
        "subject": str(_first(rec, "Subdomain", "High-level domain", "subject") or ""),
        "choices": shuffled,
        "verifier": "whetstone.verify",
    }


def _norm_humaneval(suite: str, rec: dict, idx: int) -> dict:
    stub = rec.get("prompt")
    test = rec.get("test")
    if not stub or not test:
        return {}
    return {
        "_uid": f"{suite}:{rec.get('task_id', idx)}",
        "prompt": stub,  # verbatim: indentation and docstring are the task
        "ground_truth": test,
        "level": 0,
        "suite": suite,
        "subject": "code",
        "entry_point": rec.get("entry_point", ""),
        "verifier": "code-exec",
        "grading": "code-exec-pending",
    }


def _norm_gsm8k(suite: str, rec: dict, idx: int) -> dict:
    """GSM8K: the gold is the tail after the `####` marker on the answer's last line.

    The rest of `answer` is the reference derivation, complete with the dataset's
    ``<<48/2=24>>`` calculator annotations. Handing that whole string to the
    verifier would grade a number against a paragraph, so the marker split is not
    cosmetic — it is the difference between a working suite and a 0% one.

    The tail is stored **verbatim** (stripped only), per this module's rule.
    GSM8K writes thousands separators (``1,000``) and `verify._normalize` already
    deletes commas at compare time, so normalizing here would change nothing but
    the provenance of the number.
    """
    question = _first(rec, "question", "problem")
    answer = _first(rec, "answer", "solution")
    if question is None or answer is None or "####" not in str(answer):
        return {}
    gold = str(answer).rsplit("####", 1)[1].strip()
    if not gold:
        return {}
    return {
        "_uid": f"{suite}:{idx}",
        "prompt": norm_text(question),
        "ground_truth": gold,
        "level": 0,  # GSM8K has no difficulty annotation
        "suite": suite,
        "subject": "grade-school math",
        "verifier": "whetstone.verify",
    }


NORMALIZERS = {
    "gpqa_diamond": _norm_gpqa,
    "humaneval": _norm_humaneval,
    "gsm8k_test": _norm_gsm8k,
}


def _pull(suite: str) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    repo, config, split, rev = SUITES[suite]
    print(f"[eval] loading {repo} ({config or 'default'}/{split}) @ {rev[:8]} ...", flush=True)
    kwargs = {"split": split, "revision": rev}
    ds = load_dataset(repo, config, **kwargs) if config else load_dataset(repo, **kwargs)
    fn = NORMALIZERS.get(suite, _norm_math)
    rows, skipped = [], 0
    for i in range(len(ds)):
        r = fn(suite, ds[i], i)
        if r:
            rows.append(r)
        else:
            skipped += 1
    print(f"[eval] {suite}: {len(rows)} problems ({skipped} skipped)", flush=True)
    want = EXPECTED_ROWS.get(suite)
    if want is not None and len(rows) != want:
        raise RuntimeError(
            f"{suite}: expected {want} rows at revision {rev[:8]}, got {len(rows)} "
            f"({skipped} skipped). Refusing to write a suite that is not the one "
            f"the eval plan pinned."
        )
    meta = {
        "suite": suite, "repo": repo, "config": config, "split": split,
        "revision": rev, "rows": len(rows), "skipped": skipped,
        "verifier": rows[0]["verifier"] if rows else None,
    }
    if suite == "humaneval":
        meta["grading"] = "code-exec-pending"
    return rows, meta


def _build_standard_eval(val_path: str, out_dir: str, n: int, seed: int) -> dict:
    out_path = os.path.join(out_dir, "standard_eval_300.jsonl")
    if os.path.exists(out_path):
        print(f"[eval] standard_eval_300 already exists — NOT regenerating: {out_path}",
              flush=True)
        return {"path": out_path, "rows": sum(1 for _ in open(out_path)), "regenerated": False}
    val = read_jsonl(val_path)
    rng = random.Random(seed)
    rows = stratified_sample(val, lambda r: str(r.get("level", 0)), n, rng)
    rows = [
        {**r, "suite": "standard_eval_300", "verifier": "whetstone.verify"} for r in rows
    ]
    n_w = write_jsonl(out_path, rows)
    meta = {
        "suite": "standard_eval_300", "source_file": val_path, "rows": n_w, "seed": seed,
        "sampling": "stratified by level",
        "policy": "APPEND-ONLY HISTORY — never regenerate; downstream dashboards compare across checkpoints",
    }
    write_meta(out_path, meta)
    print(f"[eval] standard_eval_300: {n_w} → {out_path}", flush=True)
    return {"path": out_path, "rows": n_w, "regenerated": True, **meta}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Build WHETSTONE v2 eval suites")
    ap.add_argument("--out_dir", required=True, help="e.g. /data/whetstone/eval")
    ap.add_argument("--suites", default=",".join(SUITES.keys()))
    ap.add_argument("--standard_eval_from", default="",
                    help="val JSONL to sample standard_eval_300 from")
    ap.add_argument("--standard_eval_n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary: dict[str, dict] = {}
    for suite in [s.strip() for s in args.suites.split(",") if s.strip()]:
        if suite not in SUITES:
            print(f"[eval] skip unknown suite: {suite}", flush=True)
            continue
        try:
            rows, meta = _pull(suite)
        except Exception as e:  # noqa: BLE001 — gated repos / transient HF failures
            print(f"[eval] FAILED {suite}: {type(e).__name__}: {e}", flush=True)
            summary[suite] = {"error": f"{type(e).__name__}: {e}", "rows": 0}
            continue
        path = os.path.join(args.out_dir, f"{suite}.jsonl")
        n = write_jsonl(path, rows)
        meta["path"] = path
        write_meta(path, meta)
        summary[suite] = meta

    if args.standard_eval_from:
        summary["standard_eval_300"] = _build_standard_eval(
            args.standard_eval_from, args.out_dir, args.standard_eval_n, args.seed
        )

    stats_path = os.path.join(args.out_dir, "eval_stats.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\n[done] eval suites:")
    for k, v in summary.items():
        note = v.get("error") or v.get("grading") or v.get("verifier") or ""
        print(f"  {k:18s} {v.get('rows', 0):6d}  {note}")
    print(f"[done] stats → {stats_path}")


if __name__ == "__main__":
    main()
