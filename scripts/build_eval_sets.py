"""Build eval suites for WHETSTONE.

Pulls each of the 7 benchmark suites and writes one JSONL per suite under
<out_dir>/<suite>.jsonl with normalized fields:

    {
        "_uid": "<suite>:<id>",
        "prompt": "<problem statement>",
        "ground_truth": "<gold answer / rubric / formal statement>",
        "suite": "aime24" | "aime25" | ... | "minif2f",
        "subject": "...",
        "level": "...",
        "verifier": "whetstone.verify" | "lean" | "judge",
    }

Numeric suites (AIME/HMMT) use whetstone.verify for grading.

Putnam-2025 is rubric-graded (no single numeric answer): verifier=judge, and
ground_truth carries the JSON rubric so a downstream LLM judge can score it.

miniF2F (openai/miniF2F on GitHub) is Lean theorem proving: verifier=lean,
ground_truth is the formal theorem statement. Grading requires Lean tooling
that is NOT bundled here — the eval runner emits generations but flags them
for separate verification.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
THEOREM_RE = re.compile(
    r"^theorem\s+(\S+)\s*\n((?:.*\n)*?)\s*:=\s*\n",
    re.MULTILINE,
)


def _write_jsonl(path: str, rows: list[dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _first(rec: dict, *names: str):
    for n in names:
        if n in rec and rec[n] not in (None, ""):
            return rec[n]
    return None


def _norm(s) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _norm_hf(suite: str, ds, idx: int, verifier: str = "whetstone.verify") -> dict:
    rec = ds[idx] if hasattr(ds, "__getitem__") else ds
    prompt = _first(rec, "Problem", "problem", "question", "prompt", "input")
    gold = _first(rec, "Answer", "answer", "gold", "ground_truth",
                  "Gold", "expected_answer")
    if gold is None:
        # math-ai/aime24 only ships `solution` with \boxed{N}; extract it.
        sol = _first(rec, "solution", "Solution")
        if sol:
            m = BOXED_RE.search(str(sol))
            if m:
                gold = m.group(1).strip()
    if prompt is None or gold is None:
        return {}
    sid = _first(rec, "id", "ID", "Id", "name", "uuid", "index", "problem_idx") or idx
    subject = _first(rec, "subject", "topic", "category", "problem_type") or ""
    level = _first(rec, "level", "difficulty") or ""
    return {
        "_uid": f"{suite}:{sid}",
        "prompt": _norm(prompt),
        "ground_truth": _norm(gold),
        "suite": suite,
        "subject": str(subject),
        "level": str(level),
        "verifier": verifier,
    }


def _norm_putnam(suite: str, ds, idx: int) -> dict:
    """Putnam-2025 is rubric-graded; carry the rubric JSON as ground_truth and
    mark verifier=judge so the eval runner skips verify_response."""
    rec = ds[idx] if hasattr(ds, "__getitem__") else ds
    prompt = _first(rec, "Problem", "problem", "question", "prompt")
    rubric = _first(rec, "grading_scheme", "rubric", "Grading_Scheme")
    if prompt is None:
        return {}
    sid = _first(rec, "id", "ID", "Id", "problem_idx", "name") or idx
    gold = json.dumps(rubric, ensure_ascii=False) if rubric else ""
    return {
        "_uid": f"{suite}:{sid}",
        "prompt": _norm(prompt),
        "ground_truth": gold,
        "suite": suite,
        "subject": "putnam",
        "level": "",
        "verifier": "judge",
    }


def _pull_hf(suite: str, repo: str, split: str = "test",
             verifier: str = "whetstone.verify",
             normalizer=_norm_hf) -> list[dict]:
    from datasets import load_dataset
    print(f"[eval] loading {repo} ({split})...", flush=True)
    try:
        ds = load_dataset(repo, split=split)
    except Exception:
        ds = load_dataset(repo, split="train")
    rows = []
    for i in range(len(ds)):
        rec = normalizer(suite, ds, i, verifier=verifier) if normalizer is _norm_hf \
              else normalizer(suite, ds, i)
        if rec:
            rows.append(rec)
    print(f"[eval] {suite}: {len(rows)} problems", flush=True)
    return rows


def _parse_lean_theorems(text: str, split: str) -> list[dict]:
    """Parse `theorem <name> ... :=` blocks from a miniF2F .lean file.
    Walks line-by-line to avoid catastrophic regex backtracking on the
    multi-line signature blocks."""
    out: list[dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if not stripped.startswith("theorem"):
            i += 1
            continue
        # First line: "theorem <name>" possibly with the start of the signature.
        header = stripped
        parts = header.split(None, 2)
        if len(parts) < 2:
            i += 1
            continue
        name = parts[1]
        # Collect signature lines until we hit a line ending in `:=`
        # (the body terminator) — typically `:=` followed by `begin` or `by`.
        sig_lines = [header[len("theorem"):].lstrip()]  # everything after `theorem `
        j = i + 1
        end = None
        while j < len(lines):
            cur = lines[j].rstrip()
            sig_lines.append(lines[j])
            if cur.endswith(":=") or cur.endswith(": ="):
                end = j
                break
            j += 1
        if end is None:
            i += 1
            continue
        sig = "\n".join(sig_lines).rstrip()
        # Trim a trailing `:=` line back to the signature
        sig = sig.rsplit(":=", 1)[0].rstrip()
        stmt = f"theorem {sig}\n  := by\n"
        out.append({
            "_uid": f"minif2f:{split}:{name}",
            "prompt": (
                f"Prove the following Lean 3 theorem. Provide only the proof "
                f"body (after `:= by`), with no commentary.\n\n{stmt}"
            ),
            "ground_truth": stmt,
            "suite": "minif2f",
            "subject": split,
            "level": "",
            "verifier": "lean",
        })
        i = end + 1
    return out


def _pull_minif2f() -> list[dict]:
    """miniF2F lives in openai/miniF2F on GitHub as `lean/src/{test,valid}.lean`.
    Parse theorem blocks; grading requires a Lean compiler."""
    base = "https://raw.githubusercontent.com/openai/miniF2F/main/lean/src"
    rows: list[dict] = []
    for split in ("test", "valid"):
        url = f"{base}/{split}.lean"
        print(f"[eval] miniF2F/{split}: {url}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[eval] WARN: miniF2F/{split} pull failed: {e}", flush=True)
            continue
        parsed = _parse_lean_theorems(text, split)
        rows.extend(parsed)
        print(f"[eval] miniF2F/{split}: {len(parsed)} theorems", flush=True)
    print(f"[eval] minif2f: {len(rows)} theorems total (Lean — separate verification)",
          flush=True)
    return rows


SUITES_HF = {
    "aime24": ("math-ai/aime24", "whetstone.verify", _norm_hf),
    "aime25": ("math-ai/aime25", "whetstone.verify", _norm_hf),
    "aime26": ("math-ai/aime26", "whetstone.verify", _norm_hf),
    "hmmt_feb_2025": ("MathArena/hmmt_feb_2025", "whetstone.verify", _norm_hf),
    "hmmt_feb_2026": ("MathArena/hmmt_feb_2026", "whetstone.verify", _norm_hf),
    "putnam_2025": ("MathArena/putnam_2025", "judge", _norm_putnam),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Build WHETSTONE eval suites")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--suites", default=",".join(list(SUITES_HF.keys()) + ["minif2f"]),
                    help="Comma list of suites to build")
    ap.add_argument("--split", default="test",
                    help="HF split for the math-answer suites")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    wanted = [s.strip() for s in args.suites.split(",") if s.strip()]
    summary = {}
    for suite in wanted:
        if suite == "minif2f":
            rows = _pull_minif2f()
        elif suite in SUITES_HF:
            repo, verifier, fn = SUITES_HF[suite]
            rows = _pull_hf(suite, repo, args.split, verifier=verifier, normalizer=fn)
        else:
            print(f"[eval] skip unknown suite: {suite}", flush=True)
            continue
        out_path = os.path.join(args.out_dir, f"{suite}.jsonl")
        n = _write_jsonl(out_path, rows)
        summary[suite] = n
    print("[done] eval suites:")
    for k, v in summary.items():
        print(f"  {k:18s} {v:6d}  -> {os.path.join(args.out_dir, k + '.jsonl')}")


if __name__ == "__main__":
    main()
