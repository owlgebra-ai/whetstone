"""Post-``</think>`` extraction and chunk parsing.

Every downstream scorer operates on either the ``<think>`` body (chunk-level
detectors, sentinel-phrase detector) or the post-``</think>`` block (terminal-
answer detectors, register-leakage detector). This module is the single source
of truth for those splits.

Guarantees:
    - ``split_think_close`` never returns ``None`` for either half; missing
      pieces come back as empty strings so downstream code can treat "no
      closure" as "post-think == ''" without ``None`` checks.
    - Chunks are ``\\n\\n``-separated, non-empty after ``strip()``.
    - The register-preservation chunk detector recognises numbered restarts
      (``^\\d+\\.``) at chunk start, matching the Gemma-4 compact SFT prior.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from whetstone.verify import extract_answer, verify_response

from .config import PROSE_TEMPLATED_ANSWER_RE

THINK_CLOSE = "</think>"

_CHUNK_RESTART_RE = re.compile(r"^\s*(\d+)\.\s")
_LINE_LEADING_NUM_RE = re.compile(r"^\s*(\d+)\.\s", re.MULTILINE)
_BOXED_FULLMATCH_RE = re.compile(
    r"^\s*\\?boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\.?\s*$"
)
_BARE_NUMERIC_FULLMATCH_RE = re.compile(
    r"^\s*-?\d+(?:[.,]\d+)*(?:\s*/\s*-?\d+(?:[.,]\d+)*)?\s*\.?\s*$"
)
# Symbolic terminal patterns: single \boxed{...}, single LaTeX expression in $...$,
# or bare identifier / letter for MCQ. Kept narrow — anything else counts as prose.
_SYMBOLIC_TERMINAL_FULLMATCH_RE = re.compile(
    r"^\s*(?:\$[^$]+\$|[A-Za-z]|\([A-Za-z]\)|\\text\{[^}]+\})\s*\.?\s*$"
)
# §6.7 prose-templated strict extractor (fallback after boxed/bare/verbose fail).
# Two patterns:
#   (a) Prefix templates: **Answer:** X, Answer: X, Final Answer: X, "The X is N."
#   (b) Bare markdown-bold: **X**  (§6.7 example: "**X** (markdown-bold bare)")
_PROSE_TEMPLATED_RE = re.compile(PROSE_TEMPLATED_ANSWER_RE, re.IGNORECASE)
_PROSE_TEMPLATED_BOLD_BARE_RE = re.compile(r"^\s*\*\*([^*]+?)\*\*\s*\.?\s*$")
# §6.11 final-block: locate the LAST \boxed{...} in post-think to define the
# byte range that r_acc considered the terminal commit.
_BOXED_SEARCH_RE = re.compile(r"\\?boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
# §4.10 last-numeric-in-think detector. Accepts integers, decimals, fractions,
# and simple negatives. Deliberately narrow — a broad numeric regex misfires on
# LaTeX identifiers and inflates false-positive contradiction flags.
_THINK_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?")
# §6.10 malformed / duplicate \boxed{} tokens.
_BOXED_OPEN_RE = re.compile(r"\\boxed\{")


@dataclass(frozen=True)
class SplitCompletion:
    """Result of splitting a completion around ``</think>``.

    ``final_block`` (§6.11): the substring of ``post_think`` from start up to
    and including the last terminal-answer commit (``\\boxed{}`` if present,
    otherwise the last non-empty line). Both ``r_acc`` and ``r_fmt`` scorers
    reference this field so structural detectors see the exact bytes that
    the verifier considered the "answer" — trailing prose after a mid-post-
    think ``\\boxed{}`` is not scored as if it were the terminal answer.
    """

    think: str          # everything before </think> (empty if closure absent)
    post_think: str     # everything after </think> (empty if closure absent)
    has_closed_think: bool
    final_block: str    # §6.11 single-source-of-truth final block


def _compute_final_block(post_think: str) -> str:
    """§6.11: post_think substring up to and including the last terminal-commit token.

    Rules:
      1. If any ``\\boxed{...}`` (or backspace-eaten ``oxed{...}``) appears,
         return the prefix ending at that last box's closing brace.
      2. Otherwise, return the prefix ending at the last non-empty line.
      3. If ``post_think`` is empty or blank, return "".

    Returning a *substring* (not just an offset) lets downstream detectors
    treat it as an ordinary string. Trailing whitespace after the commit is
    preserved when it's part of the block; the terminal-line detectors strip
    it themselves.
    """
    if not post_think or not post_think.strip():
        return ""
    matches = list(_BOXED_SEARCH_RE.finditer(post_think))
    if matches:
        return post_think[: matches[-1].end()]
    # No box — walk back from end to find the last non-empty line.
    stripped = post_think.rstrip()
    if not stripped:
        return ""
    return stripped


def split_think_close(completion: str) -> SplitCompletion:
    """Return the ``<think>`` body and post-``</think>`` block for ``completion``.

    If ``</think>`` is absent the whole string counts as the think body and the
    post-think is empty. The lone-side representation is convenient for the
    §4.9 cap-hit branch, which only rewards presence of the closure — the
    post-think being empty is already the correct downstream state.

    Also computes and attaches ``final_block`` per §6.11.
    """
    if completion is None:
        return SplitCompletion(think="", post_think="", has_closed_think=False, final_block="")
    if THINK_CLOSE in completion:
        think, post = completion.split(THINK_CLOSE, 1)
        return SplitCompletion(
            think=think,
            post_think=post,
            has_closed_think=True,
            final_block=_compute_final_block(post),
        )
    return SplitCompletion(think=completion, post_think="", has_closed_think=False, final_block="")


def split_chunks(text: str) -> List[str]:
    """Split ``text`` on ``\\n\\n`` boundaries; drop empties after strip."""
    if not text:
        return []
    return [c for c in text.split("\n\n") if c.strip()]


def count_chunk_restarts(think: str) -> int:
    """Count chunks starting with a numbered restart (``1.``, ``2.``, ...).

    Used by:
      - §3 ``chunk_restart_present`` structural reward.
      - §4.5 monolithic-think anomaly (``count == 0``).
    """
    return sum(1 for chunk in split_chunks(think) if _CHUNK_RESTART_RE.match(chunk))


def max_numbered_step(think: str) -> int:
    """Return the maximum numbered-step index anywhere in ``think``.

    Used by §4.5 to detect counter-restart evasion (steps past 50).
    Scans every line so it catches steps embedded inside long chunks as well
    as chunk-leading numbers.
    """
    if not think:
        return 0
    hits = _LINE_LEADING_NUM_RE.findall(think)
    if not hits:
        return 0
    try:
        return max(int(h) for h in hits)
    except ValueError:
        return 0


def last_line(block: str) -> str:
    """Return the last non-empty line of ``block`` (stripped)."""
    if not block:
        return ""
    for line in reversed(block.strip().splitlines()):
        s = line.strip()
        if s:
            return s
    return ""


# Legacy alias — some early callers pass ``post_think`` directly.
def post_think_last_line(post_think: str) -> str:
    """Deprecated alias for :func:`last_line`; kept for backward compat."""
    return last_line(post_think)


def terminal_is_boxed(final_block: str) -> bool:
    """§3 / §6.11: True iff ``final_block`` last line is ``\\boxed{...}``.

    Accepts both ``\\boxed{...}`` and the backspace-eaten ``oxed{...}`` variant
    per §6.9. Trailing period tolerated. Operates on the §6.11 final block —
    NOT the raw post-think — so a mid-post-think box followed by trailing
    prose still qualifies as a boxed terminal.
    """
    last = last_line(final_block)
    if not last:
        return False
    return bool(_BOXED_FULLMATCH_RE.match(last))


def terminal_is_bare(final_block: str) -> bool:
    """§3 / §6.11: True iff ``final_block`` last line is a bare numeric / symbolic answer."""
    last = last_line(final_block)
    if not last:
        return False
    if _BARE_NUMERIC_FULLMATCH_RE.match(last):
        return True
    if _SYMBOLIC_TERMINAL_FULLMATCH_RE.match(last):
        return True
    return False


def is_clean_post_think(final_block: str) -> bool:
    """Return True iff the final block is a clean terminal-answer commit.

    Clean = ≤ 3 non-empty lines with terminal in one of the accepted
    terminal-answer templates from §2.1 + §6.7:
      * bare numeric / short symbolic (``terminal_is_bare``)
      * ``\\boxed{...}`` (``terminal_is_boxed``)
      * prose-templated finalizer (``**Answer:** X``, ``The X is N.``, ``**X**``)

    Non-clean markers: numbered-step lines (``1. Define ...``) — these are
    unambiguous chunk-register leakage. Bolded headings are NOT rejected here
    because ``**Answer:**`` is a legitimate §6.7 finalizer; the §4.7 register-
    leak penalty handles illegitimate ones (``**Base Geometry:**``) via its
    own detector, which excludes finalizer forms.

    Per §6.11, this operates on the final block r_acc extracted from — NOT
    the raw post-think — so trailing prose after ``\\boxed{}`` doesn't count.
    """
    if not final_block:
        return False
    lines = [ln.strip() for ln in final_block.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    if len(lines) > 3:
        return False
    last = last_line(final_block)
    terminal_ok = (
        terminal_is_boxed(final_block)
        or terminal_is_bare(final_block)
        or line_is_prose_templated_finalizer(last)
    )
    if not terminal_ok:
        return False
    for line in lines:
        # Numbered step ("1. Define ..." leaking from <think>).
        if re.match(r"^\d+\.\s", line):
            return False
    return True


def last_k_window(post_think: str, k: int) -> str:
    """Return the last ``k`` characters of ``post_think`` (rstripped)."""
    if not post_think:
        return ""
    return post_think.rstrip()[-k:] if k > 0 else ""


def char_len(text: Optional[str]) -> int:
    """Character length; treats ``None`` as 0."""
    return len(text) if text else 0


def _template_normalize_chunk(chunk: str, prefix_tokens: int = 12) -> str:
    """Normalize a chunk for template-based repetition detection (§4.3 stage 2).

    Steps:
      1. Take the first ``prefix_tokens`` whitespace-separated tokens.
      2. Strip numeric literals, LaTeX identifiers, parenthesized args, signs.
      3. Collapse whitespace.

    The remaining shape captures the *linguistic template* of the chunk
    ("Wait, is it possible the answer is X? No.") so cycling-candidate
    paraphrase loops with varying X still collide on the normalized form.
    """
    tokens = chunk.strip().split()[:prefix_tokens]
    s = " ".join(tokens)
    # Strip LaTeX inline math.
    s = re.sub(r"\$[^$]*\$", " ", s)
    # Strip \boxed{}/\triangle-style commands with their argument.
    s = re.sub(r"\\[A-Za-z]+\{[^{}]*\}", " ", s)
    # Strip parenthesized arguments (may contain numerics/symbols).
    s = re.sub(r"\([^)]*\)", " ", s)
    # Strip numeric literals and signs.
    s = re.sub(r"[-+]?\d+(?:[.,]\d+)*", " ", s)
    # Strip stray LaTeX escapes and identifiers we don't want to distinguish on.
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    # Collapse whitespace and lowercase.
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def consecutive_runs(items: List[str]) -> List[Tuple[str, int]]:
    """Group ``items`` into ``(value, run_length)`` pairs for consecutive equals.

    Used by both the exact-string and template-normalized repetition detectors
    to convert a chunk list into "runs" whose length is checked against the
    §4.3 thresholds.
    """
    if not items:
        return []
    runs: List[Tuple[str, int]] = []
    current = items[0]
    length = 1
    for value in items[1:]:
        if value == current:
            length += 1
        else:
            runs.append((current, length))
            current = value
            length = 1
    runs.append((current, length))
    return runs


# ---------------------------------------------------------------------------
# §6.7 — Prose-templated strict extractor
# ---------------------------------------------------------------------------

def extract_prose_templated_answer(line: str) -> Optional[str]:
    """§6.7 fallback extractor for prose-templated finalizers.

    Applied to a single (terminal) line. Returns the captured answer value
    or ``None`` if the line does not match a prose template.

    Handles (in order of specificity):
      1. Prefix templates: ``**Answer:** X``, ``Answer: X``, ``Final Answer: X``,
         ``Final result is X``, ``The {quantity} is {N}.``
      2. Bare markdown-bold: ``**X**``
    """
    if not line:
        return None
    stripped = line.strip()
    # (1) Prefix templates.
    m = _PROSE_TEMPLATED_RE.match(stripped)
    if m:
        captured = (m.group(1) or "").strip().strip("*").rstrip(".").strip()
        if captured:
            return captured
    # (2) Bare markdown-bold.
    m = _PROSE_TEMPLATED_BOLD_BARE_RE.match(stripped)
    if m:
        captured = (m.group(1) or "").strip().rstrip(".").strip()
        return captured or None
    return None


def line_is_prose_templated_finalizer(line: str) -> bool:
    """Return True iff ``line`` matches a §6.7 prose-templated finalizer.

    Used by :func:`is_clean_post_think` and by the §4.7 register-leak detector
    to whitelist legitimate finalizer forms — otherwise ``**Answer:** X`` gets
    both accepted for accuracy AND penalised as register-leakage, breaking
    §5.3 (bonus-vs-penalty consistency).
    """
    return extract_prose_templated_answer(line) is not None


def candidate_matches_gold(candidate: Optional[str], gold: str) -> bool:
    """Reuse :func:`whetstone.verify.verify_response` normalization.

    Wraps ``candidate`` as ``\\boxed{candidate}`` so the boxed-first branch of
    ``extract_answer`` picks it up unambiguously, then delegates to
    ``verify_response`` for the full normalization / numeric-equivalence
    cascade (§6.2, §6.4, §6.8). Any relaxation of the equivalence rule should
    happen inside :mod:`whetstone.verify`, not here.
    """
    if not candidate or not gold:
        return False
    synthetic = "\\boxed{" + candidate + "}"
    return verify_response(synthetic, gold)


def prose_templated_matches_gold(final_block: str, gold: str) -> Optional[str]:
    """§6.7 strict-tier fallback: try prose-templated extractor on the terminal line.

    Returns the captured candidate string iff it matches ``gold``, otherwise
    ``None``. Used by :func:`whetstone.reward.tiers.classify_tier` to promote
    a rollout to strict when the primary verifier (boxed / bare / "Final
    Answer:") missed a legitimate prose finalizer.
    """
    last = last_line(final_block)
    if not last:
        return None
    candidate = extract_prose_templated_answer(last)
    if candidate is None:
        return None
    return candidate if candidate_matches_gold(candidate, gold) else None


# ---------------------------------------------------------------------------
# §4.10 — Contradiction detection helpers
# ---------------------------------------------------------------------------

def extract_last_numeric_in_think_tail(think: str, tail_chars: int = 200) -> Optional[str]:
    """§4.10: return the last numeric literal in the final ``tail_chars`` of ``<think>``.

    Deliberately narrow — only matches integers, decimals, and simple
    fractions. Broad regexes (e.g. LaTeX identifiers) inflate false-positive
    contradiction flags on legitimate arithmetic rewrites.
    """
    if not think:
        return None
    tail = think[-tail_chars:]
    hits = _THINK_NUMERIC_RE.findall(tail)
    if not hits:
        return None
    return hits[-1].strip()


def extract_terminal_answer_from_post_think(post_think: str) -> Optional[str]:
    """Return the terminal answer of ``post_think`` via the deterministic verifier.

    Reuses :func:`whetstone.verify.extract_answer`, which:
      * strips ``<think>`` (no-op here since ``post_think`` starts post-close),
      * prefers ``\\boxed{}``, then ``<answer>...``, then ``Final Answer: X``,
      * falls back to the last non-empty line.

    Returns ``None`` when nothing extractable is present (e.g. cap-hit rollouts).
    """
    if not post_think:
        return None
    return extract_answer(post_think)


def numerical_agree(
    a: Optional[str],
    b: Optional[str],
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> bool:
    """§4.10: return True iff ``a`` and ``b`` agree symbolically or numerically.

    Semantics:
      * Either being ``None`` counts as agreement (can't detect contradiction
        with missing evidence — avoids false-positive contradictions on
        cap-hit or non-numeric rollouts).
      * Delegates to :func:`candidate_matches_gold` for the primary
        comparison (uses the full verifier normalization cascade).
      * As a lightweight secondary path, tries ``float`` conversion for both
        sides and compares via :func:`math.isclose` with the design-doc
        tolerances (§4.10 numeric agreement).
    """
    if a is None or b is None:
        return True
    if candidate_matches_gold(a, b) or candidate_matches_gold(b, a):
        return True
    try:
        fa = float(a.replace(",", ""))
        fb = float(b.replace(",", ""))
    except (ValueError, AttributeError):
        return False
    return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)


# ---------------------------------------------------------------------------
# §6.10 — Malformed / duplicate \boxed{} helpers
# ---------------------------------------------------------------------------

def count_boxed_opens(final_block: str) -> int:
    """§6.10 detector-1: count ``\\boxed{`` occurrences in the final block."""
    if not final_block:
        return 0
    return len(_BOXED_OPEN_RE.findall(final_block))


def has_malformed_boxed_literal(final_block: str) -> bool:
    """§6.10 detector-2: True iff the literal token ``{boxed{`` appears.

    Evidence of an escape-eaten duplicate box (``\\boxed{X}{boxed{X}}``
    pattern). Kept as its own detector because the double-open regex above
    misses the collapsed variant.
    """
    if not final_block:
        return False
    from .config import MALFORMED_BOXED_LITERAL_TOKEN  # avoid circular at import time
    return MALFORMED_BOXED_LITERAL_TOKEN in final_block


__all__ = [
    "THINK_CLOSE",
    "SplitCompletion",
    "split_think_close",
    "split_chunks",
    "count_chunk_restarts",
    "max_numbered_step",
    "last_line",
    "post_think_last_line",
    "terminal_is_boxed",
    "terminal_is_bare",
    "is_clean_post_think",
    "last_k_window",
    "char_len",
    "consecutive_runs",
    "_template_normalize_chunk",
    # §6.7
    "extract_prose_templated_answer",
    "candidate_matches_gold",
    "prose_templated_matches_gold",
    # §4.10
    "extract_last_numeric_in_think_tail",
    "extract_terminal_answer_from_post_think",
    "numerical_agree",
    # §6.10
    "count_boxed_opens",
    "has_malformed_boxed_literal",
]
