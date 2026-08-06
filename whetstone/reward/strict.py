"""Strict-grading wrapper over :mod:`whetstone.verify` (packet P7 Part 0.2).

``verify.py`` stays deterministic and **unmodified** (CLAUDE.md invariant:
"Reward leniency lives in ``whetstone/reward/``, never in ``verify.py``").
This module is where the leniency is taken back out again, for the one caller
that cannot tolerate it: the Stage-C RL reward.

Why RL cannot use the as-scored verifier
----------------------------------------
Activity 009 finding 15 measured ``verify.py``'s two relaxations inflating a
*degenerate* model 14× more than the baseline (+3.75 pts on the round-2 student
vs +0.27 pts on the original checkpoint). Both are holes a policy-gradient
optimizer will find and sit in:

1. **Suffix fallback** — ``verify_response`` ends its cascade with
   ``npred.endswith(ngold) or ngold.endswith(npred)``, so gold ``200`` with
   prediction ``0`` grades **correct**. Under RL this rewards emitting short
   numeric fragments that happen to be suffixes of the gold.
2. **Missing-closure extraction** — ``_strip_think`` falls back to the *whole*
   text when ``</think>`` is absent, so a runaway generation that never closed
   its think block gets its scratchpad mined for ``\\boxed{}``. 009 found 4 such
   cases in 1,600 candidates: "the model knew it and could not stop". Under RL
   this pays the loop tail (2.7–3.5% of generations) for not terminating.

The strict grader removes exactly those two behaviours and **nothing else**.
Normalization (:func:`whetstone.verify._normalize`) and numeric coercion
(:func:`whetstone.verify._try_numeric`) are imported verbatim rather than
re-implemented, so the two graders can only ever differ in the two documented
ways. If the normalizer changes, both move together.

Reporting contract
------------------
Both numbers are reported side by side everywhere (packet P7 §3, finding 15).
:func:`grade` returns both in one pass so a caller cannot accidentally report
one and label it the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from whetstone.verify import (
    ANSWER_TAG_RE,
    BOXED_RE,
    FINAL_ANSWER_RE,
    _normalize,
    _try_numeric,
    verify_response,
)

THINK_CLOSE = "</think>"

#: Reasons a rollout is graded wrong. Logged per rollout so the dashboards can
#: separate "reasoned to the wrong number" from "never produced an answer".
W_NO_CLOSE = "no_think_close"       # ran to the cap; nothing is an answer
W_NO_EXTRACT = "no_answer_extracted"
W_NO_GOLD = "no_gold"
W_MISMATCH = "mismatch"


@dataclass(frozen=True)
class StrictVerdict:
    """Outcome of grading one rollout under both graders.

    Attributes
    ----------
    strict : bool
        The RL reward signal. Exact-normalized or numeric-equivalent match on
        an answer extracted from **post-``</think>`` content that exists**.
    as_scored : bool
        ``whetstone.verify.verify_response`` — the historical number, kept for
        continuity with the baseline card and every pre-P7 measurement.
    pred : str | None
        The extracted prediction (``None`` when extraction was refused or
        failed). Under ``strict`` semantics, refused when ``</think>`` is absent.
    reason : str
        ``""`` when ``strict`` is True, else one of the ``W_*`` constants.
    lenient_only : bool
        ``as_scored and not strict`` — the rollout survives only on one of the
        two relaxations. A rising rate of these during RL means the policy has
        found the grading hole; it is a first-class dashboard curve.
    """

    strict: bool
    as_scored: bool
    pred: Optional[str]
    reason: str

    @property
    def lenient_only(self) -> bool:
        return self.as_scored and not self.strict


def extract_answer_strict(text: str) -> Optional[str]:
    """Extract a final answer, refusing when ``</think>`` never closed.

    Same extraction ladder as :func:`whetstone.verify.extract_answer`
    (``\\boxed{}`` → ``<answer>`` → "Final Answer:" → last non-empty line),
    applied to post-``</think>`` content only. The difference from the
    as-scored path is the refusal: no closure means there is no answer segment,
    so there is nothing to extract — not "search the scratchpad instead".

    Returns ``None`` when the block never closed or nothing matched.
    """
    if not text or THINK_CLOSE not in text:
        return None
    post = text.split(THINK_CLOSE, 1)[1]

    m = BOXED_RE.search(post)
    if m:
        return m.group(1).strip()

    m = ANSWER_TAG_RE.search(post)
    if m:
        return m.group(1).strip()

    matches = FINAL_ANSWER_RE.findall(post)
    if matches:
        return matches[-1].strip().rstrip(".,;")

    lines = [ln.strip() for ln in post.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return None


def verify_strict(completion: str, ground_truth: str) -> StrictVerdict:
    """Grade one rollout strictly, and report the as-scored verdict alongside.

    Strict acceptance is **exact normalized equality or numeric equivalence**
    (1e-6 relative). There is no suffix fallback and no extraction from an
    unclosed think block.
    """
    as_scored = verify_response(completion, ground_truth)
    gold = (ground_truth or "").strip()

    if not completion or THINK_CLOSE not in (completion or ""):
        return StrictVerdict(False, as_scored, None, W_NO_CLOSE)
    if not gold:
        return StrictVerdict(False, as_scored, None, W_NO_GOLD)

    pred = extract_answer_strict(completion)
    if pred is None:
        return StrictVerdict(False, as_scored, None, W_NO_EXTRACT)

    npred = _normalize(pred)
    ngold = _normalize(gold)

    if npred and npred == ngold:
        return StrictVerdict(True, as_scored, pred, "")

    pp = _try_numeric(npred)
    pg = _try_numeric(ngold)
    if pp is not None and pg is not None and math.isclose(
        pp, pg, rel_tol=1e-6, abs_tol=1e-9
    ):
        return StrictVerdict(True, as_scored, pred, "")

    # NO suffix fallback here. This is the whole point of the module.
    return StrictVerdict(False, as_scored, pred, W_MISMATCH)


def grade(completion: str, ground_truth: str) -> StrictVerdict:
    """Alias for :func:`verify_strict` — the name call sites read best with."""
    return verify_strict(completion, ground_truth)


__all__ = [
    "StrictVerdict",
    "verify_strict",
    "grade",
    "extract_answer_strict",
    "W_NO_CLOSE",
    "W_NO_EXTRACT",
    "W_NO_GOLD",
    "W_MISMATCH",
]
