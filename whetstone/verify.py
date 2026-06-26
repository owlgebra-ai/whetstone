"""Deterministic answer comparator for WHETSTONE.

Reference implementation of the §2.8 / §6.2 / §7.4 verifier. The comparator is
*strictly* deterministic; any relaxation belongs in the reward shaping, not here.

v4.6.1 patches applied:
  * Lenient extraction reads ONLY post-</think> content. Scanning <think>
    for substring matches against gold produces false positives on symbolic
    non-numeric answers (e.g. "\\cot A" matching gold=1).
  * Coefficients before \\sqrt are masked ("3\\sqrt{82}/11" vs "\\sqrt{82}")
    so numeric-equivalence does not spuriously succeed on symbolic forms.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Optional

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|answer is|answer[:=])\s*[:\-]?\s*([^\n.]+)",
    re.IGNORECASE,
)
COEF_SQRT_RE = re.compile(r"(\d+)\s*\\sqrt")
FRAC_RE = re.compile(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}")
LATEX_FRAC_RE = re.compile(r"\\dfrac\{(-?[\d.]+)\}\{(-?[\d.]+)\}")


def _strip_think(text: str) -> str:
    """v4.6.1: lenient extraction is post-</think> only."""
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from a completion.

    Order: \\boxed{}, <answer>...</answer>, "Final Answer:", last non-empty line.
    Extraction is restricted to post-</think> content per v4.6.1.
    """
    if not text:
        return None
    post = _strip_think(text)

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


def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.strip("$").strip()
    s = s.replace("\\,", "").replace("\\ ", "").replace("\\!", "")
    # Substitute LaTeX fractions BEFORE stripping braces so \frac{a}{b} matches.
    s = LATEX_FRAC_RE.sub(r"\1/\2", s)
    s = FRAC_RE.sub(r"\1/\2", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|mathbf|mathsf)\{([^}]*)\}", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace(",", "")
    s = COEF_SQRT_RE.sub(r"\\sqrt", s)
    s = s.replace("\\%", "%").replace("\\$", "$").replace("\\#", "#")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "")
    s = s.rstrip(".").strip()
    s = s.lower()
    return s


def _try_numeric(s: str) -> Optional[float]:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except ZeroDivisionError:
            return None
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        return None


def verify_response(completion: str, ground_truth: str) -> bool:
    """Return True iff the completion's extracted answer matches the gold answer.

    Comparison cascade: exact normalized-string equality, then numeric / fraction
    equivalence (1e-6 rel tolerance). Returns False on extract failure.
    """
    pred = extract_answer(completion)
    if pred is None:
        return False
    gold = (ground_truth or "").strip()
    if not gold:
        return False

    npred = _normalize(pred)
    ngold = _normalize(gold)
    if npred and npred == ngold:
        return True

    pp = _try_numeric(npred)
    pg = _try_numeric(ngold)
    if pp is not None and pg is not None:
        if math.isclose(pp, pg, rel_tol=1e-6, abs_tol=1e-9):
            return True

    if ngold and npred.endswith(ngold):
        return True
    if npred and ngold.endswith(npred):
        return True

    return False


__all__ = ["verify_response", "extract_answer"]
