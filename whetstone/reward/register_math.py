"""Register-aware math normalization (activity 005 finding 14).

The compact register writes ``4√2`` where the answer segment (and the
deterministic verifier) writes ``4\\sqrt{2}``. Any detector that compares a
value taken from *inside* ``<think>`` against a value taken from the *answer*
is comparing two notations, and a naive string comparison misgrades the
register's own conventions as a disagreement.

The only such detector in Stage C is the think/answer contradiction penalty
(packet P7 §1b, kept from v1 §4.10), which compares the last ``⇒`` value in the
think body against the boxed answer. Without this module that penalty would
fire on correct rollouts whose only crime is writing mathematics the way the
register was designed to write it — a style tax dressed up as a correctness
signal, which is exactly the failure this project keeps finding.

Scope: this normalizer exists for *comparing two of the model's own strings*.
It is **not** an answer grader and must never be used as one — grading goes
through :mod:`whetstone.verify` (as-scored) or :mod:`whetstone.reward.strict`.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

#: Unicode → ASCII/LaTeX-neutral single-character substitutions.
_CHAR_MAP = {
    "−": "-",     # U+2212 minus
    "–": "-",     # en dash used as minus
    "—": "-",
    "×": "*",
    "·": "*",
    "∗": "*",
    "÷": "/",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "±": "+-",
    "π": "pi",
    "∞": "inf",
    "°": "deg",
    "⁄": "/",
    ",": "",      # thousands separators; the register does not use decimal commas
    " ": "",      # NBSP
    "\u2009": "",  # thin space
}

_LATEX_STRIP = (
    r"\left", r"\right", r"\,", r"\;", r"\!", r"\ ", r"\quad", r"\qquad",
    r"\displaystyle", r"\dollar", "$",
)

_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT_BRACED_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_SQRT_BARE_RE = re.compile(r"\\sqrt\s*([0-9a-zA-Z]+)")
_UNI_SQRT_PAREN_RE = re.compile(r"√\s*\(([^()]*)\)")
_UNI_SQRT_RE = re.compile(r"√\s*([0-9a-zA-Z]+)")
_TEXT_RE = re.compile(r"\\(?:text|mathrm|mathbf|mathsf|mathit)\s*\{([^{}]*)\}")
_CMD_RE = re.compile(r"\\(?:cdot|times)")
_POW_RE = re.compile(r"\^\s*\{([^{}]*)\}")
_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_FRACTION_RE = re.compile(r"^(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)$")


def normalize_register_math(s: Optional[str]) -> str:
    """Canonicalize a math expression written in either notation.

    ``4√2``, ``4\\sqrt{2}`` and ``4 \\sqrt 2`` all normalize to ``4sqrt(2)``.
    Returns ``""`` for ``None`` / empty input.
    """
    if not s:
        return ""
    out = s.strip()

    # LaTeX structure first — these carry braces the char map would mangle.
    out = _TEXT_RE.sub(r"\1", out)
    out = _FRAC_RE.sub(r"(\1)/(\2)", out)
    out = _SQRT_BRACED_RE.sub(r"sqrt(\1)", out)
    out = _SQRT_BARE_RE.sub(r"sqrt(\1)", out)
    out = _UNI_SQRT_PAREN_RE.sub(r"sqrt(\1)", out)
    out = _UNI_SQRT_RE.sub(r"sqrt(\1)", out)
    out = _POW_RE.sub(r"^\1", out)
    out = _CMD_RE.sub("*", out)

    for tok in _LATEX_STRIP:
        out = out.replace(tok, "")
    for src, dst in _CHAR_MAP.items():
        out = out.replace(src, dst)

    out = out.replace("{", "").replace("}", "")
    out = re.sub(r"\s+", "", out)
    out = out.rstrip(".").lower()
    return out


def to_number(s: Optional[str]) -> Optional[float]:
    """Best-effort numeric value of a normalized expression, else ``None``."""
    if not s:
        return None
    t = normalize_register_math(s)
    if _NUMERIC_RE.match(t):
        try:
            return float(t)
        except ValueError:
            return None
    m = _FRACTION_RE.match(t)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(Fraction(t))
    except (ValueError, ZeroDivisionError):
        return None


def values_agree(
    a: Optional[str],
    b: Optional[str],
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> Optional[bool]:
    """Do two expressions denote the same value?

    Returns ``True`` / ``False``, or **``None`` when undecidable** — either side
    missing, or both sides non-numeric and not string-identical after
    normalization. ``None`` is not "disagree": the contradiction penalty must
    stay silent on evidence it does not have, or it becomes a tax on symbolic
    answers (005 finding 14).
    """
    if not a or not b:
        return None
    na, nb = normalize_register_math(a), normalize_register_math(b)
    if not na or not nb:
        return None
    if na == nb:
        return True

    va, vb = to_number(a), to_number(b)
    if va is not None and vb is not None:
        import math

        return math.isclose(va, vb, rel_tol=rel_tol, abs_tol=abs_tol)

    # One or both symbolic and textually different — cannot decide safely.
    return None


__all__ = ["normalize_register_math", "to_number", "values_agree"]
