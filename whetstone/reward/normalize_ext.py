"""Deterministic equivalence extensions over ``verify._normalize`` (activity 011).

The step-0100 benchmark failure analysis (011 Run 7) measured ``verify.py``'s
normalizer rejecting **correct** answers at +6.8 pts on MATH-500 and +5.5 on
MinervaMath — seven concrete format classes, every one a deterministic
notational equivalence, not a tolerance. Both the init and the RL model were
hit equally (the paired deltas were real), but under RL the miss is worse than
a measurement bias: R_acc = 0 on a correct rollout inverts its within-group
advantage and trains *against* standard notation (Minerva's prompts literally
demand the scientific notation the verifier could not parse).

Placement: CLAUDE.md pins ``verify.py`` deterministic and unmodified — "reward
leniency lives in whetstone/reward/". These are **equivalences, not
leniencies** (no numeric tolerance is added; a 2%-rounding proposal from the
same analysis was explicitly rejected as Goodhart bait), but the same
invariant applies: the extension lives here, ``verify.py`` moves for nobody.
``strict.py`` consults this module only after its verbatim ``_normalize`` /
``_try_numeric`` path fails, so the historical strict behaviour is a strict
subset of the new one and every acceptance this module adds is fixture-pinned
in ``tests/test_normalize_ext.py``.

The seven classes (verbatim examples from the analysis):

1. internal whitespace         ``2k + 2``               ≡ ``2k+2``
2. \\dfrac / braceless \\frac    ``\\dfrac{4}{3}``          ≡ ``\\frac43`` ≡ ``4/3``
3. scientific notation         ``2.7778 \\times 10^{-6}`` ≡ ``2.7778e-6``
4. degree markers              ``45^\\circ``              ≡ ``45``
5. variable-binding prefix     ``x \\in [-2,7]``          ≡ ``[-2,7]``; ``y = 7`` ≡ ``7``
6. choice-letter wrapping      ``\\text{(C)}``            ≡ ``C``
7. symbolic fractions          ``\\frac{\\pi}{3}``         ≡ ``\\pi / 3``  (via 1+2)
"""

from __future__ import annotations

import math
import re
from typing import Optional

from whetstone.verify import _normalize, _try_numeric

#: \dfrac / \tfrac are typographic variants of \frac.
_DFRAC_RE = re.compile(r"\\[dt]frac")
#: Braceless two-digit \frac: ``\frac43`` means 4/3 (TeX single-token args).
_BRACELESS_FRAC_RE = re.compile(r"\\frac\s*(\d)\s*(\d)")
#: Symbolic \frac{..}{..} → ../.. — verify.py's FRAC_REs are digits-only, so
#: ``\frac{\pi}{3}`` survives _normalize as ``\frac\pi3``. This is a canonical
#: *form*, not an evaluation: both sides receive the same rewrite, so compound
#: arguments still only match when textually identical.
_SYM_FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
#: ``A x 10^k`` in any of its spellings, post-_normalize (braces already gone).
_SCI_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|\*|x|×)\s*10\s*\^\s*"
    r"(-?\d+)\s*$"
)
_DEGREE_RE = re.compile(r"(?:\^\s*\\circ|\\degree|°)")
#: A single leading binder: variable(s) ∈/=/≈ value. Applied at most once,
#: only when what follows is non-empty — never turns an answer into "".
_BINDER_RE = re.compile(
    r"^\s*[a-z](?:\s*,\s*[a-z])*\s*(?:\\in|=|\\approx)\s*(?=\S)"
)
_PAREN_LETTER_RE = re.compile(r"^\(([a-z])\)$")


def _ext(s: str) -> str:
    """Extended canonical form. Deterministic, order-fixed rewrites."""
    if s is None:
        return ""
    s = _DFRAC_RE.sub(r"\\frac", s)
    s = _BRACELESS_FRAC_RE.sub(r"\1/\2", s)
    for _ in range(3):                  # nested fracs, innermost outward
        s, n = _SYM_FRAC_RE.subn(r"\1/\2", s)
        if not n:
            break
    s = _normalize(s)
    s = _DEGREE_RE.sub("", s)
    s = _BINDER_RE.sub("", s)
    s = re.sub(r"\s+", "", s)          # class 1 + 7: whitespace never separates
    m = _PAREN_LETTER_RE.match(s)
    if m:
        s = m.group(1)
    return s


def _sci_to_float(s: str) -> Optional[float]:
    m = _SCI_RE.match(s)
    if m:
        try:
            return float(m.group(1)) * (10.0 ** int(m.group(2)))
        except OverflowError:
            return None
    return None


def equivalent_ext(pred: str, gold: str) -> bool:
    """True iff pred ≡ gold under the seven extension classes.

    Called only after the verbatim strict path failed. No tolerance beyond the
    strict path's own 1e-6 relative numeric parity.
    """
    ep, eg = _ext(pred), _ext(gold)
    if ep and ep == eg:
        return True
    # Numeric parity on the extended forms — catches sci-notation vs plain,
    # and \dfrac-rewritten fractions vs decimals.
    np_ = _sci_to_float(ep)
    ng_ = _sci_to_float(eg)
    if np_ is None:
        np_ = _try_numeric(ep)
    if ng_ is None:
        ng_ = _try_numeric(eg)
    if np_ is not None and ng_ is not None:
        return math.isclose(np_, ng_, rel_tol=1e-6, abs_tol=1e-9)
    return False


__all__ = ["equivalent_ext"]
