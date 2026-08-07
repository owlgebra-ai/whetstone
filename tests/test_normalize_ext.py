"""Fixtures for the seven deterministic equivalence classes (activity 011).

Every positive case is a verbatim shape from the step-0100 benchmark failure
analysis; every negative control pins that the extension added *equivalence*,
not *leniency* — near-misses, suffix shapes, and tolerance-sized gaps must
still grade wrong.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.reward.normalize_ext import equivalent_ext
from whetstone.reward.strict import verify_strict


# --- the seven classes, as (pred, gold) --------------------------------------

POSITIVE = [
    # 1. internal whitespace
    ("2k + 2", "2k+2"),
    ("\\pi / 3", "\\pi/3"),
    # 2. \dfrac and braceless \frac
    ("\\dfrac{4}{3}", "\\frac43"),
    ("\\dfrac{4}{3}", "4/3"),
    ("\\tfrac{1}{2}", "0.5"),
    # 3. scientific notation
    ("2.7778 \\times 10^{-6}", "2.7778e-6"),
    ("1.5 \\cdot 10^{3}", "1500"),
    # 4. degree markers
    ("45^\\circ", "45"),
    ("45^{\\circ}", "45"),
    # 5. variable-binding prefix
    ("x \\in [-2,7]", "[-2,7]"),
    ("y = 7", "7"),
    # 6. choice-letter wrapping
    ("\\text{(C)}", "C"),
    # 7. symbolic fractions (via 1+2)
    ("\\frac{\\pi}{3}", "\\pi / 3"),
]

NEGATIVE = [
    ("2k + 3", "2k+2"),                 # near-miss stays wrong
    ("0", "200"),                       # the suffix hole stays closed
    ("698", "700"),                     # NO rounding tolerance (rejected 2% rule)
    ("2.7778e-6", "2.7778e-5"),         # sci-notation exponent must match
    ("\\frac{4}{3}", "\\frac{3}{4}"),   # flipped fraction
    ("46^\\circ", "45"),                # degrees stripped, value still compared
    ("x \\in [-2,7]", "[-2,8]"),        # binder stripped, interval still compared
    ("(c)", "d"),                       # unwrapped letter still compared
]


@pytest.mark.parametrize("pred,gold", POSITIVE)
def test_extension_accepts_the_measured_classes(pred: str, gold: str) -> None:
    assert equivalent_ext(pred, gold) is True


@pytest.mark.parametrize("pred,gold", NEGATIVE)
def test_extension_adds_no_leniency(pred: str, gold: str) -> None:
    assert equivalent_ext(pred, gold) is False


@pytest.mark.parametrize("pred,gold", POSITIVE)
def test_strict_grader_accepts_through_the_extension(pred: str, gold: str) -> None:
    v = verify_strict(f"<think>work</think>\nThe answer is \\boxed{{{pred}}}.", gold)
    assert v.strict is True


def test_strict_refusals_unchanged_by_the_extension() -> None:
    """The two 009-f15 removals stay removed: no unclosed-think mining, and the
    extension never resurrects the suffix fallback."""
    unclosed = "<think>goal: x\n\\boxed{4/3}"
    assert verify_strict(unclosed, "\\dfrac{4}{3}").strict is False
    assert verify_strict("<think>t</think>\\boxed{0}", "200").strict is False
