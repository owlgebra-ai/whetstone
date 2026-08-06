"""Tests for the strict-grading wrapper (packet P7 Part 0.2).

The binding cases are the ones activity 009 finding 15 measured on real
rollouts. Each is asserted **both** ways: as-scored must still accept it (so we
know the case is genuinely one of the leniencies and the test has teeth), and
strict must reject it. A test that only checked ``strict is False`` would pass
just as well against a grader that rejects everything.
"""

from __future__ import annotations

import pytest

from whetstone.reward.strict import (
    W_MISMATCH,
    W_NO_CLOSE,
    W_NO_EXTRACT,
    extract_answer_strict,
    verify_strict,
)
from whetstone.verify import verify_response


def _closed(answer_text: str, think: str = "goal: x\n⇒ 5\n") -> str:
    """A well-formed rollout whose answer segment says ``answer_text``."""
    return f"<think>\n{think}</think>\n\n{answer_text}"


# --- finding 15 case A: the suffix fallback ---------------------------------
# verify.py ends its cascade with npred.endswith(ngold) / ngold.endswith(npred).

@pytest.mark.parametrize(
    "gold,pred",
    [
        ("200", "0"),      # finding 15, verbatim
        ("90", "0"),       # finding 15, verbatim
        ("2", "42"),       # finding 15, verbatim
        ("200", "1200"),   # run 13 audit: gold 200 extracted 1200
        ("20", "0"),       # run 13 audit: gold 20 extracted 0
    ],
)
def test_suffix_fallback_is_wrong_under_strict(gold: str, pred: str) -> None:
    completion = _closed(f"\\boxed{{{pred}}}")

    # The case must actually be a leniency, or this test proves nothing.
    assert verify_response(completion, gold) is True, (
        "as-scored no longer accepts this case; the fixture is stale"
    )

    v = verify_strict(completion, gold)
    assert v.strict is False
    assert v.as_scored is True
    assert v.lenient_only is True
    assert v.reason == W_MISMATCH
    assert v.pred == pred


# --- finding 15 case B: extraction from an unclosed think block -------------
# _strip_think falls back to the whole text when </think> is absent, mining the
# scratchpad. Run 13 found 4 such cases with the right answer in the scratchpad.

@pytest.mark.parametrize("gold", ["460", "243", "16", "75"])
def test_unfinished_think_is_wrong_under_strict(gold: str) -> None:
    """The loop tail: right answer in the scratchpad, block never closed.

    Shaped after run 13's four real cases — the correct value is reachable in
    the think body (here via ``\\boxed{}``, which is where ``extract_answer``'s
    ladder starts), and the generation then loops to the cap without closing.
    """
    completion = (
        f"<think>\ngoal: find n\n⇒ \\boxed{{{gold}}}\n"
        + f"chk: {gold} ✓\n" * 200  # runaway; never closes
    )
    assert "</think>" not in completion

    assert verify_response(completion, gold) is True, (
        "as-scored no longer mines the scratchpad; the fixture is stale"
    )

    v = verify_strict(completion, gold)
    assert v.strict is False
    assert v.as_scored is True
    assert v.reason == W_NO_CLOSE
    assert v.pred is None


def test_unfinished_think_with_boxed_in_scratchpad_is_wrong() -> None:
    """009's verbatim example: a \\boxed{} inside an unclosed think block."""
    completion = "<think>\ngoal: photos\n⇒ \\boxed{6}\n" + "chk: 6 ✓\n" * 300
    assert verify_response(completion, "6") is True
    v = verify_strict(completion, "6")
    assert v.strict is False
    assert v.reason == W_NO_CLOSE


def test_extract_refuses_without_closure() -> None:
    assert extract_answer_strict("<think>\n\\boxed{42}\n") is None
    assert extract_answer_strict("") is None
    assert extract_answer_strict("\\boxed{42}") is None  # no think block at all


# --- the grader must still accept genuinely correct answers ----------------

@pytest.mark.parametrize(
    "gold,answer_text",
    [
        ("18", "\\boxed{18}"),
        ("18", "The final answer is \\boxed{18}."),
        ("0.5", "\\boxed{1/2}"),           # numeric equivalence survives
        ("1/2", "\\boxed{0.5}"),
        ("72", "<answer>72</answer>"),
        ("72", "Final Answer: 72"),
        ("-3", "\\boxed{-3}"),
        ("1000", "\\boxed{1,000}"),        # normalizer strips the comma
    ],
)
def test_correct_answers_still_pass_strict(gold: str, answer_text: str) -> None:
    v = verify_strict(_closed(answer_text), gold)
    assert v.strict is True, f"strict rejected a correct answer: {v}"
    assert v.reason == ""
    assert v.lenient_only is False


def test_genuine_wrong_answer_is_wrong_both_ways() -> None:
    v = verify_strict(_closed("\\boxed{120000}"), "70000")  # run 13's real case
    assert v.strict is False
    assert v.as_scored is False
    assert v.lenient_only is False
    assert v.reason == W_MISMATCH


def test_empty_answer_segment_has_no_extraction() -> None:
    v = verify_strict("<think>\ngoal: x\n⇒ 5\n</think>\n\n", "5")
    assert v.strict is False
    assert v.reason == W_NO_EXTRACT


def test_no_gold_is_wrong_not_a_crash() -> None:
    v = verify_strict(_closed("\\boxed{5}"), "")
    assert v.strict is False
    assert v.pred is None


def test_answer_segment_only_scratchpad_is_not_mined() -> None:
    """A closed block whose *answer* is wrong is not rescued by the think body."""
    completion = "<think>\ngoal: n\n⇒ 42\n</think>\n\nI think it is 7."
    v = verify_strict(completion, "42")
    assert v.strict is False, "strict must not read the think block"
