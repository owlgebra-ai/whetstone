"""The Stage-C reward test battery (packet P7 §1b — MANDATORY GREEN before pilot step 1).

Same discipline as Round 0's meter tests: the reward is an instrument, so it is
unit-tested like one. Every ordering in the packet's table is asserted here, plus
the I1–I3 property checks over randomly-perturbed synthetic cases.

**A reward change of any kind re-runs this battery first.** The battery is cheap
and Goodhart is not.
"""

from __future__ import annotations

import random

import pytest

from whetstone.reward.register_math import normalize_register_math, values_agree
from whetstone.reward.stagec import (
    ANSWER_TARGET_TOKENS,
    I2_MIN_MARGIN,
    MIN_THINK_TOKENS,
    RolloutView,
    ThinkBudget,
    W_FMT,
    assert_invariants,
    band_multiplier,
    compute_stagec_reward,
    detect_answer_repeat,
    detect_contradiction,
    detect_register_leak,
    length_multiplier,
)

GOLD = "72"
BUDGET = 250.0


# --- fixtures: synthetic rollouts in the project's actual register ----------

COMPACT_THINK = (
    "goal: total cost\n"
    "let p = 12\n"
    "let n = 6\n"
    "⇒ 12 · 6 = 72\n"
    "chk: 72 / 6 = 12 ✓\n"
)

VERBOSE_THINK = COMPACT_THINK + "".join(
    f"sub {i}: restate and re-derive the product from first principles\n"
    f"⇒ partial {i} confirms 72\n"
    for i in range(40)
)

CLEAN_ANSWER = "The total cost is $\\boxed{72}$."


def _rollout(think: str, answer: str) -> str:
    return f"<think>\n{think}</think>\n\n{answer}"


def _view(think: str, answer: str, *, think_len=None, answer_len=None, g=1) -> RolloutView:
    """Token counts approximated by whitespace count — the battery tests the
    reward's *shape*, and the real counts come from ``parse_segments`` at
    training time. Explicit overrides are used where a length matters."""
    return RolloutView(
        completion_text=_rollout(think, answer),
        think_len=think_len if think_len is not None else len(think.split()),
        answer_len=answer_len if answer_len is not None else ANSWER_TARGET_TOKENS,
        g=g,
    )


def _score(view: RolloutView, gold: str = GOLD, **kw) -> float:
    return compute_stagec_reward(view, gold, budget_B=BUDGET, **kw).total


# --- the packet's ordering table -------------------------------------------

@pytest.fixture(scope="module")
def scores() -> dict:
    correct_compact = _view(COMPACT_THINK, CLEAN_ANSWER, think_len=40)
    correct_verbose = _view(VERBOSE_THINK, CLEAN_ANSWER, think_len=1200)
    correct_empty = RolloutView(
        completion_text="<think>\n</think>\n\n" + CLEAN_ANSWER,
        think_len=1, answer_len=ANSWER_TARGET_TOKENS, g=1,
    )
    correct_contradiction = _view(
        "goal: total\nlet p = 12\n⇒ 12 · 6 = 66\n", CLEAN_ANSWER, think_len=40
    )
    correct_leaked = _view(
        COMPACT_THINK,
        "goal: report\n⇒ the total is $\\boxed{72}$.",
        think_len=40,
    )
    wrong_wellformed = _view(
        "goal: total\nlet p = 12\n⇒ 12 · 5 = 60\n",
        "The total cost is $\\boxed{60}$.",
        think_len=40,
    )
    # g=0: never closed <think>, ran to the cap.
    loop = RolloutView(
        completion_text="<think>\ngoal: total\n" + "chk: 72 ✓\n" * 500,
        think_len=3000, answer_len=0, g=0, gate_reason="missing_think_close",
    )
    return {
        "correct_compact": _score(correct_compact),
        "correct_verbose": _score(correct_verbose),
        "correct_empty": _score(correct_empty),
        "correct_contradiction": _score(correct_contradiction),
        "correct_leaked": _score(correct_leaked),
        "wrong_wellformed": _score(wrong_wellformed),
        "loop": _score(loop),
    }


def test_compact_correct_is_highest(scores: dict) -> None:
    assert scores["correct_compact"] == max(scores.values())


def test_verbose_correct_below_compact_but_far_above_wrong(scores: dict) -> None:
    assert scores["correct_verbose"] < scores["correct_compact"]
    assert scores["correct_verbose"] > scores["wrong_wellformed"] + 0.5


def test_empty_think_is_below_verbose_correct(scores: dict) -> None:
    """The guard. Without it, `<think>\\n</think>` is the global optimum."""
    assert scores["correct_empty"] < scores["correct_verbose"]
    assert scores["correct_empty"] < scores["correct_compact"]


def test_contradiction_is_penalized_below_verbose(scores: dict) -> None:
    assert scores["correct_contradiction"] < scores["correct_verbose"]


def test_register_leak_is_penalized(scores: dict) -> None:
    assert scores["correct_leaked"] < scores["correct_compact"]


def test_wrong_wellformed_is_r_fmt_only(scores: dict) -> None:
    assert scores["wrong_wellformed"] == pytest.approx(W_FMT)


def test_loop_is_the_floor(scores: dict) -> None:
    assert scores["loop"] == pytest.approx(0.0)
    assert scores["loop"] < scores["wrong_wellformed"]


def test_full_ordering(scores: dict) -> None:
    """Every correct rollout outranks every wrong/malformed one."""
    correct = [v for k, v in scores.items() if k.startswith("correct_")]
    other = [scores["wrong_wellformed"], scores["loop"]]
    assert min(correct) >= max(other) + I2_MIN_MARGIN


# --- finding-15 leniency cases must grade WRONG ----------------------------

@pytest.mark.parametrize("gold,pred", [("200", "0"), ("20", "0"), ("2", "42")])
def test_suffix_leniency_scores_as_wrong(gold: str, pred: str) -> None:
    v = _view(COMPACT_THINK, f"$\\boxed{{{pred}}}$", think_len=40)
    b = compute_stagec_reward(v, gold, budget_B=BUDGET)
    assert b.r_acc == 0.0
    assert b.as_scored is True and b.strict is False
    assert b.lenient_only is True, "the leniency dashboard curve must see this"
    assert b.total == pytest.approx(W_FMT)


def test_unfinished_think_with_gold_inside_scores_at_the_floor() -> None:
    text = "<think>\ngoal: n\n⇒ \\boxed{72}\n" + "chk: 72 ✓\n" * 300
    v = RolloutView(completion_text=text, think_len=2000, answer_len=0, g=0,
                    gate_reason="missing_think_close")
    b = compute_stagec_reward(v, GOLD, budget_B=BUDGET)
    assert b.r_acc == 0.0
    assert b.total == pytest.approx(0.0)


# --- component shapes -------------------------------------------------------

def test_length_term_is_flat_below_budget_never_a_bonus() -> None:
    """No gradient toward shorter inside the budget — the anti-empty-think shape."""
    assert length_multiplier(10, 250) == pytest.approx(1.0)
    assert length_multiplier(249, 250) == pytest.approx(1.0)
    assert length_multiplier(250, 250) == pytest.approx(1.0)
    assert length_multiplier(500, 250) < 1.0
    assert length_multiplier(1000, 250) < length_multiplier(500, 250)


def test_length_term_gates_on_correctness() -> None:
    short_wrong = _view("goal: x\n⇒ 1\n", "$\\boxed{99}$", think_len=20)
    b = compute_stagec_reward(short_wrong, GOLD, budget_B=BUDGET)
    assert b.r_len == 0.0, "I3: a wrong rollout must not earn the length reward"
    assert b.r_band == 0.0


def test_answer_band_protects_against_the_round2_collapse() -> None:
    assert band_multiplier(288) == pytest.approx(1.0)
    assert band_multiplier(300) == pytest.approx(1.0)     # inside ±32
    assert band_multiplier(19) < band_multiplier(180) < band_multiplier(288)


def test_empty_think_guard_threshold_is_token_based() -> None:
    just_under = RolloutView(_rollout(COMPACT_THINK, CLEAN_ANSWER),
                             think_len=MIN_THINK_TOKENS - 1,
                             answer_len=ANSWER_TARGET_TOKENS, g=1)
    just_over = RolloutView(_rollout(COMPACT_THINK, CLEAN_ANSWER),
                            think_len=MIN_THINK_TOKENS,
                            answer_len=ANSWER_TARGET_TOKENS, g=1)
    bu = compute_stagec_reward(just_under, GOLD, budget_B=BUDGET)
    bo = compute_stagec_reward(just_over, GOLD, budget_B=BUDGET)
    assert bu.empty_think is True and bu.r_fmt == 0.0
    assert bo.empty_think is False and bo.r_fmt >= W_FMT
    assert bo.total > bu.total


# --- register-aware contradiction detection --------------------------------

def test_contradiction_does_not_fire_on_register_notation() -> None:
    """005 finding 14: the register writes 4√2 where the answer writes 4\\sqrt{2}."""
    split_text = _rollout("goal: x\n⇒ 4√2\n", "$\\boxed{4\\sqrt{2}}$")
    from whetstone.reward.extract import split_think_close

    d = detect_contradiction(split_think_close(split_text))
    assert d["fired"] is False, "the detector taxed the register's own notation"
    assert d["decidable"] is True


@pytest.mark.parametrize(
    "a,b",
    [
        ("4√2", "4\\sqrt{2}"),
        ("1/2", "\\frac{1}{2}"),
        ("0.5", "1/2"),
        ("12 · 6", "12 \\cdot 6"),
        ("−3", "-3"),
        ("1,000", "1000"),
    ],
)
def test_register_normalizer_unifies_notations(a: str, b: str) -> None:
    assert values_agree(a, b) is not False, (
        f"{a!r} vs {b!r} → {normalize_register_math(a)!r} vs {normalize_register_math(b)!r}"
    )


def test_contradiction_fires_on_a_real_disagreement() -> None:
    from whetstone.reward.extract import split_think_close

    d = detect_contradiction(split_think_close(_rollout("goal: x\n⇒ 6200\n",
                                                        "$\\boxed{6600}$")))
    assert d["fired"] is True   # activity 005's hand-inspected case


def test_contradiction_stays_silent_when_undecidable() -> None:
    from whetstone.reward.extract import split_think_close

    d = detect_contradiction(split_think_close(_rollout("goal: x\n⇒ the blue one\n",
                                                        "$\\boxed{\\theta}$")))
    assert d["fired"] is False and d["decidable"] is False


def test_log_dont_penalize_mode() -> None:
    v = _view("goal: total\n⇒ 66\n", CLEAN_ANSWER, think_len=40)
    penalized = compute_stagec_reward(v, GOLD, budget_B=BUDGET)
    logged = compute_stagec_reward(v, GOLD, budget_B=BUDGET,
                                   penalize_contradiction=False)
    assert penalized.penalties["contradiction"] > 0
    assert logged.penalties["contradiction"] == 0.0
    assert logged.flags["contradiction"]["fired"] is True, "still logged"


# --- register-leak detector must not tax honest English --------------------

def test_leak_detector_ignores_the_english_word_case() -> None:
    """009 finding 1: `case` appears in 10.2% of honest answers."""
    d = detect_register_leak("In this case the total is 72, so the answer is 72.")
    assert d["fired"] is False


def test_leak_detector_catches_line_initial_markers_and_symbols() -> None:
    assert detect_register_leak("goal: restate\nthe answer is 72")["fired"] is True
    assert detect_register_leak("chk: 72 ✓")["fired"] is True
    assert detect_register_leak("⇒ 72")["fired"] is True
    assert detect_register_leak("some prose\n⇒ 12 · 6 = 72")["fired"] is True


@pytest.mark.parametrize(
    "answer",
    [
        # Every one of these is a verbatim shape from the pilot's own rollouts
        # (activity 010 finding 15) — ⇒ as ordinary math notation in English.
        'If a polynomial has a root ⇒ it has a linear factor.',
        "- **If** a polynomial has a root ⇒ it has a linear factor.",
        "Total: $32 + 18 + 98 = \\$138 ⇒ this shares 72 cents for 48 fruits.",
        "4 apples = 1 watermelon ⇒ cost of 1 watermelon is $4 (per pack).",
    ],
)
def test_leak_detector_ignores_mid_prose_math_notation(answer: str) -> None:
    """A ⇒ inside a sentence is standard notation, not register leakage.

    Before this rule the detector fired on 9/9 real detections, all of them
    false — a 0.10 penalty levied on correct answers for writing mathematics.
    """
    assert detect_register_leak(answer)["fired"] is False


# --- loop detector ----------------------------------------------------------

def test_loop_detector_catches_case_enumeration() -> None:
    from whetstone.reward.stagec import detect_ngram_loop

    think = "".join(f"case {i}: try {i}\n" for i in range(1, 30))
    assert detect_ngram_loop(think)["fired"] is True


def test_loop_detector_ignores_an_honest_compact_trace() -> None:
    from whetstone.reward.stagec import detect_ngram_loop

    assert detect_ngram_loop(COMPACT_THINK)["fired"] is False


# --- ThinkBudget freeze rule ------------------------------------------------

def test_budget_freezes_when_group_is_already_tight() -> None:
    tb = ThinkBudget(400.0, anneal=0.9, std_min=40.0)
    tb.update([200, 205, 198, 203])          # std ≈ 2.7 → frozen
    assert tb.B == 400.0 and tb.frozen_steps == 1
    tb.update([100, 400, 250, 700])          # wide → tightens
    assert tb.B < 400.0


def test_budget_never_demands_below_the_realized_spread() -> None:
    tb = ThinkBudget(50.0, b_floor=10.0)
    lens = [300, 400, 500, 600]
    eff = tb.effective_B(lens)
    assert eff >= min(lens), (
        "budget demanded a length no group member produced — the reward would "
        "penalize every member equally and carry no within-group signal"
    )


# --- I1-I3 property checks over randomly perturbed cases -------------------

def test_invariants_hold_on_200_random_perturbations() -> None:
    assert_invariants()
    rng = random.Random(0)
    worst_correct, best_wrong = float("inf"), float("-inf")

    for _ in range(200):
        think_len = rng.choice([0, 5, 15, 16, 40, 300, 2000, 8000])
        answer_len = rng.choice([0, 19, 100, 288, 900])
        g = rng.choice([0, 1, 1, 1])
        correct = rng.random() < 0.5
        pred = GOLD if correct else "999"
        think = rng.choice([COMPACT_THINK, VERBOSE_THINK,
                            "case 1: a\n" * 30, "goal: x\n⇒ 66\n"])
        answer = rng.choice([
            f"$\\boxed{{{pred}}}$",
            f"chk: {pred} ✓\n$\\boxed{{{pred}}}$",
            f"{pred}\n\n{pred}\n\n{pred}",
        ])
        v = RolloutView(_rollout(think, answer), think_len, answer_len, g)
        b = compute_stagec_reward(v, GOLD, budget_B=BUDGET)

        # I1
        if b.well_formed:
            assert b.r_fmt >= W_FMT - 1e-12, f"I1 violated: r_fmt={b.r_fmt}"
        # I3
        if b.r_acc == 0.0:
            assert b.r_len == 0.0 and b.r_band == 0.0, "I3 violated"

        if b.r_acc > 0:
            worst_correct = min(worst_correct, b.total)
        else:
            best_wrong = max(best_wrong, b.total)

    # I2
    assert worst_correct - best_wrong >= I2_MIN_MARGIN, (
        f"I2 violated: worst correct {worst_correct}, best wrong {best_wrong}"
    )


def test_assert_invariants_reports_the_margin() -> None:
    out = assert_invariants()
    assert out["margin"] >= I2_MIN_MARGIN
    assert out["max_struct"] < 1.0


# --- activity 011 (Arm A scan): two more finding-15-class false positives ----

def test_leak_detector_is_case_sensitive_english_let_header_is_clean() -> None:
    """Verbatim shape from the Arm A scan: mathematical English opens lines
    with capitalized `Let:` as prose scaffolding; the register's binder is
    strictly lowercase `let:`. IGNORECASE taxed clean LaTeX answers at 3.8%."""
    english = "We take the logarithm.\n\nLet:\n\n$$\nL = \\lim_{n\\to\\infty} f(n)\n$$"
    assert detect_register_leak(english)["fired"] is False
    assert detect_register_leak("Case: when x > 0, trivial.")["fired"] is False
    # The register's own lowercase markers still fire.
    assert detect_register_leak("let: x = 3\n")["fired"] is True
    assert detect_register_leak("some prose\ncase 2: n odd\n")["fired"] is False
    assert detect_register_leak("case: n odd\n")["fired"] is True


def test_answer_repeat_ignores_latex_display_delimiters() -> None:
    """Verbatim shape from the Arm A scan: consecutive `$$` display blocks
    separated by a blank line are typesetting, not answer restatement. Fired
    on ~9% of ALL rollouts, flat across pilot 1 and Arm A — a base rate of
    honest formatting."""
    latex = ("Adding these together gives:\n$$\n12 + 12 + 0 + 10 = 34\n$$\n\n"
             "$$\n\\boxed{34}\n$$\n")
    assert detect_answer_repeat(latex)["fired"] is False
    # v1 §4.6's true target still fires: bare restated numerals.
    assert detect_answer_repeat("151\n\n151\n\n151")["fired"] is True
    assert detect_answer_repeat("\\boxed{7}\n\n\\boxed{7}")["fired"] is True
