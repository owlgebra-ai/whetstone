"""Stage-C scalar reward (design §5; packet P7 §1b).

This is the *scalar* half of Stage C. The packet's separation is the first
thing to get right and is enforced by this module's scope:

===============================================  ==================================
enters the **scalar reward** (→ group advantage)  enters the **loss** (token-level)
===============================================  ==================================
``r_acc`` (strict), ``r_fmt``, think-length tail,  TEA (think), forward-KL to π_0
answer band, the penalty catalogue                (answer), the clip objective
===============================================  ==================================

Putting the KL into the scalar reward would recreate v1's uniform-anchor
mistake by the back door, so nothing in this file knows about π_0 or entropy.

Composition (additive, v1 §2.3 magnitude-budget style)::

    total = r_acc + r_fmt
    r_acc = 1.0  iff  g == 1 AND strict-correct   else 0.0
    r_fmt = max(floor, r_struct − Σ penalties)
    floor = 0.10 iff well-formed (g == 1 and think ≥ 16 tokens) else 0.0
    r_struct = W_FMT
             + [strict-gated] W_LEN · exp(−max(0, T_think − B)/B)
             + [strict-gated] W_BAND · band(A_answer)

Three properties this shape exists to have, each traceable to a measurement:

1. **The length term is a tail, never a monotone bonus.** ``exp(−max(0,T−B)/B)``
   is flat at 1.0 for every ``T ≤ B``: there is *zero* reward for being shorter
   than the budget, only cost for exceeding it. A monotone "shorter is better"
   term makes ``<think>\\n</think>`` the global optimum on every easy problem,
   and activity 009 proved this model already knows how to emit exactly that.
2. **The empty-think guard.** ``T_think < 16`` tokens drops ``r_fmt`` to 0 and
   every structural bonus with it. ``parse_segments`` scores empty think as
   ``g = 1`` (correctly — it is not malformed), so the guard has to live here.
3. **Accuracy dominates style, always** (I2). The worst-scoring correct rollout
   outranks the best-scoring wrong one by ≥ 0.30 — checked in code, not on
   paper, by :func:`assert_invariants`.

Grading is **strict** (:mod:`whetstone.reward.strict`). The v1 *lenient* tier is
retired for RL: activity 009 finding 15 measured lenient grading inflating a
degenerate model 14× more than the baseline, i.e. it is precisely where a
policy-gradient optimizer would farm reward. Lenient survives as a logged
diagnostic (``as_scored``) and never as a reward.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from statistics import median, pstdev
from typing import Dict, List, Optional, Sequence

from .extract import SplitCompletion, split_think_close
from .register_math import values_agree
from .strict import StrictVerdict, verify_strict

# --- magnitudes (packet P7 §1b; pinned here, swept nowhere without re-running
#     the battery in tests/test_stagec_reward.py) -----------------------------

W_FMT = 0.10            # v1 invariant I1 floor: well-formed-but-wrong beats malformed
W_LEN = 0.15            # think-length tail, strict-gated
W_BAND = 0.10           # answer length band, strict-gated

MIN_THINK_TOKENS = 16   # below this the rollout is treated as a format violation

PEN_CONTRADICTION = 0.20   # v1 §4.10 (0.05) raised: it must outrank the length tail
PEN_REGISTER_LEAK = 0.10   # v1 §4.7 (0.05) raised: the answer segment must stay clean
PEN_ANSWER_REPEAT_PER = 0.05
PEN_ANSWER_REPEAT_MAX = 0.10   # v1 §4.6 max 0.15 → "reduced weight" per packet
PEN_NGRAM_LOOP = 0.10          # v1 §4.3 was 0.20 → "reduced weight" per packet

ANSWER_TARGET_TOKENS = 288     # baseline card answer median — the π_0 anchor's target
ANSWER_BAND_F = 32             # SCA-style band half-width

# Loop detectors, tuned to this register's observed failures (009 finding 14:
# a line repeated 2,729×; a 35,085-char `chk:` chain; `case N:` to `case 713:`).
LOOP_EXACT_MIN_RUN = 10        # identical consecutive lines
#: Raised 6 → 30 (activity 011 phase-2 audit): at 6 the digit-blanked rule
#: fired on honest line-oriented enumeration — 8.0% of math and 13.9% of aimeh
#: rollouts, 67%/47% of firings on strict-CORRECT work — while every true
#: degenerate loop read also tripped the exact-run rule. 97% of the false
#: firings sat at runs ≤ 20; 009's `case 1:`…`case 713:` target class runs
#: far past 30 and is still caught.
LOOP_TEMPLATE_MIN_RUN = 30     # identical after digits are blanked (`case N:`)

I2_MIN_MARGIN = 0.30           # worst correct − best wrong

# --- register-specific detectors -------------------------------------------
# Line-initial markers only, plus two symbols that never appear in honest prose.
# NOT a bare substring set: `case` is an English word occurring in 10.2% of the
# corpus's own answers (009 finding 1), so `"case" in answer` would fire on
# ordinary English and tax correct rollouts.
#: Case-SENSITIVE by design (activity 011): the register writes its markers
#: strictly lowercase (``let:``, ``goal:``), while mathematical English opens
#: lines with capitalized ``Let:``/``Case:`` as ordinary prose scaffolding —
#: IGNORECASE made the detector fire on exactly that (Arm A scan: every read
#: leak sample was a clean LaTeX answer with a ``Let:`` header, rising to 3.8%
#: of rollouts as answers grew toward the 288-token band).
_LEAK_LINE_RE = re.compile(r"^\s*(?:goal|chk|sub|let|case)\s*:")
#: Symbols count as leakage only **line-initially**. Activity 010 finding 15:
#: a bare ``[⇒✗]`` search fired on 9/9 detections in the pilot, every one of them
#: ``⇒`` used as ordinary mathematical notation inside English prose ("If a
#: polynomial has a root ⇒ it has a linear factor"). The register writes its
#: conclusions line-initially (``⇒ 12 · 6 = 72``), so requiring that is both
#: faithful to the register and free of the false positives — otherwise the
#: penalty is a tax on correct answers for using standard notation, which is the
#: exact failure class this project keeps finding.
_LEAK_SYMBOL_RE = re.compile(r"^\s*[⇒✗]", re.MULTILINE)

#: Last ``⇒`` conclusion in the think body — the contradiction detector's left side.
_ARROW_VALUE_RE = re.compile(r"⇒\s*([^\n⇒]{1,80})")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

_DIGITS_RE = re.compile(r"\d+")
#: The repeated token must contain a word character (activity 011): ``$$`` on
#: its own line — a LaTeX display-math delimiter — otherwise matches whenever
#: two display blocks are separated by a blank line, taxing normal mathematical
#: typesetting at a measured ~9% of all rollouts (flat across pilot 1 AND Arm
#: A, i.e. a base rate of honest formatting, not a behaviour). The true target
#: (v1 §4.6's ``151\n\n151\n\n151``) always contains word characters.
_ANSWER_REPEAT_RE = re.compile(
    r"^(?P<tok>(?=\S{0,39}\w)\S{1,40})(?:\s*\n\s*\n\s*(?P=tok))+\s*$", re.MULTILINE
)


@dataclass(frozen=True)
class RolloutView:
    """Everything the scalar reward needs about one rollout.

    Token counts come from :func:`whetstone.segments.parse_segments` — never
    from string offsets (packet P7 §11). The text is carried alongside for the
    string-level detectors, which are genuinely string-level concerns.
    """

    completion_text: str
    think_len: int          # tokens
    answer_len: int         # tokens
    g: int                  # SCA quality gate from parse_segments
    gate_reason: str = ""


@dataclass(frozen=True)
class StageCBreakdown:
    """Per-rollout audit trail. Every field is logged; nothing is recomputed.

    ``as_scored`` and ``lenient_only`` are diagnostics only — they never touch
    ``total``. A rising ``lenient_only`` rate during training means the policy
    found a grading hole (packet P7 §7 deliverable 3).
    """

    total: float
    r_acc: float
    r_fmt: float
    r_struct: float
    penalties_subtracted: float
    floor_applied: bool
    # --- components
    r_len: float
    r_band: float
    length_multiplier: float
    band_multiplier: float
    budget_B: float
    # --- grading
    strict: bool
    as_scored: bool
    lenient_only: bool
    strict_reason: str
    # --- structural flags
    well_formed: bool
    empty_think: bool
    think_len: int
    answer_len: int
    g: int
    gate_reason: str
    penalties: Dict[str, float] = field(default_factory=dict)
    flags: Dict[str, object] = field(default_factory=dict)


# --- think-length budget ----------------------------------------------------


class ThinkBudget:
    """Annealed think-length budget ``B`` with the freeze rule (CLAUDE.md).

    Two invariants, both from the design and both load-bearing:

    * **"A reward must never demand lengths outside the realized group spread."**
      ``B`` is floored at a low percentile of what the group actually produced,
      so at least some group members always sit inside the budget and the term
      carries within-group signal instead of penalizing everyone equally.
    * **The freeze rule.** Tightening pauses whenever within-group think-length
      std falls below ``std_min`` — a group that has already converged in length
      has nothing left to teach the budget, and tightening into it is how the
      length term starts demanding the impossible.
    """

    def __init__(
        self,
        b_init: float,
        *,
        b_floor: float = 120.0,
        anneal: float = 0.995,
        std_min: float = 40.0,
        spread_pct: float = 25.0,
    ) -> None:
        self.B = float(b_init)
        self.b_floor = float(b_floor)
        self.anneal = float(anneal)
        self.std_min = float(std_min)
        self.spread_pct = float(spread_pct)
        self.frozen_steps = 0
        self.steps = 0

    @staticmethod
    def _percentile(xs: Sequence[float], pct: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        if len(s) == 1:
            return float(s[0])
        k = (len(s) - 1) * (pct / 100.0)
        lo, hi = math.floor(k), math.ceil(k)
        if lo == hi:
            return float(s[int(k)])
        return float(s[lo] * (hi - k) + s[hi] * (k - lo))

    def effective_B(self, group_think_lens: Sequence[int]) -> float:
        """``B`` for *this* group, floored into its realized spread."""
        lens = [x for x in group_think_lens if x > 0]
        if not lens:
            return max(self.B, self.b_floor)
        return max(min(self.B, max(lens)), self._percentile(lens, self.spread_pct), 1.0)

    def update(self, group_think_lens: Sequence[int]) -> float:
        """Advance the schedule by one group. Returns the new nominal ``B``."""
        self.steps += 1
        lens = [x for x in group_think_lens if x > 0]
        if len(lens) >= 2 and pstdev(lens) < self.std_min:
            self.frozen_steps += 1          # freeze rule: do not tighten
            return self.B
        self.B = max(self.b_floor, self.B * self.anneal)
        return self.B


# --- component terms --------------------------------------------------------


def length_multiplier(think_len: int, budget_B: float) -> float:
    """``exp(−max(0, T − B)/B)`` — flat at 1.0 below the budget, decays above."""
    if budget_B <= 0:
        return 1.0
    excess = max(0.0, float(think_len) - float(budget_B))
    return math.exp(-excess / float(budget_B))


def band_multiplier(
    answer_len: int,
    target: int = ANSWER_TARGET_TOKENS,
    f: int = ANSWER_BAND_F,
) -> float:
    """SCA-style soft band: 1.0 inside ``target ± f``, decaying outside.

    Protects against the round-2 answer collapse (288 → 19 tokens) from the
    reward side; the forward-KL to π_0 protects it from the loss side. Both
    must agree on what an answer looks like, which is why the target is the
    baseline card's 288 and not the teacher corpus's 189.
    """
    if target <= 0:
        return 1.0
    dev = abs(float(answer_len) - float(target))
    if dev <= f:
        return 1.0
    return math.exp(-(dev - f) / float(target))


# --- penalty detectors ------------------------------------------------------


def _think_lines(think: str) -> List[str]:
    return [ln.strip() for ln in think.splitlines() if ln.strip()]


def detect_ngram_loop(think: str) -> Dict[str, object]:
    """Partial loops inside a *completed* rollout (v1 §4.3, register-adapted).

    ``g = 0`` already kills terminal loops — a generation that never closed
    ``</think>``. This catches the ones that looped and then recovered enough to
    close, which the gate cannot see. Line-based rather than v1's ``\\n\\n``
    chunk formalism: this register is line-oriented and has no chunks.
    """
    lines = _think_lines(think)
    # Early-exit on the SMALLER of the two thresholds — when
    # LOOP_TEMPLATE_MIN_RUN was raised to 30 (phase-2 audit) a guard on it
    # alone silently disabled the exact-run rule for any think under 30
    # lines, which is most of them. Caught by the battery before shipping.
    if len(lines) < min(LOOP_EXACT_MIN_RUN, LOOP_TEMPLATE_MIN_RUN):
        return {"fired": False, "exact_run": 0, "template_run": 0}

    def _max_run(seq: Sequence[str]) -> int:
        best = run = 1
        for i in range(1, len(seq)):
            run = run + 1 if seq[i] == seq[i - 1] else 1
            best = max(best, run)
        return best if seq else 0

    exact_run = _max_run(lines)
    # `case 1:` / `case 2:` / … `case 713:` collapses to one template.
    template_run = _max_run([_DIGITS_RE.sub("#", ln) for ln in lines])
    fired = exact_run >= LOOP_EXACT_MIN_RUN or template_run >= LOOP_TEMPLATE_MIN_RUN
    return {"fired": fired, "exact_run": exact_run, "template_run": template_run}


def detect_answer_repeat(post_think: str) -> Dict[str, object]:
    """v1 §4.6: the ``151\\n\\n151\\n\\n151`` answer-restatement shape."""
    if not post_think:
        return {"fired": False, "n_repeats": 0}
    n = 0
    for m in _ANSWER_REPEAT_RE.finditer(post_think):
        n += m.group(0).count("\n\n")
    return {"fired": n > 0, "n_repeats": n}


def detect_register_leak(post_think: str) -> Dict[str, object]:
    """v1 §4.7, register-specific: the answer segment must read as prose.

    Line-initial ``goal:``/``chk:``/``sub:``/``let:``/``case:`` plus the two
    symbols (``⇒``, ``✗``) that a natural-language answer never contains.
    Deliberately not a bare substring test — see ``_LEAK_LINE_RE``'s comment.
    """
    if not post_think:
        return {"fired": False, "line_hits": 0, "symbol_hits": 0}
    line_hits = sum(1 for ln in post_think.splitlines() if _LEAK_LINE_RE.match(ln))
    symbol_hits = len(_LEAK_SYMBOL_RE.findall(post_think))
    return {
        "fired": (line_hits + symbol_hits) > 0,
        "line_hits": line_hits,
        "symbol_hits": symbol_hits,
    }


def detect_contradiction(split: SplitCompletion) -> Dict[str, object]:
    """v1 §4.10: the think concludes X, the answer states Y.

    Observed live in this project (activity 005 hand-inspection: think said
    6,200, answer said 6,600). The comparison is **register-aware**
    (:mod:`whetstone.reward.register_math`) because the two sides are written in
    two different notations by design; a naive comparison would tax the register
    rather than detect a defect.

    Returns ``fired=False`` whenever the comparison is undecidable — missing
    evidence is not evidence of contradiction.
    """
    arrows = _ARROW_VALUE_RE.findall(split.think or "")
    boxed = _BOXED_RE.findall(split.post_think or "")
    if not arrows or not boxed:
        return {"fired": False, "decidable": False, "think_value": None, "answer_value": None}

    # The register writes conclusions as `⇒ 12 · 6 = 72`, so the *value* is the
    # right-hand side of the last `=`. Comparing the whole expression makes
    # every such line undecidable and silently disables the detector.
    think_value = arrows[-1].strip().rstrip(".;,")
    if "=" in think_value:
        think_value = think_value.rsplit("=", 1)[1].strip()
    answer_value = boxed[-1].strip()
    agree = values_agree(think_value, answer_value)
    return {
        "fired": agree is False,
        "decidable": agree is not None,
        "think_value": think_value,
        "answer_value": answer_value,
    }


def detect_counter_restart(think: str) -> Dict[str, object]:
    """Logged diagnostic ONLY (packet P7 §1b: v1 §4.5 is dropped as a penalty)."""
    nums = [int(m) for m in re.findall(r"^\s*(?:case|sub)\s*(\d+)\s*:", think or "",
                                       re.IGNORECASE | re.MULTILINE)]
    restarts = sum(1 for a, b in zip(nums, nums[1:]) if b <= a)
    return {"max_counter": max(nums) if nums else 0, "restarts": restarts}


def detect_word_stutter(text: str) -> Dict[str, object]:
    """Logged diagnostic ONLY — 009 finding 19's adjacent-word stutter.

    Measured at a 5.7% rate costing −5.9 accuracy points: "finding 14's loop in
    embryo across the whole distribution". Not a penalty (the packet's catalogue
    does not include one), but it is the earliest available warning that the
    loop mode is returning, so it is on the dashboard.
    """
    words = re.findall(r"\b\w+\b", text or "")
    stutters = sum(1 for a, b in zip(words, words[1:]) if a == b and len(a) > 2)
    return {"stutters": stutters, "rate": stutters / max(1, len(words))}


# --- the reward -------------------------------------------------------------


def compute_stagec_reward(
    view: RolloutView,
    gold: str,
    *,
    budget_B: float,
    answer_target: int = ANSWER_TARGET_TOKENS,
    penalize_contradiction: bool = True,
) -> StageCBreakdown:
    """Score one Stage-C rollout.

    Parameters
    ----------
    view : RolloutView
        Text plus **token-level** segment facts from ``parse_segments``.
    gold : str
        Ground-truth answer.
    budget_B : float
        Think-length budget for this rollout's group, from
        :meth:`ThinkBudget.effective_B`. Per-group, never global — a reward must
        never demand lengths outside the realized group spread.
    penalize_contradiction : bool
        ``False`` puts the contradiction detector in log-don't-penalize mode
        (packet P7 §1b's fallback if the register normalizer is not trusted).
    """
    split = split_think_close(view.completion_text)
    verdict: StrictVerdict = verify_strict(view.completion_text, gold)

    empty_think = view.think_len < MIN_THINK_TOKENS
    well_formed = (view.g == 1) and not empty_think

    # Malformed (g=0, incl. cap-hits) → R_acc = 0 and no structural rewards.
    # The 2.7–3.5% loop tail dies by construction; no special mechanism needed.
    is_correct = bool(verdict.strict) and view.g == 1
    r_acc = 1.0 if is_correct else 0.0

    # --- structural rewards. I3: every one gates on strict-correct. ---------
    len_mult = length_multiplier(view.think_len, budget_B)
    bnd_mult = band_multiplier(view.answer_len, target=answer_target)
    r_len = W_LEN * len_mult if (is_correct and well_formed) else 0.0
    r_band = W_BAND * bnd_mult if (is_correct and well_formed) else 0.0
    r_struct = (W_FMT if well_formed else 0.0) + r_len + r_band

    # --- penalties ----------------------------------------------------------
    loop = detect_ngram_loop(split.think)
    arep = detect_answer_repeat(split.post_think)
    leak = detect_register_leak(split.post_think)
    contra = detect_contradiction(split)

    pens: Dict[str, float] = {
        "ngram_loop": PEN_NGRAM_LOOP if loop["fired"] else 0.0,
        "answer_repeat": min(
            PEN_ANSWER_REPEAT_MAX,
            PEN_ANSWER_REPEAT_PER * int(arep["n_repeats"]),
        ),
        "register_leak": PEN_REGISTER_LEAK if leak["fired"] else 0.0,
        "contradiction": (
            PEN_CONTRADICTION if (contra["fired"] and penalize_contradiction) else 0.0
        ),
    }
    pen_sum = sum(pens.values())

    floor = W_FMT if well_formed else 0.0
    raw = r_struct - pen_sum
    r_fmt = max(floor, raw)
    total = r_acc + r_fmt

    return StageCBreakdown(
        total=total,
        r_acc=r_acc,
        r_fmt=r_fmt,
        r_struct=r_struct,
        penalties_subtracted=pen_sum,
        floor_applied=raw < floor,
        r_len=r_len,
        r_band=r_band,
        length_multiplier=len_mult,
        band_multiplier=bnd_mult,
        budget_B=float(budget_B),
        strict=bool(verdict.strict),
        as_scored=bool(verdict.as_scored),
        lenient_only=bool(verdict.lenient_only),
        strict_reason=verdict.reason,
        well_formed=well_formed,
        empty_think=empty_think,
        think_len=view.think_len,
        answer_len=view.answer_len,
        g=view.g,
        gate_reason=view.gate_reason,
        penalties=pens,
        flags={
            "loop": loop,
            "answer_repeat": arep,
            "register_leak": leak,
            "contradiction": contra,
            "counter_restart": detect_counter_restart(split.think),
            "word_stutter": detect_word_stutter(split.think),
        },
    )


# --- invariants, asserted in code (packet P7 §1b, adapted from v1 §5) -------


def assert_invariants() -> Dict[str, float]:
    """Check I1–I3 from the magnitudes alone. Call at trainer import: a reward
    mis-configuration must fail fast, not after 200 steps of quiet Goodharting.

    (I1) ``r_fmt ≥ 0.10`` whenever both boundaries are present and think ≥ 16.
    (I2) worst-scoring **correct** rollout ≥ best-scoring **wrong** + 0.30.
    (I3) every structural bonus and the length reward gate on strict-correct.
    """
    # I1 is structural: `floor = W_FMT` whenever well_formed. Assert the value.
    assert W_FMT >= 0.10, f"I1 violated: r_fmt floor {W_FMT} < 0.10"

    # I2. Worst correct: g=1 but empty think → floor 0, no bonuses → r_acc alone.
    worst_correct = 1.0 + 0.0
    # Best wrong: well-formed, floored r_fmt, no strict-gated bonuses (I3).
    best_wrong = 0.0 + W_FMT
    margin = worst_correct - best_wrong
    assert margin >= I2_MIN_MARGIN, (
        f"I2 violated: worst correct {worst_correct} − best wrong {best_wrong} "
        f"= {margin} < {I2_MIN_MARGIN}"
    )

    # I3 is enforced at the call site (`if is_correct and well_formed`); assert
    # the budget shape that makes it meaningful — a wrong rollout must not be
    # able to reach a correct one's score through structure alone.
    max_struct = W_FMT + W_LEN + W_BAND
    assert max_struct < 1.0, (
        f"I3 at risk: max structural reward {max_struct} ≥ r_acc 1.0 — style "
        "could outrank accuracy"
    )
    return {
        "worst_correct": worst_correct,
        "best_wrong": best_wrong,
        "margin": margin,
        "max_struct": max_struct,
    }


__all__ = [
    "RolloutView",
    "StageCBreakdown",
    "ThinkBudget",
    "compute_stagec_reward",
    "assert_invariants",
    "length_multiplier",
    "band_multiplier",
    "detect_ngram_loop",
    "detect_answer_repeat",
    "detect_register_leak",
    "detect_contradiction",
    "detect_counter_restart",
    "detect_word_stutter",
    "W_FMT",
    "W_LEN",
    "W_BAND",
    "MIN_THINK_TOKENS",
    "ANSWER_TARGET_TOKENS",
    "ANSWER_BAND_F",
]
