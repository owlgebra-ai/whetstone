"""§4 penalty catalogue.

Each entry from §4 is its own function returning ``(magnitude, fired, detail)``:
    - ``magnitude``: non-negative float subtracted from ``r_fmt``.
    - ``fired``: whether the detector triggered (for diagnostics / §7).
    - ``detail``: dict of detector internals — count of triggers, worst run
      length, etc. Preserved through the ``RewardBreakdown`` so postmortems
      can trace exactly why a penalty magnitude was chosen.

Invariants (§4 / §5):
    * Every penalty is INDEPENDENTLY capped at the magnitude documented
      in :mod:`.config`. Uncapped ``k × excess`` scaling was §9.2's failure
      mode; do not remove the caps.
    * Penalty *detectors* may be threshold functions (§5.5 exempts the
      binary trigger); penalty *magnitudes* are soft-linear where possible.
    * The 0.10 ``r_fmt`` floor lives in :mod:`.aggregate`, NOT here — per
      §5.1 the floor is applied at final aggregation, not per-penalty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    LENGTH_PEN_MAX,
    LENGTH_PEN_SATURATE_CHARS,
    LENGTH_PEN_START_CHARS,
    MALFORMED_BOXED_PEN_MAX,
    MALFORMED_BOXED_PEN_PER,
    META_PATTERN_PEN,
    MONOLITHIC_GROUP_VARIATION,
    MONOLITHIC_MAX_STEP,
    MONOLITHIC_PEN_MAX,
    MONOLITHIC_PEN_PER_TRIGGER,
    MONOLITHIC_ZERO_RESTARTS_MIN_CHARS,
    PLACEHOLDER_CHUNK_PATTERNS,
    PLACEHOLDER_PEN_ESCAPE_ALLOWANCE,
    PLACEHOLDER_PEN_MAX,
    PLACEHOLDER_PEN_PER_EXCESS,
    POST_TAIL_REPEAT_MIN_COUNT,
    POST_TAIL_REPEAT_PEN_MAX,
    POST_TAIL_REPEAT_PEN_PER,
    REGISTER_LEAK_PEN,
    REPETITION_EXACT_MIN_RUN,
    REPETITION_PEN,
    REPETITION_SHORT_CHUNK_EXCLUDE_TOTAL,
    REPETITION_TEMPLATE_MIN_RUN,
    RewardConfig,
    SENTINEL_PEN,
    SENTINEL_PHRASE_MIN_COUNT,
)
from .extract import (
    SplitCompletion,
    _template_normalize_chunk,
    char_len,
    consecutive_runs,
    count_boxed_opens,
    count_chunk_restarts,
    extract_last_numeric_in_think_tail,
    extract_terminal_answer_from_post_think,
    has_malformed_boxed_literal,
    line_is_prose_templated_finalizer,
    max_numbered_step,
    numerical_agree,
    split_chunks,
)


@dataclass(frozen=True)
class PenaltyResult:
    """Result of one penalty detector.

    ``magnitude`` is always non-negative — the aggregator subtracts it from
    ``r_fmt``. ``detail`` fields carry detector internals for logging and
    §7 diagnostics but are never used in the reward arithmetic.
    """

    name: str
    magnitude: float
    fired: bool
    detail: Dict[str, Any] = field(default_factory=dict)


# --- compiled patterns ---------------------------------------------------------

_PLACEHOLDER_CHUNK_RE = [re.compile(p, re.IGNORECASE) for p in PLACEHOLDER_CHUNK_PATTERNS]
_SENTINEL_PHRASE_RE = re.compile(
    r"Final Answer:|End of thought process|Output complete|"
    r"\(Placeholder|\(Self-Correction|\(Final Output",
    re.IGNORECASE,
)
_POST_TAIL_REPEAT_RE = re.compile(
    r"^(?P<tok>[\d]+|[A-Z])(?:\s*\n\s*\n\s*(?P=tok))+\s*$",
    re.MULTILINE,
)
_REGISTER_LEAK_RE = re.compile(
    r"^\s*(?:\*\*[A-Z][^*]{0,80}\*\*|\d+\.\s)",
    re.MULTILINE,
)
_META_PATTERN_RE = re.compile(
    r"Common answer|such problems often|usually the answer|Let'?s go with|"
    r"Or\s+[A-Z0-9]+\.\s*Or",
    re.IGNORECASE,
)


# --- §4.1 rumination-runaway / soft length penalty -----------------------------

def length_penalty(split: SplitCompletion, cfg: Optional[RewardConfig] = None) -> PenaltyResult:
    """§4.1 soft length tail on the ``<think>`` body (character-count based).

    Kicks in at ``start_chars``, saturates at ``saturate_chars`` to the
    per-penalty cap. NEVER a cliff — cliffs cause mode collapse (§5.5).
    """
    cfg = cfg or RewardConfig()
    n_chars = char_len(split.think)
    start = cfg.length_pen_start_chars
    saturate = cfg.length_pen_saturate_chars
    cap = cfg.length_pen_max

    if n_chars <= start:
        return PenaltyResult(name="length", magnitude=0.0, fired=False, detail={"chars": n_chars})

    denom = max(1, saturate - start)
    frac = (n_chars - start) / denom  # linear in the tail
    mag = min(cap, max(0.0, cap * frac))
    return PenaltyResult(
        name="length",
        magnitude=mag,
        fired=True,
        detail={"chars": n_chars, "saturated": n_chars >= saturate},
    )


# --- §4.2 chunk-repetition / placeholder penalty -------------------------------

def _is_placeholder_chunk(chunk: str) -> bool:
    return any(rx.match(chunk) for rx in _PLACEHOLDER_CHUNK_RE)


def placeholder_penalty(
    split: SplitCompletion,
    cfg: Optional[RewardConfig] = None,
) -> PenaltyResult:
    """§4.2 placeholder-chunk penalty (soft, capped, with escape-once allowance).

    Counts placeholder-vocab chunks across the *whole* completion (think + post-
    think). The "escape-once" allowance (``n_excess = n - 2``) lets a legitimate
    strict-correct rollout emit up to 2 restatements of the final answer before
    any penalty applies.
    """
    cfg = cfg or RewardConfig()
    all_chunks = split_chunks(split.think) + split_chunks(split.post_think)
    n_placeholder = sum(1 for c in all_chunks if _is_placeholder_chunk(c))
    excess = max(0, n_placeholder - cfg.placeholder_pen_escape_allowance)
    mag = min(cfg.placeholder_pen_max, cfg.placeholder_pen_per_excess * excess)
    return PenaltyResult(
        name="placeholder",
        magnitude=mag,
        fired=excess > 0,
        detail={"n_placeholder": n_placeholder, "excess": excess},
    )


# --- §4.3 repetition-loop penalty (n-gram / template) --------------------------

def repetition_penalty(
    split: SplitCompletion,
    cfg: Optional[RewardConfig] = None,
) -> PenaltyResult:
    """§4.3 exact-string OR template-normalized repetition detector.

    Two-stage:
      (1) Exact-string: same chunk text >= 10 consecutive.
      (2) Template-normalized: normalize each chunk (strip numerics /
          LaTeX / parens), then check for >= 5 consecutive equals on the
          normalized form.

    Short placeholder-vocab chunks (§4.2) are excluded from stage 1 when
    total chunk count is low — otherwise legitimate tail-filler in short
    strict-correct rollouts trips the detector and drops fmt (§9.5).

    Magnitude is fixed at ``REPETITION_PEN`` on detection; the aggregator
    applies the 0.10 floor (§5.1).
    """
    cfg = cfg or RewardConfig()
    chunks = split_chunks(split.think)
    if not chunks:
        return PenaltyResult(name="repetition", magnitude=0.0, fired=False, detail={})

    filtered = chunks
    if len(chunks) <= REPETITION_SHORT_CHUNK_EXCLUDE_TOTAL:
        filtered = [c for c in chunks if not _is_placeholder_chunk(c)]

    exact_hit = False
    exact_run = 0
    for _, run in consecutive_runs(filtered):
        if run >= cfg.repetition_exact_min_run:
            exact_hit = True
            exact_run = max(exact_run, run)

    template_hit = False
    template_run = 0
    if filtered:
        normalized = [_template_normalize_chunk(c) for c in filtered]
        normalized = [n for n in normalized if n]
        for _, run in consecutive_runs(normalized):
            if run >= cfg.repetition_template_min_run:
                template_hit = True
                template_run = max(template_run, run)

    fired = exact_hit or template_hit
    mag = cfg.repetition_pen if fired else 0.0
    return PenaltyResult(
        name="repetition",
        magnitude=mag,
        fired=fired,
        detail={
            "exact_hit": exact_hit,
            "template_hit": template_hit,
            "exact_run": exact_run,
            "template_run": template_run,
        },
    )


# --- §4.5 counter-restart / monolithic-think anomaly ---------------------------

def monolithic_penalty(
    split: SplitCompletion,
    cfg: Optional[RewardConfig] = None,
    *,
    group_chunk_count_variation: Optional[int] = None,
) -> PenaltyResult:
    """§4.5 counter-restart / monolithic-think / chunk-count anomaly.

    Three triggers, ``-0.10`` each, capped at ``-0.15`` total:
      (a) Any numbered step > ``MONOLITHIC_MAX_STEP`` (counter-restart
          evasion of naive step-count caps).
      (b) Zero chunk restarts AND ``|<think>|`` > ``MIN_CHARS`` (monolithic
          collapse — the model glues 100+ steps into one uninterrupted
          block).
      (c) Peer-group chunk-count variation > ``MONOLITHIC_GROUP_VARIATION``
          (this rollout is an outlier). Only checked if the caller supplied
          the group variation.
    """
    cfg = cfg or RewardConfig()
    max_step = max_numbered_step(split.think)
    restarts = count_chunk_restarts(split.think)
    think_chars = char_len(split.think)

    triggers: List[str] = []
    if max_step > cfg.monolithic_max_step:
        triggers.append("counter_restart")
    if restarts == 0 and think_chars > cfg.monolithic_zero_restarts_min_chars:
        triggers.append("monolithic")
    if (
        group_chunk_count_variation is not None
        and group_chunk_count_variation > MONOLITHIC_GROUP_VARIATION
    ):
        triggers.append("group_outlier")

    mag = min(cfg.monolithic_pen_max, cfg.monolithic_pen_per_trigger * len(triggers))
    return PenaltyResult(
        name="monolithic",
        magnitude=mag,
        fired=bool(triggers),
        detail={
            "triggers": triggers,
            "max_step": max_step,
            "restarts": restarts,
            "think_chars": think_chars,
        },
    )


# --- §4.6 post-</think> answer repetition --------------------------------------

def post_tail_repeat_penalty(
    split: SplitCompletion,
    cfg: Optional[RewardConfig] = None,
) -> PenaltyResult:
    """§4.6 post-``</think>`` answer repetition.

    Detects the ``151\\n\\n151\\n\\n151`` failure shape. Per-repeat scaling
    with a cap.
    """
    cfg = cfg or RewardConfig()
    if not split.post_think:
        return PenaltyResult(name="post_tail_repeat", magnitude=0.0, fired=False, detail={})

    matches = list(_POST_TAIL_REPEAT_RE.finditer(split.post_think))
    kept = [m for m in matches if m.group(0).count("\n\n") + 1 >= cfg.post_tail_repeat_min_count + 1]
    n_repeats_total = sum(m.group(0).count("\n\n") for m in kept)
    mag = min(cfg.post_tail_repeat_pen_max, cfg.post_tail_repeat_pen_per * n_repeats_total)
    return PenaltyResult(
        name="post_tail_repeat",
        magnitude=mag,
        fired=bool(kept),
        detail={"matches": len(kept), "n_repeats_total": n_repeats_total},
    )


# --- §4.7 post-think register leakage ------------------------------------------

def register_leak_penalty(
    split: SplitCompletion,
    cfg: Optional[RewardConfig] = None,
) -> PenaltyResult:
    """§4.7 register-leakage — fires at most once regardless of hit count.

    Excludes lines that are §6.7 prose-templated finalizers (``**Answer:** X``,
    ``**X**``, ``The X is N.``). Without that whitelist, §4.7 would double-
    punish legitimate §6.7 forms — the tier classifier accepts them as strict
    but the register-leak detector's ``\\*\\*[A-Z]`` regex fires on the same
    bytes, dropping ``r_fmt`` for a rollout the accuracy channel just praised.
    """
    cfg = cfg or RewardConfig()
    if not split.post_think:
        return PenaltyResult(name="register_leak", magnitude=0.0, fired=False, detail={})

    # Match line-by-line so we can whitelist prose-templated finalizers.
    hits = 0
    whitelisted = 0
    for line in split.post_think.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not _REGISTER_LEAK_RE.match(stripped):
            continue
        if line_is_prose_templated_finalizer(stripped):
            whitelisted += 1
            continue
        hits += 1
    fired = hits > 0
    return PenaltyResult(
        name="register_leak",
        magnitude=cfg.register_leak_pen if fired else 0.0,
        fired=fired,
        detail={"hits": hits, "whitelisted_finalizer_lines": whitelisted},
    )


# --- §4.8 sentinel-phrase rumination inside <think> ----------------------------

def sentinel_phrase_penalty(
    split: SplitCompletion,
    cfg: Optional[RewardConfig] = None,
) -> PenaltyResult:
    """§4.8 sentinel-phrase rumination.

    Auxiliary to §4.1 length penalty: length can't punish sentinel rumination
    in correct rollouts because ``acc = 1.0`` dominates. This provides a small
    direct signal (``-0.05``) subject to the 0.10 floor.
    """
    cfg = cfg or RewardConfig()
    count = len(_SENTINEL_PHRASE_RE.findall(split.think or ""))
    fired = count >= cfg.sentinel_phrase_min_count
    return PenaltyResult(
        name="sentinel",
        magnitude=cfg.sentinel_pen if fired else 0.0,
        fired=fired,
        detail={"count": count},
    )


# --- §4.10 meta-pattern / hedged-guess finalizer (discretionary) ---------------

def meta_pattern_penalty(
    split: SplitCompletion,
    cfg: Optional[RewardConfig] = None,
) -> PenaltyResult:
    """§4.10 hedged-guess finalizer. **Discretionary — off by default.**

    Enable via :attr:`RewardConfig.enable_meta_pattern_pen`. Only turn on when
    the failure mode recurs at rate > 5% across a few checkpoints; reward-only
    mitigation cannot fully solve this class (§4.10 note).
    """
    cfg = cfg or RewardConfig()
    if not cfg.enable_meta_pattern_pen:
        return PenaltyResult(name="meta_pattern", magnitude=0.0, fired=False, detail={"enabled": False})
    if not split.think:
        return PenaltyResult(name="meta_pattern", magnitude=0.0, fired=False, detail={"enabled": True})
    tail = split.think[-500:]
    fired = _META_PATTERN_RE.search(tail) is not None
    return PenaltyResult(
        name="meta_pattern",
        magnitude=cfg.meta_pattern_pen if fired else 0.0,
        fired=fired,
        detail={"enabled": True},
    )


# --- §4.10 think/post-think numerical contradiction (weak mode) ----------------

def contradiction_penalty(
    split: SplitCompletion,
    cfg: Optional[RewardConfig] = None,
) -> PenaltyResult:
    """§4.10 numerical contradiction penalty (weak mode).

    Fires when:
      * the last numeric expression in the final ``contradiction_think_tail_chars``
        of ``<think>`` disagrees numerically with the terminal answer in
        post-``</think>``, AND
      * :attr:`RewardConfig.contradiction_mode` is ``"weak"``.

    Under ``"strict_gate"`` the tier classifier (§4.10 stronger mode) handles
    the response by demoting strict → lenient — this penalty stays inert to
    avoid double-punishing. Under ``"off"`` neither fires.

    Magnitude is a small fixed ``-0.05`` per §4.10 note (weak option). Cap is
    implicit: the detector is single-shot, not per-occurrence.
    """
    cfg = cfg or RewardConfig()
    if cfg.contradiction_mode != "weak":
        return PenaltyResult(
            name="contradiction",
            magnitude=0.0,
            fired=False,
            detail={"mode": cfg.contradiction_mode},
        )
    think_last = extract_last_numeric_in_think_tail(
        split.think, cfg.contradiction_think_tail_chars
    )
    post_terminal = extract_terminal_answer_from_post_think(split.post_think)
    if think_last is None or post_terminal is None:
        return PenaltyResult(
            name="contradiction",
            magnitude=0.0,
            fired=False,
            detail={
                "think_last": think_last,
                "post_terminal": post_terminal,
                "reason": "insufficient_evidence",
            },
        )
    agree = numerical_agree(
        think_last,
        post_terminal,
        rel_tol=cfg.contradiction_numeric_rel_tol,
        abs_tol=cfg.contradiction_numeric_abs_tol,
    )
    fired = not agree
    return PenaltyResult(
        name="contradiction",
        magnitude=cfg.contradiction_pen if fired else 0.0,
        fired=fired,
        detail={"think_last": think_last, "post_terminal": post_terminal},
    )


# --- §6.10 malformed / duplicate \boxed{} penalty ------------------------------

def malformed_boxed_penalty(
    split: SplitCompletion,
    cfg: Optional[RewardConfig] = None,
) -> PenaltyResult:
    """§6.10 malformed / duplicate ``\\boxed{}`` penalty.

    Two independent triggers on the §6.11 final block:
      (a) ``> 1`` occurrence of ``\\boxed{`` (garbage double-box like
          ``\\boxed{10}{boxed{10}}``).
      (b) The literal token ``{boxed{`` anywhere (evidence of an
          escape-eaten duplicate that (a)'s regex won't count).

    Magnitude: ``-0.05`` per trigger, capped at ``-0.10``. Soft — do not
    zero the rollout; the accuracy channel is unaffected because the last-
    box parse still succeeds.
    """
    cfg = cfg or RewardConfig()
    block = split.final_block
    triggers: List[str] = []
    open_count = count_boxed_opens(block)
    if open_count > 1:
        triggers.append("duplicate_open")
    if has_malformed_boxed_literal(block):
        triggers.append("escape_eaten_literal")
    mag = min(cfg.malformed_boxed_pen_max, cfg.malformed_boxed_pen_per * len(triggers))
    return PenaltyResult(
        name="malformed_boxed",
        magnitude=mag,
        fired=bool(triggers),
        detail={"open_count": open_count, "triggers": triggers},
    )


# --- §4.9 cap-hit / closure bonus ----------------------------------------------

def closure_bonus(split: SplitCompletion, cfg: Optional[RewardConfig] = None) -> PenaltyResult:
    """§4.9 small positive reward for closing ``</think>``.

    Modeled as a *negative penalty* (i.e. a bonus added to ``r_fmt``) so it
    flows through the same aggregation code path as the penalties. Returning
    ``magnitude = -CLOSED_THINK_BONUS`` on presence keeps subtraction the
    only arithmetic direction the aggregator needs to reason about.

    §4.9 note (2): do NOT try to fix cap-hit collapse with reward alone; drop
    hard groups from the curriculum instead.
    """
    cfg = cfg or RewardConfig()
    if not split.has_closed_think:
        return PenaltyResult(name="closure_bonus", magnitude=0.0, fired=False, detail={})
    return PenaltyResult(
        name="closure_bonus",
        magnitude=-cfg.closed_think_bonus,
        fired=True,
        detail={},
    )


# --- convenience: run every penalty ---------------------------------------------

def run_all_penalties(
    split: SplitCompletion,
    cfg: Optional[RewardConfig] = None,
    *,
    group_chunk_count_variation: Optional[int] = None,
) -> Tuple[List[PenaltyResult], float]:
    """Run the full penalty stack and return ``(results, sum_magnitudes)``.

    ``sum_magnitudes`` is the un-floored subtraction total; the 0.10 floor
    is applied by the aggregator per §5.1. ``closure_bonus`` contributes a
    negative magnitude so ``sum`` naturally handles the bonus direction.
    """
    cfg = cfg or RewardConfig()
    results = [
        length_penalty(split, cfg),
        placeholder_penalty(split, cfg),
        repetition_penalty(split, cfg),
        monolithic_penalty(split, cfg, group_chunk_count_variation=group_chunk_count_variation),
        post_tail_repeat_penalty(split, cfg),
        register_leak_penalty(split, cfg),
        sentinel_phrase_penalty(split, cfg),
        contradiction_penalty(split, cfg),   # §4.10 weak
        malformed_boxed_penalty(split, cfg), # §6.10
        meta_pattern_penalty(split, cfg),
        closure_bonus(split, cfg),
    ]
    return results, sum(r.magnitude for r in results)


__all__ = [
    "PenaltyResult",
    "length_penalty",
    "placeholder_penalty",
    "repetition_penalty",
    "monolithic_penalty",
    "post_tail_repeat_penalty",
    "register_leak_penalty",
    "sentinel_phrase_penalty",
    "contradiction_penalty",
    "malformed_boxed_penalty",
    "meta_pattern_penalty",
    "closure_bonus",
    "run_all_penalties",
]
