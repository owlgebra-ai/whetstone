"""Magnitude constants for the Stage-5 reward function.

Every numeric value here traces back to a section of
``WHETSTONE_STAGE5_REWARD_DESIGN.md``. Keep them centralised so smoke tests
and design-contract self-checks import the exact same numbers the aggregator
uses at training time.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- §2.1: r_acc tier magnitudes ------------------------------------------------

R_ACC_STRICT = 1.0
R_ACC_LENIENT = 0.5
R_ACC_WRONG = 0.0


# --- §3: r_struct component magnitudes -----------------------------------------

STRUCT_CLOSED_THINK = 0.15
STRUCT_TERMINAL_BOXED = 0.10
STRUCT_TERMINAL_BARE = 0.10
STRUCT_SHORT_CLEAN_BONUS = 0.12  # midpoint of the 0.10..0.15 budget in §3
STRUCT_CHUNK_RESTART_PRESENT = 0.05


# --- §3 gates ------------------------------------------------------------------

SHORT_CLEAN_CHUNK_COUNT_MAX = 25  # §3: "chunk count <= ~25"
CHUNK_RESTART_MIN_COUNT = 2       # §3: "contains >= 2 numbered chunk restarts"
BONUS_ACC_GATE = 0.5              # §3, §4.4, §5.3: bonuses require acc >= 0.5


# --- §4.1: soft length tail penalty --------------------------------------------

LENGTH_PEN_START_CHARS = 8_000    # kicks in at 8k
LENGTH_PEN_SATURATE_CHARS = 20_000  # saturates at 20k
LENGTH_PEN_MAX = 0.30
LENGTH_PEN_SLOPE_DENOM = LENGTH_PEN_SATURATE_CHARS - LENGTH_PEN_START_CHARS  # 12k (doc says 40k / 0.30 = 40k; we use the saturation shape from §4.1 example)


# --- §4.2: placeholder / sentinel chunk repetition penalty ---------------------

PLACEHOLDER_PEN_ESCAPE_ALLOWANCE = 2  # "escape-once" -> excess = n - 2
PLACEHOLDER_PEN_PER_EXCESS = 0.05
PLACEHOLDER_PEN_MAX = 0.15
# Chunks whose *content* matches these get the placeholder treatment. Kept short
# and register-agnostic — extend when a new base-model register appears.
PLACEHOLDER_CHUNK_PATTERNS = (
    r"^\s*final answer\s*[:\-]?\s*[^\s\n]{0,40}\.?\s*$",
    r"^\s*end\.?\s*$",
    r"^\s*done\.?\s*$",
    r"^\s*output complete\.?\s*$",
    r"^\s*end of thought process\.?\s*$",
    r"^\s*\(placeholder[^)]*\)\s*$",
)


# --- §4.3: repetition-loop penalty ---------------------------------------------

REPETITION_EXACT_MIN_RUN = 10      # §4.3, §9.5: threshold 10 (NOT 5)
REPETITION_TEMPLATE_MIN_RUN = 5    # template-normalized: 5 is fine
REPETITION_PEN = 0.20
REPETITION_SHORT_CHUNK_EXCLUDE_TOTAL = 30  # §4.3: exclude placeholder-vocab when total chunks <= ~30


# --- §4.5: monolithic-think / counter-restart anomaly --------------------------

MONOLITHIC_MAX_STEP = 50               # numbered step > 50 -> trigger
MONOLITHIC_ZERO_RESTARTS_MIN_CHARS = 6_000  # 0 restarts AND think > 6k -> trigger
MONOLITHIC_GROUP_VARIATION = 100       # chunk_count peer variation > 100 -> trigger
MONOLITHIC_PEN_PER_TRIGGER = 0.10
MONOLITHIC_PEN_MAX = 0.15


# --- §4.6: post-</think> answer repetition -------------------------------------

POST_TAIL_REPEAT_PEN_PER = 0.05
POST_TAIL_REPEAT_PEN_MAX = 0.15
POST_TAIL_REPEAT_MIN_COUNT = 2  # §4.6: "match count >= 2"


# --- §4.7: post-think register leakage -----------------------------------------

REGISTER_LEAK_PEN = 0.05  # fires at most once per rollout


# --- §4.8: sentinel-phrase rumination inside <think> ---------------------------

SENTINEL_PHRASE_MIN_COUNT = 3   # §4.8: >= 3 occurrences
SENTINEL_PEN = 0.05


# --- §4.9: cap-hit / termination bonus -----------------------------------------

CLOSED_THINK_BONUS = 0.02


# --- §4.10: think/post-think numerical contradiction ---------------------------
# Two modes:
#   "weak"        -> apply CONTRADICTION_PEN as a format-channel penalty.
#   "strict_gate" -> demote a strict-tier rollout to lenient when contradiction
#                    is detected (§4.10 note: preferred when the redemption-
#                    recap attractor is observed).
#   "off"         -> disable both.

CONTRADICTION_PEN = 0.05
CONTRADICTION_THINK_TAIL_CHARS = 200  # §4.10: "final ~200 characters of <think>"
CONTRADICTION_NUMERIC_REL_TOL = 1e-6
CONTRADICTION_NUMERIC_ABS_TOL = 1e-9

# --- §4.11: meta-pattern / hedged-guess finalizer (discretionary) --------------
# (Was §4.10 in the prior doc version — renumbered when contradiction was added.)

META_PATTERN_PEN = 0.10  # discretionary; off by default


# --- §6.7: prose-templated strict extractor fallback ---------------------------
# Regex from §6.7 verbatim. Matches: **Answer:** X, **X**, Answer: X, Final
# Answer: X, Final result is X, "The {quantity} is {N}." — each with optional
# markdown wrapping and trailing period.

PROSE_TEMPLATED_ANSWER_RE = (
    r"(?:\*\*)?"                                            # optional leading **
    r"(?:Answer|Final\s+Answer|Final\s+result\s+is|"
    r"The\s+.+?\s+is)"                                      # prefix templates
    r"(?:\*\*)?"                                            # optional closing **
    r"\s*[:=]?\s*\**\s*"
    r"([^\s.*][^*]*?)"                                       # captured value
    r"\s*\**\.?\s*$"
)


# --- §6.10: malformed / duplicate \boxed{} detection ---------------------------

MALFORMED_BOXED_PEN_PER = 0.05
MALFORMED_BOXED_PEN_MAX = 0.10
MALFORMED_BOXED_LITERAL_TOKEN = "{boxed{"


# --- §2.2 / §5.1: r_fmt floors -------------------------------------------------

R_FMT_FLOOR_WITH_CLOSE = 0.10   # §2.2, §5.1: floor when </think> present
R_FMT_FLOOR_WITHOUT_CLOSE = 0.0


# --- §2.3 design-contract targets ----------------------------------------------

STRICT_MINUS_LENIENT_TARGET_LOW = 0.55
STRICT_MINUS_LENIENT_TARGET_HIGH = 0.65
STRICT_MINUS_LENIENT_MIN_ACCEPTABLE = 0.30  # §5.2: worst-case invariant


# --- §2.1 / §6.1 lenient last-K window -----------------------------------------

LENIENT_LAST_K_CHARS = 500  # §6.1: "last-K characters (K ≈ 500)"


@dataclass(frozen=True)
class RewardConfig:
    """Snapshot of the tunable magnitudes used to compute one reward.

    Constructed at aggregator entry so downstream code can trace exactly which
    magnitudes produced a given breakdown. Also serves as the anchor point for
    experiment-time overrides — bump one field, restart DAPO from SFT init.
    """

    # tiers
    r_acc_strict: float = R_ACC_STRICT
    r_acc_lenient: float = R_ACC_LENIENT
    r_acc_wrong: float = R_ACC_WRONG

    # struct
    struct_closed_think: float = STRUCT_CLOSED_THINK
    struct_terminal_boxed: float = STRUCT_TERMINAL_BOXED
    struct_terminal_bare: float = STRUCT_TERMINAL_BARE
    struct_short_clean_bonus: float = STRUCT_SHORT_CLEAN_BONUS
    struct_chunk_restart_present: float = STRUCT_CHUNK_RESTART_PRESENT
    short_clean_chunk_count_max: int = SHORT_CLEAN_CHUNK_COUNT_MAX
    bonus_acc_gate: float = BONUS_ACC_GATE

    # length
    length_pen_start_chars: int = LENGTH_PEN_START_CHARS
    length_pen_saturate_chars: int = LENGTH_PEN_SATURATE_CHARS
    length_pen_max: float = LENGTH_PEN_MAX

    # placeholder
    placeholder_pen_escape_allowance: int = PLACEHOLDER_PEN_ESCAPE_ALLOWANCE
    placeholder_pen_per_excess: float = PLACEHOLDER_PEN_PER_EXCESS
    placeholder_pen_max: float = PLACEHOLDER_PEN_MAX

    # repetition
    repetition_exact_min_run: int = REPETITION_EXACT_MIN_RUN
    repetition_template_min_run: int = REPETITION_TEMPLATE_MIN_RUN
    repetition_pen: float = REPETITION_PEN

    # monolithic
    monolithic_max_step: int = MONOLITHIC_MAX_STEP
    monolithic_zero_restarts_min_chars: int = MONOLITHIC_ZERO_RESTARTS_MIN_CHARS
    monolithic_pen_per_trigger: float = MONOLITHIC_PEN_PER_TRIGGER
    monolithic_pen_max: float = MONOLITHIC_PEN_MAX

    # tail-repeat / leakage / sentinel
    post_tail_repeat_pen_per: float = POST_TAIL_REPEAT_PEN_PER
    post_tail_repeat_pen_max: float = POST_TAIL_REPEAT_PEN_MAX
    post_tail_repeat_min_count: int = POST_TAIL_REPEAT_MIN_COUNT
    register_leak_pen: float = REGISTER_LEAK_PEN
    sentinel_phrase_min_count: int = SENTINEL_PHRASE_MIN_COUNT
    sentinel_pen: float = SENTINEL_PEN

    # closure bonus + meta pattern
    closed_think_bonus: float = CLOSED_THINK_BONUS
    meta_pattern_pen: float = META_PATTERN_PEN
    enable_meta_pattern_pen: bool = False  # §4.11: discretionary, off by default

    # §4.10 contradiction — default "weak" per §4.10 note
    contradiction_mode: str = "weak"  # {"off", "weak", "strict_gate"}
    contradiction_pen: float = CONTRADICTION_PEN
    contradiction_think_tail_chars: int = CONTRADICTION_THINK_TAIL_CHARS
    contradiction_numeric_rel_tol: float = CONTRADICTION_NUMERIC_REL_TOL
    contradiction_numeric_abs_tol: float = CONTRADICTION_NUMERIC_ABS_TOL

    # §6.7 prose-templated strict fallback
    enable_prose_templated_extractor: bool = True

    # §6.10 malformed boxed
    malformed_boxed_pen_per: float = MALFORMED_BOXED_PEN_PER
    malformed_boxed_pen_max: float = MALFORMED_BOXED_PEN_MAX

    # floor
    r_fmt_floor_with_close: float = R_FMT_FLOOR_WITH_CLOSE
    r_fmt_floor_without_close: float = R_FMT_FLOOR_WITHOUT_CLOSE

    # lenient window
    lenient_last_k_chars: int = LENIENT_LAST_K_CHARS

    # §12: bump on any magnitude/detector change. Tag every checkpoint with this.
    version: str = "5.1.0"


DEFAULT_CONFIG = RewardConfig()
