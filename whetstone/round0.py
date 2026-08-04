"""Shared Round-0 primitives: sequence construction, the R token set, scoring.

Packet P4 §4 ("do this identically everywhere"). Every Round-0 part — the R-set
builder (Part 1), the inoculation trainer's metrics (Part 3), the three meter
unit tests (Part 4) and the Stage-A branch check (Part 5) — must build the token
sequence the *same* way, because they compare numbers to each other. A record
scored under one construction and thresholded under another is exactly the
silently-inverted meter this packet exists to prevent.

The construction mirrors a native rollout (design §12.2), so a d_t measured here
is the d_t Stage A will measure at reward time:

    prompt      = apply_chat_template([user], add_generation_prompt=True,
                                      enable_thinking=True)
    completion  = "<think>\\n" + compact_think + "\\n</think>\\n\\n" + answer

Prompt and completion are tokenized **separately** and concatenated; masks come
from :func:`whetstone.segments.parse_segments`, never from string offsets
(design §12.1 — re-tokenizing a decoded split does not round-trip at the
boundary).

Scoring conventions, fixed here so no caller re-derives them:

* surprisal ``S_t = −log π(τ_t)``
* gap ``d_t = log π(top1) − log π(τ_t) ≥ 0``, and ``d_t == 0`` exactly where the
  actual token is rank 1 (asserted — it is the P0 scorer contract)
* logits at position ``t−1`` predict token ``t``. Every alignment in this module
  is written once, here.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from whetstone.segments import (
    ENDOFTEXT_ID,
    IM_END_ID,
    IM_START_ID,
    THINK_CLOSE_ID,
    THINK_OPEN_ID,
    SegmentMasks,
    parse_segments,
)

# Tokens that must never appear in text *we* wrote (card §1.6 / P3 gotcha 1).
BOUNDARY_IDS = frozenset(
    {THINK_OPEN_ID, THINK_CLOSE_ID, IM_START_ID, IM_END_ID, ENDOFTEXT_ID}
)

#: Marker classes for the per-marker-class hum readout (packet §7 metric 1).
#: The split is not cosmetic: activity 005 finding 7 measured ``case``/``✗`` at
#: ~0% of the Round-0 corpus, so the *branch* class can stay uncalibrated while
#: the overall S1 test passes — and the 32B teacher's branch-keeping traces
#: (activity 006) are written in exactly that vocabulary. Reporting one number
#: for both classes would hide the failure that matters most downstream.
MARKER_CLASSES: Dict[str, Tuple[str, ...]] = {
    "structural": ("⇒", "→", "goal", "let", ";"),
    "branch": ("case", "✗", "chk", "✓"),
}


# --------------------------------------------------------------------------
# Register card §2 — the structural whitelist
# --------------------------------------------------------------------------

def read_whitelist_strings(card_path: str | Path) -> List[str]:
    """Literal whitelist strings from register card **§2**.

    Anchored on the section *number*, never its title: activity 005 finding 3
    lost two sections from the compression prompt because a title-substring
    match stopped matching after ratification renamed the headings. "Config
    selected by prose title is config that will drift."
    """
    text = Path(card_path).read_text(encoding="utf-8")
    m = re.search(r"^##\s*2\.[^\n]*\n(.*?)(?=^##\s|\Z)", text, re.S | re.M)
    if m is None:
        raise ValueError(f"no '## 2.' section in {card_path}")
    fence = re.search(r"^```[^\n]*\n(.*?)^```", m.group(1), re.S | re.M)
    if fence is None:
        raise ValueError(f"no fenced block in §2 of {card_path}")
    strings = fence.group(1).split()
    if not strings:
        raise ValueError(f"§2 whitelist fence is empty in {card_path}")
    return strings


def whitelist_token_ids(
    tokenizer, strings: Sequence[str], *, single_token_only: bool = True
) -> Dict[int, str]:
    """Whitelist ids from card §2 strings, tokenized **bare and space-prefixed**.

    Packet §5 step 4: the whitelist enters R by fiat, below the occurrence
    floor, because its whole purpose is to install vocabulary the corpus is too
    thin to select statistically (the ``case``/``✗`` branch class).

    **Deviation from the packet's "all piece-ids included" (activity 007).**
    That instruction assumes each whitelist string is one token. Three of card
    §2's entry classes are not, and expanding their pieces admits the
    *alphabet* rather than the marker:

    * ``1.``–``9.`` → bare digits + ``.`` + ``' '`` (ids 16–24, 13, 220)
    * bare ``✗`` → undecodable byte fragments (ids 245, 25521), which occur in
      this corpus only as pieces of *other* multi-byte characters

    Measured on the Round-0 train split those pieces are **26.9% of all think
    tokens**, at 0.15–1.13 nats mean surprisal — below the p75 = 1.78 threshold,
    so the statistical rule would never have selected them. More decisively:
    meter test (c)'s value-substitution corruption edits a *digit* token, so
    training CE on digit types would train away the surprise the decisive probe
    has to detect. A marker that is not a token cannot be masked as one.

    A variant therefore contributes its id only when it tokenizes to a single
    token; multi-piece variants are dropped and reported by
    :func:`whitelist_dropped`. Pass ``single_token_only=False`` for the
    packet's literal behaviour.

    Returns ``{token_id: surface}`` where ``surface`` is the decoded piece.
    """
    ids: Dict[int, str] = {}
    for s in strings:
        for variant in (s, " " + s):
            pieces = tokenizer.encode(variant, add_special_tokens=False)
            if single_token_only and len(pieces) != 1:
                continue
            for tid in pieces:
                ids[tid] = tokenizer.decode([tid])
    return ids


def whitelist_dropped(tokenizer, strings: Sequence[str]) -> Dict[str, List[int]]:
    """``{variant: piece_ids}`` for multi-piece variants left out of R.

    Reported so the drop is visible in the journal rather than silent — the
    failure class activity 005 findings 3 and 5 both belong to.
    """
    dropped: Dict[str, List[int]] = {}
    for s in strings:
        for variant in (s, " " + s):
            pieces = tokenizer.encode(variant, add_special_tokens=False)
            if len(pieces) != 1:
                dropped[variant] = list(pieces)
    return dropped


def marker_class_ids(tokenizer) -> Dict[str, frozenset]:
    """``{class_name: {token_id, ...}}`` for :data:`MARKER_CLASSES`."""
    return {
        name: frozenset(whitelist_token_ids(tokenizer, strings))
        for name, strings in MARKER_CLASSES.items()
    }


# --------------------------------------------------------------------------
# Sequence construction
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Seq:
    """One scored sequence: ids, the prompt/completion split, and its masks."""

    uid: str
    ids: Tuple[int, ...]
    prompt_len: int
    masks: SegmentMasks
    level: int = 0

    @property
    def think_positions(self) -> List[int]:
        """Absolute positions of think-*content* tokens (boundaries excluded)."""
        return [i for i, m in enumerate(self.masks.think_mask) if m]

    @property
    def answer_positions(self) -> List[int]:
        return [i for i, m in enumerate(self.masks.answer_mask) if m]


def render_prompt(tokenizer, problem: str) -> List[int]:
    """Prompt token ids for one problem — no system prompt, thinking enabled.

    activity 003: v1's "put your reasoning between <think> tags" system prompt
    costs 8 points of accuracy and causes 6% duplicated-``</think>`` gate
    failures on Qwen3. There is no system message here, by decision.
    ``enable_thinking=True`` on every call (ROADMAP standing rule 4).
    """
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=True,
    )
    # transformers 5.x returns a BatchEncoding here, not a list (activity 003).
    ids = list(enc["input_ids"]) if hasattr(enc, "keys") else list(enc)
    return ids


def build_completion_text(compact_think: str, answer: str) -> str:
    """The native-rollout completion shape (packet §4). Body carries no tags."""
    return f"<think>\n{compact_think}\n</think>\n\n{answer}"


def build_sequence(
    tokenizer,
    *,
    uid: str,
    problem: str,
    think_body: str,
    answer: str,
    level: int = 0,
    require_gate: bool = True,
) -> Seq:
    """Build one :class:`Seq` from a corpus record's text fields.

    ``think_body`` is the think **body** with no tags (the corpus's
    ``compact_think``); the tags are added here so every caller gets the same
    boundary tokens in the same places.

    Raises:
        ValueError: if the problem text itself encodes a boundary token (it
            would parse as a real ``<think>`` and silently corrupt the masks),
            or — when ``require_gate`` — if the sequence does not clear the
            segment-parser gate ``g``.
    """
    prompt_ids = render_prompt(tokenizer, problem)
    # The chat template legitimately emits im_start/im_end at its own
    # structural positions; what must be zero is boundary tokens coming from
    # the *problem text*. Check the raw field, not the templated prompt.
    stray = BOUNDARY_IDS.intersection(
        tokenizer.encode(problem, add_special_tokens=False)
    )
    if stray:
        raise ValueError(f"{uid}: problem text encodes boundary tokens {sorted(stray)}")

    completion_ids = tokenizer.encode(
        build_completion_text(think_body, answer), add_special_tokens=False
    )
    ids = list(prompt_ids) + list(completion_ids)
    masks = parse_segments(ids, prompt_len=len(prompt_ids))
    if require_gate and masks.g != 1:
        raise ValueError(f"{uid}: segment gate g=0 ({masks.reason})")
    return Seq(
        uid=uid,
        ids=tuple(ids),
        prompt_len=len(prompt_ids),
        masks=masks,
        level=level,
    )


def build_sequence_from_ids(
    *, uid: str, prompt_ids: Sequence[int], completion_ids: Sequence[int], level: int = 0
) -> Seq:
    """Same contract, for records that already carry ``completion_token_ids``.

    Used for the verbose control set, whose native rollouts were captured with
    their token ids intact — re-tokenizing decoded text does not round-trip at
    the ``<think>`` boundary (design §12.1), so the stored ids are authoritative.
    """
    ids = list(prompt_ids) + list(completion_ids)
    masks = parse_segments(ids, prompt_len=len(prompt_ids))
    return Seq(uid=uid, ids=tuple(ids), prompt_len=len(prompt_ids), masks=masks, level=level)


# --------------------------------------------------------------------------
# Scoring — surprisal and the top-1 gap
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenScores:
    """Per-position scores for one sequence, aligned to :attr:`Seq.ids`.

    Position 0 is unconditioned and carries ``nan`` in both arrays; every other
    entry ``t`` is scored by the distribution predicted from ``ids[:t]``.
    """

    surprisal: Tuple[float, ...]   # S_t = -log pi(tau_t)
    gap: Tuple[float, ...]         # d_t = log pi(top1) - log pi(tau_t) >= 0

    def at(self, positions: Iterable[int]) -> Tuple[List[float], List[float]]:
        pos = [p for p in positions if p > 0]
        return ([self.surprisal[p] for p in pos], [self.gap[p] for p in pos])


def scores_from_prompt_logprobs(ids: Sequence[int], prompt_logprobs) -> TokenScores:
    """Convert one vLLM ``prompt_logprobs`` payload into :class:`TokenScores`.

    Accepts both the in-process object form (``entry[tid].logprob``) and the
    HTTP JSON form (``entry["<tid>"]["logprob"]``, keys stringified by JSON).

    Asserts the two P0 contract invariants at every position: the actual token
    is always present (so d_t is computable), and d_t is exactly 0 wherever the
    actual token is rank 1.
    """
    if len(prompt_logprobs) != len(ids):
        raise ValueError(
            f"prompt_logprobs has {len(prompt_logprobs)} entries for {len(ids)} tokens"
        )
    surprisal: List[float] = [math.nan]
    gap: List[float] = [math.nan]

    for pos in range(1, len(ids)):
        entry = prompt_logprobs[pos]
        if entry is None:
            raise ValueError(f"no logprob dict at position {pos}")
        actual_id = ids[pos]

        if isinstance(entry, dict) and entry and isinstance(next(iter(entry)), str):
            # HTTP form: JSON object keys are strings.
            items = {int(k): v for k, v in entry.items()}
            get_lp = lambda e: float(e["logprob"])          # noqa: E731
            get_rank = lambda e: int(e["rank"])             # noqa: E731
        else:
            items = dict(entry)
            get_lp = lambda e: float(e.logprob)             # noqa: E731
            get_rank = lambda e: int(e.rank)                # noqa: E731

        if actual_id not in items:
            raise ValueError(
                f"actual token {actual_id} missing at position {pos} — "
                "prompt_logprobs contract violated (need the actual token at "
                "every position to compute d_t)"
            )
        lp_actual = get_lp(items[actual_id])
        top1 = next(e for e in items.values() if get_rank(e) == 1)
        d = get_lp(top1) - lp_actual

        if d < -1e-6:
            raise ValueError(f"negative gap {d} at position {pos} — rank-1 is not the max")
        if get_rank(items[actual_id]) == 1 and abs(d) > 1e-6:
            raise ValueError(
                f"actual token is rank 1 at position {pos} but d_t={d} != 0"
            )
        surprisal.append(-lp_actual)
        gap.append(max(d, 0.0))

    return TokenScores(surprisal=tuple(surprisal), gap=tuple(gap))


# --------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------

def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile; ``q`` in [0, 100]. No numpy dependency."""
    xs = sorted(v for v in values if not math.isnan(v))
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def g_spike(gaps: Sequence[float], *, lam: float = 1.0, beta: float = 5.0) -> float:
    """``G_spike = exp[−(λ/β)·log((1/T)Σ_t exp(β·d_t))]`` (design §3.2).

    Computed in log space with the standard max-shift, because ``exp(β·d_t)``
    overflows for β=10 and the 40-nat gaps a corrupted trace produces.
    """
    xs = [g for g in gaps if not math.isnan(g)]
    if not xs:
        return math.nan
    m = max(xs)
    # log((1/T) Σ exp(β·d)) = β·m + log(mean(exp(β·(d−m))))
    lse = beta * m + math.log(sum(math.exp(beta * (x - m)) for x in xs) / len(xs))
    return math.exp(-(lam / beta) * lse)


def load_jsonl(path: str | Path) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


__all__ = [
    "BOUNDARY_IDS",
    "MARKER_CLASSES",
    "Seq",
    "TokenScores",
    "build_completion_text",
    "build_sequence",
    "build_sequence_from_ids",
    "g_spike",
    "load_jsonl",
    "marker_class_ids",
    "percentile",
    "read_whitelist_strings",
    "render_prompt",
    "scores_from_prompt_logprobs",
    "whitelist_dropped",
    "whitelist_token_ids",
]
