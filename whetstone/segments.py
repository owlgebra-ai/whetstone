"""Token-level ``<think>`` / answer segment parsing for Qwen3 (design §12.1).

Every v2 stage routes rewards and losses by the ``<think>`` / ``</think>``
boundaries: Stage-A's ``G_budget`` measures the think segment, Stage-B's ZPD
weights apply to think tokens, Stage-C sends think tokens to length pressure +
TEA and answer tokens to forward-KL + the SCA length band. A one-token-off mask
silently corrupts all of them, so this module is deliberately paranoid and
deliberately *token-level*.

Why token-level (packet P2 Part 1 step 4): splitting the decoded string on
``"</think>"`` and re-tokenizing each half does not round-trip — the boundary
token merges differently with its neighbours, and every downstream index shifts.
The masks here are computed on token ids only; the string-level split in
:mod:`whetstone.reward.extract` is a *separate* concern (answer extraction for
grading) and stays as it is, as does :mod:`whetstone.verify`.

Verified against the real ``Qwen/Qwen3-1.7B`` tokenizer at revision
``70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`` (transformers 5.14.1,
``Qwen2Tokenizer``) — see ``tests/test_segments.py`` for the recorded ids and
the three template behaviours this parser is built around:

1. ``enable_thinking=True`` renders a generation prompt ending at
   ``<|im_start|>assistant\\n`` — it does **not** pre-fill ``<think>``. The model
   emits ``<think>`` itself, so a completion *starts with* token 151667.
2. ``enable_thinking=False`` pre-fills ``<think>\\n\\n</think>\\n\\n``. We never
   use that mode, but such a sequence must parse as an *empty think segment*,
   not as malformed.
3. The multi-turn template strips previous-turn think blocks. All whetstone
   calls are single-turn, so :func:`parse_segments` asserts single-turn rather
   than trying to handle the multi-turn shape.

This module has no third-party imports on purpose: it must be importable (and
testable) on a laptop with no torch/vLLM. Masks are plain ``tuple[int, ...]`` of
0/1; callers wrap them in ``np.asarray`` / ``torch.tensor`` as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

# --- Qwen3 token ids (verified, not assumed — see module docstring) ----------
# Note: ``<think>``/``</think>`` are added tokens with ``special=False``, so
# ``decode(..., skip_special_tokens=True)`` does NOT strip them. They encode to
# exactly one id each inline, which is what makes token-level masking possible.
THINK_OPEN_ID = 151667      # "<think>"
THINK_CLOSE_ID = 151668     # "</think>"
IM_START_ID = 151644        # "<|im_start|>"
IM_END_ID = 151645          # "<|im_end|>"   (eos_token)
ENDOFTEXT_ID = 151643       # "<|endoftext|>" (pad_token)
ASSISTANT_ID = 77091        # "assistant" — the role token after <|im_start|>

#: Ids that terminate the answer segment. The terminator itself is excluded
#: from ``answer_mask``: the answer *content* is what length bands and the
#: answer-side KL are computed over.
EOS_IDS = frozenset({IM_END_ID, ENDOFTEXT_ID})

# --- quality-gate reasons (design §12.1 / SCA gate rule) --------------------
G_OK = ""
R_NO_CLOSE = "missing_think_close"        # cap-hit truncation
R_DUP_OPEN = "duplicated_think_open"
R_DUP_CLOSE = "duplicated_think_close"
R_CLOSE_BEFORE_OPEN = "close_before_open"
R_NO_OPEN = "missing_think_open"
R_EMPTY_ANSWER = "empty_answer"

# --- non-fatal warnings (g stays 1; visible for diagnostics) ----------------
W_EMPTY_THINK = "empty_think"
W_PREAMBLE = "preamble_before_think"
W_TRAILING_AFTER_EOS = "trailing_tokens_after_eos"


@dataclass(frozen=True)
class SegmentMasks:
    """Segment masks and quality gate for one rollout.

    ``think_mask`` and ``answer_mask`` are the same length as the token
    sequence handed to :func:`parse_segments` (including any prompt prefix,
    whose positions are 0 in both masks). They are disjoint, and both exclude
    the boundary tokens ``<think>`` / ``</think>`` and the terminating EOS —
    those are structure, not content, and counting them in ``think_len`` would
    put ``G_budget`` a token or two off from what the length schedule means.

    ``g`` is the SCA quality gate: 1 = well-formed, 0 = malformed. g=0 rollouts
    are excluded from all structural rewards and alignment losses; they still
    flow through the accuracy path (``R_acc = 0``) via the reward code.

    ``reason`` is ``""`` when ``g == 1``, otherwise the first failing rule.
    ``warnings`` records non-fatal anomalies that do not clear the gate.
    """

    think_mask: Tuple[int, ...]
    answer_mask: Tuple[int, ...]
    g: int
    reason: str = G_OK
    warnings: Tuple[str, ...] = ()
    # Half-open [start, end) content spans; (0, 0) when absent.
    think_start: int = 0
    think_end: int = 0
    answer_start: int = 0
    answer_end: int = 0
    # Boundary-token positions, -1 when the token is absent.
    open_idx: int = -1
    close_idx: int = -1
    # Tokens emitted before ``<think>`` — belong to neither segment.
    preamble_len: int = 0

    @property
    def think_len(self) -> int:
        return self.think_end - self.think_start

    @property
    def answer_len(self) -> int:
        return self.answer_end - self.answer_start

    def segment_lengths(self) -> Tuple[int, int]:
        """``(think_len, answer_len)`` — always report these two separately.

        A single combined length number is how segment drift hides (CLAUDE.md
        invariant); every dashboard and log line should use this.
        """
        return (self.think_len, self.answer_len)


def _assert_single_turn(token_ids: Sequence[int]) -> None:
    """Raise if the sequence contains more than one assistant turn.

    The Qwen3 multi-turn template strips previous-turn think blocks, so a
    multi-turn sequence has think content the masks cannot account for. Every
    whetstone call is single-turn; we assert that rather than silently
    producing masks for the wrong turn.
    """
    n_assistant = sum(
        1
        for i in range(len(token_ids) - 1)
        if token_ids[i] == IM_START_ID and token_ids[i + 1] == ASSISTANT_ID
    )
    if n_assistant > 1:
        raise ValueError(
            f"parse_segments expects a single-turn sequence; found {n_assistant} "
            "assistant headers. Multi-turn templates strip previous-turn think "
            "blocks, so the masks would be wrong."
        )


def _all_blank(
    token_ids: Sequence[int], start: int, end: int, blank_token_ids: frozenset
) -> bool:
    """True iff every id in ``[start, end)`` is a known whitespace-only token."""
    if start >= end:
        return True
    if not blank_token_ids:
        return False
    return all(token_ids[i] in blank_token_ids for i in range(start, end))


def parse_segments(
    token_ids: Sequence[int],
    *,
    prompt_len: int = 0,
    think_opened_by_prompt: bool = False,
    blank_token_ids: frozenset = frozenset(),
) -> SegmentMasks:
    """Compute think/answer masks and the quality gate ``g`` for one sequence.

    Args:
        token_ids: Full token id sequence. Either the completion alone
            (``prompt_len=0``) or the concatenated ``(q, tau)`` that the scorer
            prefills in one pass (design §12.2) — pass ``prompt_len`` in that
            case so prompt positions are masked out of both segments.
        prompt_len: Number of leading positions belonging to the prompt. Those
            positions are 0 in both masks and are never scanned for boundaries.
        think_opened_by_prompt: Set when the prompt itself ends with an open
            ``<think>`` (v1-style seeded harvest prompts). The think segment
            then starts at ``prompt_len`` and a missing ``<think>`` in the
            completion is legal. Default ``False`` matches the v2 path, where
            ``enable_thinking=True`` makes the model emit ``<think>`` itself.
        blank_token_ids: Optional set of ids that decode to whitespace only
            (build it with :func:`blank_token_ids_for`). Supplying it lets the
            empty-answer rule catch a rollout that emitted ``</think>\\n\\n``
            and then hit the token cap — one whitespace token is a non-empty
            answer by token count but an empty answer in every sense that
            matters. Without it, only a zero-length answer counts as empty.

    Returns:
        :class:`SegmentMasks`. Malformed sequences still return usable masks
        (best effort) alongside ``g = 0``; callers must check ``g``, not
        assume the masks are meaningful.

    Raises:
        ValueError: if the sequence is multi-turn, or ``prompt_len`` is out of
        range.
    """
    n = len(token_ids)
    if prompt_len < 0 or prompt_len > n:
        raise ValueError(f"prompt_len={prompt_len} out of range for {n} tokens")
    _assert_single_turn(token_ids)

    body = range(prompt_len, n)
    open_positions = [i for i in body if token_ids[i] == THINK_OPEN_ID]
    close_positions = [i for i in body if token_ids[i] == THINK_CLOSE_ID]

    zeros = (0,) * n

    def _fail(reason: str, **kw) -> SegmentMasks:
        return SegmentMasks(
            think_mask=kw.pop("think_mask", zeros),
            answer_mask=kw.pop("answer_mask", zeros),
            g=0,
            reason=reason,
            **kw,
        )

    # --- structural rules, in the packet's order ---------------------------
    if len(open_positions) > 1:
        return _fail(R_DUP_OPEN, open_idx=open_positions[0],
                     close_idx=close_positions[0] if close_positions else -1)
    if len(close_positions) > 1:
        return _fail(R_DUP_CLOSE, open_idx=open_positions[0] if open_positions else -1,
                     close_idx=close_positions[0])

    open_idx = open_positions[0] if open_positions else -1
    close_idx = close_positions[0] if close_positions else -1

    if open_idx == -1 and not think_opened_by_prompt:
        # No opener and none supplied by the prompt. Whatever this is, the
        # think segment is not locatable.
        return _fail(R_NO_OPEN, open_idx=-1, close_idx=close_idx)

    if close_idx == -1:
        # Cap-hit truncation: everything is think, there is no answer.
        think_start = open_idx + 1 if open_idx != -1 else prompt_len
        mask = tuple(1 if think_start <= i < n else 0 for i in range(n))
        return _fail(
            R_NO_CLOSE,
            think_mask=mask,
            open_idx=open_idx,
            close_idx=-1,
            think_start=think_start,
            think_end=n,
            preamble_len=(open_idx - prompt_len) if open_idx != -1 else 0,
        )

    if open_idx != -1 and close_idx < open_idx:
        return _fail(R_CLOSE_BEFORE_OPEN, open_idx=open_idx, close_idx=close_idx)

    # --- content spans -----------------------------------------------------
    think_start = open_idx + 1 if open_idx != -1 else prompt_len
    think_end = close_idx
    preamble_len = (open_idx - prompt_len) if open_idx != -1 else 0

    answer_start = close_idx + 1
    answer_end = n
    trailing_after_eos = False
    for i in range(answer_start, n):
        if token_ids[i] in EOS_IDS:
            answer_end = i
            trailing_after_eos = any(
                token_ids[j] not in EOS_IDS for j in range(i + 1, n)
            )
            break

    warnings: list = []
    if think_end <= think_start or _all_blank(token_ids, think_start, think_end, blank_token_ids):
        # enable_thinking=False pre-fill, or a model that closed <think>
        # immediately. Explicitly NOT malformed (packet P2 Part 1 step 2).
        warnings.append(W_EMPTY_THINK)
    if preamble_len > 0:
        # Tokens before <think> belong to neither segment. Not in the packet's
        # g=0 list, so the gate stays open, but it is surfaced because those
        # tokens receive no loss routing in Stage C.
        warnings.append(W_PREAMBLE)
    if trailing_after_eos:
        warnings.append(W_TRAILING_AFTER_EOS)

    think_mask = tuple(1 if think_start <= i < think_end else 0 for i in range(n))
    answer_mask = tuple(1 if answer_start <= i < answer_end else 0 for i in range(n))

    if answer_end <= answer_start or _all_blank(
        token_ids, answer_start, answer_end, blank_token_ids
    ):
        return _fail(
            R_EMPTY_ANSWER,
            think_mask=think_mask,
            answer_mask=answer_mask,
            warnings=tuple(warnings),
            open_idx=open_idx,
            close_idx=close_idx,
            think_start=think_start,
            think_end=think_end,
            answer_start=answer_start,
            answer_end=answer_end,
            preamble_len=preamble_len,
        )

    return SegmentMasks(
        think_mask=think_mask,
        answer_mask=answer_mask,
        g=1,
        reason=G_OK,
        warnings=tuple(warnings),
        think_start=think_start,
        think_end=think_end,
        answer_start=answer_start,
        answer_end=answer_end,
        open_idx=open_idx,
        close_idx=close_idx,
        preamble_len=preamble_len,
    )


def blank_token_ids_for(tokenizer, *, limit: Optional[int] = None) -> frozenset:
    """Ids whose decoded form is whitespace-only, for the empty-answer rule.

    Computed once per tokenizer and passed to :func:`parse_segments`. Scanning
    the whole 151k vocab takes a few seconds; ``limit`` caps the scan for tests.
    """
    blank = set()
    n = limit if limit is not None else len(tokenizer)
    for tid in range(n):
        piece = tokenizer.decode([tid])
        if piece and not piece.strip():
            blank.add(tid)
    return frozenset(blank)


def segment_token_counts(masks: Iterable[SegmentMasks]) -> Tuple[int, int]:
    """Summed ``(think_tokens, answer_tokens)`` over a batch — reported apart."""
    think = answer = 0
    for m in masks:
        think += m.think_len
        answer += m.answer_len
    return (think, answer)


__all__ = [
    "THINK_OPEN_ID",
    "THINK_CLOSE_ID",
    "IM_START_ID",
    "IM_END_ID",
    "ENDOFTEXT_ID",
    "ASSISTANT_ID",
    "EOS_IDS",
    "SegmentMasks",
    "parse_segments",
    "blank_token_ids_for",
    "segment_token_counts",
    "G_OK",
    "R_NO_CLOSE",
    "R_DUP_OPEN",
    "R_DUP_CLOSE",
    "R_CLOSE_BEFORE_OPEN",
    "R_NO_OPEN",
    "R_EMPTY_ANSWER",
    "W_EMPTY_THINK",
    "W_PREAMBLE",
    "W_TRAILING_AFTER_EOS",
]
