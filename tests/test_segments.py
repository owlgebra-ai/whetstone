"""Unit tests for :mod:`whetstone.segments` (packet P2 Part 1).

All token ids below were dumped from the **real** tokenizer, not assumed:

    model:      Qwen/Qwen3-1.7B
    revision:   70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
    class:      Qwen2Tokenizer (transformers 5.14.1)
    dumped:     2026-08-02 on turing

Verified template behaviour at that revision:

    apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True)
      -> '<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n'
         ... i.e. it does NOT pre-fill <think>; the model emits it.
    apply_chat_template(..., enable_thinking=False)
      -> '...<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'
    multi-turn -> previous-turn <think> blocks are stripped.
    default (kwarg omitted) == enable_thinking=True.

Run either way::

    python tests/test_segments.py          # no pytest needed
    pytest tests/test_segments.py -q
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from whetstone.segments import (  # noqa: E402
    ENDOFTEXT_ID,
    IM_END_ID,
    IM_START_ID,
    R_CLOSE_BEFORE_OPEN,
    R_DUP_CLOSE,
    R_DUP_OPEN,
    R_EMPTY_ANSWER,
    R_NO_CLOSE,
    R_NO_OPEN,
    THINK_CLOSE_ID,
    THINK_OPEN_ID,
    W_EMPTY_THINK,
    W_PREAMBLE,
    W_TRAILING_AFTER_EOS,
    parse_segments,
    segment_token_counts,
)

# --- recorded ids (see module docstring for provenance) ---------------------
TOK_THINK_OPEN = 151667      # '<think>'
TOK_THINK_CLOSE = 151668     # '</think>'
TOK_IM_START = 151644        # '<|im_start|>'
TOK_IM_END = 151645          # '<|im_end|>'
TOK_ENDOFTEXT = 151643       # '<|endoftext|>'
TOK_ASSISTANT = 77091        # 'assistant'
TOK_NL = 198                 # '\n'
TOK_NLNL = 271               # '\n\n'
TOK_SPACE = 220              # ' '

#: Whitespace-only ids used by the empty-answer rule in these tests.
BLANK = frozenset({TOK_NL, TOK_NLNL, TOK_SPACE})

#: tok.apply_chat_template([{'role':'user','content':'What is 2+2?'}],
#:                         add_generation_prompt=True, enable_thinking=True)
PROMPT_IDS = [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30,
              151645, 198, 151644, 77091, 198]

#: tok.encode('<think>\nLet me compute. 2+2 = 4.\n</think>\n\n'
#:            'The answer is \\boxed{4}.<|im_end|>', add_special_tokens=False)
WELL_FORMED = [151667, 198, 10061, 752, 12564, 13, 220, 17, 10, 17, 284, 220,
               19, 624, 151668, 271, 785, 4226, 374, 1124, 79075, 90, 19,
               7810, 151645]
#   index:      0(<think>) 1..13 body   14(</think>) 15..23 answer  24(<|im_end|>)


# ---------------------------------------------------------------------------
# well-formed
# ---------------------------------------------------------------------------

def test_well_formed_short():
    m = parse_segments(WELL_FORMED)
    assert m.g == 1, m.reason
    assert m.reason == ""
    assert m.open_idx == 0 and m.close_idx == 14
    # Boundary tokens excluded from both masks — they are structure, not content.
    assert (m.think_start, m.think_end) == (1, 14)
    assert (m.answer_start, m.answer_end) == (15, 24)
    assert m.segment_lengths() == (13, 9)
    assert len(m.think_mask) == len(m.answer_mask) == len(WELL_FORMED)
    assert m.think_mask[0] == 0 and m.think_mask[14] == 0      # <think>, </think>
    assert m.answer_mask[24] == 0                              # <|im_end|>
    assert sum(m.think_mask) == 13 and sum(m.answer_mask) == 9
    # Disjoint.
    assert all(t + a <= 1 for t, a in zip(m.think_mask, m.answer_mask))
    assert m.warnings == ()


def test_well_formed_long():
    ids = [THINK_OPEN_ID] + [1234] * 5000 + [THINK_CLOSE_ID] + [42] * 300 + [IM_END_ID]
    m = parse_segments(ids)
    assert m.g == 1
    assert m.segment_lengths() == (5000, 300)


def test_well_formed_no_eos():
    """Answer that runs to the end of the sequence with no <|im_end|>."""
    ids = WELL_FORMED[:-1]  # drop the trailing <|im_end|>
    m = parse_segments(ids)
    assert m.g == 1
    assert m.segment_lengths() == (13, 9)
    assert m.answer_end == len(ids)


def test_endoftext_also_terminates_answer():
    ids = WELL_FORMED[:-1] + [ENDOFTEXT_ID]
    m = parse_segments(ids)
    assert m.g == 1
    assert m.answer_end == 24
    assert m.answer_mask[24] == 0


def test_trailing_tokens_after_eos_warn():
    ids = WELL_FORMED + [785, 4226]
    m = parse_segments(ids)
    assert m.g == 1
    assert W_TRAILING_AFTER_EOS in m.warnings
    assert m.answer_end == 24  # answer still ends at the first EOS


# ---------------------------------------------------------------------------
# malformed -> g = 0
# ---------------------------------------------------------------------------

def test_cap_hit_missing_close():
    """Truncation at the token cap: no </think> ever arrives."""
    ids = [THINK_OPEN_ID] + [1234] * 100
    m = parse_segments(ids)
    assert m.g == 0
    assert m.reason == R_NO_CLOSE
    assert m.close_idx == -1
    # Best-effort masks still returned: everything after <think> is think.
    assert m.segment_lengths() == (100, 0)
    assert sum(m.answer_mask) == 0


def test_duplicated_think_open():
    ids = [THINK_OPEN_ID, 1, THINK_OPEN_ID, 2, THINK_CLOSE_ID, 3, IM_END_ID]
    m = parse_segments(ids)
    assert m.g == 0 and m.reason == R_DUP_OPEN


def test_duplicated_think_close():
    ids = [THINK_OPEN_ID, 1, THINK_CLOSE_ID, 2, THINK_CLOSE_ID, 3, IM_END_ID]
    m = parse_segments(ids)
    assert m.g == 0 and m.reason == R_DUP_CLOSE


def test_close_before_open():
    ids = [THINK_CLOSE_ID, 1, THINK_OPEN_ID, 2, IM_END_ID]
    m = parse_segments(ids)
    assert m.g == 0 and m.reason == R_CLOSE_BEFORE_OPEN


def test_missing_think_open():
    ids = [1, 2, THINK_CLOSE_ID, 3, IM_END_ID]
    m = parse_segments(ids)
    assert m.g == 0 and m.reason == R_NO_OPEN


def test_empty_answer_zero_tokens():
    ids = [THINK_OPEN_ID, 1, 2, THINK_CLOSE_ID, IM_END_ID]
    m = parse_segments(ids)
    assert m.g == 0 and m.reason == R_EMPTY_ANSWER
    assert m.segment_lengths() == (2, 0)


def test_empty_answer_whitespace_only():
    """`</think>\\n\\n` then the cap hits — one blank token is not an answer.

    Only detected when ``blank_token_ids`` is supplied; without it the rule is
    token-count based and this passes the gate. Both behaviours are asserted so
    the difference stays visible to whoever wires up the training loop.
    """
    ids = [THINK_OPEN_ID, 1, 2, THINK_CLOSE_ID, TOK_NLNL]
    assert parse_segments(ids).g == 1                          # count-based
    m = parse_segments(ids, blank_token_ids=BLANK)             # blank-aware
    assert m.g == 0 and m.reason == R_EMPTY_ANSWER


# ---------------------------------------------------------------------------
# empty think — explicitly NOT malformed
# ---------------------------------------------------------------------------

def test_empty_think_enable_thinking_false_shape():
    """`<think>\\n\\n</think>\\n\\n<answer>` — the enable_thinking=False pre-fill.

    We never use that mode, but the parser must classify it as an empty think
    segment rather than exploding or gating it out.
    """
    ids = [THINK_OPEN_ID, TOK_NLNL, THINK_CLOSE_ID, TOK_NLNL, 785, 19, IM_END_ID]
    m = parse_segments(ids, blank_token_ids=BLANK)
    assert m.g == 1, m.reason
    assert W_EMPTY_THINK in m.warnings
    assert m.think_len == 1          # the '\n\n' token
    assert m.answer_len == 3


def test_empty_think_zero_tokens():
    ids = [THINK_OPEN_ID, THINK_CLOSE_ID, 785, 19, IM_END_ID]
    m = parse_segments(ids)
    assert m.g == 1
    assert W_EMPTY_THINK in m.warnings
    assert m.think_len == 0


# ---------------------------------------------------------------------------
# <think> mid-text
# ---------------------------------------------------------------------------

def test_think_open_mid_text_warns_but_gate_stays_open():
    """Tokens before <think> belong to neither segment.

    Not in the packet's g=0 list, so the gate stays open — but those tokens get
    no loss routing in Stage C, so the anomaly is surfaced as a warning and
    counted in ``preamble_len``.
    """
    ids = [785, 4226, THINK_OPEN_ID, 1, 2, THINK_CLOSE_ID, 3, 4, IM_END_ID]
    m = parse_segments(ids)
    assert m.g == 1
    assert W_PREAMBLE in m.warnings
    assert m.preamble_len == 2
    assert m.think_mask[0] == 0 and m.think_mask[1] == 0
    assert m.answer_mask[0] == 0 and m.answer_mask[1] == 0
    assert m.segment_lengths() == (2, 2)


# ---------------------------------------------------------------------------
# prompt-prefixed sequences (the §12.2 one-pass scorer prefill)
# ---------------------------------------------------------------------------

def test_prompt_prefix_masked_out():
    ids = PROMPT_IDS + WELL_FORMED
    p = len(PROMPT_IDS)
    m = parse_segments(ids, prompt_len=p)
    assert m.g == 1
    assert m.segment_lengths() == (13, 9)
    assert sum(m.think_mask[:p]) == 0 and sum(m.answer_mask[:p]) == 0
    assert m.open_idx == p and m.close_idx == p + 14


def test_prompt_len_out_of_range():
    try:
        parse_segments(WELL_FORMED, prompt_len=len(WELL_FORMED) + 1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-range prompt_len")


def test_think_opened_by_prompt():
    """v1-style seeded prompt ending in `<think>\\n` — completion has no opener."""
    ids = [1, 2, 3, THINK_CLOSE_ID, 4, 5, IM_END_ID]
    assert parse_segments(ids).g == 0                       # default: malformed
    m = parse_segments(ids, think_opened_by_prompt=True)
    assert m.g == 1
    assert m.segment_lengths() == (3, 2)
    assert m.open_idx == -1


# ---------------------------------------------------------------------------
# single-turn assertion
# ---------------------------------------------------------------------------

def test_multi_turn_raises():
    ids = (
        [IM_START_ID, TOK_ASSISTANT, TOK_NL, 1, IM_END_ID]
        + [IM_START_ID, TOK_ASSISTANT, TOK_NL, 2, IM_END_ID]
    )
    try:
        parse_segments(ids)
    except ValueError as e:
        assert "single-turn" in str(e)
        return
    raise AssertionError("expected ValueError on a multi-turn sequence")


def test_single_turn_prompt_does_not_raise():
    parse_segments(PROMPT_IDS + WELL_FORMED, prompt_len=len(PROMPT_IDS))


# ---------------------------------------------------------------------------
# batch helper
# ---------------------------------------------------------------------------

def test_segment_token_counts_reports_separately():
    a = parse_segments(WELL_FORMED)
    b = parse_segments(WELL_FORMED)
    assert segment_token_counts([a, b]) == (26, 18)


# ---------------------------------------------------------------------------
# real sampled rollouts (fixture captured from vLLM on turing)
# ---------------------------------------------------------------------------

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                        "real_rollouts_qwen3_1p7b.json")


def _load_fixture():
    if not os.path.exists(_FIXTURE):
        return None
    with open(_FIXTURE) as f:
        return json.load(f)


def test_real_rollouts_all_start_with_think_open():
    """Every real rollout begins with the model's own <think> (behaviour 1)."""
    fx = _load_fixture()
    if fx is None:
        print("  SKIP (no fixture)", end=" ")
        return
    for name, rec in fx.items():
        if name == "_meta":
            continue
        ids = rec["completion_token_ids"]
        assert ids[0] == THINK_OPEN_ID, f"{name}: does not start with <think>"
        assert rec["n_think_open"] == 1, f"{name}: {rec['n_think_open']} openers"


def test_real_rollouts_parse_as_their_own_shape():
    """Data-driven: each fixture's own boundary count dictates the expected gate.

    Kept structure-driven rather than name-driven so new fixtures captured from
    later runs can be dropped in without editing assertions.
    """
    fx = _load_fixture()
    if fx is None:
        print("  SKIP (no fixture)", end=" ")
        return
    seen_ok = seen_cap = 0
    for name, rec in fx.items():
        if name == "_meta":
            continue
        ids = rec["completion_token_ids"]
        m = parse_segments(ids, blank_token_ids=BLANK)
        if rec["n_think_close"] == 0:
            # Real cap-hit truncation.
            assert rec["finish_reason"] == "length", name
            assert m.g == 0 and m.reason == R_NO_CLOSE, f"{name}: {m.reason}"
            assert m.answer_len == 0
            assert m.think_len == len(ids) - 1  # everything after <think>
            seen_cap += 1
        else:
            assert m.g == 1, f"{name}: g=0 ({m.reason})"
            assert m.think_len > 0 and m.answer_len > 0
            # Token-level split agrees with the string-level one.
            assert rec["completion_text"].count("</think>") == 1
            # think_len + answer_len + boundaries + eos accounts for everything.
            assert m.close_idx == m.think_end
            seen_ok += 1
        print(f"  [{name} g={m.g} think={m.think_len} answer={m.answer_len}]", end=" ")
    assert seen_ok >= 1 and seen_cap >= 1, "fixture must cover both shapes"


# ---------------------------------------------------------------------------
# live-tokenizer guard: recorded ids must still match the shipped tokenizer
# ---------------------------------------------------------------------------

def test_recorded_ids_match_live_tokenizer():
    """Skipped where transformers / the model snapshot is unavailable."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    except Exception as e:  # no transformers, no cache, no network
        print(f"  SKIP ({type(e).__name__})", end=" ")
        return
    assert tok.convert_tokens_to_ids(["<think>", "</think>"]) == [
        THINK_OPEN_ID, THINK_CLOSE_ID]
    assert tok.convert_tokens_to_ids("<|im_start|>") == TOK_IM_START
    assert tok.convert_tokens_to_ids("<|im_end|>") == TOK_IM_END
    assert tok.eos_token_id == IM_END_ID
    # Single-token inline encoding is what makes token-level masking possible.
    assert tok.encode("<think>", add_special_tokens=False) == [THINK_OPEN_ID]
    assert tok.encode("</think>", add_special_tokens=False) == [THINK_CLOSE_ID]
    # enable_thinking=True must NOT pre-fill <think>.
    p = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        add_generation_prompt=True, tokenize=False, enable_thinking=True)
    assert p.endswith("<|im_start|>assistant\n"), repr(p)
    assert "<think>" not in p
    # enable_thinking=False must pre-fill an empty think block.
    p0 = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)
    assert p0.endswith("<think>\n\n</think>\n\n"), repr(p0)
    # Recorded prompt ids still reproduce.
    enc = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        add_generation_prompt=True, tokenize=True, enable_thinking=True)
    ids = list(enc if isinstance(enc, list) else enc["input_ids"])
    assert ids == PROMPT_IDS, ids


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            print(f"{name} ...", end=" ")
            fn()
            print("ok")
        except Exception as e:
            failed += 1
            print(f"FAIL: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
