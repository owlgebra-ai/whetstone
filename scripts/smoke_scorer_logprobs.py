"""Smoke test: teacher-forced prompt-logprob scoring on the frozen scorer (spark / GB10).

Design §12.2: the scorer returns, in **one** teacher-forced prefill pass, both the
actual token's logprob and the rank-1 token's logprob at every position. The
per-token gap

    d_t = logprob(rank-1 token) - logprob(actual token)     (>= 0 by construction)

is the primitive underneath G_spike (Stage A), the Round-0 meter metrics, and the
Stage-B ZPD band-pass gates. If this pass is wrong, every reward in the pipeline is
wrong silently — hence this is an environment gate, not a nicety.

Two invariants are asserted:
  1. d_t is computable at every scored position (the actual token is always present
     in the returned dict even when it falls outside top-k, carrying a `rank`).
  2. d_t == 0 exactly where the actual token IS rank 1.

Usage (on spark, venv activated):
    python scripts/smoke_scorer_logprobs.py
"""

import math

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-1.7B"
TEXT = "The capital of France is Paris."


def gap(entry_dict, actual_id):
    """d_t = rank-1 logprob - actual-token logprob, from one prompt_logprobs entry."""
    actual = entry_dict[actual_id]
    top1 = next(e for e in entry_dict.values() if e.rank == 1)
    return top1.logprob - actual.logprob, actual, top1


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(TEXT).input_ids

    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.35)
    # max_tokens=1 — we want the prefill only; prompt_logprobs=2 gives top-2 plus the
    # actual token whenever it is outside the top-2.
    out = llm.generate([TEXT], SamplingParams(max_tokens=1, prompt_logprobs=2))[0]
    pl = out.prompt_logprobs

    assert pl is not None, "prompt_logprobs came back None"
    assert len(pl) == len(ids), f"len mismatch: {len(pl)} entries vs {len(ids)} tokens"
    assert pl[0] is None, "position 0 should have no logprob (nothing conditions it)"

    n_scored = n_rank1 = 0
    print(f"{'pos':>3} {'token':<12} {'rank':>4} {'lp_actual':>10} {'lp_top1':>9} {'d_t':>8}")
    for pos in range(1, len(pl)):
        entry = pl[pos]
        assert entry is not None, f"no logprob dict at position {pos}"
        actual_id = ids[pos]
        assert actual_id in entry, (
            f"actual token {actual_id} missing at position {pos} — "
            "cannot compute d_t; prompt_logprobs contract violated"
        )
        d_t, actual, top1 = gap(entry, actual_id)
        n_scored += 1

        assert d_t >= -1e-6, f"negative gap {d_t} at position {pos} — rank-1 is not the max"
        if actual.rank == 1:
            n_rank1 += 1
            assert math.isclose(d_t, 0.0, abs_tol=1e-6), (
                f"actual token is rank 1 at position {pos} but d_t={d_t} != 0"
            )

        print(
            f"{pos:>3} {tok.decode([actual_id])!r:<12} {actual.rank:>4} "
            f"{actual.logprob:>10.4f} {top1.logprob:>9.4f} {d_t:>8.4f}"
        )

    assert n_scored > 0, "no positions scored"
    assert n_rank1 > 0, "no rank-1 position in this sentence — d_t==0 branch untested"
    print(f"\nPASS: d_t computable at {n_scored}/{n_scored} positions; "
          f"{n_rank1} rank-1 positions all had d_t == 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
