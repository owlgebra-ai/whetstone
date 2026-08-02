"""Smoke test: Qwen3 hybrid-mode generation with thinking enabled (turing / RTX 5090).

Design §11: Qwen3-1.7B is a *hybrid* thinking model — `enable_thinking=True` must be
passed on **every** rollout, scoring pass and eval call, or the model silently emits an
empty `<think></think>` block and the segment parser sees no think segment at all.
This script is the environment gate for that: it fails loudly if the chat template
does not honour the flag, or if the Blackwell (sm_120) kernels are missing.

Pass = generation contains a closed `<think>…</think>` block and the correct answer
after it (verified with the deterministic verifier, `whetstone.verify`).

Usage:
    .venv/bin/python scripts/smoke_qwen3_thinking.py
"""

import sys

sys.path.insert(0, ".")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from whetstone.verify import extract_answer, verify_response

MODEL = "Qwen/Qwen3-1.7B"
QUESTION = "What is 17 * 23?"
GOLD = "391"


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # REQUIRED on every hybrid-Qwen3 call (design §11)
    )

    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.85)
    out = llm.generate(
        [prompt], SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048)
    )[0].outputs[0].text

    print("=" * 60)
    print(out[-500:])
    print("=" * 60)

    assert "</think>" in out, "no think block — enable_thinking not honored"

    think_len = len(tok(out.split("</think>")[0]).input_ids)
    answer_seg = out.split("</think>", 1)[1]
    answer_len = len(tok(answer_seg).input_ids)
    # Segment-level reporting, always — one combined length number is how drift hides.
    print(f"think_tokens={think_len}  answer_tokens={answer_len}")
    print(f"extracted_answer={extract_answer(out)!r}  gold={GOLD!r}")

    assert verify_response(out, GOLD), f"wrong answer; expected {GOLD}"
    print("PASS: thinking honored + answer correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
