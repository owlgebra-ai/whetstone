"""Smoke test for whetstone.verify — runs without torch, no GPU needed."""
import sys
sys.path.insert(0, ".")
from whetstone.verify import verify_response, extract_answer

# (completion, gold, expected)
cases = [
    # v4.6.1: post-think extraction only
    ("<think>blah</think>\\boxed{10000000}", "10000000", True),
    # thousands separator
    ("<think>blah</think>\\boxed{10,000,000}", "10000000", True),
    # fraction equivalence
    ("<think>blah</think>\\boxed{\\frac{3}{2}}", "1.5", True),
    # post-think-only extraction: \cot A in <think> must NOT cause false positive
    ("<think>\\cot A is the answer</think>\\boxed{1}", "1", True),
    # final-answer regex
    ("<think>compact reasoning</think>The answer is 5.", "5", True),
    ("<think>compact reasoning</think>Final answer: 42", "42", True),
    # wrong answer
    ("<think>x = 3</think>The answer is 5.", "7", False),
    # no think tags
    ("no think tags here. answer: 7", "7", True),
    # empty / None
    ("", "1", False),
    ("<think>x</think>", "1", False),
]

n_ok = 0
for i, (c, g, exp) in enumerate(cases):
    got = verify_response(c, g)
    ok = got == exp
    n_ok += int(ok)
    status = "OK" if ok else "FAIL"
    print(f"  case {i}: got={got} exp={exp} {status}")

print(f"{n_ok}/{len(cases)} pass")
print("extract_answer(post-think):", extract_answer("<think>x</think>\\boxed{99}"))
sys.exit(0 if n_ok == len(cases) else 1)
