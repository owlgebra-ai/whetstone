# 010 — Stage C: segment-routed DAPO recovery RL

- **Packet:** [packets/P7-stage-c-rl.md](packets/P7-stage-c-rl.md)
- **Status:** in-progress
- **Machine(s):** mac (code) / turing (rollouts + π_0 anchor) / spark (trainer)
- **Code commit(s):** `624c8b8` (packet claim) →
- **Started:** 2026-08-05

## Goal

Convert pass@k into pass@1 on the round-1 Stage-B student. 009 established the
substrate: pass@8 90.50% (89.50% strict) against Pass@1 66.50% (64.25% strict),
63% mixed groups, 2× baseline entropy, correct rollouts *shorter* than incorrect
ones. Stage C is the stage built for unreliability. Deliverables: the pre-RL
entropy card (F3c debt), a strict-grading reward, the DAPO loop with segment
routing + TEA, a 50–100 step pilot carrying the **F4 gate**, then Phase 1
(recovery, 4,000 problems) and Phase 2 (boost, fresh draw).

---

## Packet corrections found before execution

Two internal contradictions in P7, resolved here and noted so the next reader
does not re-derive them.

**(1) Bucketing temperature.** Part 0.4 says the Phase-1 K=8 bucketing runs at
`T=0.7`; the Part-1 sampling table says bucketing must run at **`T=1.0`, top-p
1.0** and explicitly names "Parts 0.4, re-buckets" as its scope, with the reason
attached: *"must match rollout sampling or buckets mis-predict group
composition."* **Resolved in favour of the table** — it is the later, reasoned
statement, it names the earlier one, and its argument is correct: a bucket table
built at a different temperature than the sampler predicts the wrong group
composition, which is the one thing the table exists to do.

**(2) Rollout sampling params.** The §11 gotcha list says rollouts run at
"T=0.7 top-p 0.95 seed-per-rollout"; the Part-1 table pins **T=1.0 / top-p 1.0**
with a policy-gradient-correctness argument (the gradient assumes samples from π
itself; top-p truncates gradient support). **Resolved in favour of the table.**
The gotcha's surviving content is the *seed-per-rollout* rule
(`sha1(uid:k:step:seed)`), which is independent of temperature and is kept:
byte-identical group members make within-group advantage zero and DAPO silently
learns nothing from them.

Both resolutions are testable at the pilot: Part 1's micro-check (200-problem
K=8 at T=1.0 vs 009's T=0.7 numbers) is exactly the measurement that would
falsify them, and it runs before any long run.

---

## Runs

### Run 1 — 2026-08-05, Part 0.3: stop `spark:8101`

Nothing to stop. `ps aux | grep -iE "vllm|python"` on spark showed **no vLLM
process at all** — neither :8101 (π-round1, Stage B's server) nor :8100
(`scorer_v1`). Both had already exited at the end of activity 009. Port 8000 is
the unrelated `llama-swap` service and was left alone, per activity 001.

Verified listening ports on spark: 22, 53, 80, 631, 8000, 8080, 9400, 11000,
plus tailscale. **No 81xx.** Precondition satisfied; nothing killed, so no risk
of the `pkill` self-match gotcha.

### Run 2 — 2026-08-05 20:00, Part 0.1: entropy card on the round-1 student (F3c)

The F3c measurement 009 left owed, doubling as TEA's calibration baseline.
Protocol copied verbatim from the audit's own `config` block so the student and
the baseline are measured identically.

- machine: turing (GPU idle, 18 MiB at launch)
- commit: `624c8b8`
- command (two-phase — vLLM does not release GPU memory promptly, so generation
  and teacher-forced scoring run as separate invocations):

```
python scripts/entropy_audit.py \
  --model /data/whetstone/ckpt/stageb/golden/round1/final \
  --pool /data/whetstone/data/pool/val_2k.jsonl \
  --out_dir /data/whetstone/runs/entropy_stagec_init \
  --n 200 --seed 0 --temperature 0.9 --top_p 0.95 \
  --max_tokens 16384 --max_len 20480 --chunk 1024 [--generate_only]
```

- generation: 200 rollouts in ~2 min → `rollouts.jsonl`
- scoring: (in progress)

Baseline to compare against (009, original checkpoint, same protocol): think
entropy mean **0.31759**, p50 **0.027817**, p80 **0.69234**, collapse mass 56.8%.

### Run 3 — 2026-08-05, Part 0.2 + Part 1b: the reward instrument (mac, CPU only)

Built and unit-tested before any GPU time, per the packet's ordering.

**`whetstone/reward/strict.py`** — the strict grader. `verify.py` untouched
(CLAUDE.md invariant). Removes exactly the two leniencies finding 15 measured
and nothing else: the `endswith` suffix fallback, and `_strip_think`'s
whole-text fallback when `</think>` is absent. Normalization
(`_normalize`, `_try_numeric`) is **imported verbatim** from `verify.py` rather
than reimplemented, so the two graders can only ever differ in the two
documented ways; if the normalizer changes they move together. Returns both
verdicts in one object (`strict`, `as_scored`, `lenient_only`) so a caller
cannot report one and label it the other.

**`whetstone/reward/register_math.py`** — register-aware math normalization
(005 finding 14: the register writes `4√2` where the answer writes
`4\sqrt{2}`). Used only by the contradiction detector, to compare two of the
model's *own* strings. `values_agree` returns **`None` when undecidable** —
missing evidence is not evidence of contradiction, or the penalty becomes a tax
on symbolic answers.

**`whetstone/reward/stagec.py`** — the scalar reward. Additive, per §1b:

```
total    = r_acc + r_fmt
r_acc    = 1.0  iff  g == 1 AND strict-correct   else 0.0
r_fmt    = max(floor, r_struct − Σ penalties)
floor    = 0.10 iff well-formed (g == 1 and think ≥ 16 tokens) else 0.0
r_struct = 0.10 + [strict-gated] 0.15·exp(−max(0,T−B)/B) + [strict-gated] 0.10·band(A)
```

Pinned magnitudes: `W_FMT 0.10`, `W_LEN 0.15`, `W_BAND 0.10`,
`MIN_THINK_TOKENS 16`, contradiction `0.20`, register-leak `0.10`,
answer-repeat `0.05/rep` capped `0.10`, n-gram loop `0.10`. The last three are
v1 §4.6/§4.3 "at reduced weight" as the packet directs; contradiction and leak
are *raised* from v1's 0.05 — a 0.05 penalty sits below the length tail's own
range and would never change an ordering.

Measured ordering (`budget_B = 250`, the battery's fixtures):

| case | total | r_acc | r_fmt | penalties |
|---|---|---|---|---|
| correct + compact register + clean | **1.3500** | 1.00 | 0.3500 | 0.00 |
| correct + register leaked into answer | 1.2500 | 1.00 | 0.2500 | 0.10 |
| correct + verbose think | 1.2034 | 1.00 | 0.2034 | 0.00 |
| correct + think contradicting the answer | 1.1500 | 1.00 | 0.1500 | 0.20 |
| correct + EMPTY think | 1.0000 | 1.00 | 0.0000 | 0.00 |
| wrong + well-formed | 0.1000 | 0.00 | 0.1000 | 0.00 |
| loop / cap-hit (g=0) | **0.0000** | 0.00 | 0.0000 | — |

Invariants, asserted in code (`assert_invariants`, called at trainer import):
worst correct **1.00**, best wrong **0.10**, **margin 0.90** against I2's
required 0.30; max structural reward **0.35 < 1.0** so style can never outrank
accuracy.

**The empty-think guard works and is the most important row.** A correct
rollout with an empty think block scores 1.0000 against the compact one's
1.3500 — it loses 0.35, so RL is never rewarded for discovering
`<think>\n</think>`. The guard is needed because `parse_segments` scores empty
think as `g = 1` (correctly — it is not malformed), so the check has to live in
the reward. The length term is the other half: `exp(−max(0,T−B)/B)` is **flat at
1.0 below the budget**, so there is zero gradient toward shorter inside it, only
cost above.

**Tests: 57 green** (`tests/test_stagec_reward.py` 34, `tests/test_strict_grading.py`
23); full suite 81 passed / 1 skipped.

> **Finding 1 — the battery caught a real defect on its first run, in the
> contradiction detector.** The register writes conclusions as `⇒ 12 · 6 = 72`,
> and the detector compared the *whole* captured expression (`12 · 6 = 66`)
> against the boxed answer (`72`). Non-numeric on the left, so `values_agree`
> returned `None` (undecidable) and the penalty **silently never fired** on the
> exact shape the register actually emits. The one contradiction case that did
> fire in testing (`⇒ 6200`) had no `=` and hid the bug. Fixed by taking the
> right-hand side of the last `=`. This is the packet's "craft it like an
> instrument, then unit-test it like one" earning its keep on day one: a
> detector that is *present, configured, logged, and inert* is the failure mode
> that unit tests exist to catch, and it would have been invisible in training —
> the curve would simply have read zero.

> **Finding 2 — two of v1's kept detectors could not be reused as written; they
> encode the *Gemma* register.** v1 §4.7 register-leak matches `**Bold Header**`
> and `1. ` numbered chunks; v1 §4.3 repetition operates on `\n\n`-separated
> "chunks". Neither exists in the v2 symbolic register, which is line-oriented
> (`goal:`, `⇒`, `chk:`). Both were rewritten against this register's actual
> markers: leak fires on line-initial `goal:`/`chk:`/`sub:`/`let:`/`case:` plus
> the symbols `⇒`/`✗`, never on bare substrings (`case` is an English word in
> 10.2% of honest answers — 009 finding 1, asserted as a test); loop fires on
> ≥10 identical consecutive *lines* or ≥6 identical after digit-blanking, which
> is what catches `case 1:`…`case 713:`. "KEEP the penalty" meant "keep the
> intent"; the implementations did not survive the register change.
