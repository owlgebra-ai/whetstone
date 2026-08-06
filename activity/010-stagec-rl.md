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
- outputs: `/data/whetstone/runs/entropy_stagec_init/{audit.json,per_token_entropy.npz,*.png}`

**F3c — the owed measurement. PASSES, on fresh rollouts.**

| think entropy | baseline (original ckpt) | **round-1 student (fresh)** | ratio |
|---|---|---|---|
| mean | 0.31759 | **0.61977** | 1.95× |
| **median** | 0.027817 | **0.27991** | **10.1×** |
| p80 | 0.69234 | **1.25825** | 1.82× |
| collapse mass (<0.1) | 56.8% | **39.6%** | — |
| fork mass (>1.5) | 2.8% (003, native) | **14.6%** | 5.2× |

F3c's criterion is "median entropy ≥ audit baseline"; the student is at **10.1×**
it. The gate is met with enormous margin, and the fresh-rollout number confirms
009's teacher-forced control slice almost exactly (that slice read mean 0.6455 /
median 0.2603 / p80 1.2282 against this run's 0.6198 / 0.2799 / 1.2582) — two
different measurement paths agreeing is worth more than either alone.

Segment split (answers are *not* where the entropy is): answer mean 0.3758,
median 0.0300, collapse mass 57.9%. Think carries the entropy; the answer
segment is still near-deterministic, which is the correct shape.

> **Finding 3 — the second entropy mode is at ≈0.725 nats on the student too,
> confirming activity 003 on a checkpoint two stages downstream.** Histogram of
> the non-collapsed think mass (60 bins over 0.1–3.1 nats) shows a monotone
> decay from the collapse spike to a **dip at 0.525** (8,475), a genuine **local
> maximum at 0.725** (11,185), a **dip at 0.925** (8,072), then a long tail.
> 003 measured this on the *original* checkpoint and warned that "TEA's τ_c = 1.0
> sits above the real second mode"; Stage B did not move it. The packet's
> τ_c ∈ {0.7, 1.0} pilot sweep is therefore well-targeted — **0.7 sits on the
> mode, 1.0 sits in the dip past it.** Mass above the design's 1.5-nat "fork"
> threshold is 14.6%, so a component hard-coding 1.5 would act on the tail and
> miss the mode entirely.

> **Finding 4 — the packet's 8,192 rollout cap was justified on GSM8K numbers
> and truncates 5.6% of well-formed generations on the actual Phase-1 pool.**
> P7 §3 pins 8,192 citing 009 Run 12's "truncates 0/377 well-formed
> generations". That run was **GSM8K only** (think p99 3,197). The Phase-1 pool
> is **50% DeepMath** (2,000 GSM8K L1 + 2,000 DeepMath L2–L10 — measured below),
> and on the DeepMath distribution this student runs far longer: think p50 2,218,
> p95 7,681; total completion p50 3,156, p95 8,445, max 10,248.
>
> | cap | well-formed generations truncated |
> |---|---|
> | 8,192 (packet) | **11/197 = 5.58%** |
> | 10,240 | 1/197 = 0.51% |
> | **12,288 (adopted)** | **0/197 = 0.00%** |
> | 16,384 | 0/197 = 0.00% |
>
> **Cap raised to 12,288** for both bucketing and RL rollouts (they must match,
> same argument as the temperature resolution). This is the same artifact class
> activity 008 caught with its answer budget — "1,024 cap-outs hit 15.4% of
> level 6 and 0% of level 1, an artifact biasing against the hard tier". Under
> RL the bias is worse than a mis-measured eval: a truncated-but-legitimate
> generation is scored `g=0 → R_acc=0`, so the reward would teach the model that
> hard problems are unsolvable when they are merely long. 12,288 still starves
> the loops (2.7× below the 32k budget at which they were characterised), and
> costs little since only ~5% of generations reach past 8,192 at all.

Also from this run, under the init checkpoint on DeepMath val at T=0.9:
`g=0` **1.50%** (3/200, all `missing_think_close`), think p25 **745** — the
latter is the `ThinkBudget` spread floor's order of magnitude on the hard half.

### Run 4 — 2026-08-05, Part 0.4 prep: Phase-1 pool composition

`/data/whetstone/corpora/stagea/subset_stagea_uids.json` (4,000) against the
golden corpus's uid list (2,414). Split confirms the packet exactly: **2,414
seen / 1,586 unseen**, 0 uids missing from `train_30k.jsonl`.

**Source is 50/50 and perfectly confounded with level:** all 2,000 level-1
problems are GSM8K; every level 2–10 problem is DeepMath.

| level | total | seen | unseen | seen % |
|---|---|---|---|---|
| 1 (GSM8K) | 2,000 | 844 | 1,156 | 42.2% |
| 2 | 38 | 5 | 33 | 13.2% |
| 3 | 100 | 88 | 12 | 88.0% |
| 4 | 110 | 105 | 5 | 95.5% |
| 5 | 353 | 327 | 26 | 92.6% |
| 6 | 546 | 494 | 52 | 90.5% |
| 7 | 276 | 234 | 42 | 84.8% |
| 8 | 314 | 235 | 79 | 74.8% |
| 9 | 250 | 82 | 168 | 32.8% |
| 10 | 13 | 0 | 13 | 0.0% |

> **Finding 5 — the seen/unseen split is severely confounded with level, so a
> pooled memorization comparison would be uninterpretable.** Seen share runs
> 42.2% at L1, **85–95% across L3–L7**, then back down to 32.8% at L9 and 0% at
> L10. A pooled "seen vs unseen Pass@1" therefore compares a mostly-easy unseen
> set against a mostly-mid-difficulty seen set, and would show a *seen-side
> deficit* from difficulty alone — the opposite sign of the memorization effect
> it is meant to detect. **The memorization watch must be read within level, and
> L2/L10 (38 and 13 problems, one side near-empty) cannot support the comparison
> at all.** The packet's Part 0.4 already says "split by seen/unseen *and* by
> level"; this is why that qualifier is load-bearing rather than decorative.

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
