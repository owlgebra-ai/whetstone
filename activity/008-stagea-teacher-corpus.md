# 008 — P5: Stage A, the teacher corpus by generate-and-select

- **Packet:** [packets/P5-stage-a-teacher-corpus.md](packets/P5-stage-a-teacher-corpus.md)
- **Status:** in-progress
- **Machine(s):** mac (code), turing (32B generation), spark (scorer_v1 scoring)
- **Code commit(s):** `8be9b5a` →
- **Started / finished:** 2026-08-04 →

## Goal

Produce the textbook: for 4,000 problems, the frozen Qwen3-32B-NVFP4 teacher —
privileged with the gold answer, the register card and (where one exists) the
student's own verified verbose trace — writes K=8 candidate compact solutions.
Each is verified on CPU, scored for student-followability under the inoculated
`scorer_v1`, structurally annotated, and 1–3 diverse survivors per problem are
selected. The teacher is a ghostwriter: never trained, never shipped. Gate **F2**
at the end.

---

## New code

| path | role |
|---|---|
| `scripts/select_stagea_subset.py` | Part 0 — the 4,000-problem subset, level floors + trace preference |
| `scripts/teacher_generate.py` | Part 1 — privileged two-phase rollout generation against the 32B |
| `scripts/score_drafts.py` | Part 2 — G_spike/G_budget under `scorer_v1` + structural annotation |
| `scripts/select_teacher_corpus.py` | Part 3 — lexicographic winner + ≤2 diverse runners-up |
| `scripts/stagea_audit_loop.py` | Part 4 — rolling GLM audit driver with the pause rule |

---

## Packet corrections made during execution

Two stale numbers, both leftovers from a pre-decision draft of the packet, were
corrected in place (README rule 2) and are recorded here:

1. §6 said "120k drafts over ~3.8 days"; §12 said "15,000/15,000 problems".
   The binding budget is the §1 user decision of **2026-08-04**: 4,000 problems
   × K=8 = **32,000 drafts**, ~24 h. §1 and DELIVERABLES already said 4,000.
2. Consequently the rolling audit's "~3,000 judgments total" becomes **~800**
   (4,000 problems ÷ 50 per round × 10 per round). The *rate* the packet
   specifies is unchanged.

---

## Runs

### Run 1 — Part 0, subset build (turing, CPU) — 2026-08-04 08:15

```bash
python scripts/select_stagea_subset.py     # all defaults
```

Inputs `/data/whetstone/data/pool/train_30k.jsonl` +
`/data/whetstone/corpora/seed/seed_verified.jsonl` (6,939 verified records →
3,784 distinct uids with a trace).
Outputs `/data/whetstone/corpora/stagea/subset_stagea.jsonl` and
`subset_stagea_uids.json` (**the resume contract**).

| level | source | n | gold+trace | frac |
|---|---|---|---|---|
| 1 | gsm8k | 2,000 | 853 | 42.6% |
| 2 | deepmath | 38 | 5 | 13.2% |
| 3 | deepmath | 100 | 90 | 90.0% |
| 4 | deepmath | 110 | 99 | 90.0% |
| 5 | deepmath | 353 | 326 | 92.4% |
| 6 | deepmath | 546 | 488 | 89.4% |
| 7 | deepmath | 276 | 238 | 86.2% |
| 8 | deepmath | 314 | 257 | 81.8% |
| 9 | deepmath | 250 | 104 | 41.6% |
| 10 | deepmath | 13 | 0 | 0.0% |

**Conditioning: 2,460 `gold+trace` / 1,540 `gold` = 61.5%**, against the
packet's "expect ~70%+". The shortfall is fully accounted for: 1,326 problems
have no verified trace at all, and **214 have one over the 12,288-token
conditioning cap** and fall back to gold-only. Traced verbose think lengths are
median 4,526 / p95 14,274 / max 28,191 tokens, so the cap bites a real tail.
The cap was left where the packet put it — those 214 are the slowest prefills in
the pool and would throttle concurrency for the whole run.

Level 9's 41.6% and level 10's 0% are the K=2 seed harvest's own difficulty
curve showing through: the hard band is exactly where the student failed to
produce a verifiable trace.

### Run 2 — smoke, native rollout (turing) — 2026-08-04 08:26

```bash
python scripts/teacher_generate.py --max_problems 25 --k 2 --concurrency 8 \
  --output /data/whetstone/corpora/stagea_raw/smoke.jsonl
```

50 drafts, 3.5 min. Result: **the register did not land at all** — see finding 1.
38/50 kept, but with native verbose prose inside the think block.

### Run 3 — smoke, with register prefill (turing) — 2026-08-04 08:35

Same command after adding the `<think>\ngoal:` prefill.
Register lands (2.98 markers/100 char); **48% of drafts fail on the think→answer
boundary instead** — see finding 2.

### Run 4 — smoke, two-phase generation (turing) — 2026-08-04 08:41

```bash
python scripts/teacher_generate.py --max_problems 25 --k 2 --concurrency 16 \
  --chunk 64 --output /data/whetstone/corpora/stagea_raw/smoke2.jsonl
```

50 drafts, 2.1 min, **23.6 drafts/min**. **48/50 kept (96%)**, 48/48 verified.

| metric | value | reference |
|---|---|---|
| markers / 100 think chars | median **2.26**, mean 2.52 | 32B raw baseline **2.10** (activity 006) |
| opens with `goal:` | **48/48** | card exemplars, Round-0 corpus 925/960 |
| carries `chk`/`✓` | 43/48 (90%) | — |
| carries `case`/`✗` | 3/48 (6%) | 32B per-sample branch rate 13.9% |
| think tokens | median **224**, p95 717 | B_target 600; native median 6,099 |
| answer tokens | median 279 | native median 679 |
| boxed leaked into think | **0** | — |
| rejects | 1 `cap_think`, 1 `cap_answer` | — |
| trailer trimmed by `clean_oneshot` | 24% | 32B unprompted trailer rate 7.8% (005) |

### Run 5 — Part 2 scoring smoke (spark, scorer_v1 on :8100) — 2026-08-04 08:52

```bash
python scripts/score_drafts.py --drafts .../smoke2.jsonl \
  --output /data/whetstone/runs/stagea/smoke_scores.jsonl --concurrency 8
```

48/48 scored, **~1,200 drafts/min** — 50× the generation rate, so the packet's
queue-depth risk does not exist at this scale and generation never has to
throttle.

| quantity | median | min | max |
|---|---|---|---|
| think `d_t` mean | 0.602 | 0.188 | 1.614 |
| think `d_t` p95 | 3.938 | 0.931 | 9.613 |
| `g_spike_b5` | 5.5e-06 | 1.6e-07 | 8.4e-04 |
| `g_spike_b10` | 3.2e-06 | 9.2e-08 | 4.8e-04 |
| `g_budget` | 1.000 | 0.280 | 1.000 |
| scored seq tokens | 550 | 138 | 1,942 |

Both d_t figures reproduce activity 007's heldout-register numbers under
`scorer_v1` (mean 0.498, p95 3.500) — the HTTP path and the construction agree
with what F1 measured, which is the check that the meter is the same meter.
Max sequence 1,942 tokens against the scorer's 8,192 limit, so `too_long` skips
are not a live concern.

Structural annotation after the finding-3 fix: 41/48 with a source,
**`verify_kept` 36/41 = 87.8%**, `branch_kept` 3/37 = 8.1%.

### Run 6 — Part 5 calibration slice (turing) — 2026-08-04 09:00 → 10:22

```bash
nohup python -u scripts/teacher_generate.py --max_problems 500 --k 8 \
  --concurrency 16 --chunk 512 \
  --output /data/whetstone/corpora/stagea_raw/drafts.jsonl \
  > /data/whetstone/runs/stagea/gen_calib.log 2>&1 &
```

Scorer worker following on spark:

```bash
nohup ~/workspace/whetstone-scorer/.venv/bin/python -u scripts/score_drafts.py \
  --follow --idle_exit_s 0 --concurrency 8 \
  > /data/whetstone/runs/stagea/score_worker.log 2>&1 &
```

Server: `vllm serve nvidia/Qwen3-32B-NVFP4 --quantization modelopt_fp4
--kv-cache-dtype fp8_e4m3 --max-model-len 32768 --gpu-memory-utilization 0.93
--port 8000`. GPU KV cache **64,224 tokens**; at concurrency 16 usage peaks at
**93%** with requests starting to queue, so 16 is the ceiling on this card and
not a tuning preference.

**Result: 4,000 drafts in 82.9 min (48.3/min), 3,841 kept = 96.0%.**

| outcome | n | rate |
|---|---|---|
| kept | 3,841 | 96.0% |
| `reject:cap_think` | 98 | 2.5% |
| `reject:verify_fail` | 60 | 1.5% |
| `reject:cap_answer` | **1** | **0.03%** |
| *flag* `boxed_in_think` (trimmed) | 966 | 24.1% |
| *flag* `final_answer_trailer` (trimmed) | 256 | 6.4% |
| *flag* `fence_trailer` (trimmed) | 5 | 0.1% |

`cap_answer` at 1 draft in 4,000 confirms finding 4's budget call. Zero request
failures; 4,000 distinct `(uid, candidate_idx)` pairs on disk with no
duplicates, so the resume contract holds exactly.

### Run 7 — full run (turing) — 2026-08-04 10:22 →

Chained behind Run 6 by `/data/whetstone/runs/stagea/run_full.sh`, which waits
for the calibration process to exit and then `exec`s:

```bash
python -u scripts/teacher_generate.py --k 8 --concurrency 16 --chunk 512 \
  --output /data/whetstone/corpora/stagea_raw/drafts.jsonl
```

Resume is a set-difference on `(uid, candidate_idx)`, so the 500 calibration
problems are skipped and the remaining 3,500 × 8 = 28,000 drafts run at ~48/min
≈ **9.7 h**. Alongside it, `/data/whetstone/runs/stagea/select_audit_cycle.sh`
re-runs selection and one 10-judgment audit round every 15 min.

---

## Findings

### 1. An instruction addressed to a thinking model's scratchpad does not reach it

The packet's §5 specifies `enable_thinking=True` and a scaffold telling the
teacher to "solve in the compact register inside a think block". Measured over
50 drafts, the 32B complied **in the answer channel and thought natively
anyway**:

| | measured | reference |
|---|---|---|
| markers / 100 think chars | **0.11** | same model's own raw baseline **2.10** |
| traces opening with `goal` | **1 / 38** | 925/960 in the Round-0 corpus |

The think blocks were ordinary verbose prose — "Okay, let's see. The problem
says…" — followed by a correctly formatted LaTeX solution. The register
instruction landed on the visible answer, not on the scratchpad, because a
thinking model's scratchpad is where it does what it always does.

**Fix: prefill `<think>\ngoal:` onto the generation prompt.** The register is
then not something the model has to decide to adopt; it is where the sampler
already is. `goal` is the register's canonical opener (every card exemplar, 925
of 960 Round-0 traces). Marker density went 0.11 → 2.98 per 100 chars.

This is a generation-side correction to the packet, not a reinterpretation of it:
the packet's own step-3 instruction ("if per-level R_acc dips below ~90%, stop
and inspect prompts before burning days") is the same reflex, and a 25-problem
smoke run is the cheap version of it.

### 2. Prefilled into the register, the model imitates the exemplars *to their end*

The card's exemplars end at `⇒ 30`. Prefilled, the teacher writes a clean compact
trace and then does the same thing — it stops, or signs off with
`$$\boxed{8}$$`, rather than closing the thinking block and writing a solution.
**48% of prefilled drafts failed on that transition alone:**

| failure | rate | what the register content looked like |
|---|---|---|
| `missing_think_close` | 20% | fine — it just never closed the block |
| `boxed_in_think` | 28% | fine — plus a boxed trailer after the `⇒` line |

Two failure modes, one cause, and the register content was good in every case.
Rejecting them would have thrown away half the corpus over punctuation.

**Fix: two-phase generation.** Phase 1 generates the think segment with
`</think>` as a stop string; phase 2 generates the solution from the *cleaned*
think body with the boundary already written. The shape stops being something
the model has to remember, and the rejection rate went 48% → **4%**.

Three details that make this safe rather than merely convenient:

* phase 1's output goes through `clean_oneshot` — the trailer cleaner activity
  005 built for exactly this artifact — and its flags are recorded per draft
  (24% `boxed_in_think`, 2% `fence_trailer`), because a rising trailer rate is
  card feedback, not a cleaning detail;
* phase 2 is conditioned on the **cleaned** think body, not the raw phase-1
  text. It costs some prefix-cache reuse and buys the property that matters:
  the stored answer is the answer to the trace the corpus actually contains;
* assembly, gating and verification all run on
  `whetstone.round0.build_completion_text` — the same construction
  `score_drafts.py` rebuilds under the student tokenizer — so a draft cannot
  pass the generation gate and fail the scoring gate.

Anything that reaches `triage` with a boxed result still inside the register is
now a genuine mid-trace violation and is rejected, not edited.

### 3. Method note — two silent data bugs, both caught by running the smoke end to end

Continuing the tally (005 findings 3/5, 007 finding 10). Neither raised an
error; both would have produced a plausible corpus and a plausible F2 table.

**(a) The structural gate had nothing to compare against.** Every draft in the
Run-5 smoke came back `no_source` — `verify_kept` and `branch_kept` were `None`
for all 48. `annotate()` read `draft["verbose_think"]`, but the raw corpus
deliberately does not carry a per-draft copy of the verbose trace (8 copies of a
~4.5k-token trace per problem), so the field was simply absent. Selection would
have run and reported "selection beats raw on `verify_kept`" as a comparison of
two vacuous `None`s — with the one criterion activity 007 finding 7
*specifically added* turned off. Fixed by joining the source in from the subset
file, with a startup refusal if that yields nothing.

Also recorded `source_seen_by_teacher`, so the 214 traces that exist but
exceeded the conditioning cap stay distinguishable from the ones the teacher
actually read.

**(b) The selected corpus had no `_uid`.** `select_teacher_corpus.py` dropped
every key starting with `_` to strip its own scratch fields, which also dropped
`_uid` — the join key for Stage B, the rolling audit and every later analysis.
A record missing its id is still valid JSON. Fixed by enumerating the scratch
keys, plus a hard refusal on emit; `poolutil.write_jsonl` whitelists `_uid` for
exactly this reason and the convention should have been followed.

The standing lesson from 007 holds and gains a corollary: checking a number
against a measurement whose answer is already known catches false *findings*;
running the whole pipeline on 50 records catches false *plumbing*. Both bugs
were invisible in unit terms and obvious the moment a real record came out the
far end.

### 4. The two generation caps are different kinds of thing

The first 1,024 calibration drafts (answer budget 1,024) put the rejections
squarely in the hard band:

| level | 1 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|
| `cap_answer` | 0.0% | 2.5% | **15.4%** | 6.2% | 8.3% | 2.1% |
| `cap_think` | 0.3% | 0.0% | 0.7% | 0.0% | 4.2% | **16.7%** |

Treating both as "the budget was too small" would have been wrong:

* **`cap_answer` is a pure artifact.** `G_budget` is think-only (design §3.2)
  and card §1.5 makes the answer a normal LaTeX solution, so nothing in Stage A
  wants the answer short. Truncating it biased the corpus against exactly the
  levels the project's claims live in ("the success criterion is specifically
  improving the low-pass-rate rows", design §6). **Raised to 2,048** — after
  which it fires **zero times** at every level, and answer length comes out p99
  1,125 / max 1,492, so the new budget has real headroom.
* **`cap_think` is signal and was left alone.** A think segment past 2,048
  tokens has already failed the register's own premise — B_target is 600 and
  the observed median is 164. Level 9's 14.6% is "the teacher could not write
  this one compactly", which is information, and those problems are Stage-C
  rescue's designated clientele. Raising the cap would launder a failure into a
  long trace.

Effect of the fix, first 1,024 drafts each way: overall kept **94.8% → 97.3%**,
level 6 kept **78.7% → 94.9%**. The 1,024 old-budget drafts were **moved to
`/data/whetstone/corpora/stagea_raw/_capA_1024/`, not deleted**, and the slice
was regenerated so the corpus has one uniform budget throughout.

### 5. Per-level R_acc clears the packet's floor, and the floor needs an honest denominator

`scripts/stagea_draft_stats.py`, first 1,024 drafts under the final budget:

| level | drafts | gate % | R_acc / gated | R_acc / all |
|---|---|---|---|---|
| 1 | 608 | 99.3% | 99.0% | 98.4% |
| 2 | 16 | 100% | 100% | 100% |
| 3 | 48 | 100% | 100% | 100% |
| 4 | 40 | 100% | 97.5% | 97.5% |
| 5 | 40 | 100% | 100% | 100% |
| 6 | 136 | 100% | 94.9% | 94.9% |
| 7 | 64 | 100% | 100% | 100% |
| 8 | 24 | 95.8% | 100% | 95.8% |
| 9 | 48 | **85.4%** | 95.1% | 81.2% |

**Every level clears the packet's ~90% R_acc floor** (min 94.9%, at level 6), so
the privileged prompt is doing its job and the run continues. Level 9's 85.4%
gate rate is the `cap_think` story above, not an accuracy story — which is
precisely why the two denominators are reported separately.

The trap worth recording: my first scratch analysis computed the verify rate
over drafts with `reject_reason is None` and reported **100.00%**. That is a
tautology — `verify_fail` *is* one of the rejection reasons — and it looked like
a result. `stagea_draft_stats.py` exists partly to make that mistake
unavailable.

Register density over kept drafts: **2.59 markers/100 think chars** (mean 2.76)
against the 32B's own single-draft baseline of 2.10. Think median 164, p95 599
— i.e. the p95 lands essentially *on* `B_target = 600` without any length
pressure having been applied, because selection has not even run yet.

### 6. Stage-B ZPD sizing — activity 006's open item 2, answered (with a caveat)

Activity 006 asked for the masked fraction to be measured on the 32B corpus
before Stage B is sized. `score_drafts.py` now stores a per-draft histogram of
student-side think-token surprisal, so it can be answered for any γ without
re-scoring. Over 244,982 think tokens from 996 drafts:

| surprisal (nats) | share | cumulative |
|---|---|---|
| < 0.5 | **72.5%** | 72.5% |
| 0.5–1 | 6.7% | 79.2% |
| 1–2 | 7.7% | 86.9% |
| 2–4 | 6.7% | 93.6% |
| 4–8 | 4.9% | 98.5% |
| 8–16 | 1.5% | 100.0% |
| ≥ 16 | 0.0% | 100.0% |

γ = 1.0 masks 20.8% of think tokens; γ = 4.0 masks 6.4%. The 32B's compact
register is **largely inside the student's reachable zone** — the fear behind
006's open item (a teacher so far ahead that Stage B masks rather than learns)
is not realised at this gap.

**Caveat, and it is load-bearing: this is measured under `scorer_v1`, not under
the checkpoint Stage B actually starts from.** `scorer_v1` has had 91% of the
register style tax removed (`goal` 39.98 → 1.21 nats, activity 007), so it finds
register text far more probable than the original checkpoint does. Stage B's
band-pass runs under the *student*, which begins from the **original**
checkpoint and has had no inoculation. **These numbers are therefore a lower
bound on the masked fraction, and P6 must re-measure under the original
checkpoint before pinning γ** — one cheap pass over the selected corpus. Quoting
this table as Stage B's γ calibration would be a genuine error.

### 7. Structural retention does **not** behave like independent draws across K

Measured over the 110 problems whose all-8 drafts survived (preview selection on
the first 190 problems). If each draft independently kept a branch at the
observed per-draft rate of 0.160, the count per problem would be
Binomial(8, 0.160). It is nothing like it:

| k of 8 keeping a branch | observed | Binomial(8, 0.16) |
|---|---|---|
| 0 | **64.5%** | 24.7% |
| 1 | 10.0% | 37.8% |
| 2 | 2.7% | 25.2% |
| 3 | 6.4% | 9.6% |
| 4 | 6.4% | 2.3% |
| 5–7 | 4.5% | 0.4% |
| 8 | **5.5%** | 0.0% |
| **P(≥1)** | **35.5%** | **75.3%** |

`verify_kept` is clustered the same way (k=8 at 50.0% observed vs 19.0%
independent). The distribution is U-shaped: a problem's compact form either
carries the property in most drafts or in none.

**This retires activity 006's open-item-2 arithmetic.** That note reasoned
"13.9% per-sample → P(≥1 branch-keeping candidate) ≈ 70% at K=8, ~91% at K=16",
and used it to argue selection would amplify branch retention. Measured, P(≥1)
is **35.5%, not 75.3%** — and because the correlation is at the *problem* level,
**raising K buys almost nothing.** K=16 would not approach 91%.

The mechanism is most likely the source detector rather than the teacher:
`structural_gate.py`'s own docstring records that `VERBOSE_BRANCH` fires on
99.5% of verbose traces ("this model says 'wait'/'actually' constantly, even on
trivial arithmetic"), and it fires on 122 of ~128 problems here. So a large part
of the k=0 mass is problems with no real case split to keep — the compact trace
cannot preserve a branch the source never had. That is why the check remains a
corpus-level diagnostic and why `--require_branch` is off by default.

**Consequence for the design:** if more branch coverage is wanted, the lever is
the register card or the conditioning, **not** K and not the selection rule. And
`branch_kept`'s denominator should be tightened before it is ever used as a
gate.

### 8. Selection captures 100% of the structure that exists

The packet's F2b targets are per **problem** ("verify ≥ 85%, branch ≥ 30% on
source-branching problems"), which is 3× different from the per-trace rate once
a problem has 3 keeps. Preview over 190 problems:

| property | eligible problems | ≥1 candidate has it | selection captured it | capture efficiency |
|---|---|---|---|---|
| verify | 119 | 98.3% | **98.3%** | **100%** |
| branch | 122 | 37.7% | **37.7%** | **100%** |

Both targets are met, and the diagnostic that matters is the last column:
**capture efficiency is 100%** — of every problem where some candidate kept the
property, selection kept it. The packet's stated worry ("if selection isn't
finding them, the rule is broken") is answered directly: the rule finds all of
them, and `available` is a fact about the teacher.

Per-trace, `verify_kept` goes raw 80.6% → selected 95.2% and `branch_kept` raw
17.1% → 27.5%. R_acc: all drafts 95.1% → **selected 100%**, so selection is not
trading correctness for style (F2a's question) — it improves it, because
`verify_ok` is a survivor condition.

Getting here required one fix. A single rank-ordered runner-up pass reached only
26.7% per-trace branch retention, because the winner rule ranks `verify_kept`
above `branch_kept`, verification is ~5× commoner, and so the winner is almost
always a verify-keeper that dropped the branch — after which the branch-keeping
draft (typically ranked 5th–8th, since branch-preserving traces are longer and
score worse on G_spike × G_budget) never gets looked at before the slots fill.
Runners-up now run in two passes: property-adding candidates first, plain
diversity second, which is what the packet's "priority to runners-up adding a
structural property the winner lacks" actually asks for.

### 9. The teacher abandons the register on exactly the problems it exists for

The prefill guarantees the *opener* — 100% of drafts at every level start with
`goal:` — so "opens with `goal:`" measures the prefill, not the model. The body
is where adherence lives, and it decays hard with difficulty:

| level | 1 | 3 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| carries `⇒` | 99.6% | 98.4% | 70.0% | 84.6% | 82.6% | 88.6% | 87.3% |
| carries `chk`/`✓` | 97.2% | 85.9% | 62.5% | 64.3% | 63.5% | 64.6% | **54.4%** |
| **no register at all** | 0.3% | 1.6% | 23.8% | 13.2% | 13.9% | 7.6% | 12.7% |
| **reads as prose** | 6.5% | 10.9% | 15.0% | 21.4% | 23.5% | 44.3% | **54.4%** |
| markers/100 char | 3.28 | 2.23 | 1.48 | 1.23 | 1.22 | 1.29 | **0.71** |

("reads as prose" = mean words per non-empty think line > 14; the 32B's own raw
baseline is 2.10 markers/100 char.)

So the teacher holds the register where it fits easily — arithmetic — and
reverts to English sentences with a `goal:` header on the hard problems the
register exists to compress. Hand-read examples confirm it: one level-8 trace is
five full sentences with no `⇒` and no `chk`, ending "Thus, we conclude that
P² = P".

**Best-of-K fixes this for free, and this is the sharp contrast with finding 7.**
Register adherence is *well mixed* across a problem's 8 samples where branch
retention is clustered: **98.9% of problems have at least one in-register
candidate — 100% at level 9, where the per-draft rate is only 42.9%.** So the
criterion went to the **top** of the lexicographic order (a prose trace with a
`goal:` header is not a compact-register example at all, and installing the
register is the whole deliverable), and it cost nothing:

| | raw per draft | selected per trace | problems captured | capture efficiency |
|---|---|---|---|---|
| in-register | 79.9% | **94.2%** | **99.2%** | 99.2% |
| verify_kept | 79.8% | 92.7% | 98.8% | **100%** |
| branch_kept | 16.3% | 25.4% | 39.6% | **100%** |

Verify and branch capture efficiency both stayed at **100%** after inserting a
criterion above them, because the runner-up property passes still reach them.

### 10. G_spike does not catch the failure the audit exists to catch

A deterministic probe for the packet's named reward-hacking signature — a trace
that *asserts* the gold rather than deriving it — flags drafts whose compact
think contains at most one numeral that is not the gold itself. It rises sharply
with level: 0.0% at level 1, 12.2% at 7, 22.8% at 8, 24.1% at 9.

The probe is a proxy and biased (abstract problems legitimately contain few
numerals — the same bias `structural_gate.py` records for `invented_frac`), so
three flagged level-8/9 traces were hand-read. All three are real:

* a ring-theory trace asserting "*a* ∈ *P* ⇒ *a* ∈ (*a*²)", a non-sequitur, as
  its load-bearing step;
* a measurability trace whose crux — that an *uncountable* supremum of
  measurable functions is measurable — is asserted in one line as "sup of
  continuous functions is measurable ✓";
* a maximisation trace that computes one test case, checks the formula at
  p = 1, 2, ∞, and then writes "confirm no configuration gives higher D" with
  no argument.

**Now the part that matters.** Design §3.2 claims G_spike "subsumes … most of
the audit's faithfulness function: a hallucinated or unsupported compact step
is, by construction, a spike." Measured against this probe, it is not:

| | n | think d_t mean | think d_t p95 | **G_spike (β=10)** | markers/100ch | think tok |
|---|---|---|---|---|---|---|
| assert-flagged | 66 | 0.825 | 4.984 | **2.76e-05** | 1.23 | 230 |
| has derivation | 1,407 | 0.613 | 4.162 | **1.48e-05** | 2.50 | 183 |
| flagged, level ≥ 7 | 51 | 0.884 | 4.925 | **2.95e-05** | 1.14 | 235 |
| derived, level ≥ 7 | 222 | 0.596 | 3.906 | **2.44e-05** | 1.07 | 382 |

Higher G_spike is *better* reward. **Asserted traces score ~1.9× better on
G_spike than derived ones**, and the gap survives restricting to level ≥ 7. The
mechanism is not mysterious: an asserted crux is *fluent* — "confirm no
configuration gives higher D" is a highly predictable sentence — while a genuine
derivation step carries real content and is therefore more surprising. G_spike
rewards fluent hand-waving over substantive derivation.

Caveats, stated plainly: the probe is a numeral-based proxy, n = 66 flagged,
and only 3 were hand-verified. The direction is nonetheless consistent across
two cuts and is the classic reward-hacking sign.

**Implications.** (i) The lexicographic winner rule is doing more work than the
packet assumed — G_spike is the *last* criterion, and on this evidence it should
stay there. (ii) The GLM audit is not a formality: it is the only instrument in
the pipeline pointed at this failure, which is why its absence is the one
genuine gap in this run. (iii) This is a concrete instance of activity 007's
"the instrument is shallow but honest — do not design a stage that needs a
sharp threshold", and it should be carried into Stage C, where G_spike-like
terms enter an actual optimiser and Goodhart pressure is no longer one-shot.

### 11. Temperature is a real but unproven lever on branch diversity

Finding 7 showed branch retention clustered per problem, which kills K as a
lever. That leaves an obvious follow-up the finding could not answer: **is the
clustering a property of the problem, or an artefact of not exploring enough at
T = 0.8?** Run as a paired arm — same 48 source-branching problems (levels 1–9,
round-robin), same K = 8, same seeds, **only the temperature differs**.

| | T = 0.8 | T = 1.0 |
|---|---|---|
| keep rate | 95.3% | 96.4% |
| `verify_ok` of kept | 100% | 100% |
| in-register per draft | 74.6% | 72.4% |
| `branch_kept` **per draft** | 20.8% | **17.8%** |
| **branch ≥1 of 8, per problem** | **52.1%** | **62.5%** |
| `verify_kept` per draft | 76.7% | 79.2% |
| verify ≥1 of 8, per problem | 91.7% | 91.7% |
| think tokens, median | 322 | 346 |
| `boxed_in_think` trailer (trimmed) | 24.1% | **41.1%** |
| `final_answer_trailer` (trimmed) | 6.4% | 13.3% |
| `cap_think` | 2.5% | 3.6% |

**The shape is exactly what decorrelation looks like:** the per-draft rate goes
*down* (20.8% → 17.8%) while per-problem availability goes *up* (52.1% →
62.5%). Same marginal rate, wider spread across the 8 samples, so more problems
land at least one branch-keeping draft. That is evidence the clustering is not
purely a property of the problem.

**But it does not survive a significance test.** McNemar exact on the 47 paired
problems: 8 gained, 3 lost, 36 unchanged, net +5, **p = 0.227**. At this n a
+10 pp effect is indistinguishable from noise. Settling it needs roughly
150–250 paired problems; only the T=1.0 arm would have to be generated, since
the T=0.8 baseline is the main corpus.

Costs, for whoever runs that: T=1.0 nearly doubles the answer-trailer rate
(24.1% → 41.1%) and raises `cap_think` by a point. Both are handled — trailers
are trimmed and counted, cap-hits are rejected — so the quality cost is
absorbed, but it is not free.

**Not applied to this run.** Switching temperature mid-corpus would split it
across two sampling regimes and make every later per-level comparison ambiguous
— the same reasoning that led to regenerating after the budget fix rather than
patching forward. This is evidence for the *next* generation, not this one.

### 12. Compression is flat in absolute terms, so the ratio rises with difficulty

Over the 5,240 drafts that have a verbose source:

| level | verbose median | compact median | ratio | fold |
|---|---|---|---|---|
| 1 | 1,060 | 122 | 0.117 | 8.6× |
| 3 | 2,051 | 250 | 0.094 | 10.6× |
| 4 | 4,242 | 340 | 0.079 | 12.7× |
| 5 | 4,754 | 408 | 0.086 | 11.6× |
| 6 | 6,172 | 434 | 0.072 | 13.9× |
| 7 | 6,189 | 382 | 0.061 | 16.5× |
| 8 | 6,365 | 337 | 0.051 | 19.5× |
| 9 | 7,943 | 300 | 0.038 | **26.4×** |
| **all** | | | **0.081** | **12.4×** |

The 32B is **far less aggressive than the alternatives** — activity 004's 1.7B
one-shot compressed to 0.043 and activity 005's GLM to 0.030 — which is the
right direction for faithfulness.

**The interesting column is the compact median: it is nearly flat.** Verbose
input grows 7.5× from level 1 to level 9 (1,060 → 7,943 tokens) while compact
output stays in a 300–434 band. The teacher is not scaling its output to the
reasoning it was given; it is writing roughly the same amount regardless.

Read against findings 9 and 10 that is not a compression story but a
**truncation** story: level 9 is simultaneously the most compressed (26×), the
most prose-reverting (54%), the most assert-flagged (24%), and the
worst-audited (57% faithful / 29% wrong). The consistent reading is that on
hard problems the teacher stops *re-encoding* the reasoning and starts
*summarising* it — same output budget, far more input, so content is dropped
rather than compressed. Whoever revises the register card should treat "the
compact trace must scale with the derivation" as the target, and
`structural_gate.py`'s `lines_per_step` (currently disabled, threshold 0) as the
measurement that would enforce it.

### 13. **The teacher confabulates when it has the answer but not the reasoning**

The pause rule fired at trailing-200 faithful 66.5% / wrong 16.5%. Investigating
it produced the most important result of this packet.

**First, most of the aggregate move was composition, not quality.** The
trailing-200 level mix went from 58% level-1 to 17% level-1 as the corpus grew.
Level 1 audits at ~95% faithful and level 9 at ~26%, so the aggregate tracks the
sampler. Per level, nothing had statistically changed (L8–9 wrong 28.6% n=14 →
46.2% n=39 is z = 1.15, p = 0.25).

**But the per-level table showed the real structure:**

| level | n | faithful | lossy | wrong |
|---|---|---|---|---|
| 1 | 99 | 94.9% | 3.0% | 2.0% |
| 3 | 39 | 84.6% | 12.8% | 2.6% |
| 4 | 28 | 82.1% | 17.9% | 0.0% |
| 5 | 33 | 69.7% | 24.2% | 6.1% |
| 6 | 40 | 55.0% | 25.0% | 20.0% |
| 7 | 30 | 56.7% | 23.3% | 20.0% |
| 8 | 29 | 65.5% | 17.2% | 17.2% |
| **9** | 27 | **25.9%** | 11.1% | **63.0%** |

And splitting the hard band by **conditioning mode** explains it:

| hard band (L≥6) | n | faithful | lossy | **wrong** |
|---|---|---|---|---|
| `gold+trace` — teacher saw the reasoning | 104 | 58.7% | 22.1% | **19.2%** |
| `gold` — teacher saw only the answer | 22 | **18.2%** | 9.1% | **72.7%** |

**A 3.8× difference in wrong-rate, driven by conditioning rather than by
difficulty.** Given a verbose trace the teacher compresses it; given only the
answer on a hard problem it *invents a derivation that reaches the known
answer*. That is exactly the failure design §3.2 assumed G_spike would catch and
finding 10 showed it does not, and it is why the packet mandates this audit.

The interaction matters: level 1 is **57.4%** gold-only and still audits at
94.9% faithful. Confabulation is not a property of gold-only conditioning — it
is what happens when the teacher is asked to derive something it *cannot*
derive. Easy problems it can.

**Exposure across the corpus:**

| | problems | gold-only | of which `trace_too_long` (fixable) | no trace at all |
|---|---|---|---|---|
| all 4,000 | 4,000 | 1,540 (38.5%) | 214 | 1,326 |
| hard band (L≥6) | 1,399 | 312 (22.3%) | **174** | 138 |
| level 9 alone | 250 | 146 (58.4%) | — | — |

**174 of the 312 risky hard-band problems are fixable and were fixed by
accident of a cap.** They *have* a verified verbose trace; the teacher was not
shown it only because the trace exceeded the 12,288-token conditioning cap that
Run 1 flagged as "biting a real tail". The server's 32,768-token context leaves
room for a ~26k-token trace beside the ~4k card, so most of those 214 could be
re-run under `gold+trace`.

**Confirmed at n = 241.** The hard-band rule tripped again two hours later, and
the same split on a 2× larger sample is no longer tentative:

| hard band (L≥6), n = 241 | n | faithful | lossy | **wrong** |
|---|---|---|---|---|
| `gold+trace` | 203 | **57.1%** | 25.1% | **17.7%** |
| `gold` | 38 | **10.5%** | 15.8% | **73.7%** |

Two-proportion z on faithful = **5.27, p < 1e-6**. This is the strongest single
effect measured in this packet.

It also explains the alarm. `gold`-only traces are ~16% of hard-band judgments
and drag the pooled hard-band faithful rate to ~46%; the `gold+trace`
population alone sits at **57.1%**, above even the original 45% floor. **The
hard-band alarm is detecting the conditioning contamination, not a teacher
that has degraded.** Split first-half vs second-half of the hard judgments at
matched level and conditioning mix: 53.3% → 46.3% faithful, z = 1.08, p = 0.28
— no established drift over time, and what movement there is goes
faithful→*lossy*, not faithful→wrong (wrong actually fell 27.5% → 25.6%).

**Threshold re-pinned, and it is worth being explicit that this is not a moved
goalpost.** The Part-5 checkpoint set the hard-band floor at 45% from n = 42
measuring 57.1%, and wrote the intent down: *"set to catch a material
degradation (≈15 pp), not to certify the current value."* At n = 241 the
hard-band rate is ~50%, so the small-sample estimate was ~7 pp high. Applying
the **same stated 15 pp rule** to the better estimate gives **35%**. A 45% floor
against a true 50% with ~83-judgment windows (SE ≈ 5.5 pp) flaps by chance
about a third of the time, which is exactly what was observed — two trips, one
at 46.3% and one at 44.6%. The **`wrong` ceiling is deliberately unchanged** at
35%: it is the metric that decides corpus usability, and it reads 25.3%.

**Two caveats that keep this honest.** (i) The audit can only judge problems
that *have* a source, so the ~1,326 no-trace problems — generated under exactly
the conditioning that produces 73.7% wrong — are **never audited at all**. The
measured rates are therefore optimistic for the corpus as a whole. (ii) Per
level, the hard band is not one population: level 9 is 32.1% faithful / 47.2%
wrong (n = 53) against level 6's 57.7% / 15.5% (n = 71), so pooling them is
itself a compositional hazard one layer down.

### 14. Throughput and the queue-depth risk

At the real chunk size (512, against the smoke's 64) throughput is **~54
drafts/min**, not the 23.6 the smoke suggested — the K=8 group shares one prompt
prefix and vLLM's prefix cache serves it, holding a 74–80% hit rate on the
shared card (3,987 rendered tokens) plus the per-problem prefix. That puts the
full 32,000 drafts at **≈ 10 h**, comfortably inside the packet's ~24 h budget.

Scoring runs at ~1,200/min on spark, **~22× faster than generation**, so the
packet's "monitor queue depth / throttle generation" contingency is dead letter
at this scale: the scorer idle-waits almost all the time.

Operational note for later packets: the `chunk` size is also the resume
granularity for phase-1 work. A kill mid-chunk discards up to 512 drafts' worth
of completed phase-1 generation (~10 min), because those results live in memory
until phase 2 finishes. Acceptable here; worth knowing before choosing a larger
chunk.

### 15. Method note — a && chain behind `&` does not do what it looks like

The scorer ran 996 drafts on **8-commit-old code** after a one-liner of the shape
`ssh box 'cd repo && git pull && rm out && nohup worker &'`. The `&` backgrounds
the *entire* chain, so the verification I thought I was doing (pull, then
launch) raced, and the launch won. It surfaced only because the new
surprisal field came back absent from every record — the same shape as
finding 3's two bugs, and caught the same way: by looking at a real record
instead of at an exit code.

Activity 001's gotcha 6 already says "first step of any packet touching a remote
box: push from the Mac, `git pull` + `git status` on the box, and reconcile."
The addition is that **the pull must be its own command with its own verified
output** — `git rev-parse --short HEAD` compared against the Mac's — not a link
in a chain whose exit code nobody reads. Scores were discarded and re-run; the
raw corpus was untouched, which is exactly what the raw/selected split is for.

---

---

## Part 5 — calibration checkpoint (go/no-go) — 2026-08-04 10:05

Measured on the first 2,560 drafts of the calibration slice (320 problems; the
subset is pre-shuffled, so this is a representative slice) plus 129 GLM
judgments. **Verdict: GO.** Thresholds pinned below, then the run continues.

### What was measured

| check | packet expectation | measured | verdict |
|---|---|---|---|
| per-level R_acc (over gate-passing drafts) | ≥ ~90% | **94.9–100%**, min at level 6 | **PASS** |
| selected `verify_kept`, per problem | ≥ 85% | **98.8%** | **PASS** |
| selected `branch_kept`, per source-branching problem | ≥ 30% | **39.6%** | **PASS** |
| selection capture efficiency | (implied 100%) | **100%** on both | **PASS** |
| think length | well under 600 | median **182**, IQR [127, 341] | **PASS** |
| R_acc, selected vs all drafts | within 3 pts | 96.0% → **100%** | **PASS** |
| throughput | ~24 h for 32k drafts | **~47/min ⇒ ~11 h** | **PASS** |
| queue depth | monitor, throttle if needed | scorer 22× faster; never backed up | **N/A** |

### GLM audit calibration — 129 judgments, of which **91 valid**

The 38 invalid ones are the finding below. Over the 91 source-bearing
judgments:

| band | n | faithful | lossy | wrong |
|---|---|---|---|---|
| level 1 (gsm8k) | 32 | 96.9% | 3.1% | 0.0% |
| levels 2–5 | 17 | 94.1% | 5.9% | 0.0% |
| levels 6–7 | 28 | 57.1% | 28.6% | 14.3% |
| levels 8–9 | 14 | 57.1% | 14.3% | **28.6%** |
| **all valid** | **91** | **78.0%** | **13.2%** | **8.8%** |

Reference: the 1.7B self-compressions judged **40% faithful / 21% wrong**
(activity 005 f13). The 32B with gold in hand is far better in aggregate — and
**at levels 8–9 it is worse than that baseline on the wrong-rate** (28.6% vs
21%). Flags over valid judgments: `dropped_branch` **24.2%** (the top flag,
corroborating findings 7–8), `invented_content` 12.1%, `fused_steps` 9.9%,
`dropped_values` 2.2%, `off_topic` 0%.

**Judgments on records with no verbose source are meaningless and were
inflating the headline.** With an empty ORIGINAL the rubric has nothing to
compare against, and the judge returned **94.7% faithful** on those 38 records
against 78.0% on the 91 real ones. They pulled the aggregate from 78.0% to
82.9%. `faithfulness_audit.py` now refuses them. Stage-A corpora legitimately
contain such records (the ~34% of problems with no verified trace), so this is
not a corpus defect — it is a sampling rule the packet did not state.

### Pinned thresholds

**The pause rule is per band, not aggregate** — this is CLAUDE.md's
segment-reporting invariant applied to the audit. An aggregate floor of 55%
would sit unbroken while levels 8–9 ran at 30%, because level-1 GSM8K is half
the corpus and scores 97%.

Over the trailing 200 **valid** judgments:

| | floor / ceiling | calibration value |
|---|---|---|
| aggregate faithful | ≥ 70% | 78.0% |
| aggregate wrong | ≤ 15% | 8.8% |
| hard band (level ≥ 6) faithful | ≥ 45% | 57.1% |
| hard band (level ≥ 6) wrong | ≤ 35% | 19.0% |

The hard-band bounds are deliberately loose relative to the measurement: n = 42
in that band, so the 1σ sampling error is ≈ 7–8 pp and a tighter bound would
trip on noise. They are set to catch a *material* degradation (≈ 15 pp), not to
certify the current value. **F2c uses the same four numbers.**

F2a/F2b targets stand as the packet wrote them: R_acc within 3 pts of the
all-drafts mean; `verify_kept` ≥ 85% and `branch_kept` ≥ 30% per problem, plus
**capture efficiency = 100%** as the diagnostic that separates a rule failure
from a teacher limitation.

### Decisions taken at the checkpoint

1. **Answer budget 1,024 → 2,048; think budget unchanged** (finding 4). Slice
   regenerated for uniformity; old drafts preserved.
2. **Register adherence added as the top selection criterion** (finding 9).
   Free — verify/branch capture efficiency stayed at 100%.
3. **Runner-up selection split into two passes** (finding 8).
4. **The register card is NOT changed — user decision, 2026-08-04.** The
   hard-level register decay (finding 9) is real, but selection recovers 99.2%
   of problems, the card is a ratified artifact, and a new exemplar deserves its
   own bake-off rather than a mid-run edit. Carried to a future card revision.
5. **Audit judgments restricted to source-bearing records.**

---

## Conclusion

*(pending — F2 verdict goes here)*
