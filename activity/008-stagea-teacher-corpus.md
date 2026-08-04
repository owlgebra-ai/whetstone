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

### Run 6 — Part 5 calibration slice (turing) — 2026-08-04 08:38 → in progress

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

### 9. Throughput and the queue-depth risk

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

### 10. Method note — a `&&` chain behind `&` does not do what it looks like

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

## Conclusion

*(pending — F2 verdict goes here)*
