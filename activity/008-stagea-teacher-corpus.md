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

### 4. Throughput and the queue-depth risk

23.6 drafts/min at concurrency 16 → **32,000 drafts ≈ 22.6 h**, in line with the
packet's ~24 h. Scoring runs at ~1,200/min on spark, **50× faster than
generation**, so the packet's "monitor queue depth / throttle generation"
contingency is dead letter at this scale: the scorer is idle-waiting almost all
the time. Prefix-cache hit rate holds at 74–80%, which is the shared card prefix
(3,987 rendered tokens) plus the K=8 per-problem prefix doing their job.

---

## Conclusion

*(pending — F2 verdict goes here)*
