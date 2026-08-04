# P5 — Stage A: teacher corpus by generate-and-select (frozen Qwen3-32B, best-of-8)

STATUS: in-progress (activity 008)
MACHINES: turing (32B generation server + client + CPU verification); spark (scorer_v1 G_spike scoring); GLM API (rolling audit)
DEPENDS ON: P4 done (activity 007 — F1 passed, `scorer_v1` frozen + serving); design decision 006 (frozen 32B teacher, generate-and-select)
BLOCKS: P6 (Stage B needs this corpus); the Stage-C rescue loop reuses this packet's generation machinery
DELIVERABLES: 4,000-problem subset file (2,000 GSM8K + 2,000 DeepMath); **raw corpus of all 32,000 drafts with full scores** (selection is a cheap re-runnable pass, never baked in); selected corpus (1–3 traces/problem); rolling GLM audit log; **F2 verdict** (redefined for a frozen teacher); Stage-B handoff file.

---

## 1. Objective

Produce the **textbook**: for 4,000 problems, the frozen 32B teacher (gold answer + register card in context, verbose trace where one exists) writes 8 candidate compact solutions; each is verified on CPU, scored for followability under the inoculated 1.7B scorer on spark, structurally annotated, and 1–3 diverse survivors per problem are selected. The teacher is a ghostwriter — never trained, never shipped. User decisions baked in (2026-08-04): K=8 over 2,000 GSM8K + 2,000 DeepMath (SCA-comparable data budget, ~24 h of generation); lexicographic selection with 2–3 diverse keeps; inline CPU verification; GLM audit every 50 problems. If Stage B later proves data-hungry, the raw-drafts design makes scaling a repeat overnight run, not a redesign.

## 2. Read first

1. Design §3 (Stage-A intent — reward semantics unchanged), §12.2 (scoring), §6 (audit)
2. [activity/006](../006-teacher-student-decoupling.md) (why generate-and-select) and [activity/007](../007-round0-inoculation.md) findings 3, 5, 7 (what the instrument can and cannot do)
3. ROADMAP facts blocks 001–007

## 3. Inputs (all exist)

| asset | value |
|---|---|
| Teacher | `nvidia/Qwen3-32B-NVFP4` (cached). Serve on **turing:8000**: `vllm serve nvidia/Qwen3-32B-NVFP4 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --max-model-len 32768 --gpu-memory-utilization 0.93 --port 8000` (activity 006 serving note; ~22 traces/min at concurrency 8; prefix caching is on by default — the shared card prefix and the per-problem prompt across K=8 both hit it) |
| Scorer | `scorer_v1` serving on **spark:8100** as `whetstone-scorer` (`VLLM_USE_FLASHINFER_SAMPLER=0`); d_t contract verified over HTTP (007 Run 7) |
| Pool | `/data/whetstone/data/pool/train_30k.jsonl` (29,998); seed subset uids `/data/whetstone/corpora/seed/subset_uids.json` (4,500); verified verbose traces `/data/whetstone/corpora/seed/seed_verified.jsonl` |
| Card | `configs/register_card.md` via `render_card()` (drop by **section number**, not title — activity 005 finding 3); `assert_no_boundary_tokens()` on the rendered prompt |
| Sequence/scoring conventions | `whetstone/round0.py` — **mandatory shared path** for every (q, τ) scoring call (CLAUDE.md: a record scored under one construction and thresholded under another is the silently-inverted meter) |
| Pinned scorer constants | τ_spike 2.25, τ_leap 3.175 (007); G_spike λ=1, **compute at both β=5 and β=10, select on β=10**, store both |
| Structural annotations | `scripts/structural_gate.py` (verify_kept / branch_kept / value_coverage) |
| GLM judge | `scripts/faithfulness_audit.py` (005) — evaluation only, output never enters any record |

## 4. Part 0 — Subset builder (`scripts/select_stagea_subset.py`, new; CPU, minutes)

**4,000 problems: 2,000 GSM8K + 2,000 DeepMath** (user decision 2026-08-04; SCA-comparable data budget — their curriculum used ~2k–6k problems, e.g. 1,400 GSM8K + 600 DeepMath in stage 2). Seed 0, via `poolutil`:

1. **GSM8K 2,000** (all level 1): prefer the ~900 seed-subset GSM8K problems first (verbose traces → `gold+trace` conditioning), fill to 2,000 at random.
2. **DeepMath 2,000, level-shaped with hard-band floors** (a purely proportional draw would leave level 9 at ~105 — too thin for the hard-tier claims):

   | level | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
   |---|---|---|---|---|---|---|---|---|---|
   | target | all 38 | 100 | ~110 | ~352 | ~546 | ~276 | ~314 | **250** | all 13 |

   (floors: all of 2 and 10, ≥100 at level 3, ≥250 at level 9; levels 4–8 proportional with the remainder.)
3. **Within every stratum, prefer problems that have a verified verbose trace** (the seed subset holds 3,600 DeepMath candidates, ~84% with a verified trace) — this maximizes `gold+trace` conditioning; report the achieved trace-conditioned fraction (expect ~70%+).

Compute: 4,000 × 8 = **32,000 drafts ≈ 24 h** at measured throughput. Write `subset_stagea_uids.json` **first** — it is the resume contract. Record the per-level/source/conditioning table in the journal. Note for reporting: GSM8K is 50% of *problems* but far less of *think tokens* (its traces are short) — publish both shares.

## 5. Part 1 — Generation (turing)

New client `scripts/teacher_generate.py` (reuse `whetstone/runio.py` — repair_tail, checkpoint sidecar, failed-request audit trail; and the 005 server-mode pattern: one `/v1/completions` request per draft, bounded in-flight window, `return_token_ids: true`).

**Prompt construction (privileged — this is the teacher, not the student):**
- System/scaffold: rendered card + instructions: solve in the compact register inside a think block, then a normal LaTeX answer ending in `\boxed{…}`. **The gold answer is provided; the trace must genuinely derive it** — assert the scaffold encodes no boundary tokens.
- Two conditioning modes, recorded per draft as `conditioned_on`:
  - `gold+trace`: problem + gold + the student's verified verbose trace — **only when the trace exists AND ≤ 12,288 tokens** (longer traces blow the KV budget and throttle concurrency; fall back to gold-only).
  - `gold`: problem + gold (the ~10.5k non-seed problems).
- `enable_thinking=True` on the template call (this is a rollout — the standing rule applies; the compression-mode `enable_thinking=False` exception was for the retired rewrite flow, not for generation).
- Sampling: T=0.8, top-p 0.95, `max_tokens 4096`, per-draft seeds `sha1(uid:k:seed)[:8]` (never one seed per group — 005 infra note 3).

**Inline, as each draft returns (CPU, no GPU):**
1. `parse_segments` on the returned token ids → g; **reject g=0** (cap-hit, malformed).
2. **Reject `\boxed{}` inside think** (card §1.5; 32B's unprompted trailer rate was 7.8% in 005 — with 8 drafts, reject beats clean).
3. `verify_response(completion, gold)` → R_acc. With gold in context this should pass at a very high rate — **if per-level R_acc dips below ~90%, stop and inspect prompts** before burning days.
4. Queue the draft for spark scoring (Part 2) and append to the **raw corpus** `/data/whetstone/corpora/stagea_raw/drafts.jsonl` — every draft, pass or fail, with its rejection reason. Raw is append-only truth; selection never mutates it.

## 6. Part 2 — Scoring & annotation (spark + CPU, concurrent with generation)

A separate worker `scripts/score_drafts.py` drains the queue against **spark:8100**:

- One teacher-forced prefill per draft via the `whetstone/round0.py` path (`prompt_logprobs=2`, student-tokenizer construction §4-of-P4 — the *unprivileged* student prompt + the draft as completion; the teacher's privileged prompt is NEVER what gets scored). Store per-draft: think mean d_t, p95 d_t, `g_spike_b5`, `g_spike_b10`, `g_budget = exp(−max(0, T_think − 600)/600)`, think/answer token counts.
- CPU: `structural_gate.py` annotations (`verify_kept`, `branch_kept`, `value_coverage`) computed against the verbose trace when one exists, else against source-independent checks only (flag `no_source`).
- Throughput budget: 32k drafts over ~24 h = a sustained ~22/min; spark prefill handles this, but **monitor queue depth** — if spark falls behind, generation must throttle rather than let the queue grow unbounded (scoring lag delays the audit's early warning). *(Corrected during execution — the "120k drafts / 3.8 days" figure was a leftover from the pre-decision 15,000-problem draft; the binding budget is the §1 user decision, 4,000 problems × K=8. Noted in activity 008.)*

## 7. Part 3 — Selection (`scripts/select_teacher_corpus.py`, cheap, re-runnable)

Per problem, over its ≤8 surviving drafts (g=1, no boxed-in-think, R_acc pass):

1. **Winner** — lexicographic: (a) prefer `verify_kept` when the source verified (the measured G_spike bias against verification, 007 finding 7, is patched here by construction); (b) prefer `branch_kept` when the source branched; (c) max `g_spike_b10 × g_budget`.
2. **Runners-up (≤2)** — must be *significantly different* from every already-kept draft: think-token **8-gram Jaccard < 0.6**, OR a differing structural signature (branch/verify presence flips, or think length differs ≥30%). Priority to runners-up adding a structural property the winner lacks. Ties by `g_spike_b10 × g_budget`.
3. Problems with zero survivors: recorded in `unserved_uids.json` with reasons — these are Stage-C rescue's future clientele, not silent losses.
4. Output `/data/whetstone/corpora/stagea_selected/selected.jsonl`: full records + `selection_rank`, `selection_reason`, `n_kept`. **Stage B must weight per problem (1/n_kept or sample-one-per-epoch), never per trace** — write this into the handoff sidecar.

Selection runs incrementally during generation (for dashboards + audit sampling) and **once more, from scratch, over the final raw corpus** — the final pass is the binding one.

## 8. Part 4 — Rolling GLM audit (every 50 problems, user-mandated)

`scripts/faithfulness_audit.py` in a loop: after every 50 problems complete selection, sample **10 kept drafts** (stratified: recent, random level) → GLM judges faithful/lossy/wrong + flags. ~3,000 judgments total (~an hour of API time spread over days). Judge output goes to `/data/whetstone/runs/stagea/audit_rolling.jsonl` and the dashboard — **never into any corpus record**.

**Pause rule (provisional until Part 5 pins it):** over the trailing 200 judgments, `faithful < 55%` or `wrong > 15%` → **pause generation**, investigate, journal before resuming. Context for calibration: the 1.7B self-compressions judged 40% faithful / 21% wrong (005 f13); the 32B *with gold in hand* must be far better — if it isn't, something is broken, likely the prompt.

## 9. Part 5 — 500-problem calibration checkpoint (go/no-go before the remaining ~21 h)

After the first 500 problems (~3 h; the work list is shuffled, so this is a representative slice): halt and measure. **Pin, then journal:**

- Per-level R_acc of drafts (expect ≳90% everywhere — gold is in context).
- Selection stats: kept-per-problem distribution; selected-corpus `verify_kept` and `branch_kept` vs the raw single-draft rates (32B raw: 70.6% / 13.9%). **Selection must beat raw meaningfully — targets: verify ≥ 85%, branch ≥ 30% on source-branching problems** (per-sample 13.9% at K=8 → ~70% of problems *should* have a branch-keeping candidate; if selection isn't finding them, the rule is broken).
- GLM audit rates over the first ~200 judgments → pin the pause thresholds.
- Throughput projection vs the 3.8-day estimate; queue depths.
- **F2 thresholds get pinned here** from the measured distributions, then the run continues.

## 10. Part 6 — F2 gate (redefined for a frozen teacher; the old convergence criteria are void)

On the completed selected corpus:

- **F2a — accuracy:** selected-corpus R_acc within 3 pts of the all-drafts mean (selection must not trade correctness for style).
- **F2b — structure:** selected `verify_kept` / `branch_kept` meet the Part-5-pinned targets; report against raw-draft baselines.
- **F2c — audit:** final stratified GLM pass (200 fresh judgments incl. hard levels) meets the pinned faithful/wrong thresholds; plus the design's reward-hacking eye: fluent filler, restated-problem no-ops, degenerate terseness.
- **F2d — dashboards:** symbol density, think-length distribution (median + IQR, expect well under 600), per-level coverage, selection-reason histogram, `unserved` rate per level.
- Segment-level reporting throughout: think and answer lengths always separate.

F2 FAIL → the raw corpus is intact; fixes are selection-rule or prompt changes and a cheap re-select — **not** a regeneration, unless the failure is generation-side (then diagnose at the level of conditioning mode / card rendering before any rerun).

## 11. Gotchas

- **Score under the student's unprivileged prompt, always** (`whetstone/round0.py`). Scoring under the teacher's privileged prompt inflates followability and is exactly the inverted-meter class.
- The instrument is **shallow but honest** (AUC 0.81 on single corruptions, 007 f3) — selection uses it as a rank signal among verified drafts, never as a hard gate on its own.
- `chk` carries a 7.92-nat residual tax under scorer_v1 in *teacher-style* contexts (007 f7) — this is WHY selection prefers `verify_kept` structurally instead of trusting G_spike there. Do not "fix" by lowering λ.
- Orphaned `VLLM::EngineCore` after kills — check `nvidia-smi --query-compute-apps`, kill by PID; never pipe a vLLM script into `head`.
- Two servers, two boxes: 32B on turing:8000, scorer on spark:8100 (spark:8000 is llama-swap — hands off).
- Per-draft seeds; resume is a set-difference on (uid, k); `--shuffle` the work list (005 f2 — level-clustered order makes interrupted runs unrepresentative).
- Long-trace conditioning: >12,288-token traces fall back to gold-only; record the fallback.
- Disk: raw corpus ~120k × ~3 KB ≈ 400 MB — trivial; keep everything.

## 12. Definition of done

- [ ] Subset built + table journaled; uids file is the resume contract.
- [ ] Calibration checkpoint (Part 5) journaled with pinned thresholds **before** the long run continued.
- [ ] 4,000/4,000 problems generated (or unserved documented); raw corpus complete with scores. *(Corrected during execution — "15,000" was a leftover from the pre-decision draft; §1 and DELIVERABLES both say 4,000.)*
- [ ] Final from-scratch selection pass; selected corpus + handoff sidecar (per-problem weighting note, conditioning-mode stats).
- [ ] Rolling audit log complete; any pause events journaled with cause.
- [ ] **F2 verdict in bold** with all four sub-gates, in `activity/008-stagea-teacher-corpus.md`.
- [ ] ROADMAP facts block for activity 008; packet status flipped; P6 unblocked with the corpus paths.
