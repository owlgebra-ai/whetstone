# P4 — Round 0: scorer inoculation + the F1 band-existence gate

STATUS: done (activity 007)
MACHINES: turing (training + eval forwards); spark (frozen-π_0 scoring passes, reward server)
DEPENDS ON: P0–P3 all done (activities 001–005). Design decision 006 (32B teacher) shapes Part 5.
BLOCKS: P5 (Stage A) and everything after — **F1 is the go/no-go for all teacher compute** (design §8 Risk 1, §11).
DELIVERABLES: R token-set JSON, SED kernel + unit tests, inoculation trainer, the four monitoring curves + per-marker-class hum, the three meter unit tests, an **F1 verdict**, frozen `scorer_v1`, and the **binding G_spike × branch-retention check** that gates P5.

---

## 1. Objective — what this run is and why it is the whole ballgame

The scorer (Qwen3-1.7B) is a verbose-CoT native: compact-register tokens read as improbable for *style* reasons. Untreated, that style tax drives every downstream reward toward verbosity. Round 0 calibrates the meter — the **smallest dose that creates recognition without infection** (design §2). The meter has three states:

| State | Behavior | Downstream cascade |
|---|---|---|
| Undertrained | register tokens spike | reward drives teacher verbose; compression dies |
| **Band (target)** | register = hum (elevated mean, no spikes); verbose ≈ baseline; genuine leaps still spike | reward separates style from leaps |
| Overtrained | register becomes argmax; residual prose reads spiky | caveman degeneration |

**F1 asks: does the band exist?** If register-hum and leap-spike are inseparable, the whole design pivots (fallback: prefix/LoRA scorer arm). This is deliberately run before any teacher GPU-hours. The product of this packet is a **trustworthy measuring instrument**, not a capable model.

## 2. Read first (in order)

1. Design doc §2 (Round-0 spec), §12.3 (type aggregation), §12.4 (SED kernel — every bullet is a bug you'd otherwise write), §7 (monitoring curves + overshoot signature), §8 Risk 1.
2. [activity/005-seed-corpus.md](../005-seed-corpus.md) — "For P4" section + finding 7 (marker-class scarcity).
3. [activity/006-teacher-student-decoupling.md](../006-teacher-student-decoupling.md) — why Part 5 exists.
4. ROADMAP "Facts pinned by activity 00N" blocks — all operational rules assumed below.

## 3. Inputs — every path exists, nothing needs rebuilding

| asset | value / path |
|---|---|
| Round-0 corpus (student's own, **unfiltered by design**) | `/data/whetstone/corpora/seed_register_qwen/` — `train.jsonl` (960), `heldout_register.jsonl` (120), `probe_pool.jsonl` (120), `verbose_control.jsonl` (200). **NEVER re-split** — a re-split moves probe traces into training and silently invalidates the corrupted-trace probe. **probe_pool is untouchable until Part 6.** |
| Record fields | `_uid, prompt, verbose_think, compact_think, answer, level, structural_*` — `compact_think` is the think **body** (no tags) |
| **H_pivot** | **0.6707 nats** (activity 005; 1,200 student traces, 243k tokens) |
| **Δ_max** | **0.7** (restoration mode, activity 003 audit verdict) |
| Entropy baseline (S3 + comparisons) | `/data/whetstone/runs/entropy_audit/per_token_entropy.npz` — raw per-token arrays; compare distributions, not just medians |
| Register card + structural whitelist | `configs/register_card.md` §2 (FILLED, ratified) |
| Boundary token ids | `<think>`=151667, `</think>`=151668, `<|im_end|>`=151645 (tokenizer @ `70d244cc`) |
| Parser | `whetstone/segments.py` (token-level; `g` gate; 24 green tests) |
| Teacher corpus for Part 5 | `/data/whetstone/corpora/seed_register_qwen32b/` (1,200, structurally annotated) |
| Stack | vllm 0.26.0 / torch 2.11.0+cu130 / transformers 5.14.1 / CPython 3.12.12 (pyproject at HEAD) |
| Scorer server (spark) | port **8100** (8000 is llama-swap — hands off), `VLLM_USE_FLASHINFER_SAMPLER=0`, launch command in activity 001 Run 6 |

**Calibration anchors from prior measurements** (use, don't re-derive):
- π_0 verbose-control baseline p95 d_t ≈ **0.750**; clean-register p95 d_t ≈ **2.375** (activity 004, bake-off corpus — your step-0 measurement on `heldout_register` is the binding reference; expect the same ballpark).
- **τ_spike's 4-nat design placeholder is DEAD ON ARRIVAL**: the pre-inoculation register p95 (~2.4) is already below it, so S1 at τ_spike=4 fires at step 0 and the run no-ops while appearing to pass. **Start τ_spike = 1.2** (strictly between the verbose baseline and the step-0 register level); pin the final value from data.

## 4. Sequence construction (do this identically everywhere)

All scoring and training operate on sequences shaped like a **native rollout**, matching what Stage-A scoring will see (design §12.2):

```
prompt_text = apply_chat_template([{user: prompt}], add_generation_prompt=True, enable_thinking=True)
completion_text = "<think>\n" + compact_think + "\n</think>\n\n" + answer
```

Tokenize prompt and completion separately, concatenate ids, and derive masks with `parse_segments(ids, prompt_len=len(prompt_ids))`. **Assert g == 1 for every record** before it enters any batch (005 shipped them clean; verify anyway). Loss/metrics apply to completion positions only; think/answer routing comes from the parser masks — never from string offsets. Remember `apply_chat_template(tokenize=True)` returns a `BatchEncoding` in transformers 5.x — use `list(enc["input_ids"])`.

Scoring conventions, used everywhere below:
- surprisal `S_t = −log π(τ_t)`; gap `d_t = log π(top1) − log π(τ_t) ≥ 0`.
- **Assert d_t == 0 wherever actual == top-1** (doubles as the P0 contract check).
- Position alignment: logits at position t−1 predict token t — off-by-one here silently shifts think/answer attribution. Sanity anchor: entropy/surprisal of the `</think>` token itself must be near zero (activities 003/005 measured ~1e-4–0.02).
- Never materialize full-vocab logits in fp32 for a sequence: chunk positions (~1024), `topk(512)` in bf16, fp32 softmax on the 512, accumulate, discard.

## 5. Part 1 — R token-set builder (`scripts/build_register_tokenset.py`, new)

Over **train.jsonl only**, think-segment positions only, surprisal under frozen π_0:

1. Score all 960 sequences (spark server is ideal: one `/v1/completions` call per record, `max_tokens=1, prompt_logprobs=2`, actual-token logprob per position; or HF on turing — either way record which).
2. Per token **type** (vocab id): collect surprisal across its occurrences. Eligibility: ≥ 10 occurrences.
3. `R_stats = { types: mean surprisal > 75th pct across eligible types AND across-occurrence std < median std }` — consistently-expensive, consistently-priced types are *style vocabulary* → train. High-surprisal type-*inconsistent* positions are content → excluded.
4. `R = R_stats ∪ whitelist` — card §2 strings, each tokenized **bare AND space-prefixed**, all piece-ids included. The whitelist enters by fiat even when below the occurrence floor (that's its purpose — see the marker-class warning in §8).
5. Dump `/data/whetstone/runs/round0/R_tokenset.json`: `{token_id: {surface, mean, std, count, source: stats|whitelist}}`. Print R size + top-50 surfaces by mean surprisal and **eyeball them into the journal** — if ordinary English words dominate, the p75 threshold is wrong for this corpus.

## 6. Part 2 — SED kernel (`whetstone/sed.py` + `tests/test_sed.py`, new — shared verbatim with Stage B later)

`SEDRegularizer(model, ema_decay=0.99, sync_every=5, tau_range=(1.1, 1.5), topk=512, H_pivot=0.6707, delta_max=0.7, gamma_e=1.0)` with `.maybe_sync(optimizer_step_idx)` and `.loss(student_logits, input_ids, think_mask)`. (γ_e has no design-table anchor — 1.0 is the declared start; record it as a pinned placeholder.)

Implementation rules — each is a named bug from design §12.4:

1. **EMA update, never replacement:** `φ ← 0.99·φ + 0.01·θ` every **5 optimizer steps** (with grad-accum 8 that is every 40 micro-batches). Initialize φ ← θ. A hard copy every 5 steps silently destroys the stabilization.
2. **Gate + temperature search on φ's logits:** one forward of φ per batch yields H_t (top-512) → target `H_t + Δ_t`, `Δ_t = 0.7·σ(1.0·(H_t − 0.6707))`.
3. **Bisection:** 20 iterations, τ̂ ∈ [1.1, 1.5], on φ's top-512 logits per token; clamp silently at range ends.
4. **K2 at the data token:** `½(log π_θ(y_t) − log π_φ,τ̂(y_t))²`, masked to think tokens, mean over unmasked.
5. φ stays on-GPU in bf16 (3.4 GB at 1.7B). **Round 0's EMA copy belongs to Round 0** — never shared with or carried into Stage B.

Unit tests (must be green before the trainer runs): EMA decay follows μ^k analytically on a toy model; bisection hits a random target within 1e-2 nats and clamps correctly; exactly one sync per 40 micro-batches under mocked grad-accum 8; K2 gradient flows to θ only (φ under `no_grad`).

## 7. Part 3 — Inoculation trainer (`scripts/inoculate_scorer.py`, new)

**Loss (design §2):** `L = Σ_{t ∈ R ∩ think} CE_t + 1.0 · L_SED(think)` (+ optional KL-to-π_0 on ¬R tokens behind a flag, **default off** — reach for it only if S2 keeps tripping). Answer tokens get no loss.

**Config:** full-FT Qwen3-1.7B, bf16, sdpa, grad checkpointing ON, AdamW, LR 1e-5, warmup 20 optimizer steps + cosine, per-device batch 1, **grad-accum 8** (effective batch 8 → **120 optimizer steps for the 1-epoch cap** over 960 records — accum 32 would give only 30 steps, too coarse to find a band). ≤ 1 epoch, hard stop.

**Checkpoint every eval (every 10 optimizer steps), keep all until the F1 verdict** — the stopping rule selects a checkpoint *retroactively*; rollback is expected, not exceptional. ~12 saves × 3.4 GB ≈ 41 GB under `/data/whetstone/ckpt/round0/` — fine; delete all but the winner after F1.

**Memory budget (32 GB):** θ 3.4 + grads 3.4 + AdamW moments ~13.6 + φ 3.4 + π_0 frozen eval copy 3.4 ≈ 27 GB + activations (short sequences, bs 1, grad-ckpt) — fits, barely. All eval forwards strictly `no_grad`. If OOM: move the π_0-side metrics to the spark server (S2's top-512 KL then needs the server launched with `--max-logprobs 512`; payloads are heavy but the control set is only 200 traces) before touching 8-bit optimizers.

**Metrics every 10 optimizer steps** — JSONL + live PNGs (the §7 curves), all four on every eval:

1. **S1 / hum:** teacher-force `heldout_register` through the trainee; p95 of d_t over think tokens. **Also per marker class** — `⇒ → goal let ;` vs **`case ✗ chk ✓`** separately. The Round-0 corpus contains almost no `case`/`✗` (005 finding 7), so the branch vocabulary can stay uncalibrated while the overall test passes — and the 32B teacher's branch-keeping traces (006) lean on exactly those tokens. Report it every eval; it feeds Part 5's interpretation.
2. **S2 / drift:** mean per-token top-512 KL(π_θ ‖ π_0) on `verbose_control`. Fires at κ_max — set it after the first 3 evals so it wouldn't fire in the first third of the run; pin the value.
3. **S3 / entropy floor:** median think-token entropy on the control set vs the audit baseline npz; fires if it drops > x% (start x = 10%).
4. **Hum trajectory:** register-token (R-set) mean surprisal on heldout — should fall to a plateau, not to zero.

**Stopping:** first of S1 (p95 < τ_spike = 1.2 to start) / S2 / S3, plus the **overshoot signature** (design §7): verbose-control likelihood falling while register p95 keeps dropping past τ_spike → roll back to the S1-crossing checkpoint. Log wall-clock per eval cycle; the whole run should be minutes-to-an-hour scale — if you project days, the eval breadth is mis-sized (shrink control sets, never the metrics).

## 8. Part 4 — Meter unit tests (all three, simultaneously, at one checkpoint)

Run at the selected checkpoint AND its two neighbors — the band may be narrow.

- **(a) Register hum:** heldout p95 d_t < τ_spike, overall **and reported per marker class** (a pass with an uncalibrated branch class is a *qualified* pass — record it as such).
- **(b) Verbose intact:** mean per-token logprob delta vs π_0 on `verbose_control` within ε (start ε = 0.2 nats/token; report the measured delta regardless).
- **(c) Corrupted-trace probe — decisive; failing it invalidates the scorer regardless of (a)/(b).** From `probe_pool` (120 traces, never trained on), build corrupted twins, ~60 per corruption type:
  1. **Chunk deletion:** remove one intermediate derivation step (a numbered line), so the next line uses a result that now comes from nowhere.
  2. **Value substitution:** replace one intermediate numeric result with a plausible wrong value (±1 on a digit), leaving later steps (which used the correct value) as non-sequiturs.
  Score twin pairs with the checkpoint. **Pass:** p95 d_t over the corrupted span (corruption point → +30 tokens) > τ_leap, while the clean twin's same span stays < τ_spike. τ_leap is pinned from the measured separation, not assumed. **The paired gap-histogram plot (corrupted vs clean) IS the band-existence evidence** — put it in `activity/assets/007/`.

**F1 verdict:**
- **PASS** (some checkpoint clears a+b+c simultaneously): freeze it to `/data/whetstone/ckpt/scorer_v1/`; load on the spark reward server (`vllm serve /data/whetstone/ckpt/scorer_v1 --port 8100 --served-model-name whetstone-scorer ...` + the standard spark flags); re-verify the d_t contract over HTTP (P0's check); record every pinned value (τ_spike, τ_leap, κ_max, x, ε, γ_e) in the journal AND the design §12.6 table AND the ROADMAP facts block.
- **FAIL:** one retry is allowed only if the failure is marginal and clearly attributable (e.g. R polluted by content tokens → rebuild R with a stricter std filter). Otherwise stop and write up precisely which test failed, by how much, at every checkpoint — the prescribed pivot is the **prefix/LoRA scorer arm** (design §8 Risk 1: adapters can't reshape generation, but scoring needs only likelihood calibration), which becomes a new packet. Do not iterate endlessly; a clean documented FAIL is a deliverable.

## 9. Part 5 — Binding G_spike × branch-retention check (gates P5; needs the inoculated scorer)

Only on F1 PASS. This is activity 006's gating check, run in its **binding** form (the pre-F1 version is confounded by the style tax the inoculation just removed).

1. Score all 1,200 records of `seed_register_qwen32b` under **scorer_v1** (same sequence construction, §4).
2. Compute per trace: `G_spike = exp[−(λ/β)·log((1/T)Σ_t exp(β·d_t))]` over think tokens, λ=1, **both β=5 and β=10**; plus mean d_t and p95 d_t.
3. Correlate against `structural_branch_kept` (field is on the records): report point-biserial correlation, the G_spike distributions split by branch_kept, and the **per-marker-class residual tax** on `case`/`✗` under scorer_v1 (ties to Part 4a's qualified-pass note).
4. Verdicts: no negative correlation → P5 proceeds with the unchanged product reward. Negative correlation attributable to residual branch-marker tax → recommend the selection-term fix (006 open item 2: `branch_kept` as a selection criterion — safe under one-shot Goodhart with a frozen teacher) and/or λ/β retuning, with numbers. Either way, **P5's packet author gets a quantitative answer, not a vibe.**

## 10. Gotchas (consolidated — each has bitten an agent already)

- `source .venv/bin/activate` always; spark needs `VLLM_USE_FLASHINFER_SAMPLER=0`; scorer port **8100**.
- Orphaned `VLLM::EngineCore` holds ~28 GB after a killed run → next start dies with a buried OOM. Check `nvidia-smi --query-compute-apps=pid,used_memory`, `kill -9` **by PID** (`pkill -f` matches its own command line). **Never pipe a vLLM script into `head`** (SIGPIPE orphans the engine).
- Any prompt text you construct must contain **zero literal think-tag strings** — they tokenize as real boundary tokens (card §1.6); `assert_no_boundary_tokens()` exists, use it.
- The trainee is the **SCORER**. The student of Stage B starts from the *original* checkpoint, not from scorer_v1.
- Do not inoculate on any teacher corpus (14B/32B/GLM): Round 0 calibrates on the student's own distribution — H_pivot reads 0.9119 on GLM text vs 0.6707 on the student's (005). Using teacher text here is the silently-inverted-meter failure this packet exists to prevent.
- Naive string metrics on this data have produced **five** confident-but-inverted findings (005 method note). Anything numeral- or marker-derived: audit before believing.
- Eval forwards build no graphs (`no_grad`); the 27 GB budget has no autograd slack.

## 11. Definition of done

- [ ] R-set built; size + top-50 eyeball in journal.
- [ ] SED kernel + 4 unit tests green, committed.
- [ ] Training run(s) with all four curves + per-marker-class hum (PNGs in `activity/assets/007/`).
- [ ] Three meter tests at winner ± neighbors; corrupted/clean histogram pair plotted.
- [ ] **F1 verdict in bold** in `activity/007-round0-inoculation.md`, with every placeholder pinned (τ_spike, τ_leap, κ_max, x, ε, γ_e) and recorded in §12.6 + ROADMAP facts.
- [ ] On PASS: `scorer_v1` frozen + serving on spark:8100, d_t contract re-verified over HTTP, non-winner checkpoints deleted.
- [ ] On PASS: Part-5 check reported with correlations and the P5 recommendation.
- [ ] Packet status flipped; ROADMAP updated (P5 unblocked with its reward verdict, or the LoRA-arm packet drafted on FAIL).
