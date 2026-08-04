# Packet roadmap — WHETSTONE v2 feasibility tier (Qwen3-1.7B on turing + spark)

```
P0 env ──► P1 data ──► P2 preconditions ──► P3a register bake-off ──► P3 seed corpus ──► P4 Round 0 ══► F1 ✅ band exists (007)
                                            (P3 Part 1 seed harvest may run in parallel with P3a)
                                                                              │
                                            PASS ◄────────────────────────────┘──► FAIL → LoRA-scorer packet
                                             │
                                             ▼
                 P5 Stage A (32B best-of-K select) ══► F2 ──► P6 Stage B (assimilation) ══► F3
                                                                        │
                                                                        ▼
                                              P7 Stage C (segment-routed DAPO) ══► F4
                                                                        │
                                                                        ▼
                                              P8 baselines + eval hardening (SCA, DeepCompress arms)
```

Only **P0–P4 are written in full detail**. P5–P7 are deliberately outlines: the design (§11, §12.6) gates them on F1's measured values (τ_spike, τ_leap, λ/β behavior, H_pivot) — writing their fine detail now would bake in numbers F1 exists to pin. **Expand each into a full packet only when its gate opens**, folding in the activity-file learnings from the packets before it.

> ⚠️ **SUPERSEDED IN PART by activity 006 (2026-08-03).** Teacher and student
> are **decoupled**: the teacher is **Qwen3-32B-NVFP4**, the student/scorer stays
> Qwen3-1.7B. Branch preservation is a scale-dependent capability (3.1% → 5.9% →
> 13.9% at 1.7B/14B/32B) that **no prompting channel transfers** (four demo-pool
> configurations, all 1–2%), so a 1.7B teacher's rollouts would never contain it
> and `G_spike` could not select on it. A 32B teacher cannot be GRPO-trained on
> one 32 GB card, so **Stage A becomes generate-and-select**: sample K, score
> with the unchanged `R_acc · G_spike · G_budget` under the frozen 1.7B scorer,
> keep the best. Gated on the `G_spike` × branch-retention correlation check in
> activity 006 — the reward may select *against* the property the decision buys.

## P5 — Stage A: teacher best-of-K selection (**UNBLOCKED — expand into a full packet next**)

> **F1 passed on its design question (activity 007): the calibration band exists.**
> Two binding inputs from that run:
> (1) `G_spike` does **not** select against branch retention (r = −0.02, p = 0.47),
>     so activity 006's 32B-teacher decision stands and the product reward is unchanged;
> (2) `G_spike` **does** select against verification retention (r = −0.113, p < 1e-4),
>     driven by a 7.92-nat residual tax on `chk`. **Add `verify_kept` as a selection
>     term; do not lower λ.** Scorer is `spark:8100` (`whetstone-scorer`), λ=1, β∈{5,10}.
> Note the outline below still says "GRPO"; activity 006 replaced that with
> generate-and-select, and the G_budget bullet is a *selection* criterion now.


- Design §3 + §12.2. Teacher = fresh Qwen3-1.7B copy, register card + exemplars + gold (+ verbose trace) **in context**; student-style prompt untouched.
- Reward `R_acc · G_spike · G_budget` — product form is non-negotiable (design A5 tests why). G_spike scored by the **frozen scorer_v1 on spark** (per-batch prefill, `prompt_logprobs≥2`, λ modest / β ∈ {5,10}).
- G_budget: B₀ = median prompted-compressed length (from P3 stats), anneal toward 600, **freeze when within-group think-length std < 40 tokens**.
- GRPO group 8, T=0.9; TEA regularization on the teacher's own updates; trl GRPOTrainer vs custom loop is an implementation decision for the packet author (evaluate trl's external-reward + vLLM-sleep support on one GPU first — the trainer and rollout engine share the 5090; vLLM sleep/wake between phases is the expected pattern).
- Dashboards: symbol density, think-length bimodality index, mean-gap vs max-gap as separate curves.
- Claude Sonnet audit spot-check (100 stratified samples/checkpoint, ≥90% pass) — `scripts/audit_compressions.py` repurposed, prompted for reward-hacking signatures.
- **Gate F2:** symbol density plateaus; bimodality resolves terse; teacher R_acc within 3 pts of prompted baseline; spot-check ≥90%. F2 fail with F1 passed → (λ,β) grid + budget schedule, do NOT touch Stage B.
- Output: teacher checkpoint + K=4 T=0.8 verifier-filtered corpus over the full pool.

## P6 — Stage B: learnability-gated, entropy-preserving assimilation (draft outline; expand after F2)

- Design §4. Student = fresh copy of the *original* checkpoint. No register card in its prompt — the register enters weights here only.
- ZPD band-pass weights `w_t = σ(κ(log π_S(τ_t) − γ)) · (1 + 0.5·min(S_t, 4))`; **γ from the measured student-on-teacher-corpus logprob histogram**; precompute w_t offline per corpus refresh (scorer pass on spark), store in the training JSONL.
- SED term (same `whetstone/sed.py`, **new EMA copy**, H_pivot from P3, Δ_max per audit verdict).
- Two rounds: train → recompute all gates under updated student (one pass) → fresh teacher batch → train. Stale gates are a named drift failure.
- **Gate F3:** within 1 pt of starting accuracy at ≤50% median think tokens; median entropy ≥ audit baseline.

## P7 — Stage C: segment-routed DAPO (draft outline; expand after F3)

- Design §5. DAPO clip 0.2/0.28, group 8, LR 1e-6; segment masks from `whetstone/segments.py`; per-segment advantage normalization.
- Think tokens: soft length tail + TEA (τ_c 1.0, λ_TEA 0.05, c 100), **no style anchor**. Answer tokens: forward KL to the *original* checkpoint + SCA band f=32.
- Curriculum per phase: fresh K=8 buckets; 1–7/8 → DAPO with difficulty amplification (α=0.5, positive think advantages only); 0/8 → pedagogy rescue (teacher M=4 with gold, G_spike-filtered, Stage-B-loss assimilation).
- Per-checkpoint rollout investigation + stop rules: v1 §7.6–7.7 verbatim.
- **Gate F4:** 50 DAPO steps, no critical rollout flag, ≥1 checkpoint Pareto-dominating the start on the easy suite.

## P8 — baselines + eval hardening (schedulable anytime after P1, needed before any write-up)

- Reproduce **SCA** and **DeepCompress** from the same Qwen3-1.7B checkpoint (design §6 — mandatory), plus the prompted-compressor-only arm.
- HumanEval code-execution grader (sandboxed) — P1 marked it `grading: code-exec-pending`.
- Headline decomposition reporting: (trim within verbose register) × (register change) as separate factors; segment-level lengths everywhere.

## Eval plan (user-ratified 2026-08-02)

Suite roles — three tiers with different touch frequencies, so headline numbers can't be overfit by repeated peeking:

| Tier | Suites | When run |
|---|---|---|
| **Primary (headline tables)** | MATH-500, AMC23, MinervaMath, AIME24, AIME25 | Stage gates (F2–F4) and final reporting only |
| **Validation** | **GSM8K test split** (1,319 problems) | Checkpoint selection, hyperparameter decisions, phase endpoints |
| **Internal continuity** | `standard_eval_300` (frozen) | Every checkpoint, cheap mode allowed |
| **Cross-domain secondary (SCA-matched)** | GPQA-Diamond; HumanEval (once the P8 code-exec grader exists) | Final reporting alongside SCA's published numbers |

- **Protocol (design §12.7, wired into `run_eval.py` by P2):** N=8, T=0.7, top-p 0.95, max_tokens 32768, `enable_thinking=True`; report Pass@1 ± seed std with **think and answer lengths as separate columns**, answer-segment-only quality. Qwen-recommended sampling (T=0.6) reported once in an appendix.
- Small suites (AIME 30, AMC 40) are noisy — never quote them without the ± std, never subsample them.
- **TODO (next executing agent, ~30 min on spark):** `gsm8k_test.jsonl` is not yet built — add the suite to `build_eval_sets.py` (`openai/gsm8k` config `main`, split `test`, same schema, pin revision) and emit to `/data/whetstone/eval/`. Contamination pre-cleared: activity 002 Run 5 checked the train pool against GSM8K-test — 0 hits.
- Baselines (SCA / DeepCompress / prompted-compressor arms) run the identical protocol from the same checkpoint — numbers are only comparable inside the same tier and protocol.

## Facts pinned by activity 007 (P4 / **F1 gate**) — binding on all later packets

- **F1 PASSES on the design question: the calibration band exists.** The register
  style tax is removable — held-out R-token mean surprisal **13.065 → 1.154 nats
  (91%)**, `goal` from **39.98 → 1.21** — while the corrupted-trace leap detector
  is essentially unchanged (**probe AUC 0.823 → 0.810**). Design §8 Risk 1 is
  retired; **the prefix/LoRA scorer arm is not needed.**
- **F1 fails the packet's literal criterion** (all three meter tests at
  τ_spike = 1.2) at every one of 13 checkpoints. Binding failure is test (a).
  **τ_spike = 1.2 was inherited from a corpus whose step-0 register p95 was
  2.375; this corpus's is 6.375.** The verbose baseline reproduces *exactly*
  (0.750), so this is a corpus difference, not a measurement one.
- **Pinned: τ_spike = 2.25, τ_leap = 3.175, κ_max = 0.3174, ε = 0.2, γ_e = 1.0,
  entropy floor x = 10% (cannot fire — see below).** S2's noise floor is
  **0.00094** (π_0 scored against its own cache), and κ_max is in those units.
- **`scorer_v1` = Round-0 step 20 — user-ratified 2026-08-04**, frozen at `/data/whetstone/ckpt/scorer_v1`,
  **serving on `spark:8100` as `whetstone-scorer`**; d_t contract re-verified
  over HTTP (4,932/4,932 positions, all 4,188 rank-1 positions exactly 0).
  Chosen on design §2's "smallest dose": step 80 removes 3% more tax for 2.3× the
  KL drift and 40× the boundary damage. Non-winner checkpoints are **kept** under
  `/data/whetstone/ckpt/round0/` (50 GB, free to delete); if pruning, keep
  `step0080` and `step0000` — π_0 is the baseline every AUC comparison is against.
- **G_spike does NOT select against branch retention** (r_pb = −0.021, p = 0.47
  at β=5; −0.023, p = 0.44 at β=10; n = 1,200 32B traces). Activity 006's
  decision stands and **P5 proceeds with the unchanged product reward.**
- **G_spike DOES select against verification retention** — r_pb = **−0.113,
  p < 0.0001** (`verify_kept`), and `structural_pass` −0.062 / p = 0.032. Cause:
  a residual tax of **7.92 nats mean / 12.0 p95 on `chk`**, the largest of any
  marker, against `⇒` 0.087 and `let` 0.065. **P5 should add `verify_kept` as a
  selection term** (006 open item 2's fix, safe under one-shot Goodhart with a
  frozen teacher). **Do not lower λ** — the tax is in the p95 tail, not the mean.
- **Calibration is context-dependent, not token-dependent.** `chk` is calibrated
  in the *student's* usage (trailing `chk:` line) and not in the *teacher's*
  (mid-trace, richer expressions), despite 308 training occurrences. Expect the
  same wherever the teacher's register differs from the Round-0 corpus.
- **The instrument is shallow but honest: probe AUC 0.81** on single localized
  corruptions. Do not design a stage that needs a sharp threshold.
- **Inoculation degrades the native `</think>` boundary**: entropy
  7.6e-05 → 0.045 (step 20) → ~1.8–2.0 (step 40+), while the *averaged* verbose
  Δlogp barely moves (−0.085). Nothing downstream scores that position today
  (it is excluded from `think_mask`); anything that starts to must re-measure.
  This is a fourth Round-0 stopping signal the packet did not specify.
- **S3 cannot fire in restoration mode.** Control think entropy *rose* (mean
  0.2910 → 0.3778, +29.8%; median 0.0173 → 0.0664). The packet's median-based
  form has no resolving power anyway — the audit median is 0.0278 nats with 56.8%
  collapse mass, so 10% of it is noise. Report mean and p80; trip on the mean.
- **Numerics (binding on every trainer):** full-FT of 1.7B on one 32 GB card
  needs **fp32 master weights** — a 1e-5 Adam update is 12× below bf16's quantum
  and every update rounds to zero silently. fp32 weights + fp32 grads + fp32 Adam
  + SED shadow = 28.9 GiB and OOMs, so **8-bit Adam moments** (`bitsandbytes`,
  now in `pyproject.toml`) are the default. **Log `theta_drift` every eval.**
  `get_cosine_schedule_with_warmup` runs the **first** optimizer step at LR 0.
- **Eval forwards need bf16 autocast** even with fp32 weights: fp32 SDPA falls
  back to the math backend and materializes a full (T,T) attention matrix
  (10 GiB on a 6.2k-token trace).
- **turing's checkout is `~/workspace/whetstone`**, not `~/git/whetstone`
  (spark's *is* `~/git/whetstone`). Do not write scratch scripts to `/tmp` on
  turing — a stale `/tmp/inspect.py` shadows the stdlib `inspect` module.
- **The `</think>` sanity anchor is a statement about NATIVE traces** (median
  6.6e-05). On compact register traces the same quantity is legitimately ~0.275
  under π_0 — applying the anchor there looks like an off-by-one bug that is not.

## Standing rules for every future packet

1. Machines: training/rollouts on turing, frozen scoring on spark, artifacts on `/data/whetstone/`.
2. Every hyperparameter starts from design §12.6; asterisked placeholders get pinned by measurement and the pin recorded in the activity file AND the §12.6 table.
3. Journals in `activity/NNN-*.md` per README conventions; failures logged as thoroughly as successes.
4. `enable_thinking=True` on every Qwen3-1.7B template call — rollout, scoring, eval, no exceptions.

## Facts pinned by activity 001 (P0) — binding on all later packets

- Stack: **vllm 0.26.0 / torch 2.11.0+cu130 / transformers 5.14.1 / CPython 3.12.12**, plain PyPI wheels on both boxes. `pyproject.toml` at HEAD is the source of truth.
- **Scorer/reward server is `spark:8100`** — port 8000 on spark belongs to an unrelated `llama-swap` service (do not kill it; a `curl :8000/v1/models` check false-greens against it). Reachable from turing as `http://192.168.1.253:8100` (LAN) or `http://198.18.0.1:8100` (direct link). Launch command verbatim in activity 001 Run 6.
- **Every vLLM invocation on spark needs `VLLM_USE_FLASHINFER_SAMPLER=0`** (GB10/sm_121 FlashInfer sampler JIT failure; the error message about sm75 is misleading). turing does not need it.
- **`source .venv/bin/activate` before running anything that starts vLLM** — never bare `.venv/bin/python` (ninja must be on PATH or engine init dies with a buried FileNotFoundError).
- **Sync checkouts before remote work:** turing's clone can lag the Mac (activity 001 gotcha 6 — Mac-local commits, scp'd stragglers). First step of any packet touching a remote box: push from the Mac, `git pull` + `git status` on the box, and reconcile stray files.

## Facts pinned by activity 002 (P1) — binding on all later packets

- Pool/eval artifacts live at `/data/whetstone/data/pool/` (train 29,998 / val 2,000), `/data/whetstone/data/sca_arm/`, `/data/whetstone/eval/` (7 suites + frozen `standard_eval_300`). Dataset revisions pinned in the `.meta.json` sidecars.
- `_uid` / normalization / dedup / stratification live in **`whetstone/poolutil.py`** — use it, never reimplement.
- **Level histogram is peaked at 5–8 and nearly empty at 2–3 and 10.** Anything "level-stratified" must stratify proportionally or merge bands — equal-count strata are impossible.
- **The SCA arm overlaps the main pool by design** (only its three stages are mutually disjoint). Never describe it as held out.
- `standard_eval_300` is frozen; the builder refuses to regenerate it. HumanEval records are self-marked `code-exec-pending` and cannot produce verifier numbers (grader is P8's).
- `run_eval.py` still runs v1 defaults and lacks `enable_thinking=True` — **required fix owned by P2** before any eval numbers are quoted.
- spark has two venvs: `~/git/whetstone/.venv` (CPU data work) and `~/workspace/whetstone-scorer/.venv` (vLLM scoring).

## Facts pinned by activity 003 (P2) — binding on all later packets

- **Segment tokens:** `<think>` = **151667**, `</think>` = **151668**, `<|im_end|>` = 151645 — each a single token inline (Qwen3-1.7B @ `70d244cc`). `enable_thinking=True` does **not** pre-fill `<think>`; the model emits it, so completions *start with* 151667. `whetstone/segments.py` is the only place masks are computed — never split the decoded string.
- **SED runs in RESTORATION mode → Stage-B `Δ_max = 0.7`** (not 0.5). Think-segment median entropy 0.0278 nats vs **0.1163 for Qwen3-1.7B-Base on identical text** (4.2× lower); collapse mass 56.8%, fork mass 2.8%.
- **The design's 80/20 fork structure does not hold for this checkpoint.** The second entropy mode sits at **≈0.7 nats, not >1.5**. Any component hard-coding a 1.5-nat fork threshold is using the wrong knife. **TEA's `τ_c = 1.0` sits above the real second mode** — P7 should add it to the run-1 sweep (currently β, H_pivot, λ_TEA).
- **Median native think length = 6,099 tokens** (median answer 679). This is the baseline `G_budget`'s B_target of 600 is measured against, and the two lengths are always reported separately.
- **No system prompt.** v1's "put your reasoning between `<think>` tags" system prompt causes **6% duplicated-`</think>` gate failures** and costs **8 points of accuracy** on Qwen3. `run_eval.py` and `harvest.py` now default to no system message; the v1 text survives as `SYS_PROMPT_V1` in both.
- **`harvest.py --prefill_think` now defaults to False.** At its old default (True) it appended `<think>\n` to the prompt, which would have made every seed-harvest completion parse as `missing_think_open` — a 100% gate-out.
- **Harvest/eval budget is 32,768 tokens.** Cap-hit 0.0% at 32k vs 10.0% at 16k. Do not lower it.
- **`run_eval.py` defaults are now the §12.7 protocol:** N=8, T=0.7, top_p=0.95, max_tokens=32768, `enable_thinking=True`, `max_model_len` 36864.
- **H_pivot is still unpinned.** P3 must re-run `scripts/entropy_audit.py --traces <seed_register.jsonl>` on the compact corpus. Native-trace think p80 = 0.6923 is *reference only*, not H_pivot.
- **~2–4% of verifier yield is lost to extraction shape**, not reasoning (unit suffixes like `290 tomatoes` vs `290`; `$$…$$` display blocks extracting as `$$`). `verify.py` was deliberately **not** changed. P8 owns the decision; P3 should expect its yield ~3 pts under the P2 probe numbers for this reason alone.
- **vLLM's `EngineCore` outlives its parent process** and can hold the whole GPU. If a vLLM start fails with "Engine core initialization failed", check `nvidia-smi --query-compute-apps=pid,used_memory` for an orphan and kill it **by PID** — `pkill -f "VLLM::EngineCore"` matches its own command line and kills the calling shell.
- **`apply_chat_template(tokenize=True)` returns a `BatchEncoding` in transformers 5.x**, not a list — use `list(enc["input_ids"])`.

## Facts pinned by activity 004 (P3a) — binding on all later packets

- **The register is ARM A (symbolic), `configs/register_card.md` — RATIFIED and FILLED 2026-08-02** (5 exemplars, levels 1–6; known gap: no true combinatorics exemplar). The card is tokenizer-audited (§1.6): zero boundary-token injections, all symbols single-token in emission contexts — keep it that way on any edit. Full card ≈ 5,150 tokens; the render step strips non-notation sections (header/§1.6/§4) before pasting into prompts, per activity 004 deviation 3. Arm B (telegraphic/caveman) was eliminated because it **never installed its register**: 0.24 register markers per 100 think tokens vs A's 3.68 (15×, stable across T = 0.4/0.7/1.0). B's output is the model's native markdown-LaTeX write-up, often with `**Final Answer** … \boxed{…}` *inside* `<think>` — a card §1.5 violation. A hybrid was rejected: B's word connectives are the component that failed.
- **v1's chunkwise prompted compression is RETIRED for Qwen3-1.7B.** The cumulative ORIGINAL+COMPACT context is a repetition attractor — 54% of arm-A traces ≥50% stalled, byte-identical consecutive chunks, register-marker density 10× lower. **`compress_local_versionB.py --mode oneshot` is the default and is what P3 Part 2 must use**; `--mode chunkwise` survives by flag for v1 comparison only.
- **The register is reachable from a notation-neutral prompt.** One-shot, the model reproduces card A's exemplar style without the prompt naming a single symbol. v1's notation-prescribing compression `SYSTEM_PROMPT` (which also banned caveman style and named the rejected `⚠`) stays retired; the scaffold is card-parametric and its rendered sha1 is recorded per run in `<output>.meta.json`.
- **Compression ratio ≈ 0.043** — median 176 compact think tokens from a 5,404-token verbose median, **80% of traces already under `B_target = 600`**. G_budget's B₀ starts far lower than v1's numbers implied.
- **H_pivot preview = 0.2276 nats** (arm A compact p80) against native 0.6923 — expect P3's pinned H_pivot to land low. Compact-register think median entropy 0.0002 (native 0.0278), collapse mass 76.1% (native 56.8%).
- **p95 `d_t` gap for a clean symbolic register = 2.375 nats**, below τ_leap ≈ 4. The register's own accent does not pre-empt the Round-0 band-existence check (Risk 1).
- **Two card edits are required before the seed corpus is built:** drop/bound §1.3's `(A)`, `(B)`, … sub-result naming (it causes a 10–18% runaway class where the model rolls over to `AAA`/`BBB`/`CCC`), and un-indent the exemplars (their 4-space markdown indentation is copied verbatim and costs **8.2% of arm A's total excess surprisal**).
- **Temperature 0.4 stays pinned** for compression. Swept 0.4/0.7/1.0: adoption, compression and the A-vs-B ordering are flat; the only effect is that arm A's runaway rate falls with T (18→12→8%), so raising T is the recorded mitigation if runaways survive the card fix.
- **`merge_and_cap` in `compress_local_versionB.py` is round-robin** and scrambles chunk order once a trace exceeds `--max-chunks`. Irrelevant in one-shot mode; if chunkwise is ever run, set `--max-chunks` above the observed max (20 at 800-token chunks).
- **`style_tax.py` exits 134 on spark after writing its JSON** (vLLM teardown). Check for the output file before treating the exit code as a failure.
- **Never pipe a vLLM script into `head`** — the SIGPIPE orphans `VLLM::EngineCore` holding the whole card (activity 003 gotcha 1, reproduced).
