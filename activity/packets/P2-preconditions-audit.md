# P2 — Preconditions: segment parser on Qwen3, entropy audit, calibration probe, register card

STATUS: done (activity 003)
MACHINES: turing (GPU forward passes + generation); parser unit tests anywhere
DEPENDS ON: P0; Parts 2–3 also need P1 (pool)
BLOCKS: P3, P4
DELIVERABLES: verified Qwen3 segment parser + unit tests, `scripts/entropy_audit.py` + audit report (preservation-vs-restoration verdict), calibration-probe results, register-card spec staged for the user.

## Objective

Four preconditions from design §1 before any corpus is built: (1) the segment parser provably matches Qwen3-1.7B's exact chat template — the design calls this out as the first thing to verify; (2) the **entropy audit** of the starting checkpoint — it decides whether SED runs in preservation or restoration mode and later pins H_pivot; (3) the v1 Step-2.1 calibration probe, run unchanged; (4) the register card, which only the user can write — this packet stages everything so that's a fill-in exercise.

## Part 1 — Segment parser vs the real Qwen3 template

The whole pipeline routes rewards and losses by `<think>`/`</think>` boundaries (design §12.1). A one-token-off mask silently corrupts every stage. Work in `whetstone/segments.py` (new) + `tests/test_segments.py` (new).

Steps:

1. **Inspect the rendered template, don't trust docs.** For Qwen3-1.7B print `tok.apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)` and `enable_thinking=False`, and dump the special-token ids: `tok.convert_tokens_to_ids(["<think>", "</think>"])`. Record the actual ids in the test file as constants with a comment naming the tokenizer revision.
2. **Known Qwen3 template behaviors to encode as tests** (verify each against the real tokenizer — do not assume):
   - With `enable_thinking=True` the generation prompt ends after the assistant header; the **model emits** `<think>` itself. So the parser must handle a completion that *starts with* `<think>`.
   - With `enable_thinking=False` the template **pre-fills an empty `<think>\n\n</think>`** — if any such completion sneaks in, the parser must classify it as *empty think segment*, not malformed. We never use this mode, but the parser mustn't explode on it.
   - Multi-turn templates strip previous-turn think blocks. All whetstone calls are single-turn; assert single-turn in the parser entry point rather than handling multi-turn.
3. **Parser contract** (`parse_segments(token_ids) -> SegmentMasks`): returns `think_mask`, `answer_mask` (answer = post-`</think>` to EOS/`<|im_end|>`), and `g` (quality gate). `g = 0` for: missing `</think>` (cap-hit truncation), duplicated `<think>` or `</think>`, empty answer segment, `</think>` before `<think>`. g=0 rollouts are excluded from all structural rewards and alignment losses (SCA gate rule) but still get R_acc = 0 handling per the reward code.
4. **Token-level, not string-level.** Masks must be computed on token ids (the boundary tokens are single special tokens in Qwen3 — verify). String-level splitting then re-tokenizing shifts offsets and is the classic bug here.
5. Reuse whatever survives of `whetstone/reward/extract.py`; keep `verify.py` untouched (extraction for *grading* stays string-level post-`</think>` — that's fine and separate).
6. Unit tests: well-formed short/long, cap-hit (no `</think>`), duplicated boundaries, empty answer, empty think, `<think>` mid-text, and a *real sampled rollout* from the P0 smoke script.

## Part 2 — Entropy audit (`scripts/entropy_audit.py`, new)

Design §12.3: HF forward pass over ~200 stratified pool problems, **top-512 entropy per token**, output histogram + verdict. This runs on the model's own rollouts:

1. Sample 1 rollout per problem for 200 level-stratified problems from `val_2k.jsonl` (vLLM, T=0.9, top-p 0.95, max 16k think budget is enough for the audit, `enable_thinking=True`). **Stratification caveat (activity 002 note 1):** the level histogram is peaked at 5–8 and nearly empty at 2–3 and 10 — use *proportional* stratification (`whetstone/poolutil.py` has the machinery); do not attempt equal counts per level.
2. Teacher-force each (prompt, rollout) through HF `Qwen/Qwen3-1.7B` (bf16, sdpa, `torch.no_grad`), and per position compute entropy of the top-512 logits (softmax over the top-512 only, per CurioSFT's convention).
3. Aggregate: overall histogram; **think vs answer segments separately** (use the Part-1 parser); per-level medians; fraction of tokens under 0.1 nats (collapse mass) and over 1.5 nats (fork mass — the design expects a usable bimodal 80/20 structure).
4. Emit `/data/whetstone/runs/entropy_audit/audit.json` + histogram PNGs (copy PNGs to `activity/assets/NNN/`), and print a **verdict**: preservation mode (healthy bimodal, median comparable to a base model) vs **restoration mode** (arrives collapsed → Stage-B `Δ_max` 0.7 instead of 0.5). There is no absolute published threshold — report the numbers and argue the verdict in the activity file; the Δ_max choice is revisited at Stage B anyway.

**Gotchas:**

1. **Never materialize full-vocab logits for a whole sequence in fp32.** 151k vocab × 16k tokens × 4 bytes ≈ 10 GB per sequence. Chunk the forward output (e.g. 1024 positions at a time), `topk(512)` in bf16, softmax in fp32 on the 512, accumulate histogram counts, discard.
2. **Position alignment:** entropy at position t is from logits at t−1 (next-token prediction). Off-by-one here shifts think/answer attribution at the boundary; the boundary tokens themselves are a sanity check (`</think>` after a terse line should be low-entropy).
3. H_pivot is **not** set by this audit — it's the 80th percentile of the *compact-register* histogram, which needs the seed register corpus (P3). This script must therefore be reusable: `--traces <jsonl>` mode that skips generation and audits provided traces. P3 re-runs it in that mode to pin H_pivot.
4. The audit doubles as the baseline for the Stage-B sanity gate ("median entropy not below the audit baseline") and the Round-0 S3 stop (entropy floor). Store raw per-token entropy arrays (npz), not just the histogram — later comparisons need distributions.

## Part 3 — Calibration probe (v1 Step-2.1, unchanged)

Read `trashed/WHETSTONE_PROCEDURE.md` Step 2.1 and execute it against Qwen3-1.7B exactly as written (it validates prompt/template/sampling/extraction yield on a small subset before any bulk generation). The design explicitly keeps this probe unchanged (design §1 precondition 3). Record: extraction-success rate, verifier yield on the probe subset, and any template fix it forced.

**Required fix, handed forward by P1 (activity 002 note 4):** `run_eval.py`'s `_build_prompt()` already uses `apply_chat_template` (no Gemma scaffolding — verified), but it does **not** pass `enable_thinking=True`, and its defaults are v1's (`K=1, T=0.0, max_tokens=12288`). Fix both here: `enable_thinking=True` on every template call, and design-§12.7 defaults (`N=8, T=0.7, top_p=0.95, max_tokens=32768`) with the old cheap settings still reachable via flags (the probe and dashboards want cheap runs). Same `enable_thinking` check applies to `harvest.py` before P3 uses it.

## Part 4 — Register card (USER deliverable — stage it, don't write it)

The card is *the* human design input (design §1 precondition 2: "specified, not discovered"). Create `configs/register_card.md` as a template with clearly marked TODO sections and hand it to the user:

- **Notation spec (~1 page):** symbol vocabulary (e.g. ⇒ → ; ✓ ⚠), step-marker convention, equation-manipulation shorthand, what may be elided vs never elided (final numeric results of a step are never elided — unsupported leaps are exactly what G_spike punishes).
- **5–10 exemplars:** verbose think-trace → compact-register rewrite pairs. Real problems from the pool, one per difficulty band, at least one algebra / one combinatorics / one geometry.
- Constraint to tell the user: exemplar style must be *executable by a 1.7B model* — clever-but-dense human shorthand that skips derivations will read as leaps to the scorer and poison Round 0. Shorter lines, not bigger jumps.

**P3 is blocked until the user fills this in.** Say so loudly in your activity file and to the user.

## Definition of done

- [ ] `whetstone/segments.py` + tests committed; all tests green; real-rollout test included; token ids recorded.
- [ ] Entropy audit run; audit.json + PNGs + verdict + npz baseline stored; summarized in activity file.
- [ ] Calibration probe executed; yields recorded; any template fixes committed.
- [ ] `configs/register_card.md` template committed; user notified it's the blocking input for P3.
- [ ] Activity file `NNN-preconditions.md` written; packet status flipped.
