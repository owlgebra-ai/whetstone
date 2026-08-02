# P3 — Seed harvest + seed register corpus

STATUS: in-progress (activity 005) — **card RATIFIED and FILLED 2026-08-02** (`configs/register_card.md`, bake-off winner arm A, required edits applied, tokenizer-audited, 5 exemplars). Parts 1 and 2 both unblocked; use `--mode oneshot` for Part 2 (activity 004).
MACHINES: turing (all generation); spark for Δlogp scoring pass — NB spark now has TWO venvs (activity 002): `~/git/whetstone/.venv` (CPU-only, data work) and `~/workspace/whetstone-scorer/.venv` (vLLM). The Δlogp pass needs the vLLM one.
DEPENDS ON: P0, P1, P2 (parser + probe + card)
BLOCKS: P4 (needs the seed register corpus), Stage A (teacher conditioning corpus)
DELIVERABLES: verified seed harvest, seed register corpus (~300–1,000 traces), Δlogp-gated, H_pivot pinned from the compact-register entropy histogram.

## Objective

Build the only two corpora that exist before the teacher is trained (design §1, preconditions 3–4): a **small blind seed harvest** (native verbose think traces, verifier-filtered) and the **seed register corpus** (the register card applied once via the v1 prompted-compression machinery). v1's GPU-days harvest is deliberately shrunk to this seed (design §10) — resist any urge to scale it up; the training corpus comes from the Stage-A teacher later, not from here.

## Part 1 — Seed harvest (blind, K=2)

Adapt `scripts/harvest.py` for Qwen3 and run:

- **Subset:** 15% of the train pool (~4,500 problems), stratified by level (*proportionally* — levels 2–3 and 10 are nearly empty, activity 002 note 1; use `whetstone/poolutil.py`), fixed seed — write the subset's `_uid` list to `/data/whetstone/corpora/seed/subset_uids.json` first (resume invariance: the subset is defined once, by file, not re-sampled).
- **Sampling:** K=2, T=0.9, top-p 0.95, `enable_thinking=True`, max_tokens 32768, `max_model_len 34816`.
- **Blindness is non-negotiable** (v1 §2): no gold in the prompt, no few-shot, no register mention. The harvest prompt is the *same unprivileged eval-style prompt* the student will see forever.
- Single worker on turing (`--worker_id 0 --n_workers 1`), `gpu_mem 0.90`. Keep the v1 per-line JSONL append + resume-by-`_uid` machinery — a 4-hour run WILL be interrupted at least once; test resume by killing it once on purpose after ~100 problems and restarting (yes, really — log that you did this).
- **Gemma scrub:** confirm `harvest.py` no longer imports `whetstone.patches.*` and builds prompts via `apply_chat_template`. (P2's probe should have fixed this; verify.)

Then the verifier gate, unchanged:

```bash
.venv/bin/python scripts/verify_harvest.py --input .../seed_harvest.jsonl --output .../seed_verified.jsonl
```

Log yield **per level band**. Reference numbers now exist (activity 003 probe, same sampling config, no system prompt): **73% overall at K=2 on 50 problems**, U-shaped per level (86% level 1, ~56% level 5, 50% level 9). Expect the bulk harvest to land ~3 points *under* the probe (known extraction-shape losses — unit suffixes, `$$` blocks; activity 003 finding 9). Substantially below that means a real bug — stop and compare per-level against the probe table before burning more GPU time. Also confirmed by P2: run with **no system prompt** (now the default; the v1 prompt costs 8 accuracy points and causes 6% gate failures) and `--prefill_think` stays False (now the default; True would gate out 100%).

## Part 2 — Seed register corpus (prompted compression, one pass)

`scripts/compress_local_versionB.py` (chunkwise prefill) with the register card:

- **Prompt scaffold:** use P3a's pinned neutral scaffold (`--card <ratified card>` — the card is the only register authority in the prompt; the v1 hardcoded SYSTEM_PROMPT is retired, see P3a "Pinned compression prompt"). The card text is versioned config — record its git sha and the rendered-prompt sha in the output header. `enable_thinking=False` on the compressor call is deliberate (prefill-trick rewrite, the one standing-rule exception).
- **Input:** the verified seed traces (think segments only — the answer segment is copied through *untouched*; compression must never touch post-`</think>` text).
- **Sampling:** T ∈ [0.3, 0.5] (design wants mild register-internal variance — use 0.4), single completion per trace.
- **USE `--mode oneshot` (activity 004).** v1's chunkwise prefill loop (§3.4) is a repetition attractor on Qwen3-1.7B — 54% of traces ≥50% stalled, register-marker density 10× lower. One-shot (whole think segment in, whole compact rewrite out) reproduces the card's exemplar style directly. `--mode chunkwise` survives by flag for v1 comparison only; do not use it to build the corpus.
- **Δlogp gate** (`scripts/perplexity_score.py`, v1 §3.6 — its only remaining use in v2):
  `delta = log P(a* | q, compact) − log P(a* | q)` under frozen Qwen3-1.7B; keep traces with delta above the v1 threshold. Run this scoring pass on **spark** (prefill-only — exactly what the GB10 is for) while turing moves on. Spark rules from activity 001: `source .venv/bin/activate`, `VLLM_USE_FLASHINFER_SAMPLER=0`, and if using the served scorer instead of in-process, it's **port 8100** (8000 is llama-swap — don't touch).
- **Target:** 300–1,000 accepted traces spanning all level bands. If acceptance is too low, loosen chunk size before touching the Δlogp threshold; if the register itself is the problem (systematically failing on one problem type), that's register-card feedback for the user, not a threshold problem — report it.

**Output:** `/data/whetstone/corpora/seed_register/seed_register.jsonl` with fields: `_uid`, `prompt`, `verbose_think`, `compact_think`, `answer`, `delta_logp`, `level`, provenance (card sha, sampling params).

## Part 3 — Pin H_pivot + build the Round-0 measurement sets

1. Re-run `scripts/entropy_audit.py --traces seed_register.jsonl --completion_field <field> --out_dir /data/whetstone/runs/entropy_audit_compact` (P2 built this mode — exact invocation in activity 003): compact-register entropy histogram → **H_pivot = its 80th percentile** (design §12.6). Record the number; P4 and Stage B consume it. For calibration: native-trace think p80 was 0.6923 (reference only, NOT H_pivot).
2. Split the seed register corpus **before P4 ever sees it**: `train` (~80%), `heldout_register` (~10%, Round-0 stop-criterion + unit test (a)), `probe_pool` (~10%, reserved for the corrupted-trace probe — these must never appear in training). Fixed seed, stratified by level. Write the three files; P4 is forbidden from re-splitting.
3. Build the **verbose control set**: ~200 verified *verbose* seed traces (from Part 1, disjoint from the compression inputs' heldout if possible) — Round 0's KL-drift gauge and unit test (b).

## Gotchas

1. **Vocabulary check: DONE — card §1.6 records the full audit** (every symbol 1 token in emission contexts; Unicode math ≤ ASCII; card text round-trips losslessly). One live rule remains for this packet: **assert the rendered compression prompt contains zero boundary-token ids** (151667/151668/151643–45) — the literal think-tag strings tokenize as REAL boundary tokens even in prose, which is why card §1.5 avoids writing them; keep it that way in any prompt text you add.
2. **Keep both think versions.** Stage-A teacher conditioning wants (gold + verbose trace); Round-0 wants compact. Never store compact-only.
3. **Answer-segment integrity:** assert `verify_response` still passes on every compressed record (same answer text ⇒ must pass). Any failure = the compressor leaked into the answer segment — a hard bug, fix before continuing.
4. **Disk/paths:** everything under `/data/whetstone/corpora/`; nothing on either root disk.
5. **This corpus is small on purpose.** ~1k traces is enough for inoculation + teacher conditioning exemplars. Scaling it up re-creates v1's dependency on lucky high-entropy sampling — the thing v2 exists to remove (design §10).

## Definition of done

- [ ] Seed harvest complete with per-level yield table; resume test performed and logged.
- [ ] Seed register corpus: acceptance rate, size, per-level coverage, 5 hand-inspected examples pasted into the activity file (verbose vs compact, side by side).
- [ ] Δlogp distribution plot; threshold recorded.
- [ ] H_pivot pinned and recorded; the three Round-0 splits + verbose control set written.
- [ ] Symbol-tokenization table recorded.
- [ ] Scripts committed; activity file `NNN-seed-corpus.md` written; packet status flipped; P4 unblocked.
