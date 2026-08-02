# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

WHETSTONE installs a compact reasoning register into a language model via self-training with verifiable rewards. The project is **mid-revision from v1 to v2 ("HONE")**: the code in this repo implements v1; the v2 design is complete and implementation is the current work.

**[whetstone-v2-design.md](whetstone-v2-design.md) is the authoritative spec.** Read it before touching any pipeline code. The v1 docs (`WHETSTONE_PROCEDURE.md`, `WHETSTONE_STAGE5_REWARD_DESIGN.md`) were moved to `trashed/` — do **not** delete them: the v2 doc cites them constantly as "v1 §N" (resume invariants, DAPO config §7.3, curriculum rules §7.6–7.7, pass-rate stratification §12), and surviving infra is specified only there.

## v2 pipeline in one screen

Starting checkpoint: an existing **thinking model** (native `<think>` traces). Two copies of it play roles: a **teacher** π_T conditioned on privileged context (gold answer, verbose trace, register card) and a **student/scorer** π_S with no privileged context. The final deliverable is the student, prompt-free.

1. **Preconditions** — entropy audit of the checkpoint (sets H_pivot, decides SED preservation vs restoration mode); human-written **register card** (the register is *specified, not discovered*); small seed harvest (K=2, 10–20% of pool); one prompted-compression pass → seed register corpus.
2. **Round 0 — scorer inoculation.** Calibrate the scorer so register tokens read as "hum" not spikes, without dulling its spike response to genuine reasoning leaps. Threshold-stopped (S1 calibration / S2 KL drift budget / S3 entropy floor). Three meter unit tests are mandatory; the **corrupted-trace probe failing invalidates the scorer regardless of the other two**. This retires the project's largest risk: a silently inverted reward meter.
3. **Stage A — compression-teacher RL.** GRPO on the teacher with the register **in context, not weights** (no register SFT for the teacher). Product reward — non-negotiable form: `r = R_acc · G_spike · G_budget`. G_spike penalizes top1-vs-actual logprob gaps under the frozen scorer (λ modest for the register accent, β high for unfollowable leaps); G_budget is a soft length tail with an annealed budget B and a **freeze rule** (tightening pauses when within-group length std < s_min).
4. **Stage B — assimilation SFT.** Student trains on teacher corpus, register card absent from its prompt (the register enters weights here, and only here). ZPD **band-pass** token weighting (gate off residual spikes, boost capped novelty in the reachable zone — the direct fix for v1's unsound surprisal *up*-weighting) plus the **SED** self-distillation term (CurioSFT: EMA teacher, entropy-gated per-token temperature) to preserve/restore entropy. Two rounds with gates recomputed under the updated student.
5. **Stage C — segment-routed RL.** DAPO backbone with `<think>`-boundary segment masks. Think tokens: length pressure + **TEA** covariance-targeted entropy protection, no style anchor. Answer tokens: forward KL to the *original* checkpoint + SCA length band. Curriculum per phase from fresh K=8; 1–7/8 → DAPO with bounded difficulty amplification; 0/8 → **pedagogy rescue** (teacher generates candidates with gold in hand, filtered, assimilated with the Stage-B loss).

Run order (gated): Round-0 band-existence check (Risk 1) → ablations A1/A4 → full pipeline on the **Qwen3-1.7B feasibility tier** → gates **F1–F4** must pass before any 4B/8B compute.

## Reference material

- **Pedagogical RL** — Ziems, "Teaching Models to Teach Themselves from Privileged Information" (https://noahziems.com/pedagogical-rl). Source of the teacher/student split, spike-aware pedagogy reward (G_spike), ZPD assimilation.
- **CurioSFT** — "Entropy-Preserving Supervised Fine-Tuning via Adaptive Self-Distillation for Large Reasoning Models" (arXiv 2602.02244). Source of the SED term and EMA-teacher mechanics.
- **Light-IF / TEA-RL** — "Light-IF: … Preview and Self-Checking for Complex Instruction Following" (arXiv 2508.03178). Source of TEA (token-wise entropy-adaptive RL regularization) and the type-aggregation selection logic reused for the register-token set R.
- **SCA / DeepCompress** — cited throughout the design for segment parsing, answer length band, difficulty amplification, and the eval setup; both are **mandatory baselines** reproduced from the same starting checkpoint.

## Models

- v2 targets **Qwen3**: Qwen3-1.7B (feasibility, hybrid mode — `enable_thinking=True` in every rollout, scoring pass, and eval), then Qwen3-4B-Thinking-2507 and Qwen3-8B (main results, matching SCA's published setup). 4B-Thinking-2507 always emits `<think>` and has no template flag — verify the segment parser against its exact template before anything else.
- The existing code targets **Gemma 4 (E4B)** — that was v1's base. `whetstone/patches/gemma4_*`, `gemma4_learnings.md`, and Gemma defaults in scripts are v1 leftovers; expect to replace model plumbing, not reuse it.

## Code map (v1 → v2 status)

| Path | v1 role | v2 status |
|---|---|---|
| `whetstone/verify.py` | Deterministic verifier (v4.6.1: post-`</think>` extraction only) | **Kept unchanged.** Is R_acc in Stage A and Stage C. Strictly deterministic — any relaxation belongs in reward shaping, never here |
| `whetstone/reward/` | Stage-5 DAPO reward (tiers, structure, penalties, diagnostics) | Base for Stage C: three-tier R_acc and diagnostics survive; uniform KL and char-count length penalty are replaced by segment routing + TEA |
| `scripts/harvest.py`, `verify_harvest.py` | Stage 1 blind harvest + gate | Demoted to **seed harvest** (K=2, 10–20% of pool) |
| `scripts/compress_local_versionB.py` | Stage 2 chunkwise prompted compression | Repurposed: one pass to build the seed register corpus only |
| `scripts/perplexity_score.py` | Δlogp sufficiency gate | Seeds only — its last remaining use; G_spike absorbs it elsewhere |
| `scripts/audit_compressions.py` | Stage 2.5 bulk cross-family audit | Demoted to 100-sample spot-check per teacher checkpoint |
| `scripts/sft_train.py` | Stage 3/4 surprisal **up**-weighted SFT | **Unsound (Diagnosis #1); replace** with Stage B (ZPD band-pass + SED) |
| `scripts/build_train_pool.py`, `build_eval_sets.py`, `run_eval.py`, `calc_metrics.py`, `merge_fsdp2_to_hf.py` | Data/eval/checkpoint infra | Kept; pool moves to DeepMath-103K + GSM8K, eval to the SCA-matched protocol |
| `scripts/smoke_verify.py` | Verifier smoke test, no GPU needed | Kept — `python scripts/smoke_verify.py` runs on this Mac |
| `whetstone/patches/gemma4_*`, `gemma4_learnings.md` | Gemma-4 workarounds | Obsolete for v2 (historical reference) |
| `trashed/*.md` | v1 procedure + Stage-5 reward design | Reference only — resolve "v1 §N" citations here |
| `deps.md` | Element-GPU dependency install (CaveThought-era) | Historical; local boxes below are the current hardware |

New v2 components with no v1 counterpart (to be built): Round-0 inoculation loss + meter unit tests, the offline analysis scripts (entropy audit, type aggregation for R — design §12.3), the teacher GRPO loop with G_spike/G_budget, the scorer "reward server" (frozen vLLM instance scoring via one teacher-forced prefill pass, `prompt_logprobs ≥ 2`), the SED kernel, and Stage-C segment routing + TEA + rescue.

## Invariants — do not violate

- **Verifier stays deterministic.** Reward leniency lives in `whetstone/reward/`, never in `verify.py`. Answer extraction is post-`</think>` only.
- **Product reward form in Stage A** (not additive). Malformed `<think>` boundaries → quality gate g = 0, excluded from all structural rewards.
- **Segment-level reporting, always:** think length and answer length as separate numbers. One combined length number is how drift hides.
- **A reward must never demand lengths outside the realized group spread** — the G_budget freeze rule exists for this.
- **EMA gotchas (design §12.4):** EMA *update* (μ=0.99), never a hard copy; count **optimizer** steps, not micro-batches; gate and temperature search run on the *teacher's* logits; Round 0 and Stage B each keep their own EMA copy — never shared or carried over.
- **Scorer gates recompute after every assimilation round** — stale gates are a known drift failure.
- **Don't trust the scorer until all three Round-0 unit tests pass simultaneously**; re-inoculate between teacher rounds rather than lowering λ.
- **Gate compute:** nothing runs on 4B/8B until F1–F4 pass on 1.7B.
- Record schema is `_uid / prompt / ground_truth / level`; keep resume invariants from v1 §2.5.

## Environment

- This Mac is for design and code work. Training/inference runs on two local GPU boxes (full survey: [activity/000-turing-reset.md](activity/000-turing-reset.md)):
  - **turing** — `ssh bajajra@192.168.1.220`, x86_64, RTX 5090 32 GB. **All training and all rollout generation.** Wiped clean 2026-08-01 (~866 GB free on root); the stale v1 checkout is gone — fresh clone per packet P0.
  - **spark** — `ssh bajajra@192.168.1.253`, DGX Spark, **aarch64 (ARM — different wheels!)**, GB10 unified memory. **Frozen scorer / reward server + CPU-heavy data prep + offline scoring passes.** Never schedule decode-heavy rollout generation here (low memory bandwidth).
  - **`/data` is the shared artifact store:** ZFS on turing (~4 TB free), NFS-mounted on spark. All corpora/checkpoints/logs go under `/data/whetstone/`; nothing large on either root disk. HF cache on turing is `~/.cache/huggingface → /data/cache/huggingface/` and already holds **Qwen3-1.7B/-Base, Qwen3-8B, Qwen3-4B-Thinking-2507-FP8**.
- System pythons are unusable (turing 3.14, too new). Use uv-managed **3.12** venvs per packet P0. The pyproject pins (`vllm==0.23.0`, `liger-kernel`, cu130 index) are v1/Gemma-era — P0 establishes and commits the new verified stack; trust the pyproject at HEAD over this sentence.
- The two-box split resolves design §12.5's 2-GPU assumption for the 1.7B tier: trainer + rollout vLLM time-multiplex on turing (vLLM sleep/wake), scorer stays resident on spark.

## Execution workflow: activity/ and instruction packets

Experiments are run by executing agents from **instruction packets** in [activity/packets/](activity/packets/), and every run is journaled in a numbered `activity/NNN-*.md` file — conventions in [activity/README.md](activity/README.md). Before doing any experimental work: read the README, claim the relevant packet (flip its STATUS header), and journal as you go — exact commands, commits, paths, failures included. The packet sequence and gating (P0→P4 detailed, P5–P7 outlined pending the F1 gate) is in [activity/packets/ROADMAP.md](activity/packets/ROADMAP.md).

## Working conventions

- When implementing a stage, cite the design-doc section in the module docstring (the v1 code does this; keep the practice — it's how the code stays auditable against the spec).
- Hyperparameters start from the consolidated table in design §12.6; asterisked values are placeholders the 1.7B run exists to pin. Sweep only β, H_pivot, λ_TEA in run 1.
- Dashboards are first-class deliverables of each stage (design §7 playbook): entropy trajectory, mean/max gap as separate curves, symbol density, bimodality index. Build them alongside the training code, not after.
