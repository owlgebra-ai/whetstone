# Packet roadmap — WHETSTONE v2 feasibility tier (Qwen3-1.7B on turing + spark)

```
P0 env ──► P1 data ──► P2 preconditions ──► P3 seed corpus ──► P4 Round 0 ══► F1 gate
                                                                              │
                                            PASS ◄────────────────────────────┘──► FAIL → LoRA-scorer packet
                                             │
                                             ▼
                       P5 Stage A (teacher GRPO) ══► F2 ──► P6 Stage B (assimilation) ══► F3
                                                                        │
                                                                        ▼
                                              P7 Stage C (segment-routed DAPO) ══► F4
                                                                        │
                                                                        ▼
                                              P8 baselines + eval hardening (SCA, DeepCompress arms)
```

Only **P0–P4 are written in full detail**. P5–P7 are deliberately outlines: the design (§11, §12.6) gates them on F1's measured values (τ_spike, τ_leap, λ/β behavior, H_pivot) — writing their fine detail now would bake in numbers F1 exists to pin. **Expand each into a full packet only when its gate opens**, folding in the activity-file learnings from the packets before it.

## P5 — Stage A: compression-teacher GRPO (draft outline; expand after F1)

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
