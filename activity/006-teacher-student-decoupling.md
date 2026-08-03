# 006 — Design decision: decouple teacher from student; Qwen3-32B teacher

- **Packet:** ad-hoc (design decision arising from activity 005 findings 15–18)
- **Status:** decided — implementation gated on the G_spike/branch check below
- **Decided by:** user, 2026-08-03
- **Machine(s):** turing (measurements)
- **Code commit(s):** `934542d` →

## The decision

**v2's teacher and student are no longer the same checkpoint.** The teacher is
**Qwen3-32B-NVFP4**; the student/scorer remains **Qwen3-1.7B**.

This overrides design §1 "Roles" ("two copies of the thinking checkpoint") and
the central-model principle (v1 §3) as they apply to Stage A. The student, the
scorer, Round 0, H_pivot and the final deliverable are unchanged.

## Why — the evidence

Branch preservation (keeping the case splits and rejected alternatives a verbose
trace contains — card §1.4's "never elided") is a **capability that appears with
scale and cannot be transferred by prompting**. Measured on 989 paired problems,
identical inputs and card:

| compressor | branch | verify | mark/100ch |
|---|---|---|---|
| Qwen3-1.7B | 3.1% | 26.2% | 1.22 |
| Qwen3-14B-FP8 | 5.9% | 60.1% | 0.82 |
| **Qwen3-32B-NVFP4** | **13.9%** | **70.6%** | **2.10** |
| GLM-5.2 | 39.9% | 95.9% | 3.15 |

And, on the same 200 traces, **every** attempt to induce it in the 1.7B by
prompting failed — static card exemplars, level-matched retrieval, and demo
pools drawn from 14B, 32B and frontier scale all leave it at **1–2%**
(activity 005 findings 15–18).

So at the 1.7B tier Stage-A rollouts would never contain branch-preserving
compressions, `G_spike` would have nothing to select on that axis, and design
§3's own argument applies against it — *"a group-relative reward can only rank
sampled lengths … it cannot jump to a register it never samples"*.

Interpolating the curve, the **4B/8B tier gate does not resolve this either**
(~4%), so this is not a decision that waiting for F1–F4 would make for us.

## What it forces: a frozen teacher, not a trained one

Design §3 makes Stage A *GRPO on the teacher*. That is not runnable here:

* full fine-tune of 32B ≈ 380 GB of weights + gradients + Adam states, against
  turing's single 32 GB RTX 5090;
* LoRA-GRPO would require trainer and rollout engine to time-multiplex that same
  card;
* spark has the memory (GB10 unified) but CLAUDE.md excludes it from
  decode-heavy rollout generation on bandwidth grounds.

**Stage A therefore becomes generate-and-select rather than train:** sample K
compressions per problem from the frozen 32B, score each with the *unchanged*
Stage-A reward `r = R_acc · G_spike · G_budget` under the frozen 1.7B scorer,
keep the best. Best-of-N against a reward is a KL-regularised approximation of
RL against that reward, so this preserves the load-bearing property — **the
reward still measures student-followability** — while dropping the optimisation
loop the hardware cannot run.

Consequences:

* `G_budget`'s annealing schedule and freeze rule become a *selection* criterion
  rather than a schedule (the freeze rule existed to stop a reward demanding
  lengths outside the realized group spread; with selection there is no
  schedule to outrun);
* "the teacher receives no register SFT" (§3) is moot — the teacher is not
  trained at all;
* **the teacher no longer improves.** If 13.9% branch retention proves
  insufficient, there is no mechanism in this design to raise it.

## The check that gates implementation

**`G_spike` may select against the property this decision exists to buy.** It
rewards traces the 1.7B finds *followable*; branch-preserving traces are longer
and structurally harder. Best-of-N could therefore systematically prefer the 32B
compressions that dropped their branches — undoing the move.

**Measure before committing compute:** score the 32B corpus under the 1.7B and
test whether `structural_branch_kept` correlates *negatively* with the d_t-based
`G_spike`. If it does, the reward needs a branch-aware term (or `λ/β` retuning)
before Stage A runs.

Second, smaller risk: Stage B's ZPD band-pass gates off tokens outside the
student's reachable zone, so a wider teacher→student gap means more of the
corpus is masked rather than learned. 32B is far closer than GLM, but the
masked fraction should be measured on the 32B corpus before Stage B is sized.

## What does not change

* student and scorer: Qwen3-1.7B;
* Round-0 inoculation and H_pivot: already built from the **1.7B** corpus
  (`seed_register_qwen`, H_pivot = 0.6707) — these measure the student's own
  distribution and must not be rebuilt from a teacher corpus;
* the register card, the verifier, segment routing, Stage C;
* the final deliverable: the student, prompt-free.

## Assets already in place

Four paired corpora over the same 1,200 inputs, all `verify_response`-clean and
structurally annotated:

| corpus | path | n |
|---|---|---|
| Qwen3-1.7B | `corpora/seed_register_qwen/` | 1,200 |
| Qwen3-14B-FP8 | `corpora/seed_register_qwen14b/` | 1,200 |
| **Qwen3-32B-NVFP4** | `corpora/seed_register_qwen32b/` | 1,200 (699 gated) |
| GLM-5.2 | `corpora/seed_register_glm/` | 989 (806 gated) |

Serving note: 32B-NVFP4 runs on one 5090 — `quantization=modelopt_fp4`,
`kv_cache_dtype=fp8_e4m3` from NVIDIA's checkpoint, 64,224-token KV cache,
~22 traces/min at concurrency 8, `--gpu-memory-utilization 0.93`.

## Open

1. Run the `G_spike` × branch-retention correlation check (above). **Gates P5.**
2. Decide K for best-of-N and whether `G_budget` selection uses the design's
   `B_target ≈ 600` (32B's median compact is 9 lines, well under it).
3. Expand P5 from "teacher GRPO" to "teacher best-of-N selection" once (1) is
   settled.
4. The GLM corpus retains its `central_model_deviation` stamp and its
   conditioning-only recommendation; with a 32B teacher in the design, its role
   narrows further — it is now the *ceiling reference*, not a planned input.
