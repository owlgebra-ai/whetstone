# 011 — Stage C rerun: entropy-regulated DAPO (pilot 2 → Phase 1)

- **Packet:** [packets/P7b-stage-c-entropy-fix.md](packets/P7b-stage-c-entropy-fix.md)
- **Status:** in-progress
- **Machine(s):** mac (code) / turing (rollout worker + all screens) / spark (trainer, fp32 AdamW, resident frozen π_0)
- **Code commit(s):** (claim commit TBD)
- **Started / finished:** 2026-08-06 → …

## Goal

Re-run the Stage-C pilot with the 010 diagnosis applied (finding 23: the think
side was entropy-raising end to end with no ceiling, on a checkpoint that
arrived at 10× baseline entropy). Arms strictly one variable at a time:
**Arm A — symmetric clipping (ε_high = ε_low = 0.20), λ_TEA = 0**, 100 steps;
Arm B (thermostat, H_hi = 1.2) only if Arm A's entropy still trends up; Arm C
(training top-p 0.995) only if boundary failure persists at *stable* entropy;
Arm D (TEA at τ_c = 3.0) only alongside a passing arm. F4 re-gated at arm
completion with the init re-screened through the identical harness, every
Pass@1 the 8-draw mean ± std, paired McNemar. On PASS → Phase 1 to its Pareto
endpoint under the fixed config.

Everything reused from 010: init checkpoint
`/data/whetstone/ckpt/stageb/golden/round1/final`, bucket table
`/data/whetstone/runs/stagec_buckets/phase1_init/` (curriculum-from-init and
the init is unchanged), reward modules incl. the post-pilot line-initial
register-leak fix, the whole loop/bus/worker infrastructure (131 tests).
Pilot-2 pins vs pilot 1: **8 problems/step (was 4), `--prefetch` ON (was off)**
— 010 findings 12/14 — everything else identical except the arm's one variable.

## Runs

(journal as executed)

## Conclusion

(TBD)
