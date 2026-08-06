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

### Run 1 — 2026-08-06, pre-flight (mac, CPU only)

Code for Arm A, all committed before any GPU time:

- **`whetstone/dapo.py` + `scripts/stagec_train.py`** (`7397f18`): clip
  epsilons threaded through `stagec_loss` → `token_level_policy_loss` and
  exposed as `--eps_low/--eps_high`. Defaults stay the v1 asymmetric values so
  a pilot-1 command reproduces pilot 1; Arm A passes `0.20 0.20` explicitly.
  New per-step diagnostics: `format.missing_think_close_rate` and
  `format.g_rate_all_cands` computed over **every candidate in the batch,
  dropped groups included** (010 f21's curve was only visible post-hoc, and a
  kept-only rate understates the failure — format death concentrates in
  all-wrong groups, which dynamic sampling removes from `kept`);
  `batch_p_hat_mean` logged and printed beside acc (010 f20).
- **Tests** (3 new, battery 138 green on mac): symmetric clip measurably
  differs from clip-higher at ratio 1.25 (gradient vanishes, clipped branch
  active); eps threading actually reaches the clip (a kwarg accepted but not
  forwarded would silently run clip-higher); **λ_TEA = 0 ⇒ `loss/tea_term`
  == 0.0 exactly** — Arm A's inert statistic per the packet's standing rule.
- **`scripts/stagec_f4_regate.py`** (`4a87078`): the re-gate harness with 010
  f22's rules hard-coded — init re-screened through the same harness in the
  same session, every Pass@1 the K-draw mean ± between-draw std, paired
  McNemar per problem per draw, think-per-correct, and the PASS/EXTEND/FAIL
  clause-2 verdict (EXTEND = endpoint within ±1σ at equal-or-less think —
  flat-but-healthy is not FAIL). `--train_log` prints the clause-1 10-step
  windows (mtc / g(all) / H / acc / p̂ / stutter / answer median / drift).
  Stats functions validated on synthetic fixtures (McNemar wins/losses/z by
  hand).
- Register-leak fix verified in place at HEAD (line-initial `_LEAK_SYMBOL_RE`,
  010 f15's four verbatim shapes as regression fixtures). Battery re-run green
  as the packet requires: mac 138, **spark (trainer venv) 138**.

The mac cannot install the project pyproject (CUDA-only wheel
`nvidia-cudnn-frontend` has no darwin build) — battery ran in a scratch
3.12 venv with torch 2.13 CPU + numpy + pytest, and again on spark's real
trainer venv.

### Run 2 — 2026-08-06 08:0x, spark reboot + the /data mount failure

Spark had just rebooted when pre-flight checks started (`up 1 minute`) and
came up **without /data**: `data.mount` (fstab, `198.18.0.2:/data`, NFS 4.2)
**failed at boot** — it raced the direct link and `nofail` let boot continue.
ssh also dropped for ~2 min mid-diagnosis. No root available non-interactively
(`sudo -n` refused), so the fix was out of reach from here; the user brought
the box back and the mount appeared shortly after ("now spark is reachable").
Verified before proceeding: `mount` shows the real NFS4.2 entry over the
direct link, write test OK, spark's own HF cache has Qwen/Qwen3-1.7B (the
resident π_0 anchor does not depend on /data).

**Ops note for the next agent:** after any spark reboot, check `systemctl
is-failed data.mount` *before* anything else — the trainer, the bus, and every
artifact path die without it, and the failure looks like missing files, not
like a mount problem. Also: spark's trainer venv is
**`~/workspace/whetstone-scorer/.venv`** (the checkout's own `.venv` has no
torch — running pytest there silently collects only 98 of 138 tests, exit
code 2 hidden in the tail). Battery must run under the venv the trainer runs
under.

### Run 3 — 2026-08-06 08:22 →, ARM A: 100 steps, symmetric clip, λ_TEA = 0

Both checkouts verified at `4a87078` (mac == turing == spark). turing GPU idle
(18 MiB), no orphan EngineCore.

- worker (turing): `python scripts/stagec_rollout_worker.py --run_dir
  /data/whetstone/runs/stagec/pilot2_armA --init_model
  /data/whetstone/ckpt/stageb/golden/round1/final` (venv activated; log
  `/data/whetstone/runs/stagec/pilot2_armA_worker.log`)
- trainer (spark): `~/chain_train_armA.sh` — waits on the worker heartbeat,
  then `python scripts/stagec_train.py --run_dir .../pilot2_armA --init_model
  .../round1/final --anchor_model Qwen/Qwen3-1.7B --buckets
  .../phase1_init/buckets.jsonl --pool .../train_30k.jsonl --steps 100
  --problems_per_step 8 --K 8 --max_tokens 12288 --eps_low 0.2 --eps_high 0.2
  --lambda_tea 0 --lambda_align 0.1 --b_init 1026 --sync_every 8 --ckpt_every
  25 --prefetch` (log `/data/whetstone/runs/stagec/pilot2_armA_train.log`)

Everything not named by the arm is pilot-1's value; the three deliberate
differences are the arm variable (eps 0.28→0.20, λ_TEA 0.05→0) and the two
packet pins 8/step + `--prefetch` (010 f12/f14). `--b_init 1026` pins B₀ at
pilot 1's measured value instead of re-measuring from batch 1 (8 problems
draw a different first batch; re-measuring would move a pinned constant).

Startup verified: banner `clip eps_low=0.2 eps_high=0.2 (SYMMETRIC) |
lambda_tea=0.0`; curriculum 3,184 mixed / bands 1,366 high / 972 mid / 846
low (the reused table, byte-identical); worker model-match check skipped the
redundant v1 publish; step-1 generation 60.9 s for 64 rollouts (8×8).

## Conclusion

(TBD)
