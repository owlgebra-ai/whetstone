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

### Run 3 (cont.) — 2026-08-06 ~11:00, Arm A COMPLETE: 100/100 steps, clean exit

Median wall ~80–110 s/step at 8 problems/step (prefetch hides generation
almost entirely — most steps log `gen 0`). Checkpoints at 25/50/75/100.
`theta_drift_rel` monotone 1.48e-05 → 2.20e-04.

**The clause-1 trajectory, 10-step windows over ALL candidates (kept +
dropped groups):**

| steps | H think | mtc (all) | g (all) | stutter | answer KL | clip-low | clip-high |
|---|---|---|---|---|---|---|---|
| 1–10 | 1.217 | 0.0750 | 0.925 | 0.0021 | 0.171 | 0.00013 | 0.00044 |
| 11–20 | 1.242 | 0.0766 | 0.920 | 0.0025 | 0.188 | 0.00014 | 0.00073 |
| 21–30 | 1.243 | 0.0719 | 0.925 | 0.0018 | 0.177 | 0.00027 | 0.00121 |
| 31–40 | 1.142 | 0.0547 | 0.945 | 0.0021 | 0.166 | 0.00023 | 0.00084 |
| 41–50 | 1.180 | 0.0453 | 0.952 | 0.0017 | 0.155 | 0.00034 | 0.00088 |
| 51–60 | 1.119 | 0.0484 | 0.942 | 0.0035 | 0.169 | 0.00029 | 0.00068 |
| 61–70 | 1.123 | 0.0547 | 0.944 | 0.0020 | 0.154 | 0.00032 | 0.00047 |
| 71–80 | 1.104 | 0.0500 | 0.944 | 0.0048 | 0.143 | 0.00033 | 0.00050 |
| 81–90 | 1.049 | 0.0453 | 0.952 | 0.0029 | 0.136 | 0.00030 | 0.00044 |
| 91–100 | **0.991** | **0.0391** | **0.955** | 0.0025 | 0.116 | 0.00027 | 0.00039 |

Side by side with pilot 1 over the same axes: H 1.05→**3.18** became
1.22→**0.99** — entropy now *declines* gently in the second half, i.e. the
policy is sharpening the way RL normally does, instead of diffusing;
`missing_think_close` 5.6%→**35.6%** became 7.5%→**3.9%** — the failure is not
merely flat, it is being *taught away* (the r_fmt=0.10 well-formed floor was
always the right gradient; it just couldn't outrun a 3× entropy rise);
`g_rate` (all candidates) rises 0.925→0.955; word stutter flat at 0.002–0.005
against pilot 1's 4.4× climb; answer-KL bounded and declining 0.171→0.116.
Clip fractions are tiny on both sides (≤0.0012) — symmetric clipping is not
suppressing learning, it is simply no longer subsidizing the upside tail.

**Reward view (kept groups):** window reward mean 0.776→0.892 with acc
0.59→0.70 against batch p̂ 0.65→0.68 — in the last three windows acc sits
**above** the bucket-table p̂ (e.g. 0.72 vs 0.62 in 81–90), the first
training-curve hint of genuine conversion, though per 010 f20 the fixed
screen remains the only binding instrument. Within-group reward std steady at
0.39–0.50; penalties ≤0.011 total; empty think ≤0.011 (guard holding);
lenient-only 0.04–0.07 not widening; drop reasons 107 all-correct vs 13
all-wrong over the run — the saturation direction, not the rot direction.

**Prediction recorded before the screen returns** (same discipline as 010
f17): if the screen shows a checkpoint with strict Pass@1 up at ≤219 think
median, Arm A PASSES and the diagnosis (entropy ceiling via symmetric
clipping alone) is confirmed end to end.

### Run 4 — 2026-08-06 11:35, the F4 re-gate (all screens one harness session)

`stagec_f4_regate.py`: init + step0025/50/75/100, 200 GSM8K problems, T=0.7 /
top-p 0.95, K=8, cap 8,192. Init **re-screened**, never quoted (010 f22).

| arm | strict P@1 (8-draw) | think med | ans med | pass@8 | g | think-per-correct† | ΔP@1 (paired) | McNemar z, p |
|---|---|---|---|---|---|---|---|---|
| **init** | 66.75% ± 1.25 | **219** | 190 | 94.00% | 95.94% | 840 | — | — |
| step0025 | 66.94% ± 1.24 | 220 | 201 | 94.50% | 96.00% | 820 | +0.19 | +0.18, 0.853 |
| step0050 | 68.88% ± 2.03 | 222 | 222 | 93.50% | 96.50% | 784 | +2.12 | +1.80, 0.072 |
| step0075 | 71.12% ± 2.94 | 227 | 244 | 94.50% | 96.50% | 711 | +4.37 | **+3.64, <0.001** |
| **step0100** | **71.62% ± 2.46** | 228 | 248 | 94.00% | 96.19% | **623** | **+4.87** | **+4.01, <0.001** |

† this journal's think-per-correct = total think tokens across all 1,600
rollouts / strict-correct count (a cost-per-correct). **Not comparable to
010's "331→367"**, which was think-median/Pass@1 — different statistic, same
direction of meaning. Within this table the init is the same-session
comparator, so the 840 → 623 (−26%) trend is internally valid.

Note the init re-screen read 66.75% ± 1.25 against 010's 66.25% ± 1.46 on the
identical protocol — half a point of between-session drift on the same
checkpoint, which is itself the argument for same-session comparators.

**The verdict, stated precisely:**

- **Clause 1 — PASS.** No named failure worsens across step windows at the
  training sampler; the two watched ones *improve* (mtc 7.5%→3.9%, H
  1.22→0.99, g(all) 0.925→0.955, stutter flat, answer-KL bounded declining).
- **Clause 2, by the letter — not met.** No checkpoint holds think median ≤
  219; the strict-Pareto row does not exist. But the letter was written
  against pilot 1's failure quadrant (accuracy *down*, think *up*,
  think-per-correct *up*). Arm A sits in a quadrant the criterion never
  anticipated: **strict Pass@1 +4.87 pts (monotone in step, McNemar p <
  0.0001), think median +9 tokens (+4.1%), think-per-correct −26%,** pass@8
  held, eval g-rate held. The +9 think creep is priced against a 7.3%
  relative accuracy gain; the Phase-1 endpoint criterion (accuracy ×
  tokens-per-correct Pareto, v1 §7.9) ranks every RL checkpoint above the
  init.
- **Arm verdict: CONTINUE (user directive 2026-08-06: "if there is
  improvement, continue the RL run to 350–400 steps").** The strict Pareto
  question is re-adjudicated at the Phase-1 endpoint, where the annealed
  budget (B at its 120 floor since ~step 50, group-relative via the
  `effective_B` spread floor) has had time to press think back down. Alarm
  condition for the continuation: **think median still rising while ΔP@1
  plateaus** — that is the drift clause 2 exists to catch.

Entropy at the eval protocol confirms the training-sampler read: between-draw
std ±1.25 → ±2.46 (grew less than pilot 1's ±1.46 → ±2.93 at 60 steps despite
100 steps and a 4.9-pt mean shift).

### Run 5 — 2026-08-06, rollout scan (6,400 rollouts) + two more detector defects

`stagec_rollout_scan.py` (validated against pilot 1 first: reproduces mtc
5.0→33.8% and the strict decay 59→42%). Arm A, 20-step windows: every rot
pattern extinct or falling — empty think 0.00% everywhere, loops ≤1.8%
falling, `case N:` enumeration ≤0.16%→0, chk-chains 0, cap-hit ≤0.94%,
mtc 7.11%→3.75%, **strict-correct at the training sampler 59.8%→75.0%**.

Two detectors flagged rollouts at rates worth reading (register_leak
0.94%→**3.83% rising**; answer_repeat ~9% flat), and the verbatim dumps showed
**both are finding-15-class false positives on honest mathematical English**:

- **register_leak**: every read sample was a clean LaTeX answer opening a line
  with capitalized **`Let:`** — mathematical prose scaffolding, not the
  register's strictly-lowercase `let:` binder. `_LEAK_LINE_RE` carried
  `re.IGNORECASE`; the "rise" tracks answers growing toward the 288-token
  band (more structured prose → more `Let:` headers), not register leakage.
  **Fixed: case-sensitive** (`144f3a7`), regression fixtures from the dumps.
- **answer_repeat**: fired on consecutive `$$` display-math blocks separated
  by a blank line — LaTeX typesetting, not v1 §4.6's `151\n\n151`
  restatement. Flat ~9% in pilot 1 AND Arm A = a base rate of honest
  formatting. **Fixed: repeated token must contain a word character**; bare
  numeral/boxed restatement asserted still firing. Battery 140 green.

Both fixes lower penalties on *correct* answers by up to ~0.20 of 1.35 and
land for the continuation (imported at trainer start), attested here.
Contradiction firings (~2–2.8% flat) were read too: mostly **true
positives** — T=1.0 think blocks reaching wrong conclusions while the answer
recovers, which is precisely the derivation-vs-luck signal the 0.20 penalty
exists for; one marginal case (extractor grabbing a verification-chain
intermediate) noted, detector left unchanged.

### Run 6 — 2026-08-06 11:50 → 17:40, primary-suite diagnostic bench (user request) + a 5.5 h stall

User direction: eval init + step0100 on **MATH-500 / AMC23 / MinervaMath /
AIME24 / AIME25** to identify further reward tweaks. Sized as a *paired
diagnostic*, not P8 numbers: K=4, T=0.7/0.95, cap 16,384 — both models through
the identical harness in one session; the full K=8/32k protocol stays
reserved for P8.

**The stall.** The first launch died ~2 min in and was not noticed for 5.5 h:
`run_eval.py` was the one script in `scripts/` without the repo-root
`sys.path` insert, and its lazy `from whetstone.segments import ...` sits
**after** `_load_model` — so the engine came up (30 GB), the import raised,
and the dying process **hung in vLLM teardown** (parent in `do_wait`,
EngineCore in `futex_do_wait`) instead of exiting. The chain's `|| echo
FAILED` never fired, the monitor saw nothing, and the GPU sat occupied by a
corpse. Fixed in `eb54d2f` (sys.path insert at top, like every other
script). Two hasty relaunches then raced the VRAM drain and OOMed at engine
init; the launch that stuck kills every pid `nvidia-smi
--query-compute-apps` reports, requires **three consecutive <200 MiB
readings**, then starts.

**Ops lessons (both earned the expensive way):** (1) crash-after-engine-load
+ hung teardown is invisible to a chain log — launch scripts must verify the
GPU is *actually* clean (compute-apps query, repeated readings), not trust
`kill` + one reading; (2) a failed vLLM init can itself orphan an EngineCore
holding full allocation, so the check must run before every launch, not once.

**Results (init vs step0100, paired per problem per draw, same session):**

| suite | n | init P@1 | step0100 P@1 | Δ | McNemar w/l | z | cap-hit i→0100 | think med i→0100 |
|---|---|---|---|---|---|---|---|---|
| MATH-500 | 500 | 58.60 ± 2.13 | 64.25 ± 1.08 | **+5.65** | 249/136 | **+5.76** | 2.9→1.2% | 830→837 |
| MinervaMath | 272 | 15.07 ± 2.33 | 17.19 ± 1.10 | **+2.11** | 69/46 | **+2.14** | 4.0→1.0% | 1,267→1,427 |
| AMC23 | 40 | 42.50 ± 5.40 | 48.75 ± 2.50 | +6.25 | 31/21 | +1.39 | 5.6→1.9% | 3,322→3,258 |
| AIME24 | 30 | 16.67 ± 2.72 | 19.17 ± 3.19 | +2.50 | 14/11 | +0.60 | 7.5→7.5% | 6,216→8,768 |
| AIME25 | 30 | 6.67 ± 2.72 | 17.50 ± 8.77 | **+10.83** | 16/3 | **+2.98** | 6.7→2.5% | 6,249→8,099 |

**The conversion generalizes off the training pool** (GSM8K+DeepMath →
+5.65 on MATH-500 at z 5.76; pooled over 872 problems ≈ **+4.6 pts**), with
between-draw std *shrinking* on every suite and cap-hit rates falling almost
everywhere — accuracy, consistency, and budget discipline moving together.
The one negative signal: **AIME think medians grew 30–41%** (6.2k→8.8k /
8.1k) — off-distribution hard problems escape the length discipline (the
training pool's effective_B never sees 6k-token groups), the same direction
as the +9-token GSM8K creep. Tweak decision from this read: **none beyond
the two detector fixes** — the gains are real and the length creep is a
length-*pressure* question for the endpoint re-read, not a new reward term
mid-run. (Diagnostic protocol: K=4 / cap 16,384 — NOT comparable to P8's
K=8/32k numbers; original-checkpoint arm queued same-session.)

### Run 7 — 2026-08-06 evening, failure-pattern analysis (subagent) → the grader extension

A subagent read the step-0100 bench rollouts (all 3,488 classified, ~55 full
manual reads): persistent failures (0/4, 1/4), the 136 MATH-500 McNemar
losses, and near-miss draw contrasts. Artifacts:
`/data/whetstone/runs/stagec/pilot2_armA/bench/failure_analysis/{report.md,patterns.json,exemplars/}`.

**Finding A — the verifier was silently rejecting correct answers at +6.8 pts
(MATH-500) / +5.5 pts (Minerva).** Seven deterministic format classes
(`2k + 2` vs `2k+2`; `\dfrac{4}{3}` vs `\frac43`; `2.7778 \times 10^{-6}` vs
`2.7778e-6` — a format Minerva prompts *demand*; `^\circ`; binder prefixes;
`\text{(C)}`; symbolic fracs). Init and RL model hit equally → every paired
delta stands; but under RL these are **false-negative rewards**: a correct
rollout scored 0 inverts its within-group advantage. **Fixed** as
`whetstone/reward/normalize_ext.py` (`6c7d0f9`): deterministic equivalence
only, tried after the verbatim strict path fails, `verify.py` untouched per
the invariant; the analysis's 2%-rounding proposal **rejected** (that is
leniency, i.e. Goodhart bait — `698` vs `700` stays wrong, pinned as a
negative control). Battery 175 green. Landed **before the continuation's
first step** — a boundary application, not a mid-run change.

**Finding B — the MATH-500 regressions are overthink, not format.** Of 136
losses: cap 7, gate 6; the rest think +252 tokens longer than init's winning
draws, with three read mechanisms: second-guessing a correct result ("derives
1997/2 … 'likely expects an integer' … boxes 998"), re-derivation slips in
long verify passes (incl. false `chk:` ✓ on 993=999), and admitted-guess
confabulation. Near-miss draws are *longer* than winning draws everywhere
except Minerva (which bails early instead). RL meanwhile **halved** eval-time
`missing_think_close` (3.9→1.8%) and cap rates — the format story is healthy;
the cost center is churn. Phase-2 candidate levers, not applied now:
think-final/answer consistency shaping; windowed anti-loop repetition penalty
(~30 cap-burning arithmetic loops read).

**Finding C — capability ceiling map.** Wrong-math dominates true failures
(70% MATH-500 → 98% AIME25). True derived-then-lost is <1%. Most Minerva 0/4
(physics/chem constants) and AIME depth are not RL-recoverable at 1.7B.

**Contamination guard, stated loudly:** the analysis's per-suite
`rescue_uids_*.json` lists are **diagnostic only** — benchmark problems must
NEVER enter the rescue/training path. Rescue's clientele remains the training
pool's 0/8 bucket (580 problems), exclusively.

**Scheduling call:** the original-checkpoint bench arm measured ~8.65
s/problem on MATH-500 (uncompressed thinking) → 4–6 h for the five suites; it
was killed at 56% of MATH-500 and **deferred to post-continuation** so the
critical-path RL run gets the night. Partial output discarded; the three-way
table will be rebuilt fresh (identical protocol; cross-session caveat to be
noted when reported). One self-inflicted ops repeat: a `pgrep -f pattern |
kill` whose pattern appeared in my own ssh command line killed the remote
shell — the bracket trick (`pgrep -f "[r]un_eval"`) or explicit PIDs, always.

### Run 8 — 2026-08-06 ~21:00 →, the continuation: steps 101–400

Worker (turing, pid in `$RUN/worker.pid`) serves
`.../pilot2_armA/ckpt/step0100`; trainer (spark) runs 300 steps, same Arm A
config, reward at `6c7d0f9` (leak + repeat + normalize_ext fixes — all three
are defect repairs, attested as the continuation's only delta beside the
restart). Run dir `/data/whetstone/runs/stagec/pilot2_armA_cont/`.
**Attested:** bf16-checkpoint restart (fp32 master + AdamW moments existed
only in the finished process), fresh drift baseline, `--b_init 120` carries
the annealed budget floor. Launch scripts now verify GPU-clean via
`--query-compute-apps` with three consecutive <200 MiB readings (Run 6's
lesson, mechanized).

### Run 9 — 2026-08-07 morning, the continuation completes + the global-400 endpoint

300 steps clean (global 101→400). 50-step windows: H **0.927 → 0.785**
(gentle monotone sharpening, never collapse), mtc **3.4% → 1.2%** against the
~6% pre-RL base rate, g(all) → **0.984**, drops 751 all-correct vs 22
all-wrong (saturating from the top), drift monotone to 3.1e-4. The
entropy-thermostat arm (B) was never triggered across 400 total steps.

**Endpoint ladder** (one session, extended grader; init re-screened,
8-draw means; global-150/200 rungs lost to a screen-cache tag collision,
fixed in the regate script afterwards):

| ckpt (global) | strict P@1 | think med | think/correct | ΔP@1 | z |
|---|---|---|---|---|---|
| init | 67.12% ± 2.03 | 218 | 818 | — | — |
| 50 | 69.94% ± 2.09 | 223 | 772 | +2.81 | +2.46 |
| 100 | 71.50% ± 2.22 | 229 | 631 | +4.37 | +3.50 |
| 250 | 72.31% ± 2.02 | 229 | 583 | +5.19 | +4.01 |
| 300 | 72.56% ± 2.56 | 227 | 573 | +5.44 | +4.19 |
| 350 | 74.75% ± 1.89 | 229 | 556 | +7.63 | +5.87 |
| **400** | **75.31% ± 2.14** | 232 | 565 | **+8.19** | **+6.37** | 

Eval g-rate 96.25% → **99.31%**. Monotone gain through 400, think flat
227–232 — the "think rising while ΔP@1 plateaus" alarm never fired. Strict
Pareto letter still unmet (+14 think tokens vs init); think-per-correct −31%.
**Endpoint checkpoint: `pilot2_armA_cont/ckpt/step0300` (global 400).** Not
plateaued → user directs continuation (below).

### Run 10 — 2026-08-07, three-way bench + Phase-2 pool + launch to global 1000

**Original-checkpoint bench arm completed** (same protocol; separate session
from the paired init/step0100 screens — identical config and seed, caveat
recorded): MATH-500 **77.50 ± 0.60** (think med 3,266), AMC23 73.12, Minerva
26.19, AIME24 38.33, AIME25 31.67 — with AIME25's think median AT the 16,384
cap, so the original's hard-suite numbers are cap-suppressed lower bounds.
The compression tax at Stage-B was ~19 pts on MATH-500 (77.5 → 58.6) for ~4×
shorter thinks; RL had recovered +5.65 by global 100. Per-token the student
dominates (64.25% @ 837 vs 77.50% @ 3,266).

**Phase-2 pool (user direction: continue to ~1000; add
`EleutherAI/hendrycks_math` + `AI-MO/aimo-validation-amc`).** Built by
`scripts/build_phase2_additions.py` + the P1 8-gram checker, two independent
contamination gates:

- AMC: 83 → **43 survivors** — the exact gate removed the 40 AMC-2023 twins
  of the `amc23` EVAL suite. Only AMC-2022 trains.
- MATH (train split only, 7 subject configs): −1,024 duplicates already in
  train_30k (DeepMath derives partly from MATH), −50 eval-exact, −75 no
  boxed answer, −5 MATH-500 near-dups caught only by the 8-gram gate →
  **3,780**. Draw: all L4 (1,285) + all L5 (1,733) + 467 L3 + 295 L1–2.
- Combined `phase2_pool.jsonl`: **7,823** = original 4,000 + 3,823. New rows
  carry native levels under their own sources; all are unseen-by-SFT, so the
  memorization read stays within the original 4,000.

Also fixed in passing: four more scripts missing the repo-root sys.path shim
(incl. `check_contamination.py` itself), found by auditing after run_eval's
instance of the same class.

**Chained** (running): re-bucket all 7,823 under the endpoint ckpt (K=8,
T=1.0/1.0, cap 12,288, seen tags carried) → memorization within-level
re-read vs the +5.32 baseline → worker + trainer, **600 steps global
400→1000**, config unchanged (eps 0.2/0.2, λ_TEA 0, 8/step, prefetch,
B floor 120), run dir `pilot2_phase2`.

### Run 11 — 2026-08-08, phase-2 launch state + standing directives

Merged table 8,674 rows; AIME census 43.4% mixed / 52.8% 0-of-8 (partly
cap-inflated — at 12,288 the endpoint model averages ~10.6k output tokens per
AIME rollout, bench p90 12.7–14.5k; **cap 16,384 for AIME-bearing pools is a
Phase-3 boundary decision**, auditor quantifying cap-burn). Curriculum
**4,957 mixed** (2,968/1,004/985 bands). Trainer running global 400→1000,
~100–135 s/step.

Memorization re-read (main table, original 4,000, within level): weighted
delta **+6.29 (SE 1.08) vs baseline +5.32 (SE 1.86)** — aggregate stable
(Δz ≈ 0.45), but the mid-band widened (L5 +9.6→+29.3, L6 +5.1→+16.0, L7
+8.2→+15.1, L9 +2.4→+10.3; small unseen ns). Confounds: curriculum tilt gave
seen problems more RL draws; L1 ceiling compresses its delta. GLM derivation
spot-check owed when GPU frees; phase-2 pool dilutes seen share 60%→28%.

**User directives on record (2026-08-08):** (1) entropy decline during this
RL phase is accepted — observed H drifting to ~0.58–0.75, at the pre-RL
card's mean; log-only, no floor term, no Arm D unless the user reopens it;
(2) the monolithic-generate bucketing limitation (22 h unresumable) spawned a
standalone chunked-resume refactor task for pre-Phase-3.

## Conclusion

(TBD — phase 2 to global 1000 in flight; owed at its endpoint: screens,
memorization re-read under the final model, three-way re-bench, rescue decision,
ROADMAP facts block, §12.6 pins, packet flips)
