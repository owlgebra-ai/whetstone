# activity/ — experiment log and instruction packets

This folder is the **operational record** of the WHETSTONE v2 ("HONE") experiments. The design doc ([whetstone-v2-design.md](../whetstone-v2-design.md)) says *what* and *why*; this folder records *who did what, when, on which machine, with what result*.

## Structure

```
activity/
  README.md            ← this file: conventions + ledger
  packets/             ← instruction packets (P0, P1, …) handed to executing agents
  NNN-short-name.md    ← one file per activity/experiment, numbered chronologically
```

## Rules

1. **One activity = one file.** Number sequentially (`000-`, `001-`, …). Claim the next free number when you start. Never renumber.
2. **Packets are contracts; activity files are journals.** An executing agent reads its packet from `packets/`, does the work, and writes its journal to a *new* `NNN-*.md`. Packets are only edited to fix errors found during execution (note the fix in your activity file) or to flip the status header.
3. **Log as you go, not after.** A failed run is a result — record it with the error and what you changed. Failures are the most valuable entries in this folder.
4. **Exact commands, exact paths, exact commits.** Every run entry must be reproducible from the file alone: git commit of the code, machine, full command line, config values, input paths, output paths.
5. **Artifacts never live in git.** Big outputs go under `/data/whetstone/` (shared between turing and spark via NFS); the activity file records the *path*. Plots and small JSON summaries (< ~200 KB) may be committed next to the activity file in `activity/assets/NNN/`.
6. **Commit the activity file with the code changes of the same work.**

## Activity file template

```markdown
# NNN — <title>

- **Packet:** packets/PX-...md (or "ad-hoc")
- **Status:** in-progress | done | failed | superseded by NNN
- **Machine(s):** turing / spark / mac
- **Code commit(s):** <sha>
- **Started / finished:** YYYY-MM-DD → YYYY-MM-DD

## Goal
One paragraph: what this run is supposed to establish.

## Runs
### Run 1 — YYYY-MM-DD HH:MM
- command: `...`
- config: key hyperparameters actually used (not the defaults you assumed)
- inputs: /data/whetstone/...
- outputs: /data/whetstone/...
- result: metrics, curves (link plot in assets/), verdict
- notes: surprises, deviations from the packet, anything the next agent must know

## Conclusion
What was established. Which gate (F1–F4 / S1–S3) passed or failed. What the next packet should do differently.
```

## Packet status headers

Every packet starts with a status line: `STATUS: draft | ready | in-progress (activity NNN) | done (activity NNN) | blocked (<on what>)`. Flip it when you claim or finish a packet.

## Ledger

| # | File | What | Status |
|---|---|---|---|
| 000 | [000-turing-reset.md](000-turing-reset.md) | Wipe stale projects + old whetstone on turing; machine survey | done |
| 001 | [001-environment-rebuild.md](001-environment-rebuild.md) | P0: rebuild turing + spark envs (vllm 0.26.0 / torch 2.11.0+cu130); verify Qwen3 thinking + d_t scoring | done |
| 002 | [002-data-pools.md](002-data-pools.md) | P1: DeepMath-103K + GSM8K pool, SCA arm, 7 eval suites, standard_eval_300, contamination check | done |
| 003 | [003-preconditions.md](003-preconditions.md) | P2: Qwen3 segment parser + tests, entropy audit (**restoration mode**), calibration probe (dropped v1 system prompt), register card staged | done |
| 004 | [004-register-bakeoff.md](004-register-bakeoff.md) | P3a: register bake-off A (symbolic) vs B (caveman) — **A wins**; v1 chunkwise compression retired for one-shot | done |
| 005 | [005-seed-corpus.md](005-seed-corpus.md) | P3: seed harvest (9,000 rollouts, 77.1% verify) + **two** register corpora (Qwen3 + GLM-5.2 bootstrap); Δlogp retired for a structural gate; **H_pivot = 0.6707** | done |
