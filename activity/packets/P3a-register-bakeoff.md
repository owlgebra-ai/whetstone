# P3a — Register bake-off: symbolic (A) vs telegraphic/caveman (B)

STATUS: ready
MACHINES: turing (compression generation, entropy pass); spark (Δlogp + scoring passes — vLLM venv, `VLLM_USE_FLASHINFER_SAMPLER=0`)
DEPENDS ON: P2 (done — uses its audit rollouts and scripts); both candidate cards at HEAD
BLOCKS: P3 Part 2 (the seed register corpus is built with the winning card). P3 Part 1 (seed harvest) is card-independent and may run in parallel with this packet.
DELIVERABLES: quantitative verdict on which register the model handles more organically, a ratified-card recommendation, activity journal.

## Objective

Decide the register form **empirically** before P3 commits the seed corpus. The
user's criterion (2026-08-02): *whichever is more organic and natural for the
model to handle, for training dynamics and entropy.* This packet turns that into
five measurements over the same 50 traces compressed under two cards:

- **A — symbolic** ([configs/register_card.md](../../configs/register_card.md)): `⇒ → ✓ ✗ !` connectives, one-op-per-arrow chains.
- **B — caveman** ([configs/register_card_caveman.md](../../configs/register_card_caveman.md)): `so / ok / no / note` word connectives, terse fragments.

Both cards share identical elision rules and structural skeleton (numbered
steps, `chk:`/`check:`, mandatory case verdicts) — the comparison isolates the
connective-tissue choice. Context: v1 drifted into *degenerate* caveman under
compression pressure (values dropped — the audit's failure class); variant B is
*disciplined* caveman and must be held to the same no-leaps standard.

## Inputs (all exist)

- Verified traces: `/data/whetstone/runs/entropy_audit/rollouts.jsonl` — 200
  rollouts, 182 gate-passing. Select the **verifier-correct AND gate-passing**
  subset, then a proportionally level-stratified 50 (`poolutil`), fixed seed.
  Record the 50 `_uid`s to `/data/whetstone/runs/register_bakeoff/subset_uids.json`.
  (16k-budget traces are fine for a register comparison; P3's real harvest uses 32k.)
- Native-trace baselines from activity 003: think median entropy 0.0278,
  p80 0.6923, per-token arrays in `per_token_entropy.npz`.

## Pinned compression prompt (do not improvise this)

`compress_local_versionB.py`'s hardcoded `SYSTEM_PROMPT` is v1's and is
**disqualifying for this experiment**: it prescribes the symbolic notation by
name (`=, ⇒, →, ✓, ⚠, ?` — biases arm A), it **explicitly bans caveman style**
(sabotages arm B), and it uses ⚠ (rejected on token cost, activity 003).

Required change (commit it): make the scaffold card-parametric —
`--card <path>` pastes the card verbatim; the surrounding scaffold becomes this
**neutral text, byte-identical across both arms**:

```
You are a compression engine. Rewrite verbose chain-of-thought reasoning into
the compact register defined by the REGISTER CARD below. The card is the only
authority on notation and style.

Rules:
- Preserve every load-bearing fact, variable, equation, constant, case-split,
  and derivation step. Never fuse steps; never drop an intermediate value.
- Do NOT invent content not in the original.
- If the original contains self-correction ("wait", "actually", "no"),
  preserve the correction and its resolution in register form.
- One to a few lines per chunk; follow the card's step-marker convention.

REGISTER CARD:
{card text, verbatim, including its exemplars}
```

Notes: v1's "no caveman" line is **dropped** (faithfulness is protected by the
never-elide rules both cards carry, and M2/M5 measure it); v1's "preserve with
⚠ or ?" is replaced by the card-agnostic self-correction rule above. Record the
sha of the fully rendered system prompt per arm in each output sidecar — the
two must differ *only* in the card block.

**`enable_thinking=False` is correct here — do not "fix" it.** The compressor
call is a prefill-trick text rewrite (v1 §3.1): we want the compact chunk
emitted directly, not a thinking preamble. The ROADMAP's `enable_thinking=True`
standing rule covers rollouts, scoring, and eval — compression is none of
those. This is the one deliberate exception; note it in your journal so the
next reader doesn't flag it as a bug.

## Procedure

### Step 1 — compress twice (turing)

`compress_local_versionB.py` with the pinned scaffold above: `--card
configs/register_card.md` → `bakeoff_A.jsonl`; `--card
configs/register_card_caveman.md` → `bakeoff_B.jsonl`. T=0.4, single
completion per trace, chunkwise machinery as P3 specifies (verify chunk
alignment on 2 traces per card before the bulk run). Record each card's git sha
in the output header sidecar. Answer segments copied through untouched;
`verify_response` must still pass on every record (hard bug if not).

### Step 2 — metrics

**M1 — Compression (tokens, not chars):** think-segment token count per trace
(tokenizer count, `whetstone/segments.py` masks): median + IQR, A vs B vs
verbose originals. Also % of traces under B_target=600.

**M2 — Δlogp answer-recoverability (spark):** `perplexity_score.py` on both
corpora, v1 threshold. Report pass rates. A corpus that compresses harder but
fails recoverability is installing sloppy reasoning — v1 §3.6's exact lesson.

**M3 — Style-tax size and concentration (spark, the decisive metric):**
teacher-forced scoring of each compact corpus under frozen π_0
(`prompt_logprobs=2` → per-token surprisal + d_t on think tokens).
  - *Size:* mean and p95 surprisal elevation vs the verbose originals' think
    tokens (same scoring pass over originals as control).
  - *Concentration:* apply the §12.3 R-recipe (per-token-type mean/std
    surprisal, min 10 occurrences) to each corpus. Report: proto-R size, and
    **what fraction of total excess surprisal the top-20 types capture**.
    Concentrated (A expected): Round 0 has a clean handle. Diffuse (B risk):
    the inoculation loss has nothing to mask — Round 0 degenerates toward
    full-SFT with its overtraining/infection risk.

**M4 — Entropy profile (turing):** `entropy_audit.py --traces` on both corpora:
median, p80 (H_pivot preview), collapse mass, fork mass — against the native
baseline (0.0278 / 0.6923). The register that keeps the distribution more open
at comparable compression is the entropy-healthy one. Note: the ⚠-token lesson
from design §4.2 applies — judge the *distribution*, not the page.

**M5 — Faithfulness eyeball:** 10 traces per corpus, hand-checked for fused
steps / dropped values / hallucinated shortcuts, verbose-vs-compact side by
side, pasted into the activity file. B's known failure mode (value-dropping
caveman) gets explicit attention.

### Step 3 — verdict rubric

1. **M2 or M5 failure disqualifies** a variant outright (faithfulness is not
   negotiable at any compression ratio).
2. If B's excess surprisal is **diffuse** (top-20 share < ~40%) while A's is
   concentrated (> ~70%) → **A wins** (training mechanics need the handle),
   unless A's *size* is extreme (clean-register p95 gap approaching τ_leap
   scale ≈ 4 nats), which would resurrect Risk 1 — then escalate to the user.
3. If B's tax is small **and** usable-R exists for it → **B wins** (organic +
   lower F1 risk).
4. Anything in between → **hybrid**: B's word connectives on A's skeleton
   (one-line card merge; both cards already share everything else).
5. M1 is a tie-breaker only within 15%; M4 flags a winner-adjustment only if
   one variant's compact entropy collapses far below the other at similar
   compression.

The winning card's compact-corpus p80 from M4 is the **H_pivot preview** (P3
still pins it on the full seed corpus).

## Gotchas

- Orphaned `VLLM::EngineCore` after kills — check `nvidia-smi
  --query-compute-apps=pid,used_memory`, kill by PID (activity 003 gotcha 1/2).
- `source .venv/bin/activate` everywhere; spark scoring needs
  `VLLM_USE_FLASHINFER_SAMPLER=0`.
- The scoring in M3 is per-**type** aggregation: min-occurrence 10 matters at
  n=50 traces — report how many types clear it, and if too few do, bump to 80
  traces before concluding "diffuse".
- Compression prompts differ ONLY in the card text. Same chunking, same
  temperature, same seed ordering — anything else contaminates the comparison.

## Definition of done

- [ ] Both corpora built, verify_response 100% on both, shas recorded.
- [ ] M1–M5 reported in the activity file with the side-by-side table.
- [ ] Verdict stated in bold with the rubric line that decided it; ratified-card
      recommendation for the user (A / B / hybrid, with the merged card drafted
      if hybrid).
- [ ] User ratifies → card status flipped to FILLED → P3 Part 2 unblocked.
- [ ] Activity file `NNN-register-bakeoff.md`; packet status flipped; ROADMAP
      facts block updated with the verdict.
