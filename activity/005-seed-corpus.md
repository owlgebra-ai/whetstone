# 005 — P3: seed harvest + seed register corpus

- **Packet:** [packets/P3-seed-corpus.md](packets/P3-seed-corpus.md)
- **Status:** in-progress
- **Machine(s):** mac (code), turing (harvest + compression), spark (Δlogp, pre-flight)
- **Code commit(s):** `ed5cd8e` → `<this commit>`
- **Started / finished:** 2026-08-02 → —

## Goal

Build the only two corpora that exist before the teacher is trained (design §1,
preconditions 3–4): a blind verifier-filtered seed harvest of native verbose
traces, and the seed register corpus produced by one prompted-compression pass
under the ratified card. Then pin **H_pivot** from the compact-register entropy
histogram and write the Round-0 measurement sets P4 consumes.

---

## Infrastructure changes made for this packet

Three changes to how generation runs, all driven by the scale jump from the
bake-off's 50 traces to this packet's 9,000 rollouts.

### 1. vLLM server/client instead of in-process batches

`harvest.py` and `compress_local_versionB.py` both gained a `--server` mode that
issues **one `/v1/completions` request per unit of work** against a resident
`vllm serve`, with a bounded in-flight window.

The offline `llm.generate(batch)` API is a **barrier**: the call does not return
until the batch's slowest member finishes, so one 32k-token rollout idles every
other slot in its batch. Independent requests let continuous batching refill a
slot the moment it frees. The server also outlives the client, so a resume costs
no model load — which matters when the run is expected to be interrupted.

`/v1/completions` is used, never `/v1/chat/completions`: prompts are rendered by
`apply_chat_template` in the calling script, which is what keeps the blindness
contract (v1 §2) and `enable_thinking` (ROADMAP rule 4) auditable and identical
to the offline path. vLLM 0.26.0 supports `return_token_ids: true`, so
`completion_token_ids` survive the HTTP hop — necessary because
`whetstone.segments` is token-level by design and re-tokenizing decoded text
does not round-trip at the `<think>` boundary (design §12.1).

Server command (turing):

```bash
source .venv/bin/activate && vllm serve Qwen/Qwen3-1.7B --port 8000 \
  --max-model-len 34816 --max-num-seqs 64 --gpu-memory-utilization 0.90
```

Reported **GPU KV cache size: 226,800 tokens**, max concurrency 6.51× at full
34,816-token requests.

### 2. Crash-safe resume (`whetstone/runio.py`)

Factored out of `harvest.py` and shared with the compressor:

- **`repair_tail()`** truncates a torn trailing line *before* the file is
  reopened for append. Skipping an unparseable line on read is **not**
  sufficient: the next append lands on that same line, fusing garbage with a
  good record and silently losing the good one on every future pass. This was a
  latent bug in v1's resume machinery.
- **`checkpoint()`** flushes + `fsync`s the corpus, then atomically replaces a
  `<output>.progress.json` sidecar. Data is synced before the sidecar so the
  sidecar can never advertise records that are not on disk. Runs on every exit
  path, including `KeyboardInterrupt`.
- Failed requests go to `<output>.failed.jsonl` as an audit trail and are
  deliberately kept **out** of the corpus, so resume retries them.

### 3. Per-rollout seeds

A single `SamplingParams.seed` shared across a K-sample group makes every
candidate for a problem byte-identical — the group collapses to K copies of one
trace. Seeds are now derived as `sha1(uid:k:seed)[:8]` in both execution paths:
candidates stay independent, and a resumed run regenerates exactly what the
interrupted one would have.

---

## Runs

### Run 1 — seed subset selection (turing, CPU) — 2026-08-02

```bash
.venv/bin/python scripts/select_seed_subset.py \
  --pool /data/whetstone/data/pool/train_30k.jsonl \
  --out_dir /data/whetstone/corpora/seed --frac 0.15 --seed 0
```

4,500 of 29,998 problems, proportional level strata preserved end to end
(including level 10's single row):

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| pool | 6002 | 38 | 767 | 1510 | 4833 | 7488 | 3788 | 4303 | 1256 | 13 |
| selected | 901 | 5 | 115 | 226 | 725 | 1124 | 569 | 646 | 188 | 1 |

Sources: deepmath 3,600 / gsm8k 900. Outputs
`/data/whetstone/corpora/seed/{subset_uids.json,subset.jsonl}`. The uid list is
written first and is the resume contract (v1 §2.5).

### Run 2 — blind seed harvest (turing) — 2026-08-02

```bash
source .venv/bin/activate && python -u scripts/harvest.py \
  --input  /data/whetstone/corpora/seed/subset.jsonl \
  --output /data/whetstone/corpora/seed/seed_harvest.jsonl \
  --model Qwen/Qwen3-1.7B --server http://127.0.0.1:8000/v1 \
  --K 2 --temperature 0.9 --top_p 0.95 --max_tokens 32768 \
  --concurrency 96 --flush_every 20 --no_system_prompt \
  --seed 0 --shuffle --worker_id 0 --n_workers 1
```

No system prompt and `--prefill_think` False, both now defaults (activity 003).
Gemma scrub confirmed: `harvest.py` imports no `whetstone.patches.*` and builds
prompts via `apply_chat_template`; the only Gemma references left are docstrings
and the unused `--assistant_model` flag.

**Resume test (packet-mandated), performed twice:**

1. `kill -9` of the client at 124 records. The kill landed between writes, so the
   file was already newline-terminated — the common case, and not a test of
   anything. So:
2. A partial record was appended by hand to simulate a kill *mid-`write()`*, and
   the client restarted. It reported
   `[resume] repaired torn tail: dropped 67 B of a partial record from the
   previous run`, then `[resume] 124 (uid, k) pairs already done` and
   `[load] 8876 rollouts to generate` — 9000 − 124, no duplicated and no skipped
   work. The server drained to `Running: 0, Waiting: 0` when the client died and
   was healthy for the restart.

**Deviation — restarted once more at 2,059 records to add `--shuffle`.** See
finding 2 below.

Note: SIGINT does not stop this client promptly (asyncio blocked in
`as_completed` with ~7k pending tasks); it needed SIGKILL after 40 s. Not a
correctness problem — the file is line-buffered and `repair_tail` covers the
torn-write case — but use `kill -9` and expect the in-flight window to be lost.

---

## Findings

### 1. `max_tokens 32768` nearly eliminates cap-hits

Activity 003 measured **9%** of rollouts hitting the 16,384-token cap. At 32,768
the cap-hit rate is **0.4%**, and the segment-parser gate passes **99.6%** of
rollouts (the only failures are `missing_think_close`, i.e. the cap-hits). The
generous budget converts what was ~9% pure waste — traces with no answer segment,
useless to both the verifier and Part 2 — into usable traces.

### 2. Pool file order is level-clustered; the harvest must shuffle

Pool JSONLs are sorted by `_uid`, so every `deepmath:*` problem is submitted
before any `gsm8k:*` one — and gsm8k **is** the entire level-1 stratum. The
partial yield report at 2,043 rollouts covered levels 2–10 and contained
**zero** level-1 rollouts.

Nothing is lost in a run that completes, but a level-stratified subset harvested
in level-clustered order has an unrepresentative prefix, so an interrupted run
is far less usable than its record count suggests. `harvest.py --shuffle`
(seeded, applied to the remaining work — resume is a set difference on
`(uid, k)`, not positional) fixes it.

It also *helped throughput*: 8.1 → 12.9 rollouts/min, because short level-1
traces now interleave with long deepmath ones and free KV slots faster.

### 3. Card rendering silently re-admitted two non-notation sections

`render_card()` dropped sections by **title substring**. Ratification renamed
the headings — `Self-check before flipping` → `4. Self-check (completed at
FILLED…)`, and the `Exemplars 3–8` ⟨PENDING⟩ stub became real exemplars — so the
bake-off's drop list stopped matching, and **§1.6 (tokenizer audit) and §4
(self-check checklist) leaked into the compressor's system prompt**.

Both are meta about the card, not register spec; §4 even states "Combinatorics:
none — known gap", which is a note to the card's authors and nonsense as an
instruction to a compressor. Drop patterns are now anchored on section
**numbers**, which are stable, and `render_card()` returns what it dropped so
the sidecar records it.

This is the class of failure that produces no error and degrades the corpus
quietly. Worth a standing rule: **config selected by prose title is config that
will drift.**

### 4. Compression prompt, pinned for this packet

`scripts/compress_local_versionB.py --dump-prompt` (CPU-only pre-flight, run on
spark):

| field | value |
|---|---|
| card blob sha | `de176e8044ab398465e7fd330f98b7c70bd399b0` |
| card raw sha1 | `fc143757ef103f21874821a67e5c11b9b164b04b` |
| card rendered sha1 | `e20ce28e111c646358fb745d2699e6a8e2f3804e` |
| **rendered prompt sha1** | `c6656806ba8de84da2b1eb5e543bd006db7b286c` |
| rendered prompt | 10,867 chars / **3,839 tokens** |
| **boundary-token hits** | **0** |
| dropped | §1.6 tokenizer audit, §2 structural whitelist, §4 self-check |
| kept | §0, §1.1–1.5, §3 + exemplars 1–5 |

**P3 gotcha 1 is enforced in code, not by inspection:**
`assert_no_boundary_tokens()` refuses to run the compressor if the rendered
scaffold+card encodes `<think>` / `</think>` / `<|im_start|>` / `<|im_end|>` /
`<|endoftext|>`. It is checked on the *system prompt* rather than the templated
prompt, because the chat template legitimately emits `im_start`/`im_end` at its
own structural positions — from text we wrote the budget is exactly zero.

### 5. The ratified card made the model put `\boxed{}` **inside** the think segment

Found by a 20-trace dry run of Part 2 against the live server, before the real
compression pass. **7 of 16 (44%) compact rewrites carried a
`**Final Answer** \boxed{...}` trailer inside `compact_think`** — a card §1.5
violation, the contamination Stage C's answer-segment KL exists to prevent, and
precisely the failure that disqualified **arm B** in the bake-off, now appearing
in arm A.

The cause is the card's own ratification. Activity 004 required the exemplars be
un-indented; they became ` ``` ` fenced blocks. The model imitates the fence
faithfully — opens it, writes the register, **closes it**, and then continues in
its native markdown voice:

```
⇒ Yes
```                 ← model closes the fence …
**Final Answer**    ← … and keeps going
\boxed{Yes}
```

v1's `clean_oneshot` stripped a fence only at the very *end* of the text, which
is exactly the case that does not occur. The closing fence is now treated as a
hard end-of-register marker wherever it appears, with a fallback cut at an
un-fenced final-answer flourish (1 of the 7 had no fence at all, just
`### **Final Answer**` + `$$\boxed{a}$$`).

Re-run on the same traces with the fix: **0/10 `think_has_boxed`**, 5 trailers
cut, and register marker density **4.23 per 100 compact think tokens** against
the bake-off's 4.74 — the cleaning removes the trailer without damaging the
register. `_finalize` now hard-fails on any surviving `\boxed{}` in think, the
same standing as the pre-existing answer-segment leak check, and the per-record
`clean_flags` make the trailer rate visible as card feedback instead of
disappearing into the cleaner.

Same function also had a latent dedent bug: `text.strip()` removed the *first*
line's indentation, so min-indent computed as 0 and uniformly-indented blocks
were never dedented — silently negating half of activity 004's un-indent fix,
whose whole point was that indentation whitespace was 8.2% of arm A's excess
surprisal.

**Two findings in this packet (3 and 5) are both "the card was edited, and a
downstream text-processing rule that keyed on its old shape stopped working
silently."** Neither produced an error. Worth treating as a standing hazard for
every future card edit.

### 6. The letter-tag ban worked — runaways are gone

Bake-off arm A had a **10% runaway rate** (cap-hit at `max_tokens_oneshot=2048`),
traced to card §1.3's `(A)`, `(B)`, … sub-result naming: the model exhausted
single letters and rolled into `AAA`/`BBB` loops. Ratification banned the
scheme. On the 20-trace dry run: **cap-hit 0.0%**, stalled-chunk rate 0.00, 0%
of traces ≥50% stalled. Activity 004's required edit 1 is confirmed effective.

### 7. Register adoption is lower than the bake-off — under investigation

Measured with the bake-off's own metric code (`bakeoff_metrics.py`, so the
numbers are directly comparable to activity 004):

| | bake-off arm A (T=0.4) | P3 dry run (n=20) |
|---|---|---|
| verbose think tokens, median | 5,404 | 8,796 |
| compact think tokens, median | 176 | 209 |
| compression ratio, median | 0.043 | **0.025** |
| % under `B_target = 600` | 80% | **95%** |
| **markers /100 think tok** | **4.74** | **2.16** |
| cap-hit (runaway) | 10% | **0%** |
| `verify_response` | 50/50 | 20/20 |

Per-marker, over 20 traces: `goal:` in 19, `let ` in 16, `⇒` in 18 — the
*structural* register is installed. What is thin is the **verification and
case-split vocabulary**: `chk:` in 2/20, `✓` in 2/20, `case ` in 0/20, `✗` in
0/20. Step bodies also carry more English connective prose ("Since x₁ is real
and between 1 and 2", "Therefore …") than the card's exemplars do.

Two candidate causes, and they are separable:

1. **Input distribution.** These traces are 8,796 verbose tokens at the median
   against the bake-off's 5,404 (a 32k cap vs 16k), and they compress 1.7×
   harder (0.025 vs 0.043). Harder summarization of longer input plausibly
   produces prose-ier step bodies.
2. **The card itself changed.** Ratification added exemplars 3–5, and those
   have measurably lower marker density than the two the bake-off card carried:

   | exemplars | markers / 100 chars |
   |---|---|
   | 1–2 (bake-off card) | **5.67** |
   | 3–5 (added at ratification) | **4.14** |

   Exemplar 5 alone is 2.44 — close to what the model now produces.

A controlled run is in flight: **the bake-off's own 50-trace subset,
recompressed with the ratified card**. Same inputs, different card, so it
isolates cause 2 from cause 1.

**This does not put the bake-off verdict in doubt** — 2.16 is still ~9× arm B's
0.24, and the register is plainly installed. It is reported as **register-card
feedback for the user** (which packet P3 Part 2 asks for explicitly), not as a
threshold to nudge, and the corpus build proceeds either way.

---

## Runbook for the rest of the packet

Executed in this order once the harvest lands. Steps 6–9 move to **spark**:
they are teacher-forcing scoring passes (exactly what the GB10 is for) and
they cannot share turing's GPU with the resident vLLM server at
`--gpu-memory-utilization 0.90`.

| # | box | command |
|---|---|---|
| 1 | turing | `verify_harvest.py --input seed_harvest.jsonl --output seed_verified.jsonl` |
| 2 | turing | `harvest_report.py` → full per-level yield table |
| 3 | turing | `select_compression_inputs.py --n 1500` |
| 4 | turing | `compress_local_versionB.py --mode oneshot --server … --temperature 0.4` |
| 5 | turing | stop the vLLM server (frees the GPU) |
| 6 | spark | `perplexity_score.py` (no `--keep-only`: annotate all rows, so the histogram has the failures too) |
| 7 | spark | `finalize_seed_register.py` → `seed_register.jsonl` |
| 8 | spark | `entropy_audit.py --traces … --completion_field completion` → **H_pivot = p80** |
| 9 | spark | `build_round0_sets.py` → 3 splits + verbose control + Δlogp plot |
| 10 | any | `show_bakeoff_examples.py --mode faithful --n 5` → the packet's hand-inspection deliverable |

`--n 1500` rather than 1,200: the dry run compresses 1.7× harder than the
bake-off, and harder compression plausibly lowers the Δlogp pass rate below the
66% the 1,200 figure assumed. The compression pass is ~30 min of GPU, so the
headroom is nearly free.

---

## Partial results (harvest at 2,043 / 9,000 rollouts)

Sanity check against the activity-003 probe before burning the rest of the GPU
time, as the packet requires.

| level | problems | rollouts | verify | solve@K | gate | usable | cap-hit | think med | answer med |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 2 | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 4115 | 585 |
| 3 | 32 | 64 | 68.8% | 68.8% | 100.0% | 68.8% | 0.0% | 4770 | 814 |
| 4 | 56 | 112 | 80.4% | 85.7% | 100.0% | 80.4% | 0.0% | 4592 | 729 |
| 5 | 211 | 421 | 77.0% | 83.9% | 99.8% | 77.0% | 0.2% | 7204 | 808 |
| 6 | 305 | 608 | 76.3% | 85.6% | 99.5% | 76.3% | 0.5% | 7572 | 831 |
| 7 | 177 | 354 | 73.5% | 83.0% | 99.2% | 73.2% | 0.9% | 8725 | 868 |
| 8 | 178 | 356 | 61.2% | 71.4% | 99.4% | 61.0% | 0.6% | 9080 | 831 |
| 9 | 61 | 122 | 51.6% | 63.9% | 100.0% | 51.6% | 0.0% | 9376 | 806 |
| 10 | 1 | 2 | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 7677 | 866 |
| **ALL** | **1023** | **2043** | **71.8%** | **80.5%** | **99.6%** | **71.7%** | **0.4%** | **7800** | **828** |

**Verdict: healthy, run continues.** 71.8% against the probe's 73% is 1.2 points
under, inside the expected ~3-point extraction-shape loss (activity 003 finding
9). The U-shape reproduces: level 9 at 51.6% vs the probe's 50%. `verify` is per
rollout; `solve@K` is per problem (either candidate correct); `usable` is
verifier-correct **and** parser-gate-passing — the pool Part 2 selects from.

*(Level 1 is absent from this table for the reason in finding 2, not because it
failed. It appears from the shuffled restart onward.)*

---

## Conclusion

⟨pending — written when the harvest, register corpus and H_pivot are done.⟩
