# 009 — P6 Stage B: assimilation SFT (ZPD band-pass + SED) and the F3 gate

- **Packet:** [packets/P6-stage-b-assimilation.md](packets/P6-stage-b-assimilation.md)
- **Status:** in-progress
- **Machine(s):** turing (baseline evals, training), spark (gate scoring, suite build), mac (code)
- **Code commit(s):** (in progress)
- **Started / finished:** 2026-08-05 → —

## Goal

Train the student — a fresh copy of the **original** Qwen3-1.7B — on the certified
Stage-A teacher corpus with an unprivileged prompt, so the register enters the
weights. Loss is cross-entropy with ZPD band-pass token weights (Diagnosis #1 fix)
plus the SED self-distillation term in restoration mode (Diagnosis #3 fix). Verdict
is **F3**: accuracy within 1 pt of the starting checkpoint at ≤50% of its median
think tokens, with median per-token entropy above the audit baseline.

## Machine state at claim time (2026-08-05)

- turing HEAD `2317e8a`, spark HEAD `65f4dc3`, Mac HEAD `04f9494` — both boxes lag;
  synced as the first step (ROADMAP standing rule / activity 001 gotcha 6).
- **turing's GPU was fully held (31,434 / 32,607 MiB) by the P5 32B teacher server**
  — `vllm serve nvidia/Qwen3-32B-NVFP4 … --port 8000`, PID 2724436 (engine 2724625),
  up 1 d 00:39, reparented to init. Idle: no established connections on :8000 and
  4 CPU ticks over a 2 s window. P5 is done (activity 008, F2 PASS) and Stage B does
  not use the teacher, so this is a leftover. Part 0 cannot start until it is stopped.
- spark: `whetstone-scorer` (`scorer_v1`) live on **:8100** — left untouched all packet.
  **:8101 free** for the π-of-round gate server.
- `/data` 3.9 TB free. Corpora present:
  `stagea_golden/golden_faithfulness.jsonl` (49.5 MB), `stagea_selected/selected.jsonl`.

## Runs

### Run 1 — Part 0a: build `gsm8k_test` (2026-08-05)

The ROADMAP TODO that has slipped since the eval plan was ratified 2026-08-02.

- code: `scripts/build_eval_sets.py` — added suite `gsm8k_test`
  (`openai/gsm8k`, config `main`, split `test`, revision
  `740312add88f781978c0658806c59bc2815b9866`, resolved from the HF API 2026-08-05)
  plus a dedicated `_norm_gsm8k` normalizer and an `EXPECTED_ROWS` build-time assert.
- **Why a dedicated normalizer:** GSM8K's `answer` field is the *whole* reference
  derivation (with `<<48/2=24>>` calculator annotations) and the gold is only the
  tail after the `####` marker on its last line. The generic `_norm_math` path takes
  `answer` verbatim via `_first(rec, "answer", …)`, which would have handed the
  verifier a paragraph and scored the suite at ~0% — a failure that looks like a
  model problem, not a data problem.
- **Gold stored verbatim (stripped only)**, against the packet's word "normalized".
  The module's standing rule is verbatim golds because `verify._normalize` already
  strips the thousands commas GSM8K writes (`1,000`); confirmed by round-trip —
  gold `1,000` verifies against both `\boxed{1000}` and `\boxed{1,000}`, and
  `\boxed{71}` against gold `72` is False. Normalizing at build time would only
  move the provenance of the number, and the module docstring warns it shifts
  measured accuracy.
- `_uid` = `gsm8k_test:<row index>`, matching the convention of the other seven
  suites in this file. Stable because the revision is pinned.
- normalizer unit-tested on the Mac before the build: plain gold, comma gold,
  negative-with-whitespace gold, and the missing-marker reject all behave.

(results pending)

## Conclusion

(pending)
