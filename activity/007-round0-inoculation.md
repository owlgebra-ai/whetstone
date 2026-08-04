# 007 — P4: Round-0 scorer inoculation and the F1 band-existence gate

- **Packet:** [packets/P4-round0-inoculation.md](packets/P4-round0-inoculation.md)
- **Status:** in-progress
- **Machine(s):** mac (code), turing (all scoring, training, meter tests)
- **Code commit(s):** `4adbabd` → `<this commit>`
- **Started / finished:** 2026-08-03 → 2026-08-03

## Goal

Calibrate the scorer so compact-register tokens read as a low hum while genuine
reasoning leaps still spike, and answer **F1: does that band exist?** If
register-hum and leap-spike are inseparable the whole design pivots to the
prefix/LoRA scorer arm. The product of this packet is a trustworthy measuring
instrument, not a capable model. On PASS, run activity 006's binding
`G_spike` × branch-retention check, which gates P5.

---

## Runs

*(filled in below)*

---

## Findings

*(filled in below)*
