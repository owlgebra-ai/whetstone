# 000 — turing reset + two-machine survey

- **Packet:** ad-hoc (pre-packet housekeeping)
- **Status:** done
- **Machine(s):** turing, spark
- **Code commit(s):** n/a (no code changes)
- **Started / finished:** 2026-08-01 → 2026-08-01

## Goal

Clear turing's full root disk and retire the stale v1 checkout so the Qwen3-1.7B feasibility experiments (design §11) start from a clean, known machine state. Survey both GPU boxes so packet P0 is grounded in reality rather than assumptions.

## What was done

Deleted from `~/workspace/` on turing (user-approved, explicitly named):

| Folder | Size |
|---|---|
| `think_ws` | 323 GB |
| `epic-sft` | 270 GB |
| `Delta-Matryoshka-Encoders` | 262 GB |
| `whetstone` (stale v1 snapshot — not a git repo, broken `.venv` symlinked to a removed python3.12) | 11 GB |

Root fs went from **12 GB free (100%)** to **~866 GB free (57%)**.

## Machine survey (2026-08-01)

### turing — `ssh bajajra@192.168.1.220`
- RTX 5090, **32 GB VRAM**, driver 595.71.05. x86_64.
- Root: 2.1 T LVM/ext4, ~866 GB free after cleanup.
- `/data`: ZFS `tank/data`, 9.9 T, ~4 T free. **HF cache lives here** (`~/.cache/huggingface → /data/cache/huggingface/`).
- Cached models incl. **Qwen3-1.7B, Qwen3-1.7B-Base, Qwen3-8B, Qwen3-4B-Thinking-2507-FP8** — no download needed for the feasibility tier.
- System python is 3.14.4 (too new for the ML stack — P0 uses a uv-managed 3.12).
- No NGC container in use; bare-metal venv is the install path.

### spark — `ssh bajajra@192.168.1.253` (DGX Spark, hostname `spark-f82c`)
- **aarch64** (ARM). GB10, unified memory (`nvidia-smi` reports memory N/A), driver 580.159.03.
- Root NVMe 3.7 T, 1.4 T free. System python 3.12.3.
- **Mounts turing's `/data` via NFS** (`198.18.0.2:/data` — the direct link between the boxes). Both machines therefore share `/data/whetstone/` artifacts and can share the HF cache path.
- Own local HF cache has small Qwen3 models only (0.6B); Qwen3-1.7B must be fetched or read via `/data`.

## Division of labor decided

- **turing (5090):** all training + all rollout generation (fast GDDR7 bandwidth; decode-heavy work).
- **spark (GB10):** the frozen **scorer / reward server** (prefill-only scoring is compute-bound, tolerates GB10's lower memory bandwidth), CPU-heavy data prep, and offline scoring passes. This resolves the single-GPU contention that design §12.5 avoids by assuming 2 GPUs.

## Conclusion

Machines are clean and surveyed. P0 (environment rebuild) is unblocked. Two facts every later packet must respect: **spark is ARM** (different wheel story for vLLM/torch) and **`/data` is the single shared artifact store** (write artifacts there, never to either root disk).
