# Gemma-4 (E2B / E4B) SFT — gotchas

Hard-won from distilling into `google/gemma-4-E4B-it` (vision SFT, long ~6.5K-token policy prompts) on H100 pods running the **vLLM image** (torch 2.11/cu13). **Pinned stack that works: transformers 5.8.1, liger-kernel 0.8.0, trl 0.27, peft 0.19.1.** E2B and E4B are MatFormer slices of one checkpoint — the attention notes apply to both.

> **TL;DR (2026-05-30):** on the pinned stack above, gemma-4-E4B LoRA SFT trains stably at **global bs=128 (bs≥16)** with sdpa — the old "sdpa+padding→nan, must use batch1" blocker no longer reproduces. Use the [verified config](#verified-working-config-2026-05-30) below.

## Attention backend — the big one

- **Flash-attention is UNUSABLE.** Gemma-4 global-attention layers have **`global_head_dim: 512`** (sliding layers use 256). FA2/FA3 — including `kernels-community/flash-attn3` and `kernels-community/vllm-flash-attn3` — reject it: `FlashAttention forward only supports head dimension at most 256`. Upstream: flash-attention #2427, transformers #45201 (per-layer FA2+SDPA dispatch not yet merged). Confirmed for **both E2B and E4B**.
- **Use `attn_implementation=sdpa`.** But sdpa has a trap (next point). `eager` is numerically fine but materializes full `[B,heads,S,S]` scores for all layers → OOMs at long seq.

## sdpa + padding → `grad_norm: nan` — NO LONGER REPRODUCES on the current stack

> **Re-verified 2026-05-30 (transformers 5.8.1, liger 0.8.0, trl 0.27): training at `bs≥16` is stable. The old `batch1` requirement is gone — train at global bs=128.**

- **History / old symptom:** earlier runs hit **finite loss but `grad_norm: nan` at a padded batch** (then loss→0 / `eval_loss: nan`) when `per_device_train_batch_size>1`. Working theory at the time: with `sliding_window: 512`, a far-right padding query whose entire 512-window is padding becomes a **fully-masked row** → sdpa's fused/mem-efficient backward returns nan (PyTorch #103749, #109517).
- **What we actually found on re-test:** a faithful `SFTTrainer` re-run at **bs=16 (global 128, r256+rslora, sdpa, FSDP2)** does **not** nan. It sails **through the exact same-seed batch that previously nan'd** (same data 0005, `num_tokens≈1.74e6 / epoch 0.138`: old run `grad_norm=nan`, re-run `grad_norm=0.73`), loss descending for 3 full epochs. Isolated probes — pure bf16, autocast fp32-master, and even a deliberate **5987-token pad (≫ the 512 window)** — all stayed finite. So the documented fully-masked-row mechanism does **not** trigger nan on this stack.
- **Recommendation: train at `bs≥16` (global 128).** `per_device_train_batch_size=1` is the old workaround — **no longer needed**, keep only as a fallback if `grad_norm=nan` ever recurs.
- **Root cause of the earlier nan was not isolated** (it was already gone here; deeper RCA was skipped per request). The transformers version was 5.8.1 both before and after, so it was likely a transient liger/trl/runtime state, *not* the transformers version. Nothing in the model/attention/loss code was changed to "fix" it.
- **Still keep `torch.backends.cuda.enable_cudnn_sdp(False)`** for gemma4 (in `sft.py`, guarded by `is_gemma4 and attn_implementation=='sdpa'`) — a *separate*, still-real issue: the cuDNN MHA backend crashes on gemma4's head_dim-512 attention (`Expected mha_graph.execute(...).is_good() to be true, but got false`; high workspace fails under ~80GB peak). Shape-dependent → fine ~10 steps then a long sequence routes to cuDNN and dies. Disabling forces math/mem-efficient (handle 512).
- `group_by_length` is **rejected by `SFTConfig` in trl 0.27** (`unexpected keyword argument`) — would need a custom `LengthGroupedSampler`. Not needed now that bs>1 is stable.
- `seq-len multiple of 16` is unrelated.

## Verified working config (2026-05-30)

`gemma-4-E4B-it`, LoRA **r=256 / lora_alpha=256 / `use_rslora=True`**, `attn_implementation=sdpa`, **FSDP2** (`fsdp2_config.yaml`: FSDP2 + `SHARD_GRAD_OP` + `NO_WRAP`), **global bs=128** (`per_device_train_batch_size=8` × accum 4 on 4×H100 — pd=8 is the max that fits, pd=16 OOMs at ~80GB), lr 2e-5, `constant_with_warmup`, 3 epochs, `freeze vision`, `max_seq_length 8192`, liger patch on. Loss **1.29→0.66**, token-acc 0.69→0.80, no nan, ~70/80 GB/GPU.

### FSDP for full fine-tuning gemma4 (round-2 full-FT, 8B on 2×H100)
- **`fsdp_auto_wrap_policy: NO_WRAP`** — per-layer `TRANSFORMER_BASED_WRAP` (wrapping `Gemma4TextDecoderLayer`) **breaks gemma4** with `KeyError: 'sliding_attention'`: gemma4 builds a per-attention-type mask dict at the top level, and per-layer FSDP wrapping disrupts that lookup. So NO_WRAP is required.
- NO_WRAP gathers all ~16GB params at once → ~80GB peak on 2×H100 (tight but fits; full-FT 7.77B trainable runs without CPU offload). Combined with cuDNN-SDPA-disable above, mem-efficient attention fits in the remaining room. Set `fsdp_sharding_strategy: FULL_SHARD`, `fsdp_cpu_ram_efficient_loading: false` (true triggers the `device_mesh` AttributeError on the frozen-siglip params).
- Net working full-FT combo: **FULL_SHARD + NO_WRAP + cpu_ram_efficient_loading:false + cuDNN-SDPA-disable + freeze vision + batch1**.

## Precision

- **bf16, not fp16.** fp16 has its own Gemma-4 nan: the attention mask fill `-1e9` exceeds fp16 max (65504) → `-inf` → nan (Unsloth note). bf16 (fp32 range) is safe.
- **fp32-master AMP (load fp32 + `mixed_precision bf16`) does not fit for full-FT** at long seq: doubles model memory (16→32 GB) and under the required `NO_WRAP` FSDP it OOMs at `per_device≥2` (the whole bf16 model is all-gathered at once). Confirmed pd=2 OOM (+2.5 GB over) / pd=1 fits. For full-FT keep plain bf16 (no fp32 master) or bs=1. LoRA doesn't need it.

## Liger kernel (memory — required at any real batch)

- Gemma-4 isn't auto-patched by `use_liger_kernel=True` for the *multimodal* `Gemma4ForConditionalGeneration` (only `gemma4_text` is in liger's registry). Without the manual patch, the full `[B, seq, 262144]` logits materialize → OOM (~50GB at batch16). Keep `walrus.train.gemma4_liger_patch` (`patch_gemma4_liger=True`).
- liger **0.8.0** `unpack_cross_entropy_result` returns a **4-tuple** `(loss, z_loss, token_accuracy, predicted_tokens)` — unpack 4, not 3.
- Gemma-4 has **`final_logit_softcapping: 30`** → compute the fused loss in **fp32** (upcast `hidden_states`/`lm_head.weight`) or the softcap + log-sum-exp overflow nans in bf16/fp16. (Done in the patch.)
- **HBM-bandwidth gotcha (chunked CE):** the fused CE chunks the *token* dim and re-reads the lm_head weight once per chunk — `num_chunks ≈ ceil(V/H) ≈ 103` (capped by tokens → ~52 for a bs=16 microbatch). With the fp32 upcast that's **~52 × 2.68 GB ≈ 139 GB HBM read per microbatch**, ~independent of how many tokens carry loss: it matmuls *every* position (prompt+pad), and `-100` only zeroes the loss, not the matmul. Only ~1% of rows (completion tokens) actually need it. If bandwidth/throughput matters, gather completion-token hidden states (`shift_labels!=-100`, ~500 rows) and do a **plain (non-chunked) CE** — logits `[~500, 262144]` ≈ 0.5 GB fit fine → weight read **once** (~3 GB, ~40× less HBM). Note: just gathering while *still* chunking does **not** help (num_chunks even rises). Not applied — the run trains fine as-is.

## KV-sharing + `use_cache=False`

- `gradient_checkpointing=True` forces `use_cache=False`, which on early transformers broke Gemma-4's KV-sharing (`num_kv_shared_layers` 18/E4B, 20/E2B) → garbage attention → garbage gradients. **Fixed in transformers ≥ 5.5.2** (PR #45312). Use ≥5.5.2 (we run 5.8.1).

## LoRA

- **LoRA `exclude_modules` — gemma4's projectors are `embed_vision` / `embed_audio`, NOT `multi_modal_projector`.** The old regex `.*(vision_tower|audio_tower|multi_modal_projector).*` matched *neither* projector (verified: `embed_vision.embedding_projection` → `excluded=False`), so `all-linear` silently put LoRA on **both** projectors — including the dead `embed_audio` (no audio inputs → no gradient → wasted params + DDP unused-param hazard). Use `exclude_modules=".*(vision_tower|audio_tower|embed_audio).*"`: freezes both encoder towers + the unused audio projector, while **intentionally keeping the vision projector `embed_vision` trainable** (cheap, helps image grounding — a sensible VLM-SFT choice). To freeze the vision projector too, add `embed_vision` to the alternation.
- Gemma-4 wraps linears in **`Gemma4ClippableLinear`** (inherits `nn.Module`, not `nn.Linear`). PEFT `all-linear` attaches fine on tf 5.8.1/peft 0.19.1, but see merge gotchas. (`all-linear` excludes `lm_head` → it stays frozen.)
- **rslora + `alpha≥rank`:** for r=256 set `lora_alpha≥256` (alpha<rank under-scales). rslora scaling is `alpha/√r` (=16 for alpha=256, r=256) → large early `grad_norm` (~95) that's clipped by `max_grad_norm=1.0` and settles (~9 by epoch 1) — finite, trains fine. `use_rslora` is now a `sft.py` runner arg (passed into `LoraConfig`).

## Env on the vLLM image (no SFT image)

- Install (**pin to the known-good stack**): `pip install fire loguru datasets tensorboard 'transformers==5.8.1' 'liger-kernel==0.8.0' 'trl==0.27.0' 'peft==0.19.1'`. The vLLM image already ships transformers 5.8.1 — **do not let installs downgrade it** (gemma4 needs ≥5; ≥5.5.2 for the KV-sharing fix). These exact pins are the combo verified stable at bs≥16 (2026-05-30); unpinned `trl`/`liger-kernel` can pull versions that reintroduce the older grad_norm=nan / API mismatches.
- If you install `kernels`, **pin to the transformers range** (`kernels>=0.12,<0.13` for tf 5.8.1). The latest `kernels` (0.15.x) makes `transformers` fail to import (`LayerRepository ... revision/version` error). For sdpa you don't need `kernels` at all.
- **Zombie processes:** an OOM'd / killed DDP run leaves python procs holding the full GPU → subsequent runs hit *false* OOMs (`77GB in use, 319MiB by PyTorch`). Always `pkill -9 -f walrus/train/sft.py; pkill -9 -f "accelerate launch"` and verify `nvidia-smi` shows 0 MiB before relaunching.
- Use **DDP** (not DeepSpeed ZeRO-3) for LoRA — ZeRO-3 silently corrupts later-layer adapters (empty tensors). e4b (~8B) fits per 80GB GPU, so DDP is fine and gives clean standard adapter checkpoints.

## Merge / vLLM serving (do at merge time)

**Standard recipe (merge → serve → eval) — use this; details below are the why.** Merge and eval are CPU/local `walrus` modules; `sft.py` writes `adapter_config.json` to the run dir so the merge needs no LoRA-config args.
```bash
# 1. FSDP2 distcp checkpoint -> vLLM-ready dir (consolidate + merge + k_norm patch). Reuses the
#    adapter_config.json sft.py wrote at the run dir; falls back to --lora_alpha/--use_rslora/--exclude_modules.
uv run python -m walrus.train.merge_fsdp_lora \
  --ckpt_dir <run>/checkpoint-N --base_model <base> --out /tmp/merged_vllm
# 2. serve (set CUDA_VISIBLE_DEVICES/--port to share a pod between two servers)
CUDA_VISIBLE_DEVICES=0 bash charts/app/scripts/vllm_gemma4_merged.sh /tmp/merged_vllm e4b_eval --port 8000
# 3. flag-decision recall/precision per L1 vs teacher (add an ENDPOINTS entry for the served name first)
uv run python -m walrus.eval.flag_recall --model <ENDPOINTS_key> --test_dir <0006_.../test> --out_dir /tmp/eval_out
```

- **FSDP2 LoRA checkpoints are sharded distcp, NOT PEFT adapter dirs.** With `fsdp_state_dict_type: SHARDED_STATE_DICT` (the `fsdp2_config.yaml` default), each `checkpoint-N/` holds `pytorch_model_fsdp_0/*.distcp` (LoRA-only keys, **no `adapter_config.json`**) — `peft.from_pretrained` can't load it directly. `walrus.train.merge_fsdp_lora` does the whole thing (consolidate → adapter → merge → k_norm patch). It reuses the **`adapter_config.json` that `sft.py` now writes at the run dir** (rank 0, on `get_peft_model`) — so r / lora_alpha / use_rslora / exclude_modules come from the real training config, no reverse-engineering. For older checkpoints that predate that, it falls back to args (`--lora_alpha`/`--use_rslora`/`--exclude_modules`; r auto-detected from the adapter). (Or set `fsdp_state_dict_type: FULL_STATE_DICT` for directly-loadable checkpoints — it all-gathers on save.) Verified on checkpoint-36: 690 LoRA keys → clean merge, 0 key warnings. **Do the merge on a pod with the pinned stack (tf ≥5.5.2)** — a host with tf 5.3.0 is below the KV-sharing fix and mis-merges.
  - *Consolidation how-to:* `torch.distributed.checkpoint.format_utils.dcp_to_torch_save(<ckpt>/pytorch_model_fsdp_0, out.pt)` (offline, single-process). `torch.load(out.pt)` returns `{"model": {…}}` → **descend into `["model"]`**. Inner keys are **already in PEFT adapter-file form** (`base_model.model.model.language_model.…lora_A.weight`, **no `.default.` infix**) and stored **fp32** → cast to bf16, `save_file` → `adapter_model.safetensors`, write a matching `LoraConfig.save_pretrained()`, then `PeftModel.from_pretrained` + `merge_and_unload`. Use a **per-checkpoint** consolidated `.pt` path — never reuse a stale one (it silently merges the wrong run). Getting `exclude_modules` exactly right matters: a *broader* exclude only adds harmless zero-init no-op modules, but a *wrong* exclude that drops a trained module (e.g. `embed_vision`) silently discards its deltas — verify a clean load (0 unexpected keys).
- **`Gemma4ClippableLinear` unwrap / `.weight`→`.linear.weight` remap** is reported necessary on some stacks (older transformers, raw state-dict merges). **On transformers 5.8.1, `peft merge_and_unload()` + `save_pretrained()` already emits base-matching `.weight` keys — no remap needed** (verified: 0 clippable keys, vLLM loads clean). What you DO still need: apply the **k_norm patch** (`gemma4_vllm_patch` logic) to add the 18 KV-shared-layer `k_norm` tensors. (The base on disk carries redundant k_proj/v_proj on shared layers 24–41 that the re-saved merge omits — expected; vLLM uses the donor K/V.)
- **vLLM has no runtime LoRA** for `Gemma4ForConditionalGeneration` (`does not support LoRA yet`) → **merge before serving**.
- Do **not** merge under DeepSpeed ZeRO-3.
- **EOS / turn terminator (corrected for gemma-4):** the tokenizer has **no `<end_of_turn>` string** — it tokenizes to `<unk>`. The turn terminator is **`<turn|>` = id 106**, and the shipped `generation_config.eos_token_id = [1, 106, 50]` already includes it → vLLM stops correctly with the **stock** config (verified: the chat template closes assistant turns with id 106; **no fix needed**). Do NOT set `eos_token="<end_of_turn>"` (resolves to `<unk>`). Gemma-4 ships a real `<pad>` (id 0); never set `pad_token=eos`.
- **Serving a merged e4b LoRA:** serve the **bf16 merge with no `--quantization`** (fp8 is only for the stock base). Working flags (mirror `vllm_gemma4_e4b_auto_sft_*.sh`): `--gpu-memory-utilization 0.9 --max-model-len 8192 --max-num-batched-tokens 32768 --async-scheduling --enable-prefix-caching --limit-mm-per-prompt '{"image":1,"video":0}' --allowed-local-media-path /mnt/nfs`. Default attn backend is fine (vLLM's native gemma4 path handles head_dim 512 — no FA needed). For plain JSON-classification eval you do **not** need `--enable-auto-tool-choice` / `--tool-call-parser gemma4` / `--reasoning-parser gemma4` (those are agentic/thinking-mode only).
- **Killing a vLLM server leaves a GPU-holding zombie.** `pkill -f api_server` kills the parent, but the **`VLLM::EngineCore` child survives holding the full ~74 GB** — and its cmdline matches **neither** `vllm` nor the model path, so `pkill -f vllm` / `pkill -f <model>` miss it. Grab the PID from `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` and `kill -9 <pid>`; verify the GPU shows 0 MiB before relaunching. Hit on every served-model **swap** on a reused pod (`CUDA_VISIBLE_DEVICES=0` confines the new server to GPU0, but the stale EngineCore still pins its GPU until killed by PID).
- **Eval = flag-decision recall/precision vs teacher, NOT token-acc.** `walrus.infer.run_inference` consumes the trl distill format natively: `--image_column images` (a list col — it auto-unwraps `[0]`) + the conversational `prompt` column as-is + `--schema …hr2.eval_split.schemas.ReasonFlag` ({reason, flag}, matching the trained output). Parse `flag` from `raw_output`, compare to the teacher `flag` in `completion[0].content`. Watch for parse failures: a higher-LR / under-regularised checkpoint can emit **degenerate repetition loops** that overflow `max_tokens` → unparseable JSON (a real instability signal — count them, don't silently drop). For on-pod eval against `localhost` (two servers sharing one pod, ingress maps only one), copy `src/walrus` to the pod + `PYTHONPATH`; the vLLM image already has openai/datasets/fire/loguru/PIL.
