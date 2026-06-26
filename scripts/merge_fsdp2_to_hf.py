"""§5.4 — Merge FSDP2 sharded checkpoint to a single HF-format model dir.

Loads the base model configuration, then overlays the FSDP2 sharded state
dict (saved by `accelerate` with `fsdp_state_dict_type: SHARDED_STATE_DICT`).
Writes a clean merged HF-format model directory that can be loaded by vLLM
or any HF API consumer.

For multimodal base models that split text and vision configs into separate
classes (TextConfig vs Config TypeError), use merge_fsdp2_to_hf_mm.py instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_sharded_state_dict(checkpoint: str) -> dict:
    shard_bins = sorted(glob(os.path.join(checkpoint, "pytorch_model*.bin")))
    shard_safes = sorted(glob(os.path.join(checkpoint, "*.safetensors")))
    state_dict: dict[str, torch.Tensor] = {}

    if shard_safes:
        from safetensors.torch import load_file
        for sf in shard_safes:
            print(f"[merge] loading {os.path.basename(sf)}", flush=True)
            state_dict.update(load_file(sf))
    elif shard_bins:
        for sf in shard_bins:
            print(f"[merge] loading {os.path.basename(sf)}", flush=True)
            state_dict.update(torch.load(sf, map_location="cpu", weights_only=False))
    elif os.path.isfile(os.path.join(checkpoint, "pytorch_model.bin")):
        state_dict.update(torch.load(
            os.path.join(checkpoint, "pytorch_model.bin"),
            map_location="cpu", weights_only=False,
        ))
    else:
        # Fall back to a single-file checkpoint path.
        state_dict.update(torch.load(checkpoint, map_location="cpu", weights_only=False))
    return state_dict


def _strip_fsdp_prefixes(state_dict: dict) -> dict:
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in (
            "_fsdp_wrapped_module.",
            "flat_param.",
            "_orig_mod.",
            "module.",
        ):
            nk = nk.replace(prefix, "")
        cleaned[nk] = v
    return cleaned


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Merge FSDP2 shards to HF format")
    ap.add_argument("--base_model", required=True,
                    help="HF model id or path (used for config + tokenizer)")
    ap.add_argument("--checkpoint", required=True,
                    help="FSDP2 checkpoint dir (with sharded pytorch_model-*.bin)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="Fail on any state_dict mismatch (default: warn)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[merge] loading base config from {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"[merge] loading shards from {args.checkpoint}", flush=True)
    state_dict = _load_sharded_state_dict(args.checkpoint)
    state_dict = _strip_fsdp_prefixes(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        msg = f"[merge] missing keys: {len(missing)} (first 5: {missing[:5]})"
        if args.strict:
            print(msg, file=sys.stderr)
            sys.exit(1)
        print(msg, flush=True)
    if unexpected:
        msg = f"[merge] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})"
        if args.strict:
            print(msg, file=sys.stderr)
            sys.exit(1)
        print(msg, flush=True)

    if missing or unexpected:
        ok = len(state_dict) - len(unexpected)
        print(f"[merge] loaded {ok}/{len(state_dict)} tensors", flush=True)

    print(f"[merge] writing merged model to {args.output_dir}", flush=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    _maybe_patch_gemma4_for_vllm(args.base_model, args.output_dir)
    print(f"[merge] done", flush=True)


def _maybe_patch_gemma4_for_vllm(base_model: str, merged_dir: str) -> None:
    """Gemma-4 uses activation-level KV sharing; the base checkpoint omits
    k_norm on consumer layers (last N layers, where N = num_kv_shared_layers).
    vLLM's Gemma-4 weight loader expects k_norm on every layer and fails to
    load the merged checkpoint without it.

    This rewrites model.safetensors in-place at merged_dir by cloning donor
    k_norm tensors into the consumer slots. No-op for non-gemma4 bases."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        # Multimodal Gemma-4 nests the text config under text_config.
        tc = getattr(cfg, "text_config", cfg)
        n_shared = getattr(tc, "num_kv_shared_layers", 0)
        model_type = getattr(cfg, "model_type", "") or ""
        if n_shared == 0 and "gemma4" not in model_type:
            return
        if n_shared == 0:
            return
    except Exception as e:  # noqa: BLE001
        print(f"[merge] WARN: gemma4 detection failed: {e}", file=sys.stderr)
        return

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from whetstone.patches.gemma4_vllm_patch import Gemma4PatchModelForVllm

    print(f"[merge] gemma4 detected — applying k_norm patch in-place at {merged_dir}",
          flush=True)
    patched_dir = merged_dir.rstrip("/") + "_vllmpatched"
    Gemma4PatchModelForVllm(merged_dir).patch(patched_dir)
    # Overwrite the merged dir with the patched artifacts so downstream
    # consumers see a single canonical path.
    import shutil
    for p in os.listdir(patched_dir):
        src = os.path.join(patched_dir, p)
        dst = os.path.join(merged_dir, p)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
    os.rmdir(patched_dir)
    print(f"[merge] k_norm patch complete", flush=True)


if __name__ == "__main__":
    main()
