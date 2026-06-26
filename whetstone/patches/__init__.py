"""Gemma-4 specific patches (Liger fused-CE for the multimodal class, vLLM k_norm patch)."""

from . import gemma4_liger_patch, gemma4_vllm_patch

__all__ = ["gemma4_liger_patch", "gemma4_vllm_patch"]
