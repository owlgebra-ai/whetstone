"""Patch a Gemma-4 checkpoint so vLLM can load it.

Background
----------
Gemma-4 uses activation-level KV sharing: config ``num_kv_shared_layers`` names
the last N layers that don't instantiate ``k_norm``/``k_proj``/``v_proj`` and
instead consume the donor layer's K/V at forward time. HF ``save_pretrained``
stores only donor weights; HF reloads via its architecture, but vLLM's
Gemma-4 weight loader expects ``k_norm.weight`` on every layer and raises
``ValueError: Following weights were not initialized from checkpoint``.

This module adds the missing ``k_norm`` entries by cloning them from the
donor layer, following the same sharing rule as
``transformers.models.gemma4.modeling_gemma4.Gemma4Attention`` (for each
consumer layer i, donor_idx = last index j in ``layer_types[:first_kv_shared]``
where ``layer_types[j] == layer_types[i]``).

Usage
-----
Programmatic::

    from whetstone.patches.gemma4_vllm_patch import Gemma4PatchModelForVllm
    Gemma4PatchModelForVllm(ckpt_dir).patch(out_dir)

CLI::

    uv run python -m whetstone.patches.gemma4_vllm_patch --ckpt <in> --out <out>
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import fire
import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file


class Gemma4PatchModelForVllm:
    """Patch a saved Gemma-4 checkpoint for vLLM compatibility.

    The patch adds the missing ``self_attn.k_norm.weight`` tensor for every
    KV-shared consumer layer, cloned from the correct donor layer.

    Parameters
    ----------
    ckpt : str | Path
        Path to the checkpoint directory (must contain ``config.json`` and
        one or more ``*.safetensors`` files).
    """

    def __init__(self, ckpt: str | Path):
        self.ckpt = Path(ckpt)
        if not (self.ckpt / "config.json").exists():
            raise FileNotFoundError(f"no config.json at {self.ckpt}")

        cfg = json.loads((self.ckpt / "config.json").read_text())
        tc = cfg.get("text_config", cfg)
        self.num_layers: int = tc["num_hidden_layers"]
        self.num_shared: int = tc.get("num_kv_shared_layers", 0)
        self.layer_types: list[str] = list(tc["layer_types"])
        self.first_kv_shared: int = self.num_layers - self.num_shared

    def _donor_index(self, consumer_idx: int) -> int:
        prev = self.layer_types[: self.first_kv_shared]
        return len(prev) - 1 - prev[::-1].index(self.layer_types[consumer_idx])

    def _load_state_dict(self) -> dict[str, torch.Tensor]:
        sd: dict[str, torch.Tensor] = {}
        for f in sorted(self.ckpt.glob("*.safetensors")):
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    sd[k] = h.get_tensor(k)
        return sd

    @staticmethod
    def _layer_prefix(sd: dict[str, torch.Tensor]) -> str:
        """Detect the prefix used for layer keys, e.g. ``model.language_model``."""
        sample = next(k for k in sd if ".self_attn.k_norm.weight" in k)
        return sample.rsplit(".layers.", 1)[0]

    def patch(self, out: str | Path, *, copy_extra: bool = True) -> Path:
        """Write the patched checkpoint to ``out``.

        Parameters
        ----------
        out : str | Path
            Destination directory. Created if missing.
        copy_extra : bool, default True
            Copy every non-safetensors / non-index file from the source
            checkpoint (``config.json``, tokenizer, processor, etc.) into
            ``out``.

        Returns
        -------
        Path
            ``out`` as a ``Path``.
        """
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"patching {self.ckpt} -> {out} "
            f"(layers={self.num_layers}, shared={self.num_shared}, "
            f"first_kv_shared={self.first_kv_shared})"
        )

        if self.num_shared == 0:
            logger.info("num_kv_shared_layers=0 — nothing to patch, copying as-is")

        if copy_extra:
            for p in self.ckpt.iterdir():
                if p.name.endswith(".safetensors") or p.name == "model.safetensors.index.json":
                    continue
                shutil.copy(p, out / p.name)

        sd = self._load_state_dict()
        logger.info(f"loaded {len(sd)} tensors")
        prefix = self._layer_prefix(sd)

        added = 0
        for i in range(self.first_kv_shared, self.num_layers):
            target = f"{prefix}.layers.{i}.self_attn.k_norm.weight"
            if target in sd:
                continue
            donor_i = self._donor_index(i)
            donor = f"{prefix}.layers.{donor_i}.self_attn.k_norm.weight"
            if donor not in sd:
                logger.warning(f"donor {donor} missing for consumer {i}")
                continue
            sd[target] = sd[donor].clone().contiguous()
            added += 1
            logger.debug(f"layer {i} ({self.layer_types[i]}) -> donor {donor_i}")
        logger.info(f"added {added} k_norm tensors (total now {len(sd)})")

        out_file = out / "model.safetensors"
        save_file(sd, str(out_file), metadata={"format": "pt"})
        logger.info(f"saved {out_file}")
        return out


def main(ckpt: str, out: str, copy_extra: bool = True) -> None:
    """CLI entrypoint. Patch ``ckpt`` -> ``out`` for vLLM compatibility."""
    Gemma4PatchModelForVllm(ckpt).patch(out, copy_extra=copy_extra)


if __name__ == "__main__":
    fire.Fire(main)
