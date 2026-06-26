"""Monkey-patch Gemma4ForConditionalGeneration forward with Liger fused linear cross entropy.

Gemma-4 is not in liger's ``MODEL_TYPE_TO_APPLY_LIGER_FN`` registry, so
``use_liger_kernel=True`` in ``SFTConfig`` does not patch it. The stock
transformers forward materializes the full ``[batch, seq, vocab≈262k]``
logits on every step, wasting ~17 GB at bs=4 seq=4096 on E2B. This patch
replaces the forward with one that calls ``LigerForCausalLMLoss``
(fused linear + shift + cross-entropy, no logit materialization) and
passes ``final_logit_softcapping`` through so the Gemma-4 softcap stays
in-kernel.

Pattern mirrors ``mistral3_liger_patch.py``.
"""
import torch
from dataclasses import dataclass
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss, unpack_cross_entropy_result
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration,
    Gemma4CausalLMOutputWithPast,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class Gemma4LCEOutput(Gemma4CausalLMOutputWithPast):
    token_accuracy: torch.Tensor = None


def _lce_forward(
    self,
    input_ids=None,
    pixel_values=None,
    pixel_values_videos=None,
    input_features=None,
    attention_mask=None,
    input_features_mask=None,
    position_ids=None,
    image_position_ids=None,
    video_position_ids=None,
    past_key_values=None,
    mm_token_type_ids=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    logits_to_keep=0,
    **kwargs,
):
    # transformers >=5.x removed/deprecated `return_dict` and the Trainer
    # injects it (plus `num_items_in_batch`) into kwargs — strip Trainer-only
    # kwargs before forwarding so we don't double-pass or pollute Liger.
    kwargs.pop("return_dict", None)
    num_items_in_batch = kwargs.pop("num_items_in_batch", None)
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        input_features=input_features,
        attention_mask=attention_mask,
        input_features_mask=input_features_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        image_position_ids=image_position_ids,
        video_position_ids=video_position_ids,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

    loss = None
    logits = None
    token_accuracy = None
    text_cfg = self.config.get_text_config()
    softcap = text_cfg.final_logit_softcapping

    if labels is not None:
        # Compute the fused loss in fp32: the final_logit_softcapping (=30 on
        # Gemma-4) and the 262k-vocab log-sum-exp are unstable in bf16/fp16 and
        # produce nan grad_norm. Upcasting here keeps the loss numerically safe
        # while still avoiding full-logit materialization (liger chunks the vocab).
        liger_kwargs = {}
        if num_items_in_batch is not None:
            liger_kwargs["num_items_in_batch"] = num_items_in_batch
        result = LigerForCausalLMLoss(
            hidden_states=hidden_states[:, slice_indices, :].float(),
            lm_head_weight=self.lm_head.weight.float(),
            labels=labels,
            hidden_size=text_cfg.hidden_size,
            final_logit_softcapping=softcap,
            **liger_kwargs,
        )
        loss, _, token_accuracy, _ = unpack_cross_entropy_result(result)
        token_accuracy = token_accuracy if token_accuracy is not None else torch.tensor(0.0)
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if softcap is not None:
            logits = logits / softcap
            logits = torch.tanh(logits)
            logits = logits * softcap

    return Gemma4LCEOutput(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        audio_hidden_states=outputs.audio_hidden_states,
        token_accuracy=token_accuracy,
    )


def patch_gemma4_liger() -> None:
    """Replace ``Gemma4ForConditionalGeneration.forward`` with the LCE variant."""
    Gemma4ForConditionalGeneration.forward = _lce_forward
    logger.info("Patched Gemma4ForConditionalGeneration forward with LigerForCausalLMLoss")
