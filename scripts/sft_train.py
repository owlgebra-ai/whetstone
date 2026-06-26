"""Stage 3 / Stage 4 — Compressibility-Aware Cold-Start SFT.

Installs the compact register as a stable policy mode via surprisal-weighted
SFT:

    w_t = max(1, 1 + alpha · S_base(x_t | x_<t)),    alpha = 0.5
    L_SFT = − E_trace Σ_t w_t · log P_policy(x_t | x_<t)

The reweighting concentrates the gradient on the ~20% of tokens that carry
novel reasoning content (high surprisal under the base) rather than on the
~80% of fluency boilerplate (low surprisal). Without it, SFT mostly teaches
formatting and the compact register fails to take.

Reference: §5 (Stage 3) and §6.6 (Stage 4, identical recipe).

Gemma-4 adaptations baked in (see gemma4_learnings.md):
  * attn_implementation defaults to "sdpa" (flash-attn rejects global head_dim 512).
  * `torch.backends.cuda.enable_cudnn_sdp(False)` (cuDNN MHA crashes on gemma4).
  * Auto-applies the Liger fused-CE patch for `Gemma4ForConditionalGeneration`
    (the multimodal class is not in liger's auto-patch registry).
  * Optional `--use_lora` with the gemma4-specific exclude_modules pattern
    (`.*(vision_tower|audio_tower|embed_audio).*`) — for dev boxes where the
    8B model does not leave room for full-param FSDP.
  * bf16 only (fp16 overflows the gemma4 attention-mask fill).
  * Pad token: gemma4 ships a real <pad> (id 0); we never override pad_token.
  * EOS: the stock generation_config already includes <turn|> (id 106).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SYS_PROMPT = (
    "Place all your step-by-step reasoning between <think> and </think> tags. "
    "After </think>, give the final answer."
)


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _is_gemma4(model) -> bool:
    cls = type(model).__name__
    return cls.startswith("Gemma4")


def _maybe_disable_cudnn_sdp(model) -> None:
    """Gemma-4's global-attention layers (head_dim=512) crash the cuDNN MHA
    backend. Force math / mem-efficient kernels instead. Safe no-op for other
    model families."""
    if _is_gemma4(model):
        torch.backends.cuda.enable_cudnn_sdp(False)
        print("[sft] gemma4 detected — disabled cuDNN SDP for head_dim=512", flush=True)


def _maybe_patch_gemma4_liger(model) -> None:
    """The multimodal Gemma4ForConditionalGeneration class is not in liger's
    MODEL_TYPE_TO_APPLY_LIGER_FN registry; the full [B, S, 262k] logits would
    materialize on every step. Patch it to use LigerForCausalLMLoss."""
    if type(model).__name__ != "Gemma4ForConditionalGeneration":
        return
    try:
        # Local import — the patch file is shipped alongside the package.
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from whetstone.patches.gemma4_liger_patch import patch_gemma4_liger
        patch_gemma4_liger()
        print("[sft] applied gemma4 Liger fused-CE patch", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[sft] WARN: gemma4 Liger patch failed: {e}", file=sys.stderr)


def _build_lora_config(args):
    """Build a peft LoraConfig with the gemma4-specific projector exclusions."""
    from peft import LoraConfig
    exclude = ".*(vision_tower|audio_tower|embed_audio).*"
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_rslora=args.use_rslora,
        target_modules="all-linear",
        exclude_modules=exclude,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


class CompactSFTDataset(Dataset):
    """Tokenizes (system, user, assistant) turns and masks loss to the
    assistant turn. Assistant-mask boundary is computed by re-tokenizing the
    prompt prefix via apply_chat_template(add_generation_prompt=True), so the
    code is model-agnostic (Qwen <|im_start|>assistant, Gemma <start_of_turn>model, etc.)."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int,
                 sys_prompt: str = SYS_PROMPT):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sys_prompt = sys_prompt

    def __len__(self) -> int:
        return len(self.rows)

    def _assistant_text(self, r: dict) -> str:
        compact = r.get("compact") or r.get("thinking_original") or ""
        gold = r.get("ground_truth", "")
        return f"<think>\n{compact}\n</think>\n{gold}"

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        r = self.rows[i]
        prompt_messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": r.get("prompt", "")},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": self._assistant_text(r)}
        ]
        prompt_prefix = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
        )
        prefix_ids = self.tokenizer(prompt_prefix, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full_text, add_special_tokens=False,
                                  truncation=True, max_length=self.max_length).input_ids

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        # Mask everything up to the assistant response. If truncation cut into
        # the prefix, mask the whole sequence (skip this row in the loss).
        boundary = min(len(prefix_ids), len(full_ids))
        labels[:boundary] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class PaddingCollator:
    """Right-pad input_ids/attention_mask; pad labels with -100."""
    pad_token_id: int

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(b["input_ids"].size(0) for b in batch)
        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            n = b["input_ids"].size(0)
            pad = max_len - n
            input_ids.append(torch.cat([b["input_ids"],
                                        torch.full((pad,), self.pad_token_id,
                                                   dtype=b["input_ids"].dtype)]))
            attention_mask.append(torch.cat([b["attention_mask"],
                                             torch.zeros(pad, dtype=b["attention_mask"].dtype)]))
            labels.append(torch.cat([b["labels"],
                                     torch.full((pad,), -100, dtype=b["labels"].dtype)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


class SurprisalWeightedTrainer(Trainer):
    """Trainer with per-token surprisal weighting against a frozen base model."""

    def __init__(self, *args: Any, surprisal_alpha: float = 0.5,
                 base_model=None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.alpha = surprisal_alpha
        self.base_model = base_model
        if self.base_model is not None:
            self.base_model.eval()
            for p in self.base_model.parameters():
                p.requires_grad = False
            # Place the frozen surprisal anchor on this rank's training device.
            # Loading via from_pretrained leaves it on CPU; we'd otherwise crash
            # the first time the forward pass sees GPU input ids.
            self.base_model = self.base_model.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        valid = shift_labels != -100
        safe = shift_labels.clone()
        safe[~valid] = 0

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        target_lp = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        per_token = -target_lp

        if self.base_model is not None and self.alpha > 0:
            with torch.no_grad():
                base_out = self.base_model(input_ids=inputs["input_ids"],
                                           attention_mask=inputs["attention_mask"])
                base_logits = base_out.logits[..., :-1, :].contiguous().float()
                base_lp = F.log_softmax(base_logits, dim=-1)
                base_target_lp = base_lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
                surprisal = -base_target_lp
                weights = torch.clamp(1.0 + self.alpha * surprisal, min=1.0)
            weights = torch.where(valid, weights, torch.zeros_like(weights))
        else:
            weights = valid.float()

        weighted = per_token * weights
        loss = weighted.sum() / (weights.sum().clamp_min(1e-8))
        return (loss, outputs) if return_outputs else loss


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="WHETSTONE Stage 3/4 surprisal-weighted SFT")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--surprisal_weight", type=float, default=0.5)
    ap.add_argument("--base_model_for_surprisal", default=None)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn_implementation", default="sdpa",
                    help="Gemma-4 requires sdpa (flash-attn rejects head_dim 512)")
    ap.add_argument("--use_lora", action="store_true",
                    help="Attach LoRA adapters instead of full-param training "
                         "(use this on dev boxes where full-param FSDP2 doesn't fit)")
    ap.add_argument("--lora_r", type=int, default=256)
    ap.add_argument("--lora_alpha", type=int, default=256)
    ap.add_argument("--use_rslora", action="store_true", default=True)
    ap.add_argument("--no_rslora", dest="use_rslora", action="store_false")
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    # Gemma-4 ships a real <pad> (id 0). Never override pad_token=eos for it.
    if tokenizer.pad_token is None and not _is_gemma4_tokenizer(tokenizer):
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    _maybe_disable_cudnn_sdp(model)
    _maybe_patch_gemma4_liger(model)

    if args.use_lora:
        from peft import get_peft_model
        peft_cfg = _build_lora_config(args)
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    base_model = None
    if args.surprisal_weight > 0 and args.base_model_for_surprisal:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_for_surprisal,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        )
        _maybe_disable_cudnn_sdp(base_model)
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad = False

    rows = _load_jsonl(args.train_jsonl)
    if not rows:
        print(f"sft_train: no rows in {args.train_jsonl}", file=sys.stderr)
        sys.exit(1)
    dataset = CompactSFTDataset(rows, tokenizer, args.max_length)
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_drop_last=False,
        seed=args.seed,
    )

    trainer = SurprisalWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        surprisal_alpha=args.surprisal_weight,
        base_model=base_model,
    )
    trainer.train()
    final_dir = os.path.join(args.output_dir, "checkpoint-FINAL")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[sft] wrote {final_dir}", flush=True)


def _is_gemma4_tokenizer(tokenizer) -> bool:
    """Heuristic: detect gemma4 tokenizers (their chat template uses <start_of_turn>)."""
    try:
        tmpl = getattr(tokenizer, "chat_template", None) or ""
        return "<start_of_turn>" in tmpl or "gemma" in (tokenizer.name_or_path or "").lower()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
