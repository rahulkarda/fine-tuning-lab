"""
Minimal HuggingFace Trainer wrapper for LoRA and full-parameter fine-tuning.

This module provides:
- Model and tokenizer loading (AutoModelForCausalLM, AutoTokenizer) for common open models
- Optional LoRA configuration via peft (parameter-efficient tuning; requires peft package)
- TrainingArguments setup: batch size, epochs, learning rate, gradient accumulation, checkpointing, etc.
- MinimalTrainer class: wraps HF Trainer and exposes .train() for easy experiment runs
- Support for resuming from checkpoint and memory-efficient training
- Utility: count_model_parameters(model, trainable_only=False) for parameter statistics (new)

LoRA integration (clarified):
- If cfg.use_lora is True, model is wrapped with LoRA using peft (must be installed).
- lora_r, lora_alpha, lora_dropout, lora_target_modules are all configurable from TrainConfig.
- For new model families, check target_modules carefully (often 'q_proj', 'v_proj' for attention).
- If peft is not installed, ImportError will be raised (explicit check).

Config tips:
- All training hyperparameters are controlled via TrainConfig dataclass (src/config.py).
- Resume-from-checkpoint is handled automatically via cfg.resume_from.
- Gradient checkpointing can be toggled via cfg.gradient_checkpointing for memory efficiency.
- For quick runs, set batch_size and grad_accum_steps to fit your GPU.

Usage Example:
    from src.config import TrainConfig
    from src.trainer import MinimalTrainer, count_model_parameters
    cfg = TrainConfig()
    model, tokenizer = setup_model_and_tokenizer(cfg)
    param_stats = count_model_parameters(model, trainable_only=True)
    # prepare train_dataset as list/dataset of tokenized samples
    trainer = MinimalTrainer(cfg, train_dataset, tokenizer)
    trainer.train()

MinimalTrainer handles model setup, Trainer instantiation, and training loop.
Extend this module for custom callbacks, evaluation, or logging as needed.

Caveats:
- LoRA requires peft; install with pip if missing (pip install peft).
- Resume-from-checkpoint expects compatible checkpoint in cfg.output_dir.
- Memory settings (fp16, gradient_checkpointing) may need adjustment per model/GPU.
- count_model_parameters is useful for reporting trainable parameter count (especially with LoRA).

Intended for quick experiment scaffolding.
"""
import os
from typing import Optional
from src.config import TrainConfig

from transformers import (
    Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
)
try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

import torch

def create_lora_config(cfg: TrainConfig):
    """
    Constructs a LoraConfig from TrainConfig.
    """
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )


def setup_model_and_tokenizer(cfg: TrainConfig):
    """
    Loads model and tokenizer, applies LoRA if configured.
    Args:
        cfg: TrainConfig instance
    Returns:
        model: HF model (possibly LoRA-wrapped)
        tokenizer: HF tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    if cfg.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("peft not installed, required for LoRA training")
        lora_config = create_lora_config(cfg)
        model = get_peft_model(model, lora_config)
    # Gradient checkpointing toggle
    if getattr(cfg, "gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            try:
                model.config.gradient_checkpointing = True
            except Exception:
                pass
    return model, tokenizer


def get_training_args(cfg: TrainConfig) -> TrainingArguments:
    """
    Builds TrainingArguments from config.
    Separates construction for clarity.
    """
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        max_steps=-1,
        seed=cfg.seed,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="no",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=True,
        report_to=[],
        resume_from_checkpoint=cfg.resume_from if cfg.resume_from else None,
        gradient_checkpointing=cfg.gradient_checkpointing,
        save_total_limit=2,
    )


class MinimalTrainer:
    """
    Minimal wrapper around HuggingFace Trainer for experiment runs.
    Handles model setup, Trainer instantiation, and exposes .train().
    Args:
        cfg: TrainConfig
        train_dataset: tokenized dataset for training
        tokenizer: tokenizer instance
    Usage:
        trainer = M
... [truncated]
th total_params, trainable_params, non_trainable_params
    """
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    non_trainable = total - trainable
    stats = {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable
    }
    if trainable_only:
        return {"trainable_params": trainable}
    else:
        return stats
