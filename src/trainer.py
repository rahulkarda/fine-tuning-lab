"""
Minimal HuggingFace Trainer wrapper for LoRA and full-parameter fine-tuning.

This module provides:
- Model and tokenizer loading (AutoModelForCausalLM, AutoTokenizer) for common open models
- Optional LoRA configuration via peft (parameter-efficient tuning)
- TrainingArguments setup: batch size, epochs, learning rate, gradient accumulation, checkpointing, etc.
- MinimalTrainer class: wraps HF Trainer and exposes .train() for easy experiment runs
- Support for resuming from checkpoint and memory-efficient training

Usage Example:
    from src.config import TrainConfig
    from src.trainer import MinimalTrainer
    cfg = TrainConfig()
    model, tokenizer = setup_model_and_tokenizer(cfg)
    # prepare train_dataset as list/dataset of tokenized samples
    trainer = MinimalTrainer(cfg, train_dataset, tokenizer)
    trainer.train()

MinimalTrainer handles model setup, Trainer instantiation, and training loop.
Extend this module for custom callbacks, evaluation, or logging as needed.

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

def create_lora_config(cfg: TrainConfig):
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


def get_training_args(cfg: TrainConfig):
    """
    Builds TrainingArguments from config.
    """
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        max_steps=-1,
        seed=cfg.seed,
        logging_steps=20,  # was 10
        save_strategy="epoch",
        evaluation_strategy="no",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=True,
        report_to=[],
        resume_from_checkpoint=cfg.resume_from if cfg.resume_from else None,
        gradient_checkpointing=cfg.gradient_checkpointing,
        save_total_limit=2,  # limit to 2 checkpoints to reduce clutter
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
        trainer = MinimalTrainer(cfg, train_dataset, tokenizer)
        trainer.train()
    """
    def __init__(self, cfg: TrainConfig, train_dataset, tokenizer):
        self.cfg = cfg
        self.model, self.tokenizer = setup_model_and_tokenizer(cfg)
        self.args = get_training_args(cfg)
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            tokenizer=tokenizer
        )

    def train(self):
        """
        Runs training loop. Handles checkpoint resume if configured.
        Returns:
            Trainer output (training state)
        """
        if self.cfg.resume_from:
            return self.trainer.train(resume_from_checkpoint=self.cfg.resume_from)
        else:
            return self.trainer.train()
