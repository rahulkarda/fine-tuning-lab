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
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    if cfg.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("peft not installed, required for LoRA training")
        lora_config = create_lora_config(cfg)
        model = get_peft_model(model, lora_config)
    return model, tokenizer

def get_training_args(cfg: TrainConfig):
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        max_steps=-1,
        seed=cfg.seed,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=True,
        report_to=[],
        resume_from_checkpoint=cfg.resume_from,
    )

class MinimalTrainer:
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
        return self.trainer.train()
