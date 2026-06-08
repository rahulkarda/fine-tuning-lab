"""Training run config. Loaded from YAML in real runs."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    dataset_path: str = "data/train.jsonl"
    output_dir: str = "outputs/run"
    epochs: int = 1
    learning_rate: float = 1e-4  # lowered default
    batch_size: int = 4  # reduced batch size for faster iteration
    grad_accum_steps: int = 4  # reduced grad accum steps for faster iteration
    max_seq_length: int = 1024
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 8  # lowered rank for memory efficiency
    lora_alpha: int = 16  # lowered alpha for memory efficiency
    lora_dropout: float = 0.15  # increased dropout for more regularization
    lora_target_modules: tuple = ("q_proj", "v_proj")
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    resume_from: Optional[str] = None
    gradient_checkpointing: bool = False  # new toggle
