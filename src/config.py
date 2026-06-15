"""Training run config. Loaded from YAML in real runs."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    dataset_path: str = "data/train.jsonl"
    output_dir: str = "outputs/run"
    epochs: int = 3  # increased default epochs for more robust training
    learning_rate: float = 1e-4  # lowered default
    batch_size: int = 4  # reduced batch size for faster iteration
    grad_accum_steps: int = 4  # reduced grad accum steps for faster iteration
    max_seq_length: int = 2048  # increased default for longer context
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 16  # increased rank for better adaptation
    lora_alpha: int = 32  # increased alpha for better adaptation
    lora_dropout: float = 0.2  # increased dropout for stronger regularization
    lora_target_modules: tuple = ("q_proj", "v_proj")
    warmup_ratio: float = 0.08  # increased warmup ratio for smoother ramp-up
    weight_decay: float = 0.0
    resume_from: Optional[str] = None
    gradient_checkpointing: bool = False  # new toggle
