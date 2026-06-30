"""Training run config. Loaded from YAML in real runs."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    dataset_path: str = "data/train.jsonl"
    output_dir: str = "outputs/run"
    epochs: int = 6  # increased default epochs for more robust training
    learning_rate: float = 1e-4  # increased learning rate for faster iteration
    batch_size: int = 2  # further reduced batch size for rapid prototyping
    grad_accum_steps: int = 2  # reduced grad accum steps for more frequent optimizer updates
    max_seq_length: int = 4096  # increased default for longer context
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 32  # increased rank for more aggressive adaptation
    lora_alpha: int = 64  # increased alpha for more aggressive adaptation
    lora_dropout: float = 0.3  # increased dropout for stronger regularization
    lora_target_modules: tuple = ("q_proj", "v_proj")
    warmup_ratio: float = 0.02  # reduced warmup ratio for faster convergence
    weight_decay: float = 0.0
    resume_from: Optional[str] = None
    gradient_checkpointing: bool = False  # new toggle
