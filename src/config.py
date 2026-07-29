"""Training run config. Loaded from YAML in real runs."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    dataset_path: str = "data/train.jsonl"
    output_dir: str = "outputs/run"
    epochs: int = 3  # reduced default epochs for quicker runs
    learning_rate: float = 1e-4  # reduced learning rate for more stability
    batch_size: int = 8  # increased batch size for stability on Phi-3-mini
    grad_accum_steps: int = 2  # reduced grad accum steps for more frequent optimizer updates
    max_seq_length: int = 4096  # increased default for longer context
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 64  # increased rank for more adaptation capacity (was 32)
    lora_alpha: int = 64  # increased alpha for more adaptation capacity (was 32)
    lora_dropout: float = 0.1  # reduced dropout for stability
    lora_target_modules: tuple = ("q_proj", "v_proj")
    warmup_ratio: float = 0.02  # reduced warmup ratio for faster convergence
    weight_decay: float = 0.0
    resume_from: Optional[str] = None
    gradient_checkpointing: bool = False  # new toggle
