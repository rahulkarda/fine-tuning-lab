"""
Training run config. Loaded from YAML in real runs.

Fields:
- base_model: HuggingFace model name or path (e.g. 'microsoft/Phi-3-mini-4k-instruct')
- dataset_path: Path to training data (JSONL, or other supported format)
- output_dir: Directory to save checkpoints and logs
- epochs: Number of training epochs
- learning_rate: Initial learning rate for optimizer
- batch_size: Per-device batch size
- grad_accum_steps: Steps to accumulate gradients before optimizer update
- max_seq_length: Maximum sequence length for tokenization/model
- seed: Random seed for reproducibility
- use_lora: Whether to enable LoRA parameter-efficient tuning
- lora_r: LoRA rank (adaptation capacity)
- lora_alpha: LoRA scaling factor
- lora_dropout: Dropout probability for LoRA layers
- lora_target_modules: Tuple of module names to apply LoRA (e.g. ('q_proj', 'v_proj'))
- warmup_ratio: Fraction of total steps for learning rate warmup
- weight_decay: Weight decay regularization
- resume_from: Optional checkpoint path to resume training
- gradient_checkpointing: Toggle for memory-efficient gradient checkpointing

Typical usage:
    from src.config import TrainConfig
    cfg = TrainConfig()
    # Override fields as needed
    cfg.base_model = 'Qwen/Qwen1.5-0.5B'
    cfg.dataset_path = 'data/qwen.jsonl'
    # Pass cfg to trainer or CLI

"""
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
