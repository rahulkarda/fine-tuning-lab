from src.config import TrainConfig
from src.sweep_lora_rank import sweep_lora_rank

"""
Minimal example/test for sweep_lora_rank utility.
Run directly to print generated configs.
"""

if __name__ == "__main__":
    base_cfg = TrainConfig()
    ranks = [8, 16, 32]
    configs = sweep_lora_rank(base_cfg, ranks, output_dir_prefix="outputs/lora_sweep_test")
    for cfg in configs:
        print(f"lora_r: {cfg.lora_r}, output_dir: {cfg.output_dir}")
