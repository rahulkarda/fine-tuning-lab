from typing import List
from src.config import TrainConfig
import copy


def sweep_lora_rank(
    base_config: TrainConfig,
    rank_values: List[int],
    output_dir_prefix: str = "outputs/lora_rank_sweep"
) -> List[TrainConfig]:
    """
    Generates a list of TrainConfig instances for a sweep over LoRA rank values.
    Args:
        base_config: starting TrainConfig
        rank_values: list of LoRA rank integers to sweep
        output_dir_prefix: directory prefix for output dirs
    Returns:
        List of TrainConfig instances, each with a different lora_r and unique output_dir
    """
    configs = []
    for rank in rank_values:
        cfg = copy.deepcopy(base_config)
        cfg.lora_r = rank
        cfg.output_dir = f"{output_dir_prefix}/r{rank}"
        configs.append(cfg)
    return configs

# Example usage:
if __name__ == "__main__":
    default_cfg = TrainConfig()
    sweep_values = [8, 16, 32, 64]
    configs = sweep_lora_rank(default_cfg, sweep_values)
    for cfg in configs:
        print(f"Rank: {cfg.lora_r}, Output: {cfg.output_dir}")
