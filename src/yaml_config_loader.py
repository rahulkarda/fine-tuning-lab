import yaml
from src.config import TrainConfig

def load_train_config_from_yaml(path: str) -> TrainConfig:
    """
    Loads a TrainConfig instance from a YAML file.
    Args:
        path: path to YAML file
    Returns:
        TrainConfig instance
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw_cfg = yaml.safe_load(f)
    # Only pass keys that exist in TrainConfig
    field_names = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    filtered_cfg = {k: v for k, v in raw_cfg.items() if k in field_names}
    return TrainConfig(**filtered_cfg)
