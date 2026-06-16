from typing import Any
from transformers import Trainer


def eval_loss_on_dataset(trainer: Trainer, eval_dataset: Any) -> float:
    """
    Runs evaluation on a held-out set and returns average loss.
    Args:
        trainer: HuggingFace Trainer instance
        eval_dataset: dataset compatible with trainer (e.g. HF Dataset, list)
    Returns:
        Average loss on eval_dataset (float)
    """
    result = trainer.evaluate(eval_dataset=eval_dataset)
    # 'eval_loss' is standard key in Trainer output
    return result.get('eval_loss', float('nan'))
