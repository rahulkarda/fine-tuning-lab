"""
Held-out loss evaluation utility for HuggingFace Trainer.

Provides:
- eval_loss_on_dataset: runs evaluation on a held-out dataset (e.g. validation split) and returns average loss.

Usage Example:
    from transformers import Trainer
    loss = eval_loss_on_dataset(trainer, eval_dataset)

Use to track loss before/after fine-tuning for comparison.
"""
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
