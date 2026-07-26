import torch
from transformers import Trainer, TrainingArguments
from typing import Any, Dict, List


def learning_rate_finder(
    trainer: Trainer,
    train_dataset: Any,
    lr_range: List[float],
    num_steps: int = 100,
    plot: bool = False
) -> Dict[str, float]:
    """
    Minimal learning rate finder utility.
    Sweeps learning rate values over lr_range, runs for num_steps, records loss.
    Args:
        trainer: HuggingFace Trainer instance (initialized)
        train_dataset: dataset for the sweep
        lr_range: list of learning rates to try
        num_steps: steps per lr
        plot: if True, plot loss vs. lr (requires matplotlib)
    Returns:
        Dict mapping learning rate to final loss
    """
    losses = {}
    original_lr = trainer.args.learning_rate
    for lr in lr_range:
        trainer.args.learning_rate = lr
        trainer.model.train()
        # Reset optimizer state to start fresh for each lr
        if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = lr
        step_losses = []
        for step, batch in enumerate(trainer.get_train_dataloader()):
            if step >= num_steps:
                break
            batch = {k: v.to(trainer.args.device) if hasattr(v, 'to') else v for k, v in batch.items()}
            outputs = trainer.model(**batch)
            loss = outputs.loss
            step_losses.append(loss.item())
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
        if step_losses:
            avg_loss = sum(step_losses) / len(step_losses)
        else:
            avg_loss = float('nan')
        losses[lr] = avg_loss
    trainer.args.learning_rate = original_lr
    # Optionally plot
    if plot:
        try:
            import matplotlib.pyplot as plt
            lrs = list(losses.keys())
            loss_vals = [losses[lr] for lr in lrs]
            plt.plot(lrs, loss_vals, marker='o')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.title('Learning Rate Finder')
            plt.show()
        except ImportError:
            pass
    return losses
