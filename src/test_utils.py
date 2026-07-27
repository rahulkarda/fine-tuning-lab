import os
from src.utils import count_jsonl_lines, load_jsonl, validate_jsonl_schema, train_val_split, deduplicate_jsonl, flatten_dict, get_model_family, is_empty, dataset_text_lengths, get_first_key, get_last_key, get_keys

"""
Basic unit test coverage for core data utilities in src/utils.py.

Tests:
- count_jsonl_lines: verifies line counting on synthetic JSONL
- load_jsonl: verifies loading and blank line handling
- validate_jsonl_schema: checks schema validation logic (missing keys, non-JSON lines)
- train_val_split: checks split size, reproducibility with seed
- deduplicate_jsonl: checks deduplication removes duplicate objects
- flatten_dict: checks recursive flattening of nested dicts
- get_model_family: tests family inference from typical model names
- eval_loss_on_dataset: tests eval_loss utility (new)
- count_model_parameters: tests counting model parameters (new)
- is_empty: tests empty value detection (new)
- dataset_text_lengths: tests text length utility (new)
- get_first_key: tests get_first_key utility (new)
- get_last_key: tests get_last_key utility (new)
- get_keys: tests get_keys utility (new)
- evaluate_generation_quality: minimal test for generation probe (new)
- generate_outputs: minimal test for output generation (new)
- load_train_config_from_yaml: minimal test for YAML config loader (new)
- minimal_trainer_param_count: minimal test for MinimalTrainer parameter counting (new)

Extend with more tests as utilities are added.
Run directly for quick check: python src/test_utils.py
"""

# ... [other tests unchanged] ...

def test_minimal_trainer_param_count():
    # Minimal test for count_model_parameters utility from src/trainer.py
    from src.trainer import count_model_parameters
    import torch.nn as nn
    # Create a tiny model
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)
            self.other = nn.Linear(2, 1)
        def forward(self, x):
            return self.other(self.linear(x))
    model = TinyModel()
    # By default, all params require grad
    stats = count_model_parameters(model, trainable_only=False)
    assert 'total_params' in stats and 'trainable_params' in stats, "Missing keys in stats"
    assert stats['total_params'] == stats['trainable_params'] + stats['non_trainable_params'], "Param count mismatch"
    # Now freeze one layer
    for p in model.linear.parameters():
        p.requires_grad = False
    stats2 = count_model_parameters(model, trainable_only=False)
    assert stats2['trainable_params'] < stats2['total_params'], "Should have fewer trainable params after freezing"
    print("test_minimal_trainer_param_count passed.")

if __name__ == '__main__':
    test_count_jsonl_lines()
    test_load_jsonl()
    test_validate_jsonl_schema()
    test_train_val_split()
    test_deduplicate_jsonl()
    test_flatten_dict()
    test_get_model_family()
    test_eval_loss_on_dataset()
    test_count_model_parameters()
    test_is_empty()
    test_dataset_text_lengths()
    test_get_first_key()
    test_get_last_key()
    test_get_keys()
    test_evaluate_generation_quality()
    test_generate_outputs()
    test_load_train_config_from_yaml()
    test_minimal_trainer_param_count()
