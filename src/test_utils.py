import os
from src.utils import count_jsonl_lines, load_jsonl, validate_jsonl_schema, train_val_split, deduplicate_jsonl, flatten_dict, get_model_family

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

Extend with more tests as utilities are added.
Run directly for quick check: python src/test_utils.py
"""


def test_count_jsonl_lines():
    # Create a temporary jsonl file with known lines
    test_path = 'test_tmp.jsonl'
    lines = [
        '{"id": 1, "text": "hello"}\n',
        '{"id": 2, "text": "world"}\n',
        '{"id": 3, "text": "!"}\n'
    ]
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    try:
        result = count_jsonl_lines(test_path)
        assert result == 3, f"Expected 3 lines, got {result}"
        print("test_count_jsonl_lines passed.")
    finally:
        try:
            os.remove(test_path)
        except FileNotFoundError:
            pass
        except PermissionError:
            pass

def test_load_jsonl():
    # Create a temporary jsonl file
    test_path = 'test_tmp_load.jsonl'
    lines = [
        '{"id": 10, "text": "foo"}\n',
        '{"id": 20, "text": "bar"}\n',
        '\n',  # Blank line should be ignored
        '{"id": 30, "text": "baz"}\n'
    ]
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    try:
        data = load_jsonl(test_path)
        assert len(data) == 3, f"Expected 3 items, got {len(data)}"
        assert data[0]["id"] == 10 and data[1]["text"] == "bar", "Data mismatch"
        print("test_load_jsonl passed.")
    finally:
        try:
            os.remove(test_path)
        except FileNotFoundError:
            pass
        except PermissionError:
            pass

def test_validate_jsonl_schema():
    # Create a temporary jsonl file with valid and invalid lines
    test_path = 'test_tmp_schema.jsonl'
    lines = [
        '{"id": 1, "text": "a"}\n',  # valid
        '{"id": 2}\n',                 # invalid (missing text)
        'not_a_json\n',                  # invalid (not JSON)
        '{"id": 3, "text": "c"}\n', # valid
        '\n',                           # blank
        '{"text": "d"}\n'             # invalid (missing id)
    ]
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    # Schema: must have both 'id' and 'text' keys
    def schema_fn(obj):
        return isinstance(obj, dict) and 'id' in obj and 'text' in obj
    try:
        invalid = validate_jsonl_schema(test_path, schema_fn)
        assert invalid == 3, f"Expected 3 invalid lines, got {invalid}"
        print("test_validate_jsonl_schema passed.")
    finally:
        try:
            os.remove(test_path)
        except FileNotFoundError:
            pass
        except PermissionError:
            pass

def test_train_val_split():
    # Minimal test for train_val_split utility
    data = list(range(10))
    val_ratio = 0.2
    seed = 123
    train, val = train_val_split(data, val_ratio=val_ratio, seed=seed)
    assert len(val) == 2, f"Expected 2 val items, got {len(val)}"
    assert len(train) == 8, f"Expected 8 train items, got {len(train)}"
    # Check reproducibility
    train2, val2 = train_val_split(data, val_ratio=val_ratio, seed=seed)
    assert val == val2 and train == train2, "Split not reproducible with same seed"
    print("test_train_val_split passed.")

def test_deduplicate_jsonl():
    # Create a temporary jsonl file with duplicates
    test_path = 'test_tmp_dedupe.jsonl'
    output_path = 'test_tmp_dedupe_out.jsonl'
    lines = [
        '{"id": 1, "text": "hello"}\n',
        '{"id": 1, "text": "hello"}\n',  # duplicate
        '{"id": 2, "text": "world"}\n',
        '{"id": 3, "text": "!"}\n',
        '{"id": 2, "text": "world"}\n',  # duplicate
        '\n',
        '{"id": 3, "text": "!"}\n',      # duplicate
    ]
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    try:
        deduplicate_jsonl(test_path, output_path)
        # Load deduplicated file
        deduped = load_jsonl(output_path)
        ids = sorted([item["id"] for item in deduped])
        assert len(deduped) == 3, f"Expected 3 unique items, got {len(deduped)}"
        assert ids == [1, 2, 3], f"Expected unique ids [1,2,3], got {ids}"
        print("test_deduplicate_jsonl passed.")
    finally:
        for p in [test_path, output_path]:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except PermissionError:
                pass

def test_flatten_dict():
    # Minimal test for flatten_dict utility
    d = {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3
            }
        },
        "f": 4
    }
    flat = flatten_dict(d)
    assert flat["a"] == 1
    assert flat["b.c"] == 2
    assert flat["b.d.e"] == 3
    assert flat["f"] == 4
    print("test_flatten_dict passed.")

def test_get_model_family():
    # Minimal test for get_model_family utility
    cases = [
        ("microsoft/Phi-3-mini-4k-instruct", "phi-3"),
        ("Qwen/Qwen2.5-7B", "qwen-2.5"),
        ("Qwen/Qwen-7B", "qwen"),
        ("meta-llama/Llama-3-8B", "llama-3"),
        ("meta-llama/Llama-2-7B", "llama-2"),
        ("unknown/model", "unknown")
    ]
    for name, expected in cases:
        fam = get_model_family(name)
        assert fam == expected, f"model_family({name}) -> {fam}, expected {expected}"
    print("test_get_model_family passed.")

# Minimal test for eval_loss_on_dataset utility
def test_eval_loss_on_dataset():
    from src.eval_loss import eval_loss_on_dataset
    class DummyTrainer:
        def evaluate(self, eval_dataset=None):
            return {'eval_loss': 1.2345}
    dummy_trainer = DummyTrainer()
    dummy_dataset = [1, 2, 3]  # not actually used
    loss = eval_loss_on_dataset(dummy_trainer, dummy_dataset)
    assert loss == 1.2345, f"Expected eval_loss 1.2345, got {loss}"
    print("test_eval_loss_on_dataset passed.")

# Minimal test for count_model_parameters utility

def test_count_model_parameters():
    from src.trainer import count_model_parameters
    import torch.nn as nn
    # Simple model with 2 parameters
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(2, 2, bias=False)  # 4 params
            self.l2 = nn.Linear(2, 1, bias=True)   # 2 weights + 1 bias = 3
        def forward(self, x):
            return self.l2(self.l1(x))
    model = Tiny()
    stats = count_model_parameters(model)
    assert stats["total_params"] == 7, f"Expected 7 total params, got {stats['total_params']}"
    assert stats["trainable_params"] == 7, f"Expected 7 trainable params, got {stats['trainable_params']}"
    assert stats["non_trainable_params"] == 0, f"Expected 0 non-trainable params, got {stats['non_trainable_params']}"
    # Make one param non-trainable
    for name, param in model.named_parameters():
        if name == "l2.bias":
            param.requires_grad = False
    stats2 = count_model_parameters(model)
    assert stats2["trainable_params"] == 6, f"Expected 6 trainable params, got {stats2['trainable_params']}"
    assert stats2["non_trainable_params"] == 1, f"Expected 1 non-trainable param, got {stats2['non_trainable_params']}"
    print("test_count_model_parameters passed.")

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
