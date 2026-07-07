import os
from src.utils import count_jsonl_lines, load_jsonl, validate_jsonl_schema, train_val_split, deduplicate_jsonl, flatten_dict, get_model_family, is_empty

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
    # Test flatten_dict utility
    nested = {
        "a": 1,
        "b": {"c": 2, "d": {"e": 3}},
        "f": 4
    }
    flat = flatten_dict(nested)
    assert flat["a"] == 1, "Missing key a"
    assert flat["b.c"] == 2, "Missing key b.c"
    assert flat["b.d.e"] == 3, "Missing key b.d.e"
    assert flat["f"] == 4, "Missing key f"
    assert len(flat) == 4, f"Expected 4 keys, got {len(flat)}"
    print("test_flatten_dict passed.")

def test_get_model_family():
    # Test get_model_family utility
    assert get_model_family("microsoft/Phi-3-mini-4k-instruct") == "phi"
    assert get_model_family("Qwen2.5-1.8B") == "qwen"
    assert get_model_family("Llama-3-8B") == "llama"
    assert get_model_family("unknown-model") == "unknown"
    print("test_get_model_family passed.")

# Minimal test for eval_loss_on_dataset utility
from src.eval_loss import eval_loss_on_dataset
class DummyTrainer:
    def evaluate(self, eval_dataset=None):
        return {'eval_loss': 1.23}

def test_eval_loss_on_dataset():
    trainer = DummyTrainer()
    loss = eval_loss_on_dataset(trainer, eval_dataset=[1,2,3])
    assert loss == 1.23, f"Expected loss 1.23, got {loss}"
    print("test_eval_loss_on_dataset passed.")

# Minimal test for count_model_parameters utility
from src.trainer import count_model_parameters
class DummyModel:
    def parameters(self):
        class P:
            def __init__(self, n, req):
                self._n = n
                self.requires_grad = req
            def numel(self):
                return self._n
        return [P(3, True), P(5, False), P(1, True)]

def test_count_model_parameters():
    model = DummyModel()
    stats1 = count_model_parameters(model, trainable_only=True)
    stats2 = count_model_parameters(model, trainable_only=False)
    assert stats1["trainable_params"] == 4, f"Expected 4 trainable params, got {stats1['trainable_params']}"
    assert stats2["total_params"] == 9, f"Expected 9 total params, got {stats2['total_params']}"
    assert stats2["trainable_params"] == 4, f"Expected 4 trainable params, got {stats2['trainable_params']}"
    assert stats2["non_trainable_params"] == 5, f"Expected 5 non-trainable param, got {stats2['non_trainable_params']}"
    print("test_count_model_parameters passed.")

# Minimal test for is_empty utility

def test_is_empty():
    assert is_empty(None) == True, "None should be empty"
    assert is_empty("") == True, "Empty string should be empty"
    assert is_empty("   ") == True, "Whitespace string should be empty"
    assert is_empty([]) == True, "Empty list should be empty"
    assert is_empty({}) == True, "Empty dict should be empty"
    assert is_empty(set()) == True, "Empty set should be empty"
    assert is_empty(0) == False, "Zero should not be empty"
    assert is_empty([1]) == False, "Non-empty list should not be empty"
    assert is_empty({"a": 1}) == False, "Non-empty dict should not be empty"
    assert is_empty("foo") == False, "Non-empty string should not be empty"
    assert is_empty(False) == False, "False should not be empty"
    print("test_is_empty passed.")

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
