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
        assert ids == [1, 2, 3], f"Expected deduped ids [1,2,3], got {ids}"
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
    nested = {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3
            }
        },
        "f": 4
    }
    flat = flatten_dict(nested)
    assert flat["a"] == 1
    assert flat["b.c"] == 2
    assert flat["b.d.e"] == 3
    assert flat["f"] == 4
    print("test_flatten_dict passed.")

def test_get_model_family():
    names = [
        "microsoft/Phi-3-mini-4k-instruct",
        "Qwen/Qwen1.5-7B-Chat",
        "meta-llama/Meta-Llama-3-8B",
        "llama-3-Open",
        "qwen2.5-14b",
        "phi3-mixed",
        "llama3",
        "llama-3",
        "foobar"
    ]
    expected = [
        "phi",
        "qwen",
        "llama",
        "llama",
        "qwen",
        "phi",
        "llama",
        "llama",
        "unknown"
    ]
    for name, fam in zip(names, expected):
        result = get_model_family(name)
        assert result == fam, f"Expected '{fam}' for '{name}', got '{result}'"
    print("test_get_model_family passed.")

def test_eval_loss_on_dataset():
    # Dummy Trainer and dataset
    class DummyTrainer:
        def evaluate(self, eval_dataset=None):
            return {'eval_loss': 1.23}
    trainer = DummyTrainer()
    from src.eval_loss import eval_loss_on_dataset
    loss = eval_loss_on_dataset(trainer, eval_dataset=[1,2,3])
    assert abs(loss - 1.23) < 1e-5, f"Expected loss 1.23, got {loss}"
    print("test_eval_loss_on_dataset passed.")

def test_count_model_parameters():
    # Dummy model: 3 params, 2 trainable
    class DummyParam:
        def __init__(self, n, req_grad):
            self._n = n
            self.requires_grad = req_grad
        def numel(self):
            return self._n
    class DummyModel:
        def parameters(self):
            return [DummyParam(10, True), DummyParam(5, False), DummyParam(2, True)]
    from src.trainer import count_model_parameters
    model = DummyModel()
    stats = count_model_parameters(model)
    assert stats["total_params"] == 17
    assert stats["trainable_params"] == 12
    assert stats["non_trainable_params"] == 5
    train_only = count_model_parameters(model, trainable_only=True)
    assert train_only["trainable_params"] == 12
    print("test_count_model_parameters passed.")

def test_is_empty():
    assert is_empty(None) == True
    assert is_empty("") == True
    assert is_empty([]) == True
    assert is_empty({}) == True
    assert is_empty("   ") == True
    assert is_empty("a") == False
    assert is_empty([1]) == False
    assert is_empty({"x": 1}) == False
    print("test_is_empty passed.")

def test_dataset_text_lengths():
    # Each item should be dict with 'text' key
    data = [
        {"text": "abc"},
        {"text": "defg"},
        {"text": ""},
        {"foo": "bar"}  # missing text
    ]
    lengths = dataset_text_lengths(data)
    assert lengths == [3, 4, 0, 0], f"Expected [3,4,0,0], got {lengths}"
    print("test_dataset_text_lengths passed.")

def test_get_first_key():
    d1 = {"a": 1, "b": 2}
    d2 = {"x": 42}
    d3 = {}
    d4 = []
    d5 = None
    assert get_first_key(d1) == "a", f"Expected 'a' as first key"
    assert get_first_key(d2) == "x", f"Expected 'x' as first key"
    assert get_first_key(d3) is None, f"Expected None for empty dict"
    assert get_first_key(d4) is None, f"Expected None for non-dict"
    assert get_first_key(d5) is None, f"Expected None for None input"
    print("test_get_first_key passed.")

def test_get_last_key():
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"x": 42}
    d3 = {}
    d4 = []
    d5 = None
    assert get_last_key(d1) == "c", f"Expected 'c' as last key"
    assert get_last_key(d2) == "x", f"Expected 'x' as last key"
    assert get_last_key(d3) is None, f"Expected None for empty dict"
    assert get_last_key(d4) is None, f"Expected None for non-dict"
    assert get_last_key(d5) is None, f"Expected None for None input"
    print("test_get_last_key passed.")

def test_get_keys():
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"x": 42}
    d3 = {}
    d4 = []
    d5 = None
    assert get_keys(d1) == ["a", "b", "c"], f"Expected ['a','b','c'] as keys"
    assert get_keys(d2) == ["x"], f"Expected ['x'] as keys"
    assert get_keys(d3) == [], f"Expected [] for empty dict"
    assert get_keys(d4) == [], f"Expected [] for non-dict"
    assert get_keys(d5) == [], f"Expected [] for None input"
    print("test_get_keys passed.")

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
