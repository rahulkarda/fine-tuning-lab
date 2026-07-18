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
        ids = set([obj["id"] for obj in deduped])
        assert len(deduped) == 3, f"Expected 3 deduped items, got {len(deduped)}"
        assert ids == {1,2,3}, f"Expected ids {{1,2,3}}, got {ids}"
        print("test_deduplicate_jsonl passed.")
    finally:
        try:
            os.remove(test_path)
        except FileNotFoundError:
            pass
        except PermissionError:
            pass
        try:
            os.remove(output_path)
        except FileNotFoundError:
            pass
        except PermissionError:
            pass

def test_flatten_dict():
    # Minimal test for flatten_dict utility
    from src.utils import flatten_dict
    d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}, "f": 4}
    flat = flatten_dict(d)
    assert flat["a"] == 1 and flat["b.c"] == 2 and flat["b.d.e"] == 3 and flat["f"] == 4, f"Flatten failed: {flat}"
    print("test_flatten_dict passed.")

def test_get_model_family():
    families = [
        ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
        ("Qwen/Qwen1.5-7B-Chat", "qwen"),
        ("meta-llama/Meta-Llama-3-8B", "llama3"),
        ("unknown-model/foobar", "unknown"),
        ("Llama-3-Open", "llama3"),
        ("qwen2.5-14b", "qwen"),
        ("phi3-mixed", "phi3"),
        ("llama3", "llama3"),
        ("llama-3", "llama3")
    ]
    for name, expected in families:
        fam = get_model_family(name)
        assert fam == expected, f"Family mismatch for {name}: got {fam}, expected {expected}"
    print("test_get_model_family passed.")

def test_eval_loss_on_dataset():
    # Minimal stub test for eval_loss_on_dataset utility
    from src.eval_loss import eval_loss_on_dataset
    class DummyTrainer:
        def evaluate(self, eval_dataset=None):
            return {"eval_loss": 1.23}
    trainer = DummyTrainer()
    loss = eval_loss_on_dataset(trainer, [1,2,3])
    assert abs(loss - 1.23) < 1e-6, f"Expected 1.23, got {loss}"
    print("test_eval_loss_on_dataset passed.")

def test_count_model_parameters():
    # Minimal stub test for count_model_parameters utility
    from src.trainer import count_model_parameters
    class DummyModel:
        def parameters(self):
            class P:
                def __init__(self, numel, req_grad):
                    self._n = numel
                    self.requires_grad = req_grad
                def numel(self):
                    return self._n
            return [P(100, True), P(200, False), P(50, True)]
    model = DummyModel()
    stats = count_model_parameters(model)
    assert stats["total_params"] == 350, f"Total params mismatch: {stats}"
    assert stats["trainable_params"] == 150, f"Trainable params mismatch: {stats}"
    assert stats["non_trainable_params"] == 200, f"Non-trainable params mismatch: {stats}"
    only = count_model_parameters(model, trainable_only=True)
    assert only["trainable_params"] == 150, f"Trainable only mismatch: {only}"
    print("test_count_model_parameters passed.")

def test_is_empty():
    from src.utils import is_empty
    assert is_empty(None), "None should be empty"
    assert is_empty(""), "Empty string should be empty"
    assert is_empty("   "), "Whitespace string should be empty"
    assert is_empty([]), "Empty list should be empty"
    assert is_empty({}, "Empty dict should be empty")
    assert not is_empty("abc"), "Non-empty string should not be empty"
    assert not is_empty([1]), "Non-empty list should not be empty"
    print("test_is_empty passed.")

def test_dataset_text_lengths():
    from src.utils import dataset_text_lengths
    data = [
        {"text": "hello"},
        {"text": "world!"},
        {"text": ""},
        {"foo": "bar"}
    ]
    lengths = dataset_text_lengths(data)
    assert lengths == [5,6,0,0], f"Text lengths mismatch: {lengths}"
    print("test_dataset_text_lengths passed.")

def test_get_first_key():
    from src.utils import get_first_key
    d1 = {"a": 1, "b": 2}
    d2 = {}
    d3 = {"x": 10}
    d4 = "notadict"
    assert get_first_key(d1) == "a", f"Expected 'a' for d1"
    assert get_first_key(d2) is None, f"Expected None for empty dict"
    assert get_first_key(d3) == "x", f"Expected 'x' for d3"
    assert get_first_key(d4) is None, f"Expected None for non-dict"
    print("test_get_first_key passed.")

def test_get_last_key():
    from src.utils import get_last_key
    d1 = {"a": 1, "b": 2}
    d2 = {}
    d3 = {"x": 10}
    d4 = "notadict"
    assert get_last_key(d1) == "b", f"Expected 'b' for d1"
    assert get_last_key(d2) is None, f"Expected None for empty dict"
    assert get_last_key(d3) == "x", f"Expected 'x' for d3"
    assert get_last_key(d4) is None, f"Expected None for non-dict"
    print("test_get_last_key passed.")

def test_get_keys():
    from src.utils import get_keys
    d1 = {"a": 1, "b": 2}
    d2 = {}
    d3 = {"x": 10, "y": 20}
    d4 = "notadict"
    d5 = None
    assert get_keys(d1) == ["a", "b"], f"Expected ['a', 'b'] for d1"
    assert get_keys(d2) == [], f"Expected [] for empty dict"
    assert get_keys(d3) == ["x", "y"], f"Expected ['x', 'y'] for d3"
    assert get_keys(d4) == [], f"Expected [] for non-dict"
    assert get_keys(d5) == [], f"Expected [] for None input"
    print("test_get_keys passed.")

def test_evaluate_generation_quality():
    from src.eval_generation import evaluate_generation_quality
    class DummyTokenizer:
        def encode(self, s):
            return list(s)
        def batch_decode(self, ids, skip_special_tokens=True):
            # ids is list of lists; decode to string
            if isinstance(ids[0], list):
                return ["".join(map(str, idseq)) for idseq in ids]
            else:
                return ["".join(map(str, ids))]
        eos_token_id = 0
    class DummyModel:
        def to(self, device):
            return self
        def generate(self, **kwargs):
            prompts = kwargs['input_ids'].tolist() if 'input_ids' in kwargs else [[1,2,3]]
            # For each prompt, return input_ids + [100]
            import torch
            return torch.tensor([ids + [100] for ids in prompts])
    dummy_tokenizer = DummyTokenizer()
    dummy_model = DummyModel()
    prompts = ["abc", "123"]
    # Simulate input_ids
    def dummy_tokenizer_call(prompts, return_tensors, padding, truncation):
        import torch
        return torch.tensor([[ord(c) for c in p] for p in prompts])
    dummy_tokenizer.__call__ = dummy_tokenizer_call
    results = evaluate_generation_quality(
        dummy_model, dummy_tokenizer, prompts,
        references=["100", "100"],
        max_new_tokens=1,
        metrics=["length", "exact_match"],
        batch_size=1,
        device="cpu"
    )
    assert "outputs" in results and "lengths" in results and "exact_match" in results, "Missing keys in results"
    assert isinstance(results["lengths"], list), "Lengths should be a list"
    assert 0 <= results["exact_match"] <= 1, f"Exact match out of range: {results['exact_match']}"
    print("test_evaluate_generation_quality passed.")

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
