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
        # Load dedu
        deduped = load_jsonl(output_path)
        assert len(deduped) == 3, f"Expected 3 unique items, got {len(deduped)}"
        ids = set(obj['id'] for obj in deduped)
        assert ids == {1, 2, 3}, f"Unexpected ids: {ids}"
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
    # Test recursive flattening
    nested = {
        'a': 1,
        'b': {
            'c': 2,
            'd': {
                'e': 3
            }
        }
    }
    flat = flatten_dict(nested)
    expected = {
        'a': 1,
        'b.c': 2,
        'b.d.e': 3
    }
    assert flat == expected, f"Expected {expected}, got {flat}"
    print("test_flatten_dict passed.")

def test_get_model_family():
    # Typical model names
    names = [
        "microsoft/Phi-3-mini-4k-instruct",
        "Qwen/Qwen1.5-7B-Chat",
        "meta-llama/Meta-Llama-3-8B",
        "unknown-model/foobar",
        "Llama-3-Open",
        "qwen2.5-14b",
        "phi3-mixed",
        "llama3",
        "llama-3"
    ]
    expected = [
        "phi3",
        "qwen",
        "llama3",
        "unknown",
        "llama3",
        "qwen",
        "phi3",
        "llama3",
        "llama3"
    ]
    results = [get_model_family(name) for name in names]
    assert results == expected, f"Expected {expected}, got {results}"
    print("test_get_model_family passed.")

# Minimal test for eval_loss_on_dataset
from src.eval_loss import eval_loss_on_dataset
class DummyTrainer:
    def evaluate(self, eval_dataset=None):
        return {'eval_loss': 1.23}
def test_eval_loss_on_dataset():
    trainer = DummyTrainer()
    loss = eval_loss_on_dataset(trainer, eval_dataset=[1,2,3])
    assert abs(loss - 1.23) < 1e-6, f"Expected loss 1.23, got {loss}"
    print("test_eval_loss_on_dataset passed.")

# Minimal test for count_model_parameters
from src.trainer import count_model_parameters
class DummyModel:
    def parameters(self):
        class P:
            def __init__(self, numel, grad):
                self._numel = numel
                self.requires_grad = grad
            def numel(self):
                return self._numel
        # 2 trainable, 1 frozen
        return [P(100, True), P(200, True), P(300, False)]
def test_count_model_parameters():
    model = DummyModel()
    stats = count_model_parameters(model)
    assert stats['total_params'] == 600, f"Total params mismatch: {stats}"
    assert stats['trainable_params'] == 300, f"Trainable mismatch: {stats}"
    assert stats['non_trainable_params'] == 300, f"Non-trainable mismatch: {stats}"
    stats2 = count_model_parameters(model, trainable_only=True)
    assert stats2['trainable_params'] == 300, f"Trainable only mismatch: {stats2}"
    print("test_count_model_parameters passed.")

def test_is_empty():
    # Should detect None, empty string, empty list, blank spaces
    cases = [None, '', [], {}, '   ', '\n', 'x', [1], {'a': 1}]
    expected = [True, True, True, True, True, True, False, False, False]
    results = [is_empty(x) for x in cases]
    assert results == expected, f"Expected {expected}, got {results}"
    print("test_is_empty passed.")

def test_dataset_text_lengths():
    # Should return correct character lengths
    dataset = [
        {'text': 'hello'},
        {'text': 'world!'},
        {'text': ''}
    ]
    lens = dataset_text_lengths(dataset)
    expected = [5, 6, 0]
    assert lens == expected, f"Expected {expected}, got {lens}"
    print("test_dataset_text_lengths passed.")

def test_get_first_key():
    d1 = {'a': 1, 'b': 2}
    d2 = {}
    d3 = ['not_a_dict']
    assert get_first_key(d1) == 'a', f"Expected 'a', got {get_first_key(d1)}"
    assert get_first_key(d2) is None, f"Expected None, got {get_first_key(d2)}"
    assert get_first_key(d3) is None, f"Expected None, got {get_first_key(d3)}"
    print("test_get_first_key passed.")

def test_get_last_key():
    d1 = {'a': 1, 'b': 2, 'c': 3}
    d2 = {}
    assert get_last_key(d1) == 'c', f"Expected 'c', got {get_last_key(d1)}"
    assert get_last_key(d2) is None, f"Expected None, got {get_last_key(d2)}"
    print("test_get_last_key passed.")

def test_get_keys():
    d1 = {'x': 10, 'y': 20}
    d2 = {}
    d3 = ['not_a_dict']
    assert get_keys(d1) == ['x', 'y'], f"Expected ['x', 'y'], got {get_keys(d1)}"
    assert get_keys(d2) == [], f"Expected [], got {get_keys(d2)}"
    assert get_keys(d3) == [], f"Expected [], got {get_keys(d3)}"
    print("test_get_keys passed.")

# Minimal test for evaluate_generation_quality
from src.eval_generation import evaluate_generation_quality
class DummyTokenizer:
    def __init__(self):
        pass
    def encode(self, text):
        return list(text)
    def __call__(self, batch_prompts, return_tensors=None, padding=None, truncation=None):
        # Fake tokenization
        return {'input_ids': [[1,2,3]]*len(batch_prompts), 'attention_mask': [[1,1,1]]*len(batch_prompts), 'to': lambda device: self}
    def batch_decode(self, batch_ids, skip_special_tokens=True):
        # Just return prompt + " response"
        return ["prompt response" for _ in batch_ids]
    @property
    def eos_token_id(self):
        return 0
class DummyModel:
    def to(self, device):
        return self
    def generate(self, **inputs):
        # Return dummy ids
        batch_size = len(inputs['input_ids'])
        return [[1,2,3,4] for _ in range(batch_size)]
def test_evaluate_generation_quality():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    prompts = ["prompt1", "prompt2"]
    refs = ["response1", "response2"]
    results = evaluate_generation_quality(model, tokenizer, prompts, references=refs, metrics=["length", "exact_match"])
    assert "outputs" in results and "lengths" in results and "exact_match" in results, f"Missing keys in results"
    assert isinstance(results["lengths"], list) and isinstance(results["exact_match"], float), f"Wrong types"
    print("test_evaluate_generation_quality passed.")

# Minimal test for generate_outputs utility
from src.eval_generation import generate_outputs
class DummyTokenizer2(DummyTokenizer):
    def batch_decode(self, batch_ids, skip_special_tokens=True):
        # Return original prompt plus " test"
        return ["prompt1 test", "prompt2 test"]
    def __call__(self, batch_prompts, return_tensors=None, padding=None, truncation=None):
        class Fake:
            def to(self, device):
                return self
        return Fake()
class DummyModel2(DummyModel):
    def generate(self, **inputs):
        # Returns dummy ids for two prompts
        return [None, None]
def test_generate_outputs():
    model = DummyModel2()
    tokenizer = DummyTokenizer2()
    prompts = ["prompt1", "prompt2"]
    outputs = generate_outputs(model, tokenizer, prompts, batch_size=2, device=None)
    assert isinstance(outputs, list), f"Expected list, got {type(outputs)}"
    assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"
    assert outputs[0] == "test" and outputs[1] == "test", f"Unexpected outputs: {outputs}"
    print("test_generate_outputs passed.")

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
