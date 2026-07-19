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
        deduped = load_jsonl(output_path)
        assert len(deduped) == 3, f"Expected 3 unique items, got {len(deduped)}"
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
    nested = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    flat = flatten_dict(nested)
    # Expect keys: 'a', 'b.c', 'b.d.e'
    assert flat['a'] == 1
    assert flat['b.c'] == 2
    assert flat['b.d.e'] == 3
    print("test_flatten_dict passed.")

def test_get_model_family():
    # Minimal test for get_model_family utility
    names = [
        'microsoft/Phi-3-mini-4k-instruct',
        'Qwen/Qwen1.5-7B-Chat',
        'meta-llama/Meta-Llama-3-8B',
        'unknown-model/foobar',
        'Llama-3-Open',
        'qwen2.5-14b',
        'phi3-mixed',
        'llama3',
        'llama-3'
    ]
    expected = [
        'phi', 'qwen', 'llama', 'unknown', 'llama', 'qwen', 'phi', 'llama', 'llama'
    ]
    for name, exp in zip(names, expected):
        fam = get_model_family(name)
        assert fam == exp, f"Expected {exp}, got {fam} for {name}"
    print("test_get_model_family passed.")

def test_eval_loss_on_dataset():
    # Smoke test for eval_loss_on_dataset utility
    from src.eval_loss import eval_loss_on_dataset
    class DummyTrainer:
        def evaluate(self, eval_dataset=None):
            return {'eval_loss': 1.234}
    dummy_trainer = DummyTrainer()
    dummy_dataset = [1, 2, 3]
    loss = eval_loss_on_dataset(dummy_trainer, dummy_dataset)
    assert abs(loss - 1.234) < 1e-6, f"Expected 1.234, got {loss}"
    print("test_eval_loss_on_dataset passed.")

def test_count_model_parameters():
    # Smoke test for count_model_parameters utility
    from src.trainer import count_model_parameters
    class DummyModule:
        def parameters(self):
            class P:
                def __init__(self, n, grad):
                    self._n = n
                    self.requires_grad = grad
                def numel(self):
                    return self._n
            return [P(10, True), P(20, False), P(30, True)]
    dummy_model = DummyModule()
    stats = count_model_parameters(dummy_model)
    assert stats['total_params'] == 60
    assert stats['trainable_params'] == 40
    assert stats['non_trainable_params'] == 20
    only = count_model_parameters(dummy_model, trainable_only=True)
    assert only['trainable_params'] == 40
    print("test_count_model_parameters passed.")

def test_is_empty():
    # Minimal test for is_empty utility
    assert is_empty(None)
    assert is_empty('')
    assert is_empty([])
    assert not is_empty('abc')
    assert not is_empty([1])
    print("test_is_empty passed.")

def test_dataset_text_lengths():
    # Minimal test for dataset_text_lengths utility
    items = [
        {'text': 'abc'},
        {'text': 'defgh'},
        {'text': ''},
        {'text': 'ijkl'}
    ]
    lengths = dataset_text_lengths(items)
    assert lengths == [3, 5, 0, 4], f"Expected [3,5,0,4], got {lengths}"
    print("test_dataset_text_lengths passed.")

def test_get_first_key():
    # Minimal test for get_first_key utility
    d = {'x': 1, 'y': 2}
    k = get_first_key(d)
    assert k == 'x', f"Expected 'x', got {k}"
    k_none = get_first_key(None)
    assert k_none is None
    print("test_get_first_key passed.")

def test_get_last_key():
    # Minimal test for get_last_key utility
    d = {'a': 1, 'b': 2, 'c': 3}
    k = get_last_key(d)
    assert k == 'c', f"Expected 'c', got {k}"
    k_none = get_last_key({})
    assert k_none is None
    print("test_get_last_key passed.")

def test_get_keys():
    # Minimal test for get_keys utility
    d = {'foo': 1, 'bar': 2}
    keys = get_keys(d)
    assert keys == ['foo', 'bar'], f"Expected ['foo', 'bar'], got {keys}"
    keys_empty = get_keys(None)
    assert keys_empty == []
    print("test_get_keys passed.")

def test_evaluate_generation_quality():
    # Minimal test for evaluate_generation_quality utility
    from src.eval_generation import evaluate_generation_quality
    class DummyModel:
        def to(self, device):
            return self
        def generate(self, **kwargs):
            # just return input_ids unchanged
            return kwargs['input_ids']
    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 0
        def __call__(self, prompts, return_tensors=None, padding=None, truncation=None):
            # just return dict with input_ids
            return type('Dummy', (), {'input_ids': [[1,2],[3,4]], 'to': lambda self, device: self})()
        def batch_decode(self, gen_ids, skip_special_tokens=None):
            return ['prompt A response', 'prompt B response']
        def encode(self, text):
            return [0] * len(text)
    model = DummyModel()
    tokenizer = DummyTokenizer()
    prompts = ['prompt A', 'prompt B']
    references = ['response', 'response']
    metrics = ['length', 'exact_match']
    results = evaluate_generation_quality(model, tokenizer, prompts, references=references, metrics=metrics)
    assert 'outputs' in results and 'lengths' in results and 'exact_match' in results
    print("test_evaluate_generation_quality passed.")

def test_generate_outputs():
    # Minimal test for generate_outputs utility
    from src.eval_generation import generate_outputs
    class DummyModel:
        def to(self, device):
            return self
        def generate(self, **kwargs):
            return kwargs['input_ids']
    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 0
        def __call__(self, prompts, return_tensors=None, padding=None, truncation=None):
            return type('Dummy', (), {'input_ids': [[1,2],[3,4]], 'to': lambda self, device: self})()
        def batch_decode(self, gen_ids, skip_special_tokens=None):
            return ['prompt one output', 'prompt two output']
    model = DummyModel()
    tokenizer = DummyTokenizer()
    prompts = ['prompt one', 'prompt two']
    outputs = generate_outputs(model, tokenizer, prompts, batch_size=2, device='cpu')
    assert len(outputs) == 2
    print("test_generate_outputs passed.")

# New test for yaml config loader

def test_load_train_config_from_yaml():
    import yaml
    from src.yaml_config_loader import load_train_config_from_yaml
    # Minimal YAML config
    yaml_content = '''
base_model: "test/model"
dataset_path: "data/example.jsonl"
output_dir: "outputs/example"
epochs: 2
learning_rate: 0.001
batch_size: 8
grad_accum_steps: 1
max_seq_length: 1024
seed: 99
use_lora: true
lora_r: 4
lora_alpha: 8
lora_dropout: 0.2
lora_target_modules: ["q_proj"]
warmup_ratio: 0.05
weight_decay: 0.01
resume_from: null
gradient_checkpointing: true
'''
    test_yaml_path = 'test_tmp_config.yaml'
    with open(test_yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    try:
        cfg = load_train_config_from_yaml(test_yaml_path)
        assert cfg.base_model == "test/model"
        assert cfg.dataset_path == "data/example.jsonl"
        assert cfg.epochs == 2
        assert cfg.batch_size == 8
        assert cfg.gradient_checkpointing is True
        print("test_load_train_config_from_yaml passed.")
    finally:
        try:
            os.remove(test_yaml_path)
        except FileNotFoundError:
            pass
        except PermissionError:
            pass

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
