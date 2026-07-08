"""
Data utility functions for JSONL dataset loading, validation, deduplication, and splitting.

Includes:
- count_jsonl_lines: count non-empty lines in JSONL file
- load_jsonl: load objects from JSONL, skip blank lines
- validate_jsonl_schema: check each line against a schema function
- deduplicate_jsonl: remove duplicate JSONL objects based on hash
- train_val_split: reproducible train/val split by ratio and seed
- get_model_family: infer major model family from model name string
- dataset_stats: quick stats for token length and label balance
- normalize_text: text normalization for robust comparison (new)
- flatten_dict: recursively flatten nested dicts for easier metric aggregation
- is_numeric: check if value is a numeric type (int/float, not bool)
- sample_jsonl: randomly sample N lines from JSONL file (new)
- get_first_non_empty: returns the first non-empty item from a list (new)
- get_last_non_empty: returns the last non-empty item from a list (new)
- get_non_empty: returns all non-empty items from a list (new)
- dataset_sample_stats: basic stats on a random sample of dataset items (new)
- deduplicate_texts: remove duplicate text items from a list (new)
- pad_or_truncate: pad or truncate a sequence to a fixed length (new)
- is_empty: returns True if value is None, empty, or only whitespace (new)
- dataset_token_counts: computes token counts for each item in a dataset (new)

Usage notes:
- All utilities are designed for quick experiment scaffolding: call directly or batch in scripts.
- load_jsonl and validate_jsonl_schema are robust to blank lines and minor format errors.
- train_val_split and deduplicate_jsonl support reproducible workflow for small datasets.
- get_model_family is handy for auto-selecting prompt templates by model name.
- Use dataset_stats before/after deduplication to check dataset quality.
- sample_jsonl enables rapid prototyping and debugging on small data slices.
- flatten_dict is ideal for preparing results for aggregate_metrics or dashboard reporting.
- dataset_sample_stats provides basic statistics on a random subset (lengths, keys, label balance).
- deduplicate_texts is a quick utility for removing duplicate text strings (case-sensitive).
- normalize_text can be used for robust deduplication and comparison of text samples (lowercase, whitespace, punctuation-stripping).
- pad_or_truncate is handy for prepping tokenized inputs to a fixed length (for batching, eval, etc).
- is_empty is a simple helper for filtering or validation of values (None, '', [], etc).
- dataset_token_counts helps analyze tokenization stats for batching and memory planning (new).

Example:
    # Load and deduplicate dataset
    data = load_jsonl('data.jsonl')
    train, val = train_val_split(data, val_ratio=0.1, seed=42)
    stats = dataset_stats(train)
    print(stats)
    deduplicate_jsonl('data.jsonl', 'data_deduped.jsonl')
    # Pad tokens for batching
    tokens = [101, 102, 103]
    tokens = pad_or_truncate(tokens, length=5)
    # Filter empty items
    filtered = [x for x in data if not is_empty(x)]
    # Compute token counts per item
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
    counts = dataset_token_counts(data, tokenizer)
    print(counts)

Caveats:
- These utilities are not optimized for large-scale datasets (>100k items); use for experiment prototyping.
- Deduplication is based on object hash; small differences (ordering, whitespace) may defeat it.
- Text normalization is aggressive; tune for your use-case if needed.
- train_val_split uses random.shuffle; reproducibility is controlled by seed.
- is_empty checks for None, '', [], {}, and whitespace-only strings. It does not check for False.
- dataset_token_counts assumes you provide a compatible tokenizer and optionally the text field name.

"""
import json
import random
from typing import List, Any, Callable, Optional, Tuple, Dict
import re


def count_jsonl_lines(path: str) -> int:
    """
    Counts number of non-empty lines in JSONL file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def load_jsonl(path: str) -> List[Any]:
    """
    Loads JSONL file, skips blank lines.
    Ignores invalid JSON lines, prints one warning per file if any are encountered.
    """
    data = []
    had_invalid = False
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                if not had_invalid:
                    print(f"Warning: invalid JSON line detected in {path}, line {idx+1}. Skipping.")
                    had_invalid = True
                continue
            except Exception as e:
                if not had_invalid:
                    print(f"Warning: unexpected error loading line {idx+1} of {path}: {e}. Skipping.")
                    had_invalid = True
                continue
    return data


def validate_jsonl_schema(path: str, schema_fn: Callable[[Any], bool]) -> int:
    """
    Checks each line in JSONL file against schema_fn.
    Returns number of invalid lines. Prints one warning if any parsing errors are encountered.
    """
    invalid = 0
    had_error = False
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not schema_fn(obj):
                    invalid += 1
            except Exception:
                if not had_error:
                    print(f"Warning: error parsing line {idx+1} in {path}. Skipping.")
                    had_error = True
                invalid += 1
    return invalid


def deduplicate_jsonl(path_in: str, path_out: str) -> None:
    """
    Deduplicates JSONL file (by object hash), writes unique items to path_out.
    """
    seen = set()
    unique = []
    with open(path_in, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                h = hash(json.dumps(obj, sort_keys=True))
                if h not in seen:
                    seen.add(h)
                    unique.append(obj)
            except Exception:
                continue
    with open(path_out, 'w', encoding='utf-8') as f:
        for obj in unique:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def train_val_split(data: List[Any], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Any], List[Any]]:
    """
    Splits data into train/val sets by ratio. Shuffle with seed.
    """
    n = len(data)
    idxs = list(range(n))
    random.seed(seed)
    random.shuffle(idxs)
    val_size = int(n * val_ratio)
    val_idxs = idxs[:val_size]
    train_idxs = idxs[val_size:]
    train = [data[i] for i in train_idxs]
    val = [data[i] for i in val_idxs]
    return train, val


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Recursively flattens nested dicts. Keys are joined by sep.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_model_family(model_name: str) -> str:
    """
    Infer model family from model name string.
    """
    s = model_name.lower()
    if 'phi' in s:
        return 'phi'
    if 'qwen' in s:
        return 'qwen'
    if 'llama' in s or 'lama' in s:
        return 'llama'
    if 'mistral' in s:
        return 'mistral'
    if 'falcon' in s:
        return 'falcon'
    if 'gemma' in s:
        return 'gemma'
    return 'other'


def dataset_stats(data: List[Any], text_key: str = 'text', label_key: Optional[str] = None) -> dict:
    """
    Computes basic stats for dataset: token lengths, label balance.
    """
    lengths = []
    labels = []
    for item in data:
        txt = item.get(text_key, '') if isinstance(item, dict) else str(item)
        lengths.append(len(txt))
        if label_key and label_key in item:
            labels.append(item[label_key])
    stats = {
        'count': len(data),
        'mean_length': float(sum(lengths) / len(lengths)) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'label_balance': {}
    }
    if labels:
        from collections import Counter
        stats['label_balance'] = dict(Counter(labels))
    return stats


def normalize_text(text: str) -> str:
    """
    Aggressive text normalization: lowercase, strip whitespace, remove punctuation.
    """
    text = text.strip().lower()
    text = re.sub(r'[\s]+', ' ', text)
    text = re.sub(r'[\p{P}\p{S}]', '', text) if hasattr(re, 'UNICODE') else re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@[\\]^_`{|}~]', '', text)
    return text


def is_numeric(val: Any) -> bool:
    """
    Returns True if val is int or float, but not bool.
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def sample_jsonl(path: str, n: int, seed: int = 42) -> List[Any]:
    """
    Randomly samples n items from JSONL file.
    """
    data = load_jsonl(path)
    random.seed(seed)
    if n >= len(data):
        return data
    idxs = random.sample(range(len(data)), n)
    return [data[i] for i in idxs]


def get_first_non_empty(items: List[Any]) -> Any:
    """
    Returns first non-empty item from list, or None.
    """
    for item in items:
        if not is_empty(item):
            return item
    return None


def get_last_non_empty(items: List[Any]) -> Any:
    """
    Returns last non-empty item from list, or None.
    """
    for item in reversed(items):
        if not is_empty(item):
            return item
    return None


def get_non_empty(items: List[Any]) -> List[Any]:
    """
    Returns all non-empty items from list.
    """
    return [item for item in items if not is_empty(item)]


def dataset_sample_stats(data: List[Any], sample_size: int = 20, text_key: str = 'text', seed: int = 42) -> dict:
    """
    Computes stats on a random sample of dataset items.
    """
    random.seed(seed)
    sample = random.sample(data, min(sample_size, len(data)))
    keys = set()
    lengths = []
    for item in sample:
        if isinstance(item, dict):
            keys.update(item.keys())
            txt = item.get(text_key, '')
        else:
            txt = str(item)
        lengths.append(len(txt))
    return {
        'sample_size': len(sample),
        'mean_length': float(sum(lengths)/len(lengths)) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'keys': sorted(keys),
    }


def deduplicate_texts(texts: List[str]) -> List[str]:
    """
    Removes duplicate strings (case-sensitive).
    """
    seen = set()
    unique = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def pad_or_truncate(seq: List[Any], length: int, pad_value: Any = 0) -> List[Any]:
    """
    Pads or truncates sequence to fixed length.
    """
    if len(seq) > length:
        return seq[:length]
    elif len(seq) < length:
        return seq + [pad_value] * (length - len(seq))
    else:
        return seq


def is_empty(val: Any) -> bool:
    """
    Returns True if value is None, empty string, empty list/dict, or only whitespace.
    False for numbers, True/False, non-empty containers.
    """
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip() == ''
    if isinstance(val, (list, dict, set, tuple)):
        return len(val) == 0
    return False


def dataset_token_counts(
    data: List[Any],
    tokenizer,
    text_key: str = 'text',
    add_special_tokens: bool = True
) -> List[int]:
    """
    Computes token counts for each item in a dataset using the provided tokenizer.
    Args:
        data: list of items (dicts, each with text_key or string)
        tokenizer: HuggingFace tokenizer instance
        text_key: key to extract text from dict (default: 'text')
        add_special_tokens: whether to add special tokens when encoding
    Returns:
        List of token counts for each item
    """
    counts = []
    for item in data:
        if isinstance(item, dict):
            txt = item.get(text_key, '')
        else:
            txt = str(item)
        tokens = tokenizer.encode(txt, add_special_tokens=add_special_tokens)
        counts.append(len(tokens))
    return counts
