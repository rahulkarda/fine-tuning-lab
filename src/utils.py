"""
Data utility functions for JSONL dataset loading, validation, deduplication, and splitting.

Includes:
- count_jsonl_lines: count non-empty lines in JSONL file
- load_jsonl: load objects from JSONL, skip blank lines
- validate_jsonl_schema: check each line against a schema function
- deduplicate_jsonl: remove duplicate JSONL objects based on hash
- train_val_split: reproducible train/val split by ratio and seed
- get_model_family: infer major model family from model name string (refactored, clarified logic)
- dataset_stats: quick stats for token length and label balance
- normalize_text: text normalization for robust comparison (new)
- flatten_dict: recursively flatten nested dicts for easier metric aggregation (refactored, clarified)
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
- dataset_text_lengths: computes text length (number of characters) per item (new)
- dataset_label_counts: computes label distribution for quick stats (new)
- get_first_key: returns the first key from a dict, or None if empty (new)
- get_last_key: returns the last key from a dict, or None if empty (new)
- get_keys: returns a list of keys from a dict, or empty list if not a dict (new)

Usage notes:
- All utilities are designed for quick experiment scaffolding: call directly or batch in scripts.
- load_jsonl and validate_jsonl_schema are robust to blank lines and minor format errors.
- train_val_split and deduplicate_jsonl support reproducible workflow for small datasets.
- get_model_family is handy for auto-selecting prompt templates by model name (clarified logic).
- Use dataset_stats before/after deduplication to check dataset quality.
- sample_jsonl enables rapid prototyping and debugging on small data slices.
- flatten_dict is ideal for preparing results for aggregate_metrics or dashboard reporting (clarified logic).
- dataset_sample_stats provides basic statistics on a random subset (lengths, keys, label balance).
- deduplicate_texts is a quick utility for removing duplicate text strings (case-sensitive).
- normalize_text can be used for robust deduplication and comparison of text samples (lowercase, whitespace, punctuation-stripping).
- pad_or_truncate is handy for prepping tokenized inputs to a fixed length (for batching, eval, etc).
- is_empty is a simple helper for filtering or validation of values (None, '', [], etc).
- dataset_token_counts helps analyze tokenization stats for batching and memory planning (new; pass a HuggingFace tokenizer).
- dataset_text_lengths computes text length (in characters) for each item, useful for quick stats (new).
- dataset_label_counts computes label distribution statistics for classification tasks (new; pass dataset and label key).
- get_first_key returns the first key from a dict, useful for quick schema inspection (new).
- get_last_key returns the last key from a dict, useful for quick schema inspection (new).
- get_keys returns a list of keys from a dict, useful for schema inspection or iterating fields (new).

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
    # Compute text lengths per item
    lengths = dataset_text_lengths(data)
    print(lengths)
    # Compute label distribution
    label_counts = dataset_label_counts(data, label_key='label')
    print(label_counts)
    # Get first key from schema dict
    key = get_first_key(data[0])
    print(key)
    # Get last key from schema dict
    last_key = get_last_key(data[0])
    print(last_key)
    # Get all keys from schema dict
    keys = get_keys(data[0])
    print(keys)

Caveats:
- These utilities are not optimized for large-scale datasets (>100k items); use for experiment prototyping.
- Deduplication is based on object hash; small differences (ordering, whitespace) may defeat it.
- Template selection in get_model_family is clarified: prefers substring match, case-insensitive, with fallback.
"""
import json
import random
import hashlib
import re
from typing import Any, Callable, List, Tuple, Optional


def count_jsonl_lines(path: str) -> int:
    """
    Counts non-empty lines in a JSONL file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def load_jsonl(path: str) -> List[Any]:
    """
    Loads objects from JSONL file, skipping blank lines.
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except Exception:
                continue
    return data


def validate_jsonl_schema(path: str, schema_fn: Callable[[Any], bool]) -> bool:
    """
    Validates each JSONL object against schema_fn.
    Returns True if all pass, False otherwise.
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                return False
            if not schema_fn(obj):
                return False
    return True


def deduplicate_jsonl(input_path: str, output_path: str) -> None:
    """
    Deduplicates JSONL objects by hash and writes to output_path.
    """
    seen = set()
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            h = hashlib.md5(line.strip().encode('utf-8')).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            fout.write(line)


def train_val_split(data: List[Any], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Any], List[Any]]:
    """
    Train/val split with seed control and ratio.
    """
    n = len(data)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    val_size = int(n * val_ratio)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    return train, val


def get_model_family(model_name: str) -> str:
    """
    Infers model family from model name string.
    Returns one of: 'phi', 'qwen', 'llama', 'mistral', 'gemma', 'falcon', or 'other'.
    Case-insensitive, prefers substring match, clarified for ambiguous names.
    """
    name = model_name.lower()
    if 'phi' in name:
        return 'phi'
    elif 'qwen' in name:
        return 'qwen'
    elif 'llama' in name or 'llama-3' in name or 'llama3' in name:
        return 'llama'
    elif 'mistral' in name:
        return 'mistral'
    elif 'gemma' in name:
        return 'gemma'
    elif 'falcon' in name:
        return 'falcon'
    else:
        return 'other'


def dataset_stats(data: List[Any]) -> Any:
    """
    Prints quick stats for token length and label balance.
    """
    if not data:
        return {}
    lengths = [len(str(x.get('text', ''))) for x in data]
    labels = [x.get('label', None) for x in data if 'label' in x]
    stats = {
        'count': len(data),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'mean_length': sum(lengths)/len(lengths),
        'label_counts': {l: labels.count(l) for l in set(labels)}
    }
    return stats


def normalize_text(text: str) -> str:
    """
    Lowercase, strip, remove punctuation for robust comparison.
    """
    text = text.lower().strip()
    text = re.sub(r'[\W_]+', '', text)
    return text


def flatten_dict(d: Any, parent_key: str = '', sep: str = '.') -> dict:
    """
    Recursively flattens a nested dict. Keys are joined by sep.
    """
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
    else:
        items[parent_key] = d
    return items


def is_numeric(val: Any) -> bool:
    """
    Returns True if val is int or float (but not bool).
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def sample_jsonl(path: str, n: int, seed: int = 42) -> List[Any]:
    """
    Returns random sample of n lines from JSONL.
    """
    data = load_jsonl(path)
    random.Random(seed).shuffle(data)
    return data[:n]


def get_first_non_empty(lst: List[Any]) -> Any:
    """
    Returns first non-empty item from list, or None.
    """
    for x in lst:
        if not is_empty(x):
            return x
    return None


def get_last_non_empty(lst: List[Any]) -> Any:
    """
    Returns last non-empty item from list, or None.
    """
    for x in reversed(lst):
        if not is_empty(x):
            return x
    return None


def get_non_empty(lst: List[Any]) -> List[Any]:
    """
    Returns all non-empty items from list.
    """
    return [x for x in lst if not is_empty(x)]


def dataset_sample_stats(data: List[Any], sample_size: int = 10, seed: int = 42) -> dict:
    """
    Basic stats on a random sample of dataset items.
    """
    sample = random.sample(data, min(sample_size, len(data))) if data else []
    keys = set()
    for item in sample:
        if isinstance(item, dict):
            keys.update(item.keys())
    stats = {
        'sample_size': len(sample),
        'keys': sorted(list(keys)),
        'lengths': [len(str(item.get('text', ''))) for item in sample if isinstance(item, dict)],
        'labels': [item.get('label', None) for item in sample if isinstance(item, dict) and 'label' in item]
    }
    return stats


def deduplicate_texts(texts: List[str]) -> List[str]:
    """
    Deduplicates text items (case-sensitive).
    """
    seen = set()
    out = []
    for t in texts:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def pad_or_truncate(seq: List[Any], length: int, pad_token: Any = 0) -> List[Any]:
    """
    Pads or truncates a sequence to fixed length.
    """
    if len(seq) > length:
        return seq[:length]
    else:
        return seq + [pad_token] * (length - len(seq))


def is_empty(val: Any) -> bool:
    """
    Returns True if val is None, empty, or only whitespace.
    """
    if val is None:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    if isinstance(val, (list, dict)) and not val:
        return True
    return False


def dataset_token_counts(data: List[Any], tokenizer: Any, text_key: str = 'text') -> List[int]:
    """
    Computes token counts for each item in dataset.
    """
    counts = []
    for x in data:
        text = x.get(text_key, '')
        tokens = tokenizer.encode(text)
        counts.append(len(tokens))
    return counts


def dataset_text_lengths(data: List[Any], text_key: str = 'text') -> List[int]:
    """
    Computes text length (characters) per item.
    """
    return [len(str(x.get(text_key, ''))) for x in data]


def dataset_label_counts(data: List[Any], label_key: str = 'label') -> dict:
    """
    Computes label distribution for quick stats.
    """
    labels = [x.get(label_key, None) for x in data if label_key in x]
    return {l: labels.count(l) for l in set(labels)}


def get_first_key(d: Any) -> Optional[str]:
    """
    Returns the first key from a dict, or None if empty/not a dict.
    """
    if not isinstance(d, dict) or not d:
        return None
    for k in d:
        return k
    return None


def get_last_key(d: Any) -> Optional[str]:
    """
    Returns the last key from a dict, or None if empty/not a dict.
    """
    if not isinstance(d, dict) or not d:
        return None
    last = None
    for k in d:
        last = k
    return last


def get_keys(d: Any) -> List[str]:
    """
    Returns a list of keys from a dict, or empty list if not a dict.
    """
    if not isinstance(d, dict):
        return []
    return list(d.keys())
