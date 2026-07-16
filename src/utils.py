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
- dataset_text_lengths: computes text length (number of characters) per item (new)
- dataset_label_counts: computes label distribution for quick stats (new)
- get_first_key: returns the first key from a dict, or None if empty (new)
- get_last_key: returns the last key from a dict, or None if empty (new)
- get_keys: returns a list of keys from a dict, or empty list if not a dict (new)

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
- dataset_text_lengths computes text length (in characters) for each item, useful for quick stats (new).
- dataset_label_counts computes label distribution statistics for classification tasks (new).
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
    label_counts = dataset_label_counts(data)
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
- Text normalization is aggressive; tune for your use-case if needed.
- train_val_split uses random.shuffle; reproducibility is controlled by seed.
- is_empty checks for None, '', [], {}, and whitespace-only strings. It does not check for False.
- dataset_token_counts assumes you provide a compatible tokenizer and optionally the text field name.
- dataset_text_lengths computes character length using len(str(text)).
- dataset_label_counts computes label distribution statistics for classification tasks.
- get_first_key returns the first key from a dict, useful for quick schema inspection.
- get_last_key returns the last key from a dict, useful for quick schema inspection.
- get_keys returns a list of keys from a dict, useful for schema inspection or iterating fields.
"""
import json
import random
from typing import Any, Callable, List, Optional, Dict, Tuple, Union
import string


def count_jsonl_lines(path: str) -> int:
    """
    Counts non-empty lines in a JSONL file.
    """
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_jsonl(path: str) -> List[Any]:
    """
    Loads JSON objects from JSONL file, skipping blank lines.
    """
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                pass
    return items


def validate_jsonl_schema(path: str, schema_fn: Callable[[Any], bool]) -> int:
    """
    Validates each line against schema_fn. Returns count of invalid lines.
    """
    invalid = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not schema_fn(obj):
                    invalid += 1
            except Exception:
                invalid += 1
    return invalid


def deduplicate_jsonl(input_path: str, output_path: str) -> None:
    """
    Removes duplicate JSONL objects based on hash.
    """
    seen = set()
    out_items = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                h = hash(json.dumps(obj, sort_keys=True))
                if h not in seen:
                    seen.add(h)
                    out_items.append(obj)
            except Exception:
                pass
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in out_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def train_val_split(data: List[Any], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Any], List[Any]]:
    """
    Reproducible train/val split by ratio and seed.
    """
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    val_size = int(round(len(data) * val_ratio))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    return train, val


def get_model_family(model_name: str) -> str:
    """
    Infers major model family from model name string.
    """
    name = model_name.lower()
    if "phi" in name:
        return "phi"
    if "qwen" in name:
        return "qwen"
    if "llama" in name or "meta-llama" in name or "llama3" in name or "llama-3" in name:
        return "llama"
    return "other"


def flatten_dict(d: Any, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Recursively flattens nested dicts for easier metric aggregation.
    """
    items = {}
    if not isinstance(d, dict):
        return items
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def is_numeric(val: Any) -> bool:
    """
    Checks if value is a numeric type (int/float, not bool).
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def sample_jsonl(path: str, n: int, seed: int = 42) -> List[Any]:
    """
    Randomly sample N lines from JSONL file.
    """
    data = load_jsonl(path)
    rng = random.Random(seed)
    rng.shuffle(data)
    return data[:n]


def get_first_non_empty(lst: List[Any]) -> Optional[Any]:
    """
    Returns the first non-empty item from a list.
    """
    for item in lst:
        if not is_empty(item):
            return item
    return None


def get_last_non_empty(lst: List[Any]) -> Optional[Any]:
    """
    Returns the last non-empty item from a list.
    """
    for item in reversed(lst):
        if not is_empty(item):
            return item
    return None


def get_non_empty(lst: List[Any]) -> List[Any]:
    """
    Returns all non-empty items from a list.
    """
    return [x for x in lst if not is_empty(x)]


def dataset_sample_stats(data: List[Any], n: int = 10, seed: int = 42) -> Dict[str, Any]:
    """
    Basic stats on a random sample of dataset items.
    """
    sample = data.copy()
    random.Random(seed).shuffle(sample)
    sample = sample[:n]
    keys = set()
    lengths = []
    for item in sample:
        if isinstance(item, dict):
            keys.update(item.keys())
        lengths.append(len(str(item)))
    return {
        "num_sample": len(sample),
        "keys": sorted(list(keys)),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
    }


def deduplicate_texts(texts: List[str]) -> List[str]:
    """
    Removes duplicate text items from a list (case-sensitive).
    """
    seen = set()
    out = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def normalize_text(text: str) -> str:
    """
    Text normalization: lowercase, strip, remove punctuation.
    """
    t = text.lower().strip()
    t = t.translate(str.maketrans('', '', string.punctuation))
    t = ' '.join(t.split())
    return t


def pad_or_truncate(seq: List[Any], length: int, pad_val: Any = 0) -> List[Any]:
    """
    Pad or truncate a sequence to a fixed length.
    """
    if len(seq) < length:
        return seq + [pad_val] * (length - len(seq))
    else:
        return seq[:length]


def is_empty(val: Any) -> bool:
    """
    Returns True if value is None, empty, or only whitespace.
    """
    if val is None:
        return True
    if isinstance(val, (str, bytes)):
        return not val.strip()
    if isinstance(val, (list, dict, set)):
        return len(val) == 0
    return False


def dataset_token_counts(data: List[Any], tokenizer: Any, text_field: str = "text") -> List[int]:
    """
    Computes token counts for each item in a dataset.
    """
    counts = []
    for item in data:
        if isinstance(item, dict) and text_field in item:
            text = item[text_field]
        else:
            text = str(item)
        try:
            tokens = tokenizer.encode(text)
            counts.append(len(tokens))
        except Exception:
            counts.append(0)
    return counts


def dataset_text_lengths(data: List[Any], text_field: str = "text") -> List[int]:
    """
    Computes text length (number of characters) per item.
    """
    lengths = []
    for item in data:
        if isinstance(item, dict) and text_field in item:
            text = item[text_field]
        else:
            text = str(item)
        lengths.append(len(text))
    return lengths


def dataset_label_counts(data: List[Any], label_field: str = "label") -> Dict[Any, int]:
    """
    Computes label distribution for quick stats.
    """
    counts = {}
    for item in data:
        if isinstance(item, dict) and label_field in item:
            label = item[label_field]
        elif isinstance(item, str):
            label = item
        else:
            label = None
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
    return counts


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
