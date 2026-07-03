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
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                # Skip invalid JSON lines silently
                continue
            except Exception:
                # Other exceptions (rare), also skip
                continue
    return data


def validate_jsonl_schema(path: str, schema_fn: Callable[[Any], bool]) -> int:
    """
    Checks each line in JSONL file against schema_fn.
    Returns number of invalid lines.
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


def deduplicate_jsonl(path: str, output_path: str) -> None:
    """
    Deduplicates JSONL file based on hash of each object.
    Blank lines are ignored.
    """
    seen = set()
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    with open(output_path, 'w', encoding='utf-8') as out:
        for line in lines:
            try:
                obj = json.loads(line)
                key = json.dumps(obj, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    out.write(line + '\n')
            except json.JSONDecodeError:
                # Skip invalid JSON lines silently
                continue
            except Exception:
                # Other exceptions (rare), also skip
                continue


def deduplicate_texts(texts: List[str]) -> List[str]:
    """
    Removes duplicate text entries (case-sensitive).
    Preserves order (first occurrence kept).
    Args:
        texts: list of strings
    Returns:
        list of unique strings (order preserved)
    """
    seen = set()
    unique = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def normalize_text(text: str) -> str:
    """
    Normalizes text for robust comparison and deduplication.
    - Lowercases
    - Removes leading/trailing whitespace
    - Collapses internal whitespace to single space
    - Strips punctuation
    Args:
        text: input string
    Returns:
        normalized string
    """
    if not isinstance(text, str):
        return ''
    # Lowercase
    text = text.lower()
    # Remove leading/trailing whitespace
    text = text.strip()
    # Collapse internal whitespace to single space
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[\p{P}\p{S}]', '', text)
    # For broader punctuation removal (unicode), fallback:
    text = re.sub(r'[^\w\s]', '', text)
    return text


def train_val_split(
    data: List[Any],
    val_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[List[Any], List[Any]]:
    """
    Randomly split dataset into train and validation sets with seed control.
    Args:
        data: list of items
        val_ratio: fraction of items
        seed: random seed (optional)
    Returns:
        train, val splits (lists)
    """
    n = len(data)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_size = max(1, int(n * val_ratio))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    return train, val


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Recursively flattens a nested dict.
    Args:
        d: dict to flatten
        parent_key: prefix for keys
        sep: separator
    Returns:
        flat dict with dotted keys
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def is_numeric(val: Any) -> bool:
    """
    Returns True if val is int or float, not bool.
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def sample_jsonl(path: str, n: int, seed: Optional[int] = None) -> List[Any]:
    """
    Randomly samples n lines from JSONL file.
    Args:
        path: JSONL path
        n: number of samples
        seed: random seed
    Returns:
        list of sampled objects
    """
    all_data = load_jsonl(path)
    rng = random.Random(seed)
    if n >= len(all_data):
        return all_data
    idx = rng.sample(range(len(all_data)), n)
    return [all_data[i] for i in idx]


def get_first_non_empty(items: List[Any]) -> Any:
    """
    Returns the first non-empty item from a list.
    Empty means None, '', [], {}
    """
    for item in items:
        if item not in (None, '', [], {}):
            return item
    return None


def get_last_non_empty(items: List[Any]) -> Any:
    """
    Returns the last non-empty item from a list.
    Empty means None, '', [], {}
    """
    for item in reversed(items):
        if item not in (None, '', [], {}):
            return item
    return None


def get_non_empty(items: List[Any]) -> List[Any]:
    """
    Returns all non-empty items from a list.
    Empty means None, '', [], {}
    """
    return [item for item in items if item not in (None, '', [], {})]


def get_model_family(model_name: str) -> str:
    """
    Infers model family from model name string.
    Returns one of: 'phi', 'qwen', 'llama3', or 'unknown'
    """
    name = model_name.lower()
    if 'phi' in name:
        return 'phi'
    if 'qwen' in name:
        return 'qwen'
    if 'llama-3' in name or 'llama3' in name:
        return 'llama3'
    return 'unknown'


def dataset_stats(data: List[Any], text_key: str = "text", label_key: str = "label") -> dict:
    """
    Quick stats for token length and label balance.
    Args:
        data: list of dicts
        text_key: which key to measure lengths
        label_key: which key for label balance
    Returns:
        dict of stats
    """
    lengths = []
    label_counter = {}
    for item in data:
        text = item.get(text_key, "")
        lengths.append(len(text))
        label = item.get(label_key, None)
        if label is not None:
            label_counter[label] = label_counter.get(label, 0) + 1
    stats = {
        "count": len(data),
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "mean_len": sum(lengths) / len(lengths) if lengths else 0,
        "label_counts": label_counter
    }
    return stats


def dataset_sample_stats(data: List[Any], n: int = 20, seed: Optional[int] = None) -> dict:
    """
    Basic stats on a random sample of dataset items.
    Args:
        data: list of dicts
        n: sample size
        seed: random seed
    Returns:
        dict of stats
    """
    if not data:
        return {}
    rng = random.Random(seed)
    sample = rng.sample(data, min(n, len(data)))
    lengths = []
    keys_counter = {}
    label_counter = {}
    for item in sample:
        text = item.get("text", "")
        lengths.append(len(text))
        keys = list(item.keys())
        for k in keys:
            keys_counter[k] = keys_counter.get(k, 0) + 1
        label = item.get("label", None)
        if label is not None:
            label_counter[label] = label_counter.get(label, 0) + 1
    stats = {
        "sample_size": n,
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "mean_len": sum(lengths) / len(lengths) if lengths else 0,
        "most_common_keys": sorted(keys_counter.items(), key=lambda x: -x[1]),
        "label_counts": label_counter
    }
    return stats
