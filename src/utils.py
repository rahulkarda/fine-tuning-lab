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
- normalize_text: text normalization for robust comparison
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


def train_val_split(
    data: List[Any],
    val_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[List[Any], List[Any]]:
    """
    Randomly split dataset into train and validation sets with seed control.
    Args:
        data: list of items
        val_ratio: fraction of items to assign to val set (0 < val_ratio <= 1)
        seed: random seed for reproducibility
    Returns:
        train, val: (list, list). If val_ratio==1, all data is validation; if val_ratio==0, all data is train.
        For empty input, returns two empty lists.
    """
    num_items = len(data)
    if num_items == 0:
        return [], []
    if not 0 < val_ratio <= 1:
        raise ValueError("val_ratio must be in (0, 1]")
    # Edge case: val_ratio==1 means all data is validation
    if val_ratio == 1:
        return [], list(data)
    # Edge case: val_ratio very small (but not zero)
    indices = list(range(num_items))
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    val_size = int(round(num_items * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train = [data[i] for i in train_indices]
    val = [data[i] for i in val_indices]
    return train, val


def get_model_family(model_name: str) -> str:
    """
    Infers model family name from model_name string.
    Returns one of: 'phi', 'qwen', 'llama', 'llama3', etc.
    """
    lower = model_name.lower()
    if "phi" in lower:
        return "phi"
    if "qwen" in lower:
        return "qwen"
    if "llama-3" in lower or "llama3" in lower:
        return "llama3"
    if "llama" in lower:
        return "llama"
    return "unknown"


def dataset_stats(data: List[Any]) -> Dict[str, Any]:
    """
    Computes quick stats for token length and label balance.
    Args:
        data: list of dicts
    Returns:
        dict with stats
    """
    lengths = []
    label_counter = {}
    for item in data:
        text = item.get("text", "")
        lengths.append(len(text.split()))
        label = item.get("label", None)
        if label is not None:
            label_counter[label] = label_counter.get(label, 0) + 1
    stats = {
        "num_items": len(data),
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "mean_len": sum(lengths) / len(lengths) if lengths else 0,
        "label_counts": label_counter
    }
    return stats


def normalize_text(text: str) -> str:
    """
    Normalizes text: lowercasing, strip, and collapse whitespace.
    """
    return re.sub(r'\s+', ' ', text.strip().lower())


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Recursively flattens nested dicts.
    Args:
        d: dict
        parent_key: prefix for keys
        sep: separator
    Returns:
        flat dict
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
    Returns True if value is int or float, not bool.
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def sample_jsonl(path: str, n: int, seed: Optional[int] = None) -> List[Any]:
    """
    Randomly sample n lines from JSONL file.
    Args:
        path: jsonl file
        n: number of samples
        seed: random seed
    Returns:
        list of sampled objects
    """
    data = load_jsonl(path)
    if n >= len(data):
        return data
    rnd = random.Random(seed)
    indices = rnd.sample(range(len(data)), n)
    return [data[i] for i in indices]


def get_first_non_empty(lst: List[Any]) -> Optional[Any]:
    """
    Returns the first non-empty item from a list.
    """
    for x in lst:
        if x:
            return x
    return None


def get_last_non_empty(lst: List[Any]) -> Optional[Any]:
    """
    Returns the last non-empty item from a list.
    """
    for x in reversed(lst):
        if x:
            return x
    return None


def get_non_empty(lst: List[Any]) -> List[Any]:
    """
    Returns all non-empty items from a list.
    """
    return [x for x in lst if x]


def dataset_sample_stats(data: List[Any], n: int = 20, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Computes basic stats on a random sample of dataset items.
    Args:
        data: list of dicts
        n: sample size
        seed: random seed
    Returns:
        dict with min/max/mean text length, common keys, label balance
    """
    if not data:
        return {}
    rnd = random.Random(seed)
    sample = rnd.sample(data, min(n, len(data)))
    lengths = []
    keys_counter = {}
    label_counter = {}
    for item in sample:
        text = item.get("text", "")
        lengths.append(len(text.split()))
        for k in item.keys():
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
