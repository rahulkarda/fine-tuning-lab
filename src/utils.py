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

Usage notes:
- All utilities are designed for quick experiment scaffolding: call directly or batch in scripts.
- load_jsonl and validate_jsonl_schema are robust to blank lines and minor format errors.
- train_val_split and deduplicate_jsonl support reproducible workflow for small datasets.
- get_model_family is handy for auto-selecting prompt templates by model name.
- Use dataset_stats before/after deduplication to check dataset quality.
- sample_jsonl enables rapid prototyping and debugging on small data slices.
- flatten_dict is ideal for preparing results for aggregate_metrics or dashboard reporting.
- dataset_sample_stats provides basic statistics on a random subset (lengths, keys, label balance).

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
    # Edge case: val_ratio very small (but >0) will always reserve at least 1 for val if possible
    val_size = int(num_items * val_ratio)
    if val_size == 0 and num_items > 1:
        val_size = 1
    elif val_size > num_items:
        val_size = num_items
    indices = list(range(num_items))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    else:
        random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    val_set = [data[i] for i in val_indices]
    train_set = [data[i] for i in train_indices]
    return train_set, val_set


def get_model_family(model_name: str) -> str:
    """
    Infers major model family (phi, qwen, llama3, unknown) from model name string.
    """
    mn = model_name.lower()
    if "phi" in mn:
        return "phi"
    if "qwen" in mn:
        return "qwen"
    if "llama-3" in mn or "llama3" in mn:
        return "llama3"
    return "unknown"


def dataset_stats(dataset: List[Any], label_key: str = "label") -> Dict[str, Any]:
    """
    Computes token length distribution and label balance for a dataset.
    Args:
        dataset: list of dicts
        label_key: key for label field
    Returns:
        Dict with length stats and label counts
    """
    lengths = []
    label_counts = {}
    for item in dataset:
        text = item.get("text", "")
        lengths.append(len(text.split()))
        label = item.get(label_key, None)
        if label is not None:
            label_counts[label] = label_counts.get(label, 0) + 1
    stats = {
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "mean_len": sum(lengths) / len(lengths) if lengths else 0,
        "label_counts": label_counts
    }
    return stats


def normalize_text(text: str) -> str:
    """
    Lowercases and strips whitespace for robust string comparison.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def flatten_dict(d: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Recursively flattens nested dicts for easy metric aggregation.
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
    Checks if value is numeric (int/float, not bool).
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def sample_jsonl(path: str, n: int = 10, seed: Optional[int] = None) -> List[Any]:
    """
    Randomly sample N lines from JSONL file.
    """
    data = load_jsonl(path)
    if not data or n >= len(data):
        return data
    rng = random.Random(seed) if seed is not None else random
    indices = rng.sample(range(len(data)), n)
    return [data[i] for i in indices]


def get_first_non_empty(items: List[Any]) -> Any:
    """
    Returns the first non-empty (not None, not blank string, not empty list/dict) item from a list.
    """
    for item in items:
        if item is None:
            continue
        if isinstance(item, str) and not item.strip():
            continue
        if isinstance(item, (list, dict)) and not item:
            continue
        return item
    return None


def get_last_non_empty(items: List[Any]) -> Any:
    """
    Returns the last non-empty (not None, not blank string, not empty list/dict) item from a list.
    """
    for item in reversed(items):
        if item is None:
            continue
        if isinstance(item, str) and not item.strip():
            continue
        if isinstance(item, (list, dict)) and not item:
            continue
        return item
    return None


def get_non_empty(items: List[Any]) -> List[Any]:
    """
    Returns all non-empty (not None, not blank string, not empty list/dict) items from a list.
    Args:
        items: list of items (Any type)
    Returns:
        List of all non-empty items.
    """
    result = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, str) and not item.strip():
            continue
        if isinstance(item, (list, dict)) and not item:
            continue
        result.append(item)
    return result


def dataset_sample_stats(dataset: List[Any], sample_size: int = 10, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Computes basic stats on a random sample of dataset items:
    - min/max/mean text length
    - most common keys
    - label balance (if label key present)
    Args:
        dataset: list of dicts
        sample_size: number of items to sample (default 10)
        seed: random seed for reproducibility
    Returns:
        Dict of stats
    """
    if not dataset:
        return {}
    n = min(sample_size, len(dataset))
    rng = random.Random(seed) if seed is not None else random
    indices = rng.sample(range(len(dataset)), n)
    sample = [dataset[i] for i in indices]
    # Compute text length stats
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
