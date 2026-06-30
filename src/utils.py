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

Usage notes:
- All utilities are designed for quick experiment scaffolding: call directly or batch in scripts.
- load_jsonl and validate_jsonl_schema are robust to blank lines and minor format errors.
- train_val_split and deduplicate_jsonl support reproducible workflow for small datasets.
- get_model_family is handy for auto-selecting prompt templates by model name.
- Use dataset_stats before/after deduplication to check dataset quality.
- sample_jsonl enables rapid prototyping and debugging on small data slices.
- flatten_dict is ideal for preparing results for aggregate_metrics or dashboard reporting.

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
        random.seed(seed)
    random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    val = [data[i] for i in val_indices]
    train = [data[i] for i in train_indices]
    return train, val


def get_model_family(name: str) -> str:
    """
    Infers model family (phi, qwen, llama3, unknown) from model name string.
    """
    name = name.lower()
    if 'phi' in name:
        return 'phi'
    if 'qwen' in name:
        return 'qwen'
    if re.search(r'llama[-_]?3', name) or 'llama3' in name:
        return 'llama3'
    return 'unknown'


def dataset_stats(data: List[Dict[str, Any]], text_key: str = 'text', label_key: Optional[str] = None, tokenizer=None) -> Dict[str, Any]:
    """
    Computes token length stats and label balance for a dataset.
    Args:
        data: list of dicts
        text_key: key for text field
        label_key: optional key for label field
        tokenizer: optional tokenizer for token count
    Returns:
        dict of stats
    """
    lengths = []
    label_counts = {}
    for item in data:
        text = item.get(text_key, '')
        if tokenizer is not None:
            length = len(tokenizer(text)['input_ids'])
        else:
            length = len(text.split())
        lengths.append(length)
        if label_key:
            label = item.get(label_key, None)
            if label is not None:
                label_counts[label] = label_counts.get(label, 0) + 1
    stats = {
        'count': len(data),
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'mean_length': float(sum(lengths) / len(lengths)) if lengths else 0.0,
    }
    if label_key:
        stats['label_counts'] = label_counts
    return stats


def normalize_text(s: str) -> str:
    """
    Normalizes text for robust comparison (lowercase, whitespace, punctuation).
    """
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[\.,!?;:"]', '', s)
    return s


def is_numeric(val: Any) -> bool:
    """
    Checks if val is a numeric type (int/float, not bool).
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Recursively flattens nested dicts.
    Args:
        d: input dict
        parent_key: prefix for recursion
        sep: separator for keys
    Returns:
        flat dict with keys joined by sep
    """
    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def sample_jsonl(path: str, n: int, seed: Optional[int] = None) -> List[Any]:
    """
    Randomly samples n lines from JSONL file.
    Args:
        path: path to JSONL file
        n: number of items to sample
        seed: random seed
    Returns:
        list of sampled JSON objects
    """
    all_data = load_jsonl(path)
    if seed is not None:
        random.seed(seed)
    if n > len(all_data):
        n = len(all_data)
    return random.sample(all_data, n)


def get_first_non_empty(items: List[Any]) -> Optional[Any]:
    """
    Returns the first non-empty (not None, not blank string, not empty list/dict) item from a list.
    Args:
        items: list of items (Any type)
    Returns:
        The first non-empty item, or None if none found.
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


def get_last_non_empty(items: List[Any]) -> Optional[Any]:
    """
    Returns the last non-empty (not None, not blank string, not empty list/dict) item from a list.
    Args:
        items: list of items (Any type)
    Returns:
        The last non-empty item, or None if none found.
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
