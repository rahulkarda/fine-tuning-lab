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
    rand = random.Random(seed) if seed is not None else random
    rand.shuffle(indices)
    val_indices = set(indices[:val_size])
    train = [data[i] for i in range(num_items) if i not in val_indices]
    val = [data[i] for i in range(num_items) if i in val_indices]
    return train, val


def get_model_family(model_name: str) -> str:
    """
    Infers model family (phi, qwen, llama3) from model name string.
    Returns 'phi', 'qwen', 'llama3', or 'unknown'.
    """
    name = model_name.lower()
    if 'phi' in name:
        return 'phi'
    if 'qwen' in name:
        return 'qwen'
    if 'llama-3' in name or 'llama3' in name or re.search(r'\bllama3\b', name):
        return 'llama3'
    return 'unknown'


def dataset_stats(data: List[Any], tokenizer=None, text_key: str = 'text', label_key: str = 'label') -> Dict[str, Any]:
    """
    Quick stats for token length and label balance in dataset.
    Args:
        data: list of dicts
        tokenizer: optional tokenizer for token length distribution
        text_key: which key contains text
        label_key: which key contains label/class
    Returns:
        dict with counts, token lengths, label balance
    """
    stats = {}
    stats['num_examples'] = len(data)
    # Token length stats
    if tokenizer:
        lens = []
        for item in data:
            text = item.get(text_key, '')
            if not text:
                continue
            tokens = tokenizer(text, return_tensors=None)['input_ids']
            lens.append(len(tokens))
        stats['token_length_mean'] = sum(lens) / len(lens) if lens else 0
        stats['token_length_min'] = min(lens) if lens else 0
        stats['token_length_max'] = max(lens) if lens else 0
        stats['token_length_std'] = (sum((x - stats['token_length_mean']) ** 2 for x in lens) / len(lens)) ** 0.5 if lens else 0
    # Label balance stats
    label_counts = {}
    for item in data:
        label = item.get(label_key, None)
        if label is not None:
            label_counts[label] = label_counts.get(label, 0) + 1
    stats['label_balance'] = label_counts
    return stats


def normalize_text(text: str) -> str:
    """
    Normalizes text for robust comparison: lower, strip, collapse whitespace.
    """
    if not isinstance(text, str):
        return ''
    return re.sub(r'\s+', ' ', text.strip().lower())


def flatten_dict(d: Any, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Recursively flattens nested dicts. Keys joined with sep.
    """
    items = {}
    if not isinstance(d, dict):
        return items
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def is_numeric(val: Any) -> bool:
    """
    Returns True if val is int or float (but not bool).
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def sample_jsonl(path: str, n: int, seed: Optional[int] = None) -> List[Any]:
    """
    Randomly sample n lines from JSONL file.
    """
    data = load_jsonl(path)
    if not data:
        return []
    rand = random.Random(seed) if seed is not None else random
    return rand.sample(data, min(n, len(data)))


def get_first_non_empty(items: List[Any]) -> Any:
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


def get_last_non_empty(items: List[Any]) -> Any:
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
