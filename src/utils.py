"""
Dataset/file utilities for fine-tuning-lab.

Includes:
- count_jsonl_lines: quick count of dataset examples
- validate_jsonl_schema: schema validation for JSONL datasets
- load_jsonl: load JSONL as list of dicts
- get_token_length_distribution: token length stats for dataset
- train_val_split: random split of dataset with seed control

Useful for dataset stats, validation, and loading.
"""
import json
from typing import Dict, Any, Callable, List, Optional, Tuple
import random


def count_jsonl_lines(path: str) -> int:
    """
    Count the number of lines (examples) in a jsonl file.
    Useful for quick dataset stats.
    Ignores empty or blank lines.
    """
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def validate_jsonl_schema(path: str, schema_fn: Callable[[Dict[str, Any]], bool]) -> int:
    """
    Validate each line in a jsonl file against a schema_fn.
    Returns the number of invalid examples.
    schema_fn: function taking a dict, returns True if valid.
    Ignores empty or blank lines.
    """
    invalid_count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                invalid_count += 1
                continue
            if not schema_fn(obj):
                invalid_count += 1
    return invalid_count


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a jsonl file into a list of dicts.
    Each line must be valid JSON.
    Ignores empty or blank lines.
    """
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(obj)
    return items


def get_token_length_distribution(
    data: List[Dict[str, Any]],
    text_key: str = "text",
    tokenizer = None,
    max_items: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute token length distribution for dataset.
    Args:
      data: list of dicts (from load_jsonl)
      text_key: key in dict to tokenize
      tokenizer: HuggingFace tokenizer (must be provided)
      max_items: if set, only process this many items
    Returns:
      dict with stats: min, max, mean, median, lengths
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    token_lengths = []
    processed_count = 0
    for item in data:
        if max_items is not None and processed_count >= max_items:
            break
        if text_key not in item:
            continue
        text = item[text_key]
        tokens = tokenizer.encode(text, add_special_tokens=True)
        # Handle possible return types from tokenizer.encode
        if tokens is None:
            continue
        if hasattr(tokens, 'tolist'):
            tokens_list = tokens.tolist()
        elif isinstance(tokens, dict):
            # If encode returns dict, skip this item
            continue
        else:
            tokens_list = tokens
        token_lengths.append(len(tokens_list))
        processed_count += 1  # Only increment for valid, non-skipped items
    if not token_lengths:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "lengths": []}
    sorted_lengths = sorted(token_lengths)
    n = len(sorted_lengths)
    mean = sum(sorted_lengths) / n
    if n % 2 == 0:
        median = (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2
    else:
        median = sorted_lengths[n // 2]
    return {
        "min": min(sorted_lengths),
        "max": max(sorted_lengths),
        "mean": mean,
        "median": median,
        "lengths": sorted_lengths
    }


def train_val_split(
    data: List[Any],
    val_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[List[Any], List[Any]]:
    """
    Randomly split dataset into train and val sets with seed control.
    Args:
      data: list of items
      val_ratio: fraction of items to assign to val set (between 0 and 1)
      seed: random seed for reproducibility
    Returns:
      train, val: (list, list)
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1)")
    indices = list(range(len(data)))
    rnd = random.Random(seed) if seed is not None else random
    rnd.shuffle(indices)
    val_size = int(len(data) * val_ratio)
    val_indices = set(indices[:val_size])
    train, val = [], []
    for idx, item in enumerate(data):
        # Assignment should use shuffled indices
        if idx in val_indices:
            val.append(item)
        else:
            train.append(item)
    return train, val
