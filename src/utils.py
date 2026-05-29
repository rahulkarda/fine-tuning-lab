"""
Dataset/file utilities for fine-tuning-lab.

Includes:
- count_jsonl_lines: quick count of dataset examples
- validate_jsonl_schema: schema validation for JSONL datasets
- load_jsonl: load JSONL as list of dicts
- get_token_length_distribution: token length stats for dataset

Useful for dataset stats, validation, and loading.
"""
import json
from typing import Dict, Any, Callable, List, Optional


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
    lengths = []
    for i, item in enumerate(data):
        if text_key not in item:
            continue
        text = item[text_key]
        tokens = tokenizer.encode(text, add_special_tokens=True)
        # Some tokenizers (e.g. SentencePiece) return a dict or np.ndarray, not always a list
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        elif isinstance(tokens, dict):
            # unusual, but just skip
            continue
        lengths.append(len(tokens))
        if max_items is not None and i + 1 >= max_items:
            break
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "lengths": []}
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    mean = sum(lengths_sorted) / n
    if n % 2 == 0:
        median = (lengths_sorted[n // 2 - 1] + lengths_sorted[n // 2]) / 2
    else:
        median = lengths_sorted[n // 2]
    return {
        "min": min(lengths_sorted),
        "max": max(lengths_sorted),
        "mean": mean,
        "median": median,
        "lengths": lengths_sorted
    }
