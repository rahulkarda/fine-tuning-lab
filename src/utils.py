"""
Dataset/file utilities for fine-tuning-lab.

Includes:
- count_jsonl_lines: quick count of dataset examples
- validate_jsonl_schema: schema validation for JSONL datasets
- load_jsonl: load JSONL as list of dicts
- save_jsonl: save list of dicts to JSONL file
- get_token_length_distribution: token length stats for dataset
- train_val_split: random split of dataset with seed control
- get_model_family_from_name: extract model family string from base model name
- deduplicate_jsonl: remove duplicate lines from a JSONL file
- filter_jsonl_by_schema: filter a JSONL file by a schema function and save only valid lines
- shard_jsonl: split a JSONL file into N shards of roughly equal size

Useful for dataset stats, validation, and loading.
"""
import json
from typing import Dict, Any, Callable, List, Optional, Tuple
import random
import os

def count_jsonl_lines(path: str) -> int:
    """
    Count the number of lines (examples) in a jsonl file.
    Returns the number of non-empty lines.
    """
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def validate_jsonl_schema(path: str, schema_fn: Callable[[Dict[str, Any]], bool]) -> int:
    """
    Validate each line in a jsonl file against a schema function.
    Returns the number of invalid examples.
    Ignores empty or blank lines.
    """
    invalid_count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
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

def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """
    Save a list of dicts to a jsonl file.
    Each dict is written as a line of JSON.
    """
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_token_length_distribution(
    data: List[Dict[str, Any]],
    text_key: str = "text",
    tokenizer = None,
    max_items: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute token length distribution for a dataset.
    Args:
      data: list of dicts (from load_jsonl)
      text_key: key in dict to tokenize
      tokenizer: HuggingFace tokenizer (required)
      max_items: if set, only process up to this many items
    Returns:
      dict with stats: min, max, mean, median, lengths
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    token_lengths = []
    processed = 0
    for item in data:
        if max_items is not None and processed >= max_items:
            break
        if text_key not in item:
            continue
        text = item[text_key]
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if tokens is None:
            continue
        # Accept either list or tensor, but skip dicts
        if hasattr(tokens, 'tolist'):
            tokens_list = tokens.tolist()
        elif isinstance(tokens, dict):
            continue
        else:
            tokens_list = tokens
        token_lengths.append(len(tokens_list))
        processed += 1
    if not token_lengths:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "lengths": []}
    sorted_lengths = sorted(token_lengths)
    n = len(sorted_lengths)
    mean = sum(sorted_lengths) / n
    median = (
        (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2
        if n % 2 == 0 else sorted_lengths[n // 2]
    )
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
    Randomly split dataset into train and validation sets with seed control.
    Args:
      data: list of items
      val_ratio: fraction of items to assign to val set (0 < val_ratio < 1)
      seed: random seed for reproducibility
    Returns:
      train, val: (list, list)
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1)")
    num_items = len(data)
    # Ensure at least 1 item in val set if val_ratio > 0 and num_items > 0
    val_size = max(1, int(num_items * val_ratio)) if num_items > 0 and val_ratio > 0 else 0
    if val_size > num_items:
        val_size = num_items
    indices = list(range(num_items))
    rnd = random.Random(seed) if seed is not None else random
    rnd.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    val = [data[i] for i in val_indices]
    train = [data[i] for i in train_indices]
    return train, val

def get_model_family_from_name(model_name: str) -> str:
    """
    Returns model family string ('phi', 'qwen', 'llama3', etc) from base model name.
    """
    lower_name = model_name.lower()
    if "phi" in lower_name:
        return "phi"
    if "qwen" in lower_name:
        return "qwen"
    if "llama" in lower_name or "llama-3" in lower_name:
        return "llama3"
    return "unknown"

def deduplicate_jsonl(input_path: str, output_path: str) -> int:
    """
    Removes duplicate lines from a JSONL file.
    Args:
        input_path: path to input JSONL
        output_path: path to output JSONL
    Returns:
        Number of unique lines written
    """
    seen = set()
    count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line_strip = line.strip()
            if not line_strip:
                continue
            if line_strip not in seen:
                fout.write(line)
                seen.add(line_strip)
                count += 1
    return count

def filter_jsonl_by_schema(input_path: str, output_path: str, schema_fn: Callable[[Dict[str, Any]], bool]) -> int:
    """
    Filters a JSONL file by a schema function. Writes only valid lines.
    Args:
        input_path: path to input JSONL
        output_path: path to output JSONL
        schema_fn: function taking dict, returns bool
    Returns:
        Number of valid lines written
    """
    written = 0
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line_strip = line.strip()
            if not line_strip:
                continue
            try:
                obj = json.loads(line_strip)
            except Exception:
                continue
            if schema_fn(obj):
                fout.write(line)
                written += 1
    return written

def shard_jsonl(input_path: str, output_dir: str, num_shards: int) -> List[str]:
    """
    Splits a JSONL file into N shards of roughly equal size.
    Args:
        input_path: path to input JSONL
        output_dir: directory for output shards
        num_shards: number of shards
    Returns:
        List of shard file paths
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    total = len(lines)
    if num_shards < 1 or total == 0:
        return []
    base = total // num_shards
    remainder = total % num_shards
    shard_sizes = [base + 1 if i < remainder else base for i in range(num_shards)]
    shard_paths = []
    idx = 0
    os.makedirs(output_dir, exist_ok=True)
    for i, size in enumerate(shard_sizes):
        shard_path = os.path.join(output_dir, f"shard_{i+1}.jsonl")
        with open(shard_path, 'w', encoding='utf-8') as f:
            f.writelines(lines[idx:idx+size])
        shard_paths.append(shard_path)
        idx += size
    return shard_paths
