"""
Dataset/file utilities for fine-tuning-lab.

Includes:
- count_jsonl_lines: quick count of dataset examples
- load_jsonl: load JSONL as list of dicts
- save_jsonl: save list of dicts to JSONL file
- validate_jsonl_schema: schema validation for JSONL datasets
- train_val_split: random split of dataset with seed control
- get_token_length_distribution: token length stats for dataset
- deduplicate_jsonl: remove duplicate lines from a JSONL file
- filter_jsonl_by_schema: filter a JSONL file by a schema function and save only valid lines
- shard_jsonl: split a JSONL file into N shards of roughly equal size
- get_model_family_from_name: extract model family string from base model name

Useful for dataset stats, validation, and loading.
"""
import json
from typing import Dict, Any, Callable, List, Optional, Tuple
import random
import os

def count_jsonl_lines(path: str) -> int:
    """
    Count the number of non-empty lines (examples) in a JSONL file.
    Returns:
        int: number of non-empty lines
    """
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dicts.
    Ignores empty or blank lines.
    Returns:
        List[Dict[str, Any]]: loaded items
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
    Save a list of dicts to a JSONL file.
    Each dict is written as a line of JSON.
    """
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def validate_jsonl_schema(path: str, schema_fn: Callable[[Dict[str, Any]], bool]) -> int:
    """
    Validate each line in a JSONL file against a schema function.
    Returns:
        int: number of invalid examples
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
    val_size = max(1, int(num_items * val_ratio)) if num_items > 0 and val_ratio > 0 else 0
    indices = list(range(num_items))
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    val = [data[i] for i in val_indices]
    train = [data[i] for i in train_indices]
    return train, val


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
        (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2 if n % 2 == 0 else sorted_lengths[n // 2]
    )
    return {
        "min": min(sorted_lengths),
        "max": max(sorted_lengths),
        "mean": mean,
        "median": median,
        "lengths": token_lengths
    }


def deduplicate_jsonl(input_path: str, output_path: str) -> int:
    """
    Removes duplicate lines from a JSONL file and writes unique lines to output.
    Ignores empty or blank lines. Returns the number of lines written.
    Args:
        input_path: source JSONL file
        output_path: deduplicated output JSONL file
    Returns:
        int: number of unique lines written
    """
    seen = set()
    unique_lines = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            sline = line.strip()
            if not sline:
                continue
            if sline not in seen:
                seen.add(sline)
                unique_lines.append(line)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(unique_lines)
    return len(unique_lines)


def filter_jsonl_by_schema(
    input_path: str,
    output_path: str,
    schema_fn: Callable[[Dict[str, Any]], bool]
) -> int:
    """
    Filters a JSONL file by a schema function and saves only valid lines.
    Ignores empty or blank lines.
    Returns the number of valid lines written.
    Args:
        input_path: source JSONL file
        output_path: filtered output JSONL file
        schema_fn: function taking dict, returns True if valid
    Returns:
        int: number of valid lines written
    """
    valid_lines = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            sline = line.strip()
            if not sline:
                continue
            try:
                obj = json.loads(sline)
            except Exception:
                continue
            if schema_fn(obj):
                valid_lines.append(line)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
    return len(valid_lines)


def shard_jsonl(input_path: str, output_dir: str, num_shards: int) -> List[str]:
    """
    Splits a JSONL file into N shards of roughly equal size.
    Ignores empty or blank lines.
    Args:
        input_path: source JSONL file
        output_dir: directory to write shards
        num_shards: number of shards
    Returns:
        List[str]: paths to shard files
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    n = len(lines)
    if n == 0:
        return []
    shard_size = max(1, n // num_shards)
    shard_paths = []
    for i in range(num_shards):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_shards - 1 else n
        shard_lines = lines[start:end]
        shard_path = os.path.join(output_dir, f"shard_{i+1}.jsonl")
        with open(shard_path, 'w', encoding='utf-8') as f:
            f.writelines(shard_lines)
        shard_paths.append(shard_path)
    return shard_paths


def get_model_family_from_name(model_name: str) -> Optional[str]:
    """
    Extract model family string from base model name.
    Args:
        model_name: e.g. 'microsoft/Phi-3-mini-4k-instruct', 'Qwen/Qwen1.5-1.8B', etc.
    Returns:
        str: family ('phi', 'qwen', 'llama3'), or None if not recognized
    """
    name = model_name.lower()
    if 'phi' in name:
        return 'phi'
    if 'qwen' in name:
        return 'qwen'
    if 'llama-3' in name or 'llama3' in name:
        return 'llama3'
    return None
