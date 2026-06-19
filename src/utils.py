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

Designed for flexible experiment scaffolding and quick data checks.
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
        train, val: (list, list)
    """
    if not 0 < val_ratio <= 1:
        raise ValueError("val_ratio must be in (0, 1]")
    num_items = len(data)
    if num_items == 0:
        return [], []
    if val_ratio == 1:
        return [], list(data)
    val_size = max(1, int(num_items * val_ratio)) if num_items > 0 and val_ratio > 0 else 0
    indices = list(range(num_items))
    rand = random.Random(seed) if seed is not None else random
    rand.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    val = [data[i] for i in val_indices]
    train = [data[i] for i in train_indices]
    return train, val


def get_model_family(model_name: str) -> str:
    """
    Infers model family ('phi', 'qwen', 'llama3') from model name string.
    Useful for auto-selecting prompt template.
    Args:
        model_name: model identifier (e.g. 'microsoft/Phi-3-mini-4k-instruct')
    Returns:
        family: str ('phi', 'qwen', 'llama3') or 'unknown'
    """
    name = model_name.lower()
    if 'phi' in name:
        return 'phi'
    if 'qwen' in name:
        return 'qwen'
    if 'llama-3' in name or 'llama3' in name:
        return 'llama3'
    return 'unknown'


def dataset_stats(
    data: List[Dict[str, Any]],
    tokenizer,
    prompt_key: str = 'user',
    label_key: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes quick stats for dataset: token length distribution and label balance.
    Args:
        data: list of dicts
        tokenizer: HF tokenizer
        prompt_key: which key to tokenize ('user' by default)
        label_key: if provided, counts label frequencies
        max_samples: limit number of examples for fast stats
    Returns:
        Dict: {token_lengths, length_stats, (optional) label_counts}
    """
    items = data[:max_samples] if max_samples is not None else data
    token_lengths = []
    for ex in items:
        prompt = ex.get(prompt_key, "")
        token_lengths.append(len(tokenizer.encode(prompt)))
    length_stats = {
        'min': min(token_lengths) if token_lengths else 0,
        'max': max(token_lengths) if token_lengths else 0,
        'mean': sum(token_lengths)/len(token_lengths) if token_lengths else 0,
        'median': sorted(token_lengths)[len(token_lengths)//2] if token_lengths else 0
    }
    stats = {
        'token_lengths': token_lengths,
        'length_stats': length_stats
    }
    if label_key is not None:
        label_counts = {}
        for ex in items:
            label = ex.get(label_key, None)
            if label is not None:
                label_counts[label] = label_counts.get(label, 0) + 1
        stats['label_counts'] = label_counts
    return stats


def normalize_text(text: str) -> str:
    """
    Normalizes text for robust comparison:
    - Lowercases
    - Strips leading/trailing whitespace
    - Collapses multiple spaces/tabs/newlines to single space
    Args:
        text: input string
    Returns:
        normalized string
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text
