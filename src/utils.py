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


def dataset_stats(data: List[Dict], label_key: Optional[str] = None, text_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes quick stats for token length and label balance.
    Args:
        data: list of dicts (dataset objects)
        label_key: key for label (optional)
        text_key: key for text (optional)
    Returns:
        stats dict: {num_examples, mean_len, std_len, min_len, max_len, label_counts}
    """
    from collections import Counter
    lengths = []
    labels = []
    for obj in data:
        if text_key and text_key in obj:
            txt = obj[text_key]
            lengths.append(len(txt.split()))
        elif not text_key:
            # Try to guess text field
            for k in ['text', 'prompt', 'user']:
                if k in obj:
                    txt = obj[k]
                    lengths.append(len(str(txt).split()))
                    break
        if label_key and label_key in obj:
            labels.append(obj[label_key])
    stats = {}
    if lengths:
        import numpy as np
        arr = np.array(lengths)
        stats['num_examples'] = len(arr)
        stats['mean_len'] = float(np.mean(arr))
        stats['std_len'] = float(np.std(arr))
        stats['min_len'] = int(np.min(arr))
        stats['max_len'] = int(np.max(arr))
    if labels:
        stats['label_counts'] = dict(Counter(labels))
    return stats


def normalize_text(text: str) -> str:
    """
    Normalize text for robust comparison:
    - Lowercase
    - Remove surrounding whitespace
    - Collapse all internal whitespace to single space
    - Remove control characters
    Useful for deduplication, exact match eval, or comparison across generations.
    Args:
        text: input string
    Returns:
        normalized string
    """
    if not isinstance(text, str):
        return ""
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    # Lowercase
    text = text.lower()
    # Strip
    text = text.strip()
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Recursively flattens a nested dict.
    E.g. {'a': {'b': 2}} -> {'a.b': 2}
    Useful for aggregating metrics from nested result dicts.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def is_numeric(val: Any) -> bool:
    """
    Returns True if val is a numeric scalar (int/float, not bool).
    Useful for robust metric filtering.
    """
    return (isinstance(val, (int, float)) and not isinstance(val, bool))
