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
- normalize_text: text normalization for robust comparison (new)
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
- normalize_text can be used for robust deduplication and comparison of text samples (lowercase, whitespace, punctuation-stripping).

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
    Ignores invalid JSON lines, prints one warning per file if any are encountered.
    """
    data = []
    had_invalid = False
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                if not had_invalid:
                    print(f"Warning: invalid JSON line detected in {path}, line {idx+1}. Skipping.")
                    had_invalid = True
                continue
            except Exception as e:
                if not had_invalid:
                    print(f"Warning: unexpected error loading line {idx+1} of {path}: {e}. Skipping.")
                    had_invalid = True
                continue
    return data


def validate_jsonl_schema(path: str, schema_fn: Callable[[Any], bool]) -> int:
    """
    Checks each line in JSONL file against schema_fn.
    Returns number of invalid lines. Prints one warning if any parsing errors are encountered.
    """
    invalid = 0
    had_error = False
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not schema_fn(obj):
                    invalid += 1
            except Exception as e:
                invalid += 1
                if not had_error:
                    print(f"Warning: error parsing line {idx+1} in {path}: {e}. Counting as invalid.")
                    had_error = True
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
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip invalid JSON lines silently
                continue
            except Exception:
                continue
            key = json.dumps(obj, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            out.write(json.dumps(obj, ensure_ascii=False) + '\n')


def train_val_split(data: List[Any], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Any], List[Any]]:
    """
    Splits data into train/val sets by ratio, reproducible by seed.
    """
    n = len(data)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_size = int(n * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train = [data[i] for i in train_indices]
    val = [data[i] for i in val_indices]
    return train, val


def get_model_family(model_name: str) -> str:
    """
    Infers major model family from model name string.
    Returns one of: 'phi', 'qwen', 'llama', 'mistral', 'gemma', 'other'.
    Case-insensitive, robust to common HuggingFace naming patterns.
    """
    name = model_name.lower()
    # Remove common prefixes and suffixes for clarity
    tokens = re.split(r'[\-/:_]', name)
    # Direct family keywords
    for family in ["phi", "qwen", "llama", "mistral", "gemma"]:
        if any(family in t for t in tokens):
            return family
    # Fallback heuristic for common variants
    if "phi3" in name or "phi-3" in name:
        return "phi"
    if "qwen2" in name or "qwen-2" in name:
        return "qwen"
    if "llama3" in name or "llama-3" in name:
        return "llama"
    return "other"


def dataset_stats(data: List[Any], text_key: str = "text", label_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes stats for token length and label balance in dataset.
    """
    lengths = []
    label_counter = {}
    for item in data:
        text = item.get(text_key, "")
        lengths.append(len(text))
        if label_key:
            label = item.get(label_key, None)
            if label is not None:
                label_counter[label] = label_counter.get(label, 0) + 1
    stats = {
        "count": len(data),
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "mean_len": sum(lengths) / len(lengths) if lengths else 0,
        "label_counts": label_counter
    }
    return stats


def normalize_text(text: str) -> str:
    """
    Normalizes text for robust comparison: lowercase, strips whitespace and punctuation.
    """
    t = text.lower()
    t = re.sub(r'[\s]+', ' ', t)
    t = re.sub(r'[\p{P}\p{S}]', '', t)
    t = t.strip()
    return t


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Recursively flattens nested dicts. Useful for metrics aggregation.
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
    Returns True if val is int or float, but not bool.
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def sample_jsonl(path: str, n: int, seed: int = 42) -> List[Any]:
    """
    Randomly samples n lines from JSONL file.
    """
    data = load_jsonl(path)
    random.Random(seed).shuffle(data)
    return data[:n]


def get_first_non_empty(items: List[Any]) -> Optional[Any]:
    """
    Returns first non-empty item from a list.
    """
    for item in items:
        if item:
            return item
    return None


def get_last_non_empty(items: List[Any]) -> Optional[Any]:
    """
    Returns last non-empty item from a list.
    """
    for item in reversed(items):
        if item:
            return item
    return None


def get_non_empty(items: List[Any]) -> List[Any]:
    """
    Returns all non-empty items from a list.
    """
    return [item for item in items if item]


def deduplicate_texts(texts: List[str]) -> List[str]:
    """
    Removes duplicate text items from a list (case-sensitive).
    """
    seen = set()
    out = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def dataset_sample_stats(data: List[Any], n: int = 100) -> Dict[str, Any]:
    """
    Computes basic stats on a random sample of dataset items.
    """
    if not data:
        return {}
    sample = random.sample(data, min(n, len(data)))
    lengths = []
    keys_counter = {}
    label_counter = {}
    for item in sample:
        text = item.get("text", "")
        lengths.append(len(text))
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
