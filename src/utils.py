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
- pad_or_truncate: pad or truncate a sequence to a fixed length (new)

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
- pad_or_truncate is handy for prepping tokenized inputs to a fixed length (for batching, eval, etc).

Example:
    # Load and deduplicate dataset
    data = load_jsonl('data.jsonl')
    train, val = train_val_split(data, val_ratio=0.1, seed=42)
    stats = dataset_stats(train)
    print(stats)
    deduplicate_jsonl('data.jsonl', 'data_deduped.jsonl')
    # Pad tokens for batching
    tokens = [101, 102, 103]
    tokens = pad_or_truncate(tokens, length=5)

Caveats:
- These utilities are not optimized for large-scale datasets (>100k items); use for experiment prototyping.
- Deduplication is based on object hash; small differences (ordering, whitespace) may defeat it.
- Text normalization is aggressive; tune for your use-case if needed.
- train_val_split uses random.shuffle; reproducibility is controlled by seed.

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
            h = hash(json.dumps(obj, sort_keys=True))
            if h not in seen:
                seen.add(h)
                out.write(json.dumps(obj, ensure_ascii=False) + '\n')


def train_val_split(data: List[Any], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Any], List[Any]]:
    """
    Reproducible train/val split for small datasets.
    Args:
        data: input list
        val_ratio: fraction for validation set
        seed: random seed
    Returns:
        train, val lists
    """
    if not data:
        return [], []
    idxs = list(range(len(data)))
    random.seed(seed)
    random.shuffle(idxs)
    split_at = int(len(data) * val_ratio)
    val_idxs = idxs[:split_at]
    train_idxs = idxs[split_at:]
    val = [data[i] for i in val_idxs]
    train = [data[i] for i in train_idxs]
    return train, val


def get_model_family(model_name: str) -> str:
    """
    Infers model family (phi, qwen, llama, mistral, etc) from model name string.
    Returns lowercased family name.
    """
    m = model_name.lower()
    if 'phi' in m:
        return 'phi'
    if 'qwen' in m:
        return 'qwen'
    if 'llama' in m:
        return 'llama'
    if 'mistral' in m:
        return 'mistral'
    if 'gemma' in m:
        return 'gemma'
    if 'falcon' in m:
        return 'falcon'
    return 'unknown'


def dataset_stats(data: List[Any]) -> Dict[str, Any]:
    """
    Computes quick stats for token length and label balance.
    Args:
        data: list of dicts (each item)
    Returns:
        dict with stats (token_length, label distribution)
    """
    if not data:
        return {}
    lengths = []
    labels = {}
    for item in data:
        if isinstance(item, dict):
            txt = item.get('text', '')
            lengths.append(len(txt.split()))
            label = item.get('label', None)
            if label is not None:
                labels[label] = labels.get(label, 0) + 1
    stats = {
        'count': len(data),
        'mean_length': float(sum(lengths) / len(lengths)) if lengths else 0.0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'label_dist': labels
    }
    return stats


def normalize_text(txt: str) -> str:
    """
    Normalizes text for robust comparison: lowercase, remove extra whitespace and punctuation.
    Args:
        txt: input string
    Returns:
        normalized string
    """
    txt = txt.lower()
    txt = re.sub(r'[\s]+', ' ', txt)
    txt = re.sub(r'[\p{P}\p{S}]', '', txt)
    txt = txt.strip()
    return txt


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Recursively flattens nested dicts.
    Args:
        d: input dict
        parent_key: for recursive calls
        sep: separator
    Returns:
        flattened dict
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
    Returns True if value is int or float, not bool.
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def sample_jsonl(path: str, n: int, seed: int = 42) -> List[Any]:
    """
    Randomly sample N lines from JSONL file.
    """
    data = load_jsonl(path)
    random.seed(seed)
    random.shuffle(data)
    return data[:n]


def get_first_non_empty(lst: List[Any]) -> Optional[Any]:
    """
    Returns first non-empty item from list, or None.
    """
    for item in lst:
        if item:
            return item
    return None


def get_last_non_empty(lst: List[Any]) -> Optional[Any]:
    """
    Returns last non-empty item from list, or None.
    """
    for item in reversed(lst):
        if item:
            return item
    return None


def get_non_empty(lst: List[Any]) -> List[Any]:
    """
    Returns all non-empty items from list.
    """
    return [item for item in lst if item]


def dataset_sample_stats(data: List[Any], n: int = 10, seed: int = 42) -> Dict[str, Any]:
    """
    Basic stats on a random sample of dataset items.
    Args:
        data: list of dicts
        n: sample size
        seed: random seed
    Returns:
        dict with sample keys, lengths, label balance
    """
    if not data:
        return {}
    random.seed(seed)
    sample = random.sample(data, min(n, len(data)))
    keys = set()
    lengths = []
    labels = {}
    for item in sample:
        if isinstance(item, dict):
            keys.update(item.keys())
            txt = item.get('text', '')
            lengths.append(len(txt.split()))
            label = item.get('label', None)
            if label is not None:
                labels[label] = labels.get(label, 0) + 1
    stats = {
        'sample_size': len(sample),
        'keys': sorted(list(keys)),
        'mean_length': float(sum(lengths) / len(lengths)) if lengths else 0.0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'label_dist': labels
    }
    return stats


def deduplicate_texts(texts: List[str]) -> List[str]:
    """
    Removes duplicate text strings from list. Case-sensitive.
    """
    seen = set()
    output = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            output.append(t)
    return output


def pad_or_truncate(seq: List[Any], length: int, pad_value: Any = 0) -> List[Any]:
    """
    Pads or truncates a sequence to a fixed length.
    Args:
        seq: input sequence (list)
        length: target length
        pad_value: value to pad with (default 0)
    Returns:
        list of length 'length'
    """
    if len(seq) > length:
        return seq[:length]
    elif len(seq) < length:
        return seq + [pad_value] * (length - len(seq))
    else:
        return seq
