"""
Data utility functions for JSONL dataset loading, validation, deduplication, and splitting.

Includes:
- count_jsonl_lines: count non-empty lines in JSONL file
- load_jsonl: load objects from JSONL, skip blank lines
- validate_jsonl_schema: check each line against a schema function
- deduplicate_jsonl: remove duplicate JSONL objects based on hash
- train_val_split: reproducible train/val split by ratio and seed
- get_model_family: infer major model family from model name string (clarified logic)
- dataset_stats: quick stats for token length and label balance
- normalize_text: text normalization for robust comparison (new)
- flatten_dict: recursively flatten nested dicts for easier metric aggregation (refactored)
- is_numeric: check if value is a numeric type (int/float, not bool)
- sample_jsonl: randomly sample N lines from JSONL file (new)
- get_first_non_empty: returns the first non-empty item from a list (new)
- get_last_non_empty: returns the last non-empty item from a list (new)
- get_non_empty: returns all non-empty items from a list (new)
- dataset_sample_stats: basic stats on a random sample of dataset items (new)
- deduplicate_texts: remove duplicate text items from a list (new)
- pad_or_truncate: pad or truncate a sequence to a fixed length (new)
- is_empty: returns True if value is None, empty, or only whitespace (new)
- dataset_token_counts: computes token counts for each item in a dataset (new)
- dataset_text_lengths: computes text length (number of characters) per item (new)
- dataset_label_counts: computes label distribution for quick stats (new)
- get_first_key: returns the first key from a dict, or None if empty (new)
- get_last_key: returns the last key from a dict, or None if empty (new)
- get_keys: returns a list of keys from a dict, or empty list if not a dict (new)

Usage notes:
- All utilities are designed for quick experiment scaffolding: call directly or batch in scripts.
- load_jsonl and validate_jsonl_schema are robust to blank lines and minor format errors.
- train_val_split and deduplicate_jsonl support reproducible workflow for small datasets.
- get_model_family is handy for auto-selecting prompt templates by model name (now clearer logic).
- Use dataset_stats before/after deduplication to check dataset quality.
- sample_jsonl enables rapid prototyping and debugging on small data slices.
- flatten_dict is ideal for preparing results for aggregate_metrics or dashboard reporting (now clearer logic).
- dataset_sample_stats provides basic statistics on a random subset (lengths, keys, label balance).
- deduplicate_texts is a quick utility for removing duplicate text strings (case-sensitive).
- normalize_text can be used for robust deduplication and comparison of text samples (lowercase, whitespace, punctuation-stripping).
- pad_or_truncate is handy for prepping tokenized inputs to a fixed length (for batching, eval, etc).
- is_empty is a simple helper for filtering or validation of values (None, '', [], etc).
- dataset_token_counts helps analyze tokenization stats for batching and memory planning (new).
- dataset_text_lengths computes text length (in characters) for each item, useful for quick stats (new).
- dataset_label_counts computes label distribution statistics for classification tasks (new).
- get_first_key returns the first key from a dict, useful for quick schema inspection (new).
- get_last_key returns the last key from a dict, useful for quick schema inspection (new).
- get_keys returns a list of keys from a dict, useful for schema inspection or iterating fields (new).

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
    # Filter empty items
    filtered = [x for x in data if not is_empty(x)]
    # Compute token counts per item
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
    counts = dataset_token_counts(data, tokenizer)
    print(counts)
    # Compute text lengths per item
    lengths = dataset_text_lengths(data)
    print(lengths)
    # Compute label distribution
    label_counts = dataset_label_counts(data)
    print(label_counts)
    # Get first key from schema dict
    key = get_first_key(data[0])
    print(key)
    # Get last key from schema dict
    last_key = get_last_key(data[0])
    print(last_key)
    # Get all keys from schema dict
    keys = get_keys(data[0])
    print(keys)

Caveats:
- These utilities are not optimized for large-scale datasets (>100k items); use for experiment prototyping.
- Deduplication is based on object hash; small differences (ordering, whitespace) may defeat it.
- Text normalization is aggressive; tune for your use-case if needed.
- train_val_split uses random.shuffle; reproducibil
"""
import json
import random
import string
from typing import Any, Callable, List, Dict, Tuple, Optional


def count_jsonl_lines(path: str) -> int:
    """
    Counts the number of non-empty lines in a JSONL file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def load_jsonl(path: str) -> List[Any]:
    """
    Loads objects from a JSONL file, skipping blank lines.
    """
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                continue
    return items


def validate_jsonl_schema(path: str, schema_fn: Callable[[Any], bool]) -> int:
    """
    Checks each line against a schema function. Returns number of invalid lines.
    """
    invalid = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                invalid += 1
                continue
            if not schema_fn(obj):
                invalid += 1
    return invalid


def deduplicate_jsonl(input_path: str, output_path: str) -> None:
    """
    Removes duplicate JSONL objects based on hash.
    """
    seen = set()
    deduped = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                hashable = json.dumps(obj, sort_keys=True)
                if hashable not in seen:
                    seen.add(hashable)
                    deduped.append(obj)
            except Exception:
                continue
    with open(output_path, 'w', encoding='utf-8') as f:
        for obj in deduped:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def train_val_split(data: List[Any], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Any], List[Any]]:
    """
    Split data into train and val sets by ratio, with seed control.
    """
    n = len(data)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    val_size = int(n * val_ratio)
    val_idxs = idxs[:val_size]
    train_idxs = idxs[val_size:]
    train = [data[i] for i in train_idxs]
    val = [data[i] for i in val_idxs]
    return train, val


def get_model_family(model_name: str) -> str:
    """
    Infers major model family from a model name string.
    Returns 'phi', 'qwen', 'llama', or 'unknown'.
    """
    name = model_name.lower()
    if 'phi' in name:
        return 'phi'
    if 'qwen' in name:
        return 'qwen'
    if 'llama' in name or 'meta-llama' in name:
        return 'llama'
    return 'unknown'


def dataset_stats(data: List[Any]) -> Dict[str, Any]:
    """
    Quick stats for token length and label balance.
    """
    stats = {}
    stats['num_items'] = len(data)
    if not data:
        return stats
    first = data[0]
    if isinstance(first, dict):
        stats['keys'] = list(first.keys())
    stats['lengths'] = [len(str(x)) for x in data]
    return stats


def normalize_text(text: str) -> str:
    """
    Aggressive normalization: lowercase, strip, remove punctuation, collapse whitespace.
    """
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


def flatten_dict(d: Any, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Recursively flattens nested dicts. Key paths joined by sep.
    Example: {'a': {'b': 2}} -> {'a.b': 2}
    Only dicts are recursed; lists/other types are kept as-is.
    """
    items = {}
    if not isinstance(d, dict):
        return items
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def is_numeric(val: Any) -> bool:
    """
    True if val is int or float (not bool).
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def sample_jsonl(path: str, n: int, seed: int = 42) -> List[Any]:
    """
    Randomly sample n lines from JSONL.
    """
    items = load_jsonl(path)
    random.Random(seed).shuffle(items)
    return items[:n]


def get_first_non_empty(lst: List[Any]) -> Any:
    """
    Returns the first non-empty item from a list or None.
    """
    for item in lst:
        if not is_empty(item):
            return item
    return None


def get_last_non_empty(lst: List[Any]) -> Any:
    """
    Returns the last non-empty item from a list or None.
    """
    for item in reversed(lst):
        if not is_empty(item):
            return item
    return None


def get_non_empty(lst: List[Any]) -> List[Any]:
    """
    Returns all non-empty items from a list.
    """
    return [item for item in lst if not is_empty(item)]


def dataset_sample_stats(data: List[Any], sample_size: int = 10, seed: int = 123) -> Dict[str, Any]:
    """
    Basic stats on a random sample: length, keys, label balance.
    """
    if not data:
        return {}
    sample = data.copy()
    random.Random(seed).shuffle(sample)
    sample = sample[:sample_size]
    stats = {}
    stats['num_sampled'] = len(sample)
    stats['lengths'] = [len(str(x)) for x in sample]
    if isinstance(sample[0], dict):
        stats['keys'] = list(sample[0].keys())
        if 'label' in sample[0]:
            labels = [x.get('label') for x in sample]
            stats['label_counts'] = {l: labels.count(l) for l in set(labels)}
    return stats


def deduplicate_texts(texts: List[str]) -> List[str]:
    """
    Remove duplicate text items from a list (case-sensitive).
    """
    seen = set()
    out = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def pad_or_truncate(seq: List[Any], length: int, pad_value: Any = 0) -> List[Any]:
    """
    Pad or truncate a sequence to a fixed length.
    """
    seq = list(seq)
    if len(seq) < length:
        seq += [pad_value] * (length - len(seq))
    else:
        seq = seq[:length]
    return seq


def is_empty(val: Any) -> bool:
    """
    Returns True if val is None, empty, or only whitespace.
    """
    if val is None:
        return True
    if isinstance(val, str):
        return not val.strip()
    if isinstance(val, (list, dict)):
        return not val
    return False


def dataset_token_counts(data: List[Any], tokenizer: Any, text_key: str = 'text') -> List[int]:
    """
    Computes token counts for each item in a dataset.
    """
    counts = []
    for item in data:
        if isinstance(item, dict):
            text = item.get(text_key, '')
        else:
            text = str(item)
        tokens = tokenizer.encode(text)
        counts.append(len(tokens))
    return counts


def dataset_text_lengths(data: List[Any], text_key: str = 'text') -> List[int]:
    """
    Computes text length (number of characters) per item.
    """
    lengths = []
    for item in data:
        if isinstance(item, dict):
            text = item.get(text_key, '')
        else:
            text = str(item)
        lengths.append(len(text))
    return lengths


def dataset_label_counts(data: List[Any], label_key: str = 'label') -> Dict[Any, int]:
    """
    Computes label distribution for quick stats.
    """
    counts = {}
    for item in data:
        if isinstance(item, dict) and label_key in item:
            label = item[label_key]
            counts[label] = counts.get(label, 0) + 1
    return counts


def get_first_key(d: Any) -> Optional[str]:
    """
    Returns the first key from a dict, or None if empty/not a dict.
    """
    if not isinstance(d, dict) or not d:
        return None
    for k in d:
        return k
    return None


def get_last_key(d: Any) -> Optional[str]:
    """
    Returns the last key from a dict, or None if empty/not a dict.
    """
    if not isinstance(d, dict) or not d:
        return None
    last = None
    for k in d:
        last = k
    return last


def get_keys(d: Any) -> List[str]:
    """
    Returns a list of keys from a dict, or empty list if not a dict.
    """
    if not isinstance(d, dict):
        return []
    return list(d.keys())
