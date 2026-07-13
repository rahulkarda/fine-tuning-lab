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
- is_empty: returns True if value is None, empty, or only whitespace (new)
- dataset_token_counts: computes token counts for each item in a dataset (new)
- dataset_text_lengths: computes text length (number of characters) per item (new)
- dataset_label_counts: computes label distribution for quick stats (new)
- get_first_key: returns the first key from a dict, or None if empty (new)

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
- is_empty is a simple helper for filtering or validation of values (None, '', [], etc).
- dataset_token_counts helps analyze tokenization stats for batching and memory planning (new).
- dataset_text_lengths computes text length (in characters) for each item, useful for quick stats (new).
- dataset_label_counts computes label distribution statistics for classification tasks (new).
- get_first_key returns the first key from a dict, useful for quick schema inspection (new).

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

Caveats:
- These utilities are not optimized for large-scale datasets (>100k items); use for experiment prototyping.
- Deduplication is based on object hash; small differences (ordering, whitespace) may defeat it.
- Text normalization is aggressive; tune for your use-case if needed.
- train_val_split uses random.shuffle; reproducibility is controlled by seed.
- is_empty checks for None, '', [], {}, and whitespace-only strings. It does not check for False.
- dataset_token_counts assumes you provide a compatible tokenizer and optionally the text field name.
- dataset_text_lengths computes character length using len(str(text)).
- dataset_label_counts assumes a 'label' field is present in each item (can override label_key).
- get_first_key returns None if dict is empty or not a dict.
"""
import json
import random
from typing import List, Any, Callable, Optional, Tuple, Dict
import re


def count_jsonl_lines(path: str) -> int:
    """
    Counts non-empty lines in a JSONL file.
    """
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_jsonl(path: str) -> List[Any]:
    """
    Loads objects from JSONL, skips blank lines.
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
    Checks each line against a schema function.
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


def deduplicate_jsonl(input_path: str, output_path: str) -> None:
    """
    Removes duplicate JSONL objects based on hash.
    """
    seen = set()
    unique = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                h = json.dumps(obj, sort_keys=True)
                if h not in seen:
                    seen.add(h)
                    unique.append(obj)
            except Exception:
                continue
    with open(output_path, 'w', encoding='utf-8') as f:
        for obj in unique:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def train_val_split(data: List[Any], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Any], List[Any]]:
    """
    Reproducible train/val split by ratio and seed.
    """
    data = list(data)
    random.seed(seed)
    random.shuffle(data)
    n_val = int(len(data) * val_ratio)
    val = data[:n_val]
    train = data[n_val:]
    return train, val


def flatten_dict(d: Any, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Recursively flattens nested dicts for easier metric aggregation.
    """
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
    else:
        items[parent_key] = d
    return items


def get_model_family(model_name: str) -> str:
    """
    Infers major model family from model name string.
    """
    name = model_name.lower()
    if 'phi' in name:
        return 'phi'
    if 'qwen' in name:
        return 'qwen'
    if 'llama' in name or 'lama' in name:
        return 'llama'
    if 'mistral' in name:
        return 'mistral'
    if 'gemma' in name:
        return 'gemma'
    return 'other'


def normalize_text(text: str) -> str:
    """
    Text normalization for robust comparison (lowercase, whitespace, punctuation-stripping).
    """
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\.,;:!?"\'\-]', '', text)
    return text


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


def get_first_non_empty(items: List[Any]) -> Any:
    """
    Returns the first non-empty item from a list.
    """
    for item in items:
        if not is_empty(item):
            return item
    return None


def get_last_non_empty(items: List[Any]) -> Any:
    """
    Returns the last non-empty item from a list.
    """
    for item in reversed(items):
        if not is_empty(item):
            return item
    return None


def get_non_empty(items: List[Any]) -> List[Any]:
    """
    Returns all non-empty items from a list.
    """
    return [item for item in items if not is_empty(item)]


def dataset_sample_stats(data: List[Any], n: int = 20, seed: int = 42) -> Dict[str, Any]:
    """
    Basic stats on a random sample of dataset items (lengths, keys, label balance).
    """
    random.seed(seed)
    sample = random.sample(data, min(n, len(data)))
    lengths = [len(str(item.get('text', ''))) for item in sample if isinstance(item, dict)]
    keys = set()
    for item in sample:
        if isinstance(item, dict):
            keys.update(item.keys())
    label_counts = {}
    for item in sample:
        if isinstance(item, dict) and 'label' in item:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
    return {
        'sample_size': len(sample),
        'avg_length': sum(lengths) / len(lengths) if lengths else 0,
        'keys': list(keys),
        'label_counts': label_counts
    }


def deduplicate_texts(texts: List[str]) -> List[str]:
    """
    Remove duplicate text items from a list (case-sensitive).
    """
    seen = set()
    unique = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def pad_or_truncate(seq: List[Any], length: int, pad_token: Any = 0) -> List[Any]:
    """
    Pad or truncate a sequence to a fixed length.
    """
    if len(seq) > length:
        return seq[:length]
    else:
        return seq + [pad_token] * (length - len(seq))


def is_empty(val: Any) -> bool:
    """
    Returns True if value is None, empty, or only whitespace.
    """
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip() == ''
    if isinstance(val, (list, dict)):
        return len(val) == 0
    return False


def dataset_token_counts(data: List[Any], tokenizer: Any, text_key: str = 'text') -> List[int]:
    """
    Computes token counts for each item in a dataset.
    """
    counts = []
    for item in data:
        text = item.get(text_key, '') if isinstance(item, dict) else str(item)
        if not is_empty(text):
            tokens = tokenizer(text, return_tensors=None, add_special_tokens=True)['input_ids']
            counts.append(len(tokens))
    return counts


def dataset_text_lengths(data: List[Any], text_key: str = 'text') -> List[int]:
    """
    Computes text length (number of characters) per item.
    """
    lengths = []
    for item in data:
        text = item.get(text_key, '') if isinstance(item, dict) else str(item)
        if not is_empty(text):
            lengths.append(len(str(text)))
    return lengths


def dataset_label_counts(data: List[Any], label_key: str = 'label') -> Dict[Any, int]:
    """
    Computes label distribution for quick stats.
    """
    counts = {}
    for item in data:
        if isinstance(item, dict):
            label = item.get(label_key, None)
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
        elif isinstance(item, str):
            # Accept direct string label
            if item.strip() != '':
                counts[item] = counts.get(item, 0) + 1
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
