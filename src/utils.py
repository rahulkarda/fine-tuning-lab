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

Caveats:
- These utilities are not optimized for large-scale datasets (>100k items); use for experiment prototyping.
- Deduplication is based on object hash; small differences (ordering, whitespace) may defeat it.
- Text normalization is aggressive; tune for your use-case if needed.
- train_val_split uses random.shuffle; reproducibility is controlled by seed.
- is_empty checks for None, '', [], {}, and whitespace-only strings. It does not check for False.
- dataset_token_counts assumes you provide a compatible tokenizer and optionally the text field name.
- dataset_text_lengths computes character length using len(str(text)).
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
                    print(f"Warning: error loading JSON line in {path}, line {idx+1}: {e}")
                    had_invalid = True
                continue
    return data


def validate_jsonl_schema(path: str, schema_fn: Callable[[Any], bool]) -> int:
    """
    Validates each line in JSONL file against schema_fn.
    Returns number of invalid lines.
    """
    invalid_count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                invalid_count += 1
                continue
            if not schema_fn(obj):
                invalid_count += 1
    return invalid_count


def deduplicate_jsonl(path: str, output_path: str) -> None:
    """
    Deduplicates objects in JSONL file based on hash, writes to output_path.
    """
    seen = set()
    deduped = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            h = hash(json.dumps(obj, sort_keys=True))
            if h not in seen:
                seen.add(h)
                deduped.append(obj)
    with open(output_path, 'w', encoding='utf-8') as f:
        for obj in deduped:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def train_val_split(data: List[Any], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Any], List[Any]]:
    """
    Splits data into train and val sets by ratio and seed.
    """
    n = len(data)
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    val_n = int(round(n * val_ratio))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    return train, val


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Recursively flattens nested dicts.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_model_family(model_name: str) -> str:
    """
    Infers model family from model name string.
    """
    name = model_name.lower()
    if 'phi' in name:
        return 'phi'
    elif 'qwen' in name:
        return 'qwen'
    elif 'llama' in name:
        return 'llama'
    elif 'mistral' in name:
        return 'mistral'
    else:
        return 'unknown'


def normalize_text(text: str) -> str:
    """
    Aggressively normalizes text: lower, strip, remove punctuation, collapse whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_numeric(val: Any) -> bool:
    """
    Returns True if val is int or float, not bool.
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def sample_jsonl(path: str, n: int, seed: int = 42) -> List[Any]:
    """
    Randomly samples n lines from JSONL file.
    """
    data = load_jsonl(path)
    random.seed(seed)
    random.shuffle(data)
    return data[:n]


def get_first_non_empty(items: List[Any]) -> Optional[Any]:
    """
    Returns first non-empty item from list.
    """
    for item in items:
        if not is_empty(item):
            return item
    return None


def get_last_non_empty(items: List[Any]) -> Optional[Any]:
    """
    Returns last non-empty item from list.
    """
    for item in reversed(items):
        if not is_empty(item):
            return item
    return None


def get_non_empty(items: List[Any]) -> List[Any]:
    """
    Returns all non-empty items from list.
    """
    return [item for item in items if not is_empty(item)]


def dataset_stats(data: List[Any], text_key: str = 'text') -> dict:
    """
    Computes quick stats on token length and label balance.
    """
    lengths = []
    labels = []
    for item in data:
        if isinstance(item, dict):
            txt = item.get(text_key, '')
            label = item.get('label', None)
        else:
            txt = str(item)
            label = None
        lengths.append(len(txt))
        labels.append(label)
    label_dist = {}
    for l in labels:
        if l is not None:
            label_dist[l] = label_dist.get(l, 0) + 1
    return {
        'count': len(data),
        'mean_length': sum(lengths)/len(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'label_dist': label_dist
    }


def dataset_sample_stats(data: List[Any], n: int = 20, seed: int = 42, text_key: str = 'text') -> dict:
    """
    Computes stats on a random sample of dataset items.
    """
    sample = data[:]
    random.seed(seed)
    random.shuffle(sample)
    sample = sample[:n]
    lengths = []
    keys = set()
    labels = []
    for item in sample:
        if isinstance(item, dict):
            txt = item.get(text_key, '')
            keys.update(item.keys())
            label = item.get('label', None)
        else:
            txt = str(item)
            label = None
        lengths.append(len(txt))
        labels.append(label)
    label_dist = {}
    for l in labels:
        if l is not None:
            label_dist[l] = label_dist.get(l, 0) + 1
    return {
        'sample_count': len(sample),
        'mean_length': sum(lengths)/len(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'sample_keys': list(keys),
        'sample_label_dist': label_dist
    }


def deduplicate_texts(texts: List[str]) -> List[str]:
    """
    Removes duplicate text items (case-sensitive).
    """
    seen = set()
    result = []
    for txt in texts:
        if txt not in seen:
            seen.add(txt)
            result.append(txt)
    return result


def pad_or_truncate(seq: List[Any], length: int, pad_token: Any = 0) -> List[Any]:
    """
    Pads or truncates a sequence to fixed length.
    """
    if len(seq) > length:
        return seq[:length]
    else:
        return seq + [pad_token] * (length - len(seq))


def is_empty(val: Any) -> bool:
    """
    Returns True if val is None, empty, or only whitespace.
    """
    if val is None:
        return True
    if isinstance(val, str):
        return not val.strip()
    if isinstance(val, (list, dict)):
        return len(val) == 0
    return False


def dataset_token_counts(data: List[Any], tokenizer, text_key: str = 'text', add_special_tokens: bool = True) -> List[int]:
    """
    Computes token counts for each item in dataset using tokenizer.
    Args:
        data: list of items (dicts or str)
        tokenizer: transformers tokenizer instance
        text_key: key to extract text from dict (default: 'text')
        add_special_tokens: whether to add special tokens when encoding
    Returns:
        List of token counts for each item
    """
    counts = []
    for item in data:
        if isinstance(item, dict):
            txt = item.get(text_key, '')
        else:
            txt = str(item)
        tokens = tokenizer.encode(txt, add_special_tokens=add_special_tokens)
        counts.append(len(tokens))
    return counts


def dataset_text_lengths(data: List[Any], text_key: str = 'text') -> List[int]:
    """
    Computes the length (number of characters) of text for each item.
    Args:
        data: list of items (dicts or str)
        text_key: key to extract text from dict (default: 'text')
    Returns:
        List of text lengths for each item
    """
    lengths = []
    for item in data:
        if isinstance(item, dict):
            txt = item.get(text_key, '')
        else:
            txt = str(item)
        lengths.append(len(txt))
    return lengths
