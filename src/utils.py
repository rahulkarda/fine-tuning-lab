"""
Data utility functions for JSONL dataset loading, validation, deduplication, and splitting.

Includes:
- count_jsonl_lines: count non-empty lines in JSONL file
- load_jsonl: load objects from JSONL, skip blank lines
- validate_jsonl_schema: check each line against a schema function
- deduplicate_jsonl: remove duplicate JSONL objects based on hash
- train_val_split: reproducible train/val split by ratio and seed
- get_model_family: infer major model family from model name string (refactored, clarified logic)
- dataset_stats: quick stats for token length and label balance
- normalize_text: text normalization for robust comparison (new)
- flatten_dict: recursively flatten nested dicts for easier metric aggregation (refactored, clarified)
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
- get_model_family is handy for auto-selecting prompt templates by model name (clarified logic).
- Use dataset_stats before/after deduplication to check dataset quality.
- sample_jsonl enables rapid prototyping and debugging on small data slices.
- flatten_dict is ideal for preparing results for aggregate_metrics or dashboard reporting (clarified logic).
- dataset_sample_stats provides basic statistics on a random subset (lengths, keys, label balance).
- deduplicate_texts is a quick utility for removing duplicate text strings (case-sensitive).
- normalize_text can be used for robust deduplication and comparison of text samples (lowercase, whitespace, punctuation-stripping).
- pad_or_truncate is handy for prepping tokenized inputs to a fixed length (for batching, eval, etc).
- is_empty is a simple helper for filtering or validation of values (None, '', [], etc).
- dataset_token_counts helps analyze tokenization stats for batching and memory planning (new; pass a HuggingFace tokenizer).
- dataset_text_lengths computes text length (in characters) for each item, useful for quick stats (new).
- dataset_label_counts computes label distribution statistics for classification tasks (new; pass dataset and label key).
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
    label_counts = dataset_label_counts(data, label_key='label')
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
- Deduplication is based on object hash; small differences (ordering, whitespace) may defeat deduplication.
- Most stats functions assume lists of dicts as input.
- For tokenization functions, HuggingFace tokenizer must be provided.
- pad_or_truncate truncates or pads to fixed length; be careful with sequences shorter/longer than target.
- is_empty returns True for None, empty string, empty list, empty dict, or whitespace-only string.
- get_model_family logic is heuristic; check output for new model names.
"""
# ... [rest of file unchanged] ...
