from src.utils import load_jsonl, validate_jsonl_schema, train_val_split

"""
Minimal loader for Phi-3-mini instruction dataset.

Intended for phase 4 experiment:
- Loads JSONL dataset
- Validates schema (system, user, assistant keys)
- Splits train/val

Run directly to print stats and first few examples.
"""

def phi3mini_schema(obj):
    """
    Checks if obj has required keys for Phi-3-mini instruction data.
    """
    return (
        isinstance(obj, dict) and
        'user' in obj and
        'assistant' in obj
        # optional 'system'
    )

if __name__ == "__main__":
    dataset_path = "data/phi3mini_instruct.jsonl"
    print(f"Loading dataset from {dataset_path}...")
    try:
        data = load_jsonl(dataset_path)
    except FileNotFoundError:
        print("Dataset not found: skipping.")
        exit(0)
    print(f"Loaded {len(data)} examples.")

    # Schema validation
    invalid = validate_jsonl_schema(dataset_path, phi3mini_schema)
    if invalid > 0:
        print(f"WARNING: {invalid} invalid examples (missing keys or not JSON)")
    else:
        print("All examples valid.")

    # Split train/val
    train, val = train_val_split(data, val_ratio=0.1, seed=42)
    print(f"Train set: {len(train)} examples, Val set: {len(val)} examples.")

    # Show first 2 valid examples
    for i, ex in enumerate(train[:2]):
        print(f"--- Example {i+1} ---")
        print(f"User: {ex.get('user','')}")
        print(f"Assistant: {ex.get('assistant','')}")
        if 'system' in ex:
            print(f"System: {ex['system']}")
