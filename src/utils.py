import json
from typing import Dict, Any, Callable

def count_jsonl_lines(path: str) -> int:
    """
    Count the number of lines (examples) in a jsonl file.
    Useful for quick dataset stats.
    """
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def validate_jsonl_schema(path: str, schema_fn: Callable[[Dict[str, Any]], bool]) -> int:
    """
    Validate each line in a jsonl file against a schema_fn.
    Returns the number of invalid examples.
    schema_fn: function taking a dict, returns True if valid.
    """
    invalid_count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except Exception:
                invalid_count += 1
                continue
            if not schema_fn(obj):
                invalid_count += 1
    return invalid_count
