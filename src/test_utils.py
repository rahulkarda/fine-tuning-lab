import os
from src.utils import count_jsonl_lines

"""
Basic unit test for data utilities.
Currently tests count_jsonl_lines on a small synthetic jsonl file.
Extend with more tests as utilities are added.
"""

def test_count_jsonl_lines():
    # Create a temporary jsonl file with known lines
    test_path = 'test_tmp.jsonl'
    lines = [
        '{"id": 1, "text": "hello"}\n',
        '{"id": 2, "text": "world"}\n',
        '{"id": 3, "text": "!"}\n'
    ]
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    try:
        result = count_jsonl_lines(test_path)
        assert result == 3, f"Expected 3 lines, got {result}"
        print("test_count_jsonl_lines passed.")
    finally:
        os.remove(test_path)

if __name__ == '__main__':
    test_count_jsonl_lines()
