from src.utils import flatten_dict

"""
Minimal example usage for flatten_dict utility.
Run directly to see output.
"""

if __name__ == "__main__":
    nested = {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3
            }
        },
        "f": 4
    }
    flat = flatten_dict(nested)
    print("Original nested dict:")
    print(nested)
    print("\nFlattened dict:")
    print(flat)
