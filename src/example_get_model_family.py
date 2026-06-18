from src.utils import get_model_family

"""
Example usage for get_model_family utility function.

Demonstrates model family inference from typical model names.
Run directly to see outputs.
"""

if __name__ == "__main__":
    model_names = [
        "microsoft/Phi-3-mini-4k-instruct",
        "Qwen/Qwen1.5-7B-Chat",
        "meta-llama/Meta-Llama-3-8B",
        "unknown-model/foobar",
        "Llama-3-Open",
        "qwen2.5-14b",
        "phi3-mixed",
        "llama3",
        "llama-3"
    ]
    for name in model_names:
        family = get_model_family(name)
        print(f"Model name: {name}\n  Family: {family}\n")
