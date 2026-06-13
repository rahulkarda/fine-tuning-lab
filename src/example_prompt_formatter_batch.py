from src.prompt_formatter import format_prompts_batch

"""
Example usage for prompt_formatter.py: formatting batches of chat examples.

Demonstrates how to use format_prompts_batch to generate training-ready
prompt strings for multiple examples across different model families.
Run directly to see formatted outputs for a batch.
"""

if __name__ == "__main__":
    # Batch of examples to format
    batch_examples = [
        {
            "system": "You are a helpful assistant.",
            "user": "How do I make pancakes?",
            "assistant": "To make pancakes, mix flour, eggs, milk, and cook on a skillet."
        },
        {
            "system": "You are a code expert.",
            "user": "Write a Python function to add two numbers.",
            "assistant": "def add(a, b):\n    return a + b"
        }
    ]

    # Format batch for each model family
    families = ["phi", "qwen", "llama3"]
    for family in families:
        formatted_batch = format_prompts_batch(batch_examples, model_family=family)
        print(f"--- {family.capitalize()} batch formatted prompts ---")
        for idx, prompt in enumerate(formatted_batch):
            print(f"Prompt {idx+1}:")
            print(prompt)
