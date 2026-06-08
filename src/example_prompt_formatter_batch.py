from src.prompt_formatter import format_prompts_batch

if __name__ == "__main__":
    examples = [
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
    prompts_phi = format_prompts_batch(examples, model_family="phi")
    print("--- Phi batch formatted prompts ---")
    for i, prompt in enumerate(prompts_phi):
        print(f"Prompt {i+1}:")
        print(prompt)
    prompts_qwen = format_prompts_batch(examples, model_family="qwen")
    print("--- Qwen batch formatted prompts ---")
    for i, prompt in enumerate(prompts_qwen):
        print(f"Prompt {i+1}:")
        print(prompt)
    prompts_llama3 = format_prompts_batch(examples, model_family="llama3")
    print("--- Llama-3 batch formatted prompts ---")
    for i, prompt in enumerate(prompts_llama3):
        print(f"Prompt {i+1}:")
        print(prompt)
