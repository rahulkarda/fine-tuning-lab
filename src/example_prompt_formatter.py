from src.prompt_formatter import format_prompt

if __name__ == "__main__":
    # Example for Phi-family prompt formatting
    example = {
        "system": "You are a helpful assistant.",
        "user": "How do I make pancakes?",
        "assistant": "To make pancakes, mix flour, eggs, milk, and cook on a skillet."
    }
    prompt = format_prompt(example, model_family="phi")
    print("--- Phi formatted prompt ---")
    print(prompt)

    # Example for Qwen-family prompt formatting
    prompt_qwen = format_prompt(example, model_family="qwen")
    print("--- Qwen formatted prompt ---")
    print(prompt_qwen)

    # Example for Llama-3-family prompt formatting
    prompt_llama3 = format_prompt(example, model_family="llama3")
    print("--- Llama-3 formatted prompt ---")
    print(prompt_llama3)
