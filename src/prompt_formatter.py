"""
Prompt formatting utility for chat-style datasets.

Supports:
- Simple chat templates for major model families: Phi, Qwen, Llama-3
- Format examples (dicts with 'system', 'user', 'assistant') into training-ready prompt strings

Templates:
  - Phi: <|system|> <|user|> <|assistant|> tokens
  - Qwen: <|im_start|>role ... <|im_end|> blocks
  - Llama-3: <|system|> <|user|> <|assistant|> blocks (newline separated)

Usage:
    prompt = format_prompt(example, model_family="phi")
    prompt = format_prompt(example, model_family="qwen")
    prompt = format_prompt(example, model_family="llama3")

Args:
    example: dict with keys ('system', 'user', 'assistant')
    model_family: str, one of 'phi', 'qwen', 'llama3'
    add_system: whether to include system message if present

Returns:
    Prompt string formatted for model family
"""
from typing import List, Dict, Any

CHAT_TEMPLATES = {
    "phi": {
        "system": "<|system|>{system}\n",
        "user": "<|user|>{user}\n",
        "assistant": "<|assistant|>{assistant}\n"
    },
    "qwen": {
        "system": "<|im_start|>system\n{system}\n<|im_end|>\n",
        "user": "<|im_start|>user\n{user}\n<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{assistant}\n<|im_end|>\n"
    },
    "llama3": {
        "system": "<|system|>\n{system}\n",
        "user": "<|user|>\n{user}\n",
        "assistant": "<|assistant|>\n{assistant}\n"
    }
}

def format_prompt(
    example: Dict[str, Any],
    model_family: str = "phi",
    add_system: bool = True
) -> str:
    """
    Formats a chat example to a prompt string for given model family.
    Args:
        example: dict with keys ('user', 'assistant', optionally 'system')
        model_family: one of 'phi', 'qwen', 'llama3'
        add_system: whether to prepend system message if present
    Returns:
        prompt string
    """
    if model_family not in CHAT_TEMPLATES:
        raise ValueError(f"Unknown model_family: {model_family}")
    template = CHAT_TEMPLATES[model_family]
    parts = []
    if add_system and 'system' in example and example['system']:
        parts.append(template['system'].format(system=example['system']))
    # Handle missing keys gracefully
    user_val = example.get('user', None)
    if user_val is not None:
        parts.append(template['user'].format(user=user_val))
    assistant_val = example.get('assistant', None)
    if assistant_val is not None:
        parts.append(template['assistant'].format(assistant=assistant_val))
    return ''.join(parts)


def format_prompts_batch(
    examples: List[Dict[str, Any]],
    model_family: str = "phi",
    add_system: bool = True
) -> List[str]:
    """
    Formats a list of chat examples into prompt strings for the given model family.
    Args:
        examples: list of dicts with keys ('user', 'assistant', optionally 'system')
        model_family: one of 'phi', 'qwen', 'llama3'
        add_system: whether to prepend system message if present
    Returns:
        List of prompt strings
    """
    return [format_prompt(ex, model_family=model_family, add_system=add_system) for ex in examples]
