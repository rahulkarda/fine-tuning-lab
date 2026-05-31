from typing import List, Dict, Any

"""
Prompt formatting utility for chat-style datasets.
Supports simple templates per model family (Phi, Qwen, Llama-3).
"""

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
    if 'user' in example:
        parts.append(template['user'].format(user=example['user']))
    if 'assistant' in example:
        parts.append(template['assistant'].format(assistant=example['assistant']))
    return ''.join(parts)

