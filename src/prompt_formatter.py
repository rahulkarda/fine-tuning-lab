"""
Prompt formatting utility for chat-style datasets.

Supports:
- Simple chat templates for major model families: Phi, Qwen, Llama-3
- Format examples (dicts with 'system', 'user', 'assistant') into training-ready prompt strings
- Multi-turn dialogue formatting (list of turns)
- Role mapping for template compatibility

Templates:
  - Phi: <|system|> <|user|> <|assistant|> tokens
  - Qwen: <|im_start|>role ... <|im_end|> blocks
  - Llama-3: <|system|> <|user|> <|assistant|> blocks (newline separated)

Usage:
    prompt = format_prompt(example, model_family="phi")
    prompt = format_prompt(example, model_family="qwen")
    prompt = format_prompt(example, model_family="llama3")
    prompt = format_multi_turn_prompt(dialogue, model_family="phi")

Args:
    example: dict with keys ('system', 'user', 'assistant')
    model_family: str, one of 'phi', 'qwen', 'llama3'
    add_system: whether to include system message if present
    dialogue: list of dicts with 'role' and 'content' (for multi-turn)
    system_message: optional str for multi-turn prompt (prepended)

Returns:
    Prompt string formatted for model family.
    For multi-turn: roles are mapped to template keys if possible.
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

ROLE_MAP = {
    "phi": {
        "system": "system",
        "user": "user",
        "assistant": "assistant"
    },
    "qwen": {
        "system": "system",
        "user": "user",
        "assistant": "assistant"
    },
    "llama3": {
        "system": "system",
        "user": "user",
        "assistant": "assistant"
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


def format_multi_turn_prompt(
    dialogue: List[Dict[str, Any]],
    model_family: str = "phi",
    system_message: str = None
) -> str:
    """
    Formats a multi-turn dialogue (list of turns) for the given model family.
    Args:
        dialogue: list of dicts with keys ('role', 'content')
        model_family: one of 'phi', 'qwen', 'llama3'
        system_message: optional str, prepended as system message if present
    Returns:
        prompt string
    Notes:
        - Roles are mapped to template keys (system, user, assistant) if possible.
        - If role is not directly supported, attempts mapping by lowercasing.
        - Ignores turns with missing role/content.
    """
    if model_family not in CHAT_TEMPLATES:
        raise ValueError(f"Unknown model_family: {model_family}")
    template = CHAT_TEMPLATES[model_family]
    role_map = ROLE_MAP[model_family]
    parts = []
    if system_message:
        parts.append(template['system'].format(system=system_message))
    for turn in dialogue:
        role = turn.get('role', None)
        content = turn.get('content', None)
        if role is None or content is None:
            continue
        # Only format roles that are supported in template
        if role in role_map.values():
            parts.append(template[role].format(**{role: content}))
        else:
            # Try mapping common role names to template keys
            if role.lower() in role_map:
                template_key = role_map[role.lower()]
                parts.append(template[template_key].format(**{template_key: content}))
            else:
                continue
    return ''.join(parts)
