"""
Prompt formatting utility for chat-style datasets.

Supports:
- Simple chat templates for major model families: Phi, Qwen, Llama-3
- Format examples (dicts with 'system', 'user', 'assistant') into training-ready prompt strings
- Multi-turn dialogue formatting (list of turns)
- Role mapping for template compatibility
- Prompt trimming utility for length control

Templates:
  - Phi: <|system|> <|user|> <|assistant|> tokens
  - Qwen: <|im_start|>role ... <|im_end|> blocks
  - Llama-3: <|system|> <|user|> <|assistant|> blocks (newline separated)

Usage:
    prompt = format_prompt(example, model_family="phi")
    prompt = format_prompt(example, model_family="qwen")
    prompt = format_prompt(example, model_family="llama3")
    prompt = format_multi_turn_prompt(dialogue, model_family="phi")
    trimmed = trim_prompt(prompt, max_length=2048, tokenizer=my_tokenizer)

Args:
    example: dict with keys ('system', 'user', 'assistant')
    model_family: str, one of 'phi', 'qwen', 'llama3'
    add_system: whether to include system message if present
    dialogue: list of dicts with 'role' and 'content' (for multi-turn)
    system_message: optional str for multi-turn prompt (prepended)
    trim_prompt: trims a string prompt to max tokens, optionally preserving suffix/answer

Returns:
    Prompt string formatted for model family.
    For multi-turn: roles are mapped to template keys if possible.
    For trimming: string with tokens <= max_length
"""
from typing import List, Dict, Any, Optional

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
    if system_message is not None and system_message != "":
        parts.append(template['system'].format(system=system_message))
    for turn in dialogue:
        # Robustly handle missing or None keys
        role = turn.get('role', None)
        content = turn.get('content', None)
        if role is None or content is None:
            continue
        role_str = str(role).lower() if isinstance(role, str) else None
        content_str = str(content) if content is not None else None
        if not role_str or not content_str:
            continue
        # Only add non-empty content
        content_str = content_str.strip()
        if not content_str:
            continue
        mapped_role = role_map.get(role_str, None)
        if mapped_role is None:
            # Try lowercasing, fallback to original
            mapped_role = role_map.get(role_str.lower(), role_str)
        if mapped_role in template:
            parts.append(template[mapped_role].format(**{mapped_role: content_str}))
        else:
            # Fallback: just append content
            parts.append(content_str + "\n")
    return ''.join(parts)


def trim_prompt(
    prompt: str,
    max_length: int,
    tokenizer,
    answer: Optional[str] = None
) -> str:
    """
    Trims a string prompt to max tokens using tokenizer.
    If answer is provided, preserves answer as suffix and trims the prefix.
    Args:
        prompt: string prompt
        max_length: max tokens after tokenization
        tokenizer: HuggingFace tokenizer
        answer: optional string to preserve as suffix
    Returns:
        Trimmed string
    """
    if answer:
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        total_tokens = len(prompt_tokens) + len(answer_tokens)
        if total_tokens <= max_length:
            return prompt + answer
        prefix_budget = max_length - len(answer_tokens)
        if prefix_budget > 0:
            prefix_tokens = prompt_tokens[:prefix_budget]
            trimmed_prefix = tokenizer.decode(prefix_tokens, skip_special_tokens=False)
            return trimmed_prefix + answer
        else:
            # Only enough room for answer
            return answer
    else:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        if len(tokens) <= max_length:
            return prompt
        tokens = tokens[:max_length]
        trimmed = tokenizer.decode(tokens, skip_special_tokens=False)
        return trimmed
