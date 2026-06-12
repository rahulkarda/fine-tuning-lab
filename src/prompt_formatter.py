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


def trim_prompt(
    prompt: str,
    max_length: int,
    tokenizer,
    preserve_suffix: bool = False,
    answer_delimiter: Optional[str] = None
) -> str:
    """
    Trims a string prompt to max_length tokens using the provided tokenizer.
    Optionally preserves the answer/suffix if answer_delimiter is provided.
    Args:
        prompt: string prompt to trim
        max_length: max tokens (inclusive)
        tokenizer: HF tokenizer (must have encode/decode)
        preserve_suffix: if True and answer_delimiter is present, preserve answer after delimiter
        answer_delimiter: string delimiter signaling start of answer (e.g. '<|assistant|>')
    Returns:
        trimmed string prompt
    """
    if max_length <= 0:
        return prompt
    # If preserving suffix/answer, try to keep answer section
    if preserve_suffix and answer_delimiter and answer_delimiter in prompt:
        parts = prompt.split(answer_delimiter, 1)
        prefix = parts[0]
        answer = answer_delimiter + parts[1]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=True)
        # If answer itself is too long, truncate answer
        if len(answer_tokens) > max_length:
            answer_tokens = answer_tokens[:max_length]
            trimmed = tokenizer.decode(answer_tokens, skip_special_tokens=False)
            return trimmed
        # Truncate prefix to fit max_length - answer
        prefix_budget = max_length - len(answer_tokens)
        if prefix_budget > 0:
            prefix_tokens = prefix_tokens[:prefix_budget]
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
