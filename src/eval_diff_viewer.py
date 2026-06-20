from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer


def diff_viewer_generation(
    base_model: PreTrainedModel,
    tuned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 4,
    device: Optional[str] = "cuda"
) -> List[Dict[str, Any]]:
    """
    Runs generation for base and tuned models on same prompts, returns diffs.
    Args:
        base_model: original (untuned) model
        tuned_model: finetuned model
        tokenizer: HF tokenizer
        prompts: list of prompt strings
        max_new_tokens: max tokens to generate
        temperature: sampling temperature
        top_p: nucleus sampling
        batch_size: batch size
        device: str
    Returns:
        List of dicts: {
            'prompt': str,
            'base_output': str,
            'tuned_output': str,
            'diff': str (simple inline diff)
        }
    """
    import torch
    base_model = base_model.to(device)
    tuned_model = tuned_model.to(device)
    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            base_ids = base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            tuned_ids = tuned_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        base_outputs = tokenizer.batch_decode(base_ids, skip_special_tokens=True)
        tuned_outputs = tokenizer.batch_decode(tuned_ids, skip_special_tokens=True)
        for prompt, base_out, tuned_out in zip(batch_prompts, base_outputs, tuned_outputs):
            base_resp = base_out[len(prompt):].strip() if base_out.startswith(prompt) else base_out.strip()
            tuned_resp = tuned_out[len(prompt):].strip() if tuned_out.startswith(prompt) else tuned_out.strip()
            diff_str = simple_inline_diff(base_resp, tuned_resp)
            results.append({
                'prompt': prompt,
                'base_output': base_resp,
                'tuned_output': tuned_resp,
                'diff': diff_str
            })
    return results


def simple_inline_diff(a: str, b: str) -> str:
    """
    Returns a simple inline diff between two strings.
    - Words unique to base: [-word-]
    - Words unique to tuned: {+word+}
    Args:
        a: base string
        b: tuned string
    Returns:
        diff string
    """
    import difflib
    a_words = a.split()
    b_words = b.split()
    sm = difflib.SequenceMatcher(None, a_words, b_words)
    diff = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            diff.extend(a_words[i1:i2])
        elif tag == 'replace':
            diff.extend([f"[-{w}-]" for w in a_words[i1:i2]])
            diff.extend([f"{{+{w}+}}" for w in b_words[j1:j2]])
        elif tag == 'delete':
            diff.extend([f"[-{w}-]" for w in a_words[i1:i2]])
        elif tag == 'insert':
            diff.extend([f"{{+{w}+}}" for w in b_words[j1:j2]])
    return ' '.join(diff)
