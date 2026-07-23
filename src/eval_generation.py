from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_outputs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.98,  # increased from 0.95 for more diversity
    min_length: int = 0,  # new param for minimal generation length
    batch_size: int = 4,
    device: Optional[str] = "cuda"
) -> List[str]:
    """
    Generates outputs for a list of prompts using the model and tokenizer.
    Args:
        model: HuggingFace model (PreTrainedModel)
        tokenizer: HuggingFace tokenizer
        prompts: list of prompt strings
        max_new_tokens: maximum tokens to generate per prompt
        temperature: sampling temperature
        top_p: nucleus sampling probability
        min_length: minimal length for output sequence (new)
        batch_size: number of prompts per generation batch
        device: device string ("cuda" or "cpu")
    Returns:
        List of generated strings (responses)
    """
    import torch
    # Fix: handle device=None gracefully (fall back to CPU)
    resolved_device = device if device is not None else "cpu"
    model = model.to(resolved_device)
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(resolved_device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_length=inputs.input_ids.shape[1] + min_length if min_length > 0 else None,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        batch_outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        # Only keep generated portion after prompt
        for prompt, output in zip(batch_prompts, batch_outputs):
            if output.startswith(prompt):
                outputs.append(output[len(prompt):].strip())
            else:
                outputs.append(output.strip())
    return outputs


def evaluate_generation_quality(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    references: Optional[List[str]] = None,
    max_new_tokens: int = 128,
    metrics: Optional[List[str]] = None,
    batch_size: int = 4,
    device: Optional[str] = "cuda",
    top_p: float = 0.98,
    min_length: int = 0
) -> Dict[str, Any]:
    """
    Runs a generation quality probe on fixed prompts and computes metrics.
    Args:
        model: HF model
        tokenizer: HF tokenizer
        prompts: list of prompt strings
        references: optional gold/reference responses (same length as prompts)
        max_new_tokens: generation length
        metrics: list of metrics to compute ("length", "exact_match")
        batch_size: batch size
        device: device string
        top_p: nucleus sampling probability (new, default 0.98)
        min_length: minimal length for output sequence (new)
    Returns:
        Dict with generated outputs and metrics
    """
    outputs = generate_outputs(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
        top_p=top_p,
        min_length=min_length
    )
    results = {
        "outputs": outputs
    }
    if metrics is None:
        metrics = ["length"]
    if "length" in metrics:
        results["lengths"] = [len(tokenizer.encode(out)) for out in outputs]
    if references is not None and "exact_match" in metrics:
        matches = [out.strip() == ref.strip() for out, ref in zip(outputs, references)]
        results["exact_match"] = sum(matches) / len(matches)
    return results
