from __future__ import annotations

import torch

from .cache_utils import cache_memory_bytes, to_legacy_cache
from .compression import LayerAdaptiveCompressor
from .scheduler import RuntimeScheduler


@torch.no_grad()
def generate_layer_adaptive(
    model,
    tokenizer,
    prompt: str,
    scheduler: RuntimeScheduler,
    global_compression: float,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = encoded.input_ids
    policies = scheduler.allocate(global_compression)
    compressor = LayerAdaptiveCompressor(
        min_tokens=policies[0].min_tokens,
        recent_window=policies[0].recent_window,
    )
    if max_new_tokens <= 0:
        return {
            "text": tokenizer.decode(input_ids[0], skip_special_tokens=True),
            "compression_ratio": global_compression,
            "avg_cache_bytes_before": 0,
            "avg_cache_bytes_after": 0,
            "policies": [policy.__dict__ for policy in policies],
        }

    outputs = model(input_ids=input_ids, use_cache=True, output_attentions=True)
    cache = to_legacy_cache(outputs.past_key_values)
    before_bytes = cache_memory_bytes(cache)
    cache = compressor.compress_cache(cache, outputs.attentions, policies)
    after_bytes = cache_memory_bytes(cache)

    generated = [input_ids]
    next_token = _sample_next(outputs.logits[:, -1, :], temperature=temperature, top_p=top_p)
    generated.append(next_token)

    next_position = input_ids.shape[1]
    for _ in range(max_new_tokens - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break
        position_ids = torch.tensor([[next_position]], device=device, dtype=torch.long)
        outputs = model(
            input_ids=next_token,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
        )
        next_position += 1
        cache = to_legacy_cache(outputs.past_key_values)
        before_bytes += cache_memory_bytes(cache)
        cache = compressor.compress_cache(cache, outputs.attentions, policies)
        after_bytes += cache_memory_bytes(cache)
        next_token = _sample_next(outputs.logits[:, -1, :], temperature=temperature, top_p=top_p)
        generated.append(next_token)

    full_ids = torch.cat(generated, dim=1)
    return {
        "text": tokenizer.decode(full_ids[0], skip_special_tokens=True),
        "compression_ratio": global_compression,
        "avg_cache_bytes_before": before_bytes / max(len(generated), 1),
        "avg_cache_bytes_after": after_bytes / max(len(generated), 1),
        "policies": [policy.__dict__ for policy in policies],
    }


def _sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[..., 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs * keep)
        probs = filtered / filtered.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)
