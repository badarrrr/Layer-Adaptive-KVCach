from __future__ import annotations

import inspect
from dataclasses import replace

import torch

from .cache_utils import cache_memory_bytes, to_legacy_cache
from .compression import CompressionPolicy, LayerAdaptiveCompressor
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
    compression_start_tokens: int = 128,
    max_layer_compression: float = 0.30,
    repetition_penalty: float = 1.10,
    no_repeat_ngram_size: int = 4,
    format_instruction: bool = True,
) -> dict:
    device = next(model.parameters()).device
    model_prompt = f"Question: {prompt}\nAnswer:" if format_instruction else prompt
    encoded = tokenizer(model_prompt, return_tensors="pt").to(device)
    input_ids = encoded.input_ids
    supports_cache_position = _supports_cache_position(model)
    policies = _cap_policies(scheduler.allocate(global_compression), max_layer_compression)
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
            "actual_cache_compression": 0.0,
            "policies": [policy.__dict__ for policy in policies],
        }

    prompt_positions = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)
    model_kwargs = {
        "input_ids": input_ids,
        "position_ids": prompt_positions.unsqueeze(0),
        "use_cache": True,
        "output_attentions": True,
    }
    if supports_cache_position:
        model_kwargs["cache_position"] = prompt_positions
    outputs = model(**model_kwargs)
    cache = to_legacy_cache(outputs.past_key_values)
    initial_cache_bytes = cache_memory_bytes(cache)
    bytes_per_token = initial_cache_bytes / max(input_ids.shape[1], 1)
    before_bytes = initial_cache_bytes
    if input_ids.shape[1] >= compression_start_tokens:
        cache = compressor.compress_cache(
            cache,
            outputs.attentions,
            policies,
            sequence_length=input_ids.shape[1],
            target_compression_ratio=global_compression,
        )
    after_bytes = cache_memory_bytes(cache)

    generated_ids = input_ids
    next_token = _sample_next(
        outputs.logits[:, -1, :],
        generated_ids=generated_ids,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    generated_ids = torch.cat([generated_ids, next_token], dim=1)

    next_position = input_ids.shape[1]
    for _ in range(max_new_tokens - 1):
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

        model_position = _model_position(next_position, cache, supports_cache_position)
        position_ids = torch.tensor([[model_position]], device=device, dtype=torch.long)
        model_kwargs = {
            "input_ids": next_token,
            "position_ids": position_ids,
            "past_key_values": cache,
            "use_cache": True,
            "output_attentions": True,
        }
        if supports_cache_position:
            model_kwargs["cache_position"] = torch.tensor([next_position], device=device, dtype=torch.long)
        outputs = model(**model_kwargs)
        next_position += 1

        cache = to_legacy_cache(outputs.past_key_values)
        before_bytes += bytes_per_token * next_position
        if next_position >= compression_start_tokens:
            cache = compressor.compress_cache(
                cache,
                outputs.attentions,
                policies,
                sequence_length=next_position,
                target_compression_ratio=global_compression,
            )
        after_bytes += cache_memory_bytes(cache)

        next_token = _sample_next(
            outputs.logits[:, -1, :],
            generated_ids=generated_ids,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    generated_tokens = max(generated_ids.shape[1] - input_ids.shape[1], 1)
    return {
        "text": tokenizer.decode(generated_ids[0], skip_special_tokens=True),
        "compression_ratio": global_compression,
        "avg_cache_bytes_before": before_bytes / generated_tokens,
        "avg_cache_bytes_after": after_bytes / generated_tokens,
        "actual_cache_compression": 1.0 - (after_bytes / max(before_bytes, 1)),
        "policies": [policy.__dict__ for policy in policies],
    }


def _cap_policies(
    policies: list[CompressionPolicy],
    max_layer_compression: float,
) -> list[CompressionPolicy]:
    cap = min(max(max_layer_compression, 0.0), 0.95)
    return [
        replace(policy, compression_ratio=min(policy.compression_ratio, cap))
        for policy in policies
    ]


def _sample_next(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    logits = logits.clone()
    if repetition_penalty > 1.0:
        for batch_idx in range(logits.shape[0]):
            seen_tokens = torch.unique(generated_ids[batch_idx])
            token_logits = logits[batch_idx, seen_tokens]
            logits[batch_idx, seen_tokens] = torch.where(
                token_logits < 0,
                token_logits * repetition_penalty,
                token_logits / repetition_penalty,
            )

    if no_repeat_ngram_size > 1:
        _ban_repeated_ngrams(logits, generated_ids, no_repeat_ngram_size)

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


def _ban_repeated_ngrams(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    ngram_size: int,
) -> None:
    if generated_ids.shape[1] < ngram_size - 1:
        return

    prefix_len = ngram_size - 1
    for batch_idx in range(generated_ids.shape[0]):
        tokens = generated_ids[batch_idx].tolist()
        current_prefix = tuple(tokens[-prefix_len:])
        banned_tokens = []
        for start in range(len(tokens) - ngram_size + 1):
            ngram = tokens[start : start + ngram_size]
            if tuple(ngram[:-1]) == current_prefix:
                banned_tokens.append(ngram[-1])
        if banned_tokens:
            logits[batch_idx, banned_tokens] = -torch.inf


def _supports_cache_position(model) -> bool:
    try:
        return "cache_position" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _model_position(
    absolute_position: int,
    cache,
    supports_cache_position: bool,
) -> int:
    if supports_cache_position or cache is None:
        return absolute_position
    return int(cache[0][0].shape[-2])
