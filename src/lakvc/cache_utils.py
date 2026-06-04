from __future__ import annotations

from typing import Iterable, Sequence

import torch


LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def to_legacy_cache(cache) -> LegacyCache:
    """Normalize Hugging Face cache objects to tuple[(key, value), ...]."""
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return cache


def cache_sequence_length(cache: LegacyCache) -> int:
    if not cache:
        return 0
    return int(cache[0][0].shape[-2])


def cache_memory_bytes(cache: LegacyCache) -> int:
    total = 0
    for key, value in cache:
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return total


def gather_tokens(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather sequence positions from a KV tensor shaped [batch, heads, seq, dim]."""
    expanded = indices[:, None, :, None].expand(
        tensor.shape[0], tensor.shape[1], indices.shape[1], tensor.shape[3]
    )
    return torch.gather(tensor, dim=2, index=expanded)


def compress_layer_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    keep_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return gather_tokens(key, keep_indices), gather_tokens(value, keep_indices)


def replace_layer(
    cache: LegacyCache,
    layer_idx: int,
    layer_cache: tuple[torch.Tensor, torch.Tensor],
) -> LegacyCache:
    updated = list(cache)
    updated[layer_idx] = layer_cache
    return tuple(updated)


def mean_attention_scores(attention: torch.Tensor | None, seq_len: int, device) -> torch.Tensor:
    """Convert attention [batch, heads, query, kv] into per-token scores [batch, kv]."""
    if attention is None:
        return torch.ones((1, seq_len), device=device)
    scores = attention.mean(dim=(1, 2))
    if scores.shape[-1] != seq_len:
        scores = scores[..., -seq_len:]
    return scores


def ensure_sorted_unique(indices: torch.Tensor) -> torch.Tensor:
    values = torch.unique(indices, sorted=True)
    return values


def stack_batch_indices(per_batch: Sequence[torch.Tensor], fallback_device) -> torch.Tensor:
    if not per_batch:
        return torch.empty((0, 0), dtype=torch.long, device=fallback_device)
    return torch.stack(per_batch, dim=0)
