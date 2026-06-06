from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache_utils import (
    LegacyCache,
    compress_layer_cache,
    ensure_sorted_unique,
    mean_attention_scores,
    stack_batch_indices,
)


@dataclass(frozen=True)
class CompressionPolicy:
    compression_ratio: float
    strategy: str
    recent_window: int = 32
    heavy_ratio: float = 0.5
    min_tokens: int = 16

    def keep_count(self, seq_len: int) -> int:
        ratio = min(max(self.compression_ratio, 0.0), 0.95)
        return max(self.min_tokens, 1, int(round(seq_len * (1.0 - ratio))))


class LayerAdaptiveCompressor:
    """Apply layer-specific token pruning to a legacy KV cache."""

    def __init__(self, min_tokens: int = 16, recent_window: int = 32):
        self.min_tokens = min_tokens
        self.recent_window = recent_window

    @torch.no_grad()
    def compress_cache(
        self,
        cache: LegacyCache,
        attentions: tuple[torch.Tensor, ...] | None,
        policies: list[CompressionPolicy],
        sequence_length: int | None = None,
    ) -> LegacyCache:
        if not cache:
            return cache

        # Transformers 4.41 LLaMA/Mistral create one causal mask for all layers.
        # Keep the cache length uniform across layers, but let each layer choose
        # its own retained token positions according to its policy.
        target_lengths = [
            min(policy.keep_count(sequence_length or key.shape[-2]), key.shape[-2])
            for policy, (key, _) in zip(policies, cache)
        ]
        target_len = min(target_lengths)

        compressed_layers = []
        for layer_idx, (key, value) in enumerate(cache):
            policy = policies[layer_idx]
            seq_len = key.shape[-2]
            if target_len >= seq_len:
                compressed_layers.append((key, value))
                continue

            keep_indices = self.select_keep_indices(
                seq_len=seq_len,
                policy=policy,
                attention=None if attentions is None else attentions[layer_idx],
                device=key.device,
                batch_size=key.shape[0],
                keep_n=target_len,
            )
            compressed_layers.append(compress_layer_cache(key, value, keep_indices))

        return tuple(compressed_layers)

    def select_keep_indices(
        self,
        seq_len: int,
        policy: CompressionPolicy,
        attention: torch.Tensor | None,
        device,
        batch_size: int,
        keep_n: int | None = None,
    ) -> torch.Tensor:
        keep_n = min(policy.keep_count(seq_len) if keep_n is None else keep_n, seq_len)
        if keep_n >= seq_len:
            base = torch.arange(seq_len, device=device, dtype=torch.long)
            return base[None, :].expand(batch_size, -1)

        if policy.compression_ratio <= 0.0:
            return self._recent_indices(seq_len, keep_n, device, batch_size)
        if policy.strategy == "recent":
            return self._recent_indices(seq_len, keep_n, device, batch_size)
        if policy.strategy == "heavy_hitter":
            return self._heavy_hitter_indices(seq_len, keep_n, policy, attention, device, batch_size)
        if policy.strategy == "hybrid":
            return self._hybrid_indices(seq_len, keep_n, policy, attention, device, batch_size)
        if policy.strategy == "snapkv":
            return self._snapkv_indices(seq_len, keep_n, policy, attention, device, batch_size)
        raise ValueError(f"Unknown compression strategy: {policy.strategy}")

    def _recent_indices(self, seq_len: int, keep_n: int, device, batch_size: int) -> torch.Tensor:
        start = seq_len - keep_n
        idx = torch.arange(start, seq_len, device=device, dtype=torch.long)
        return idx[None, :].expand(batch_size, -1)

    def _heavy_hitter_indices(
        self,
        seq_len: int,
        keep_n: int,
        policy: CompressionPolicy,
        attention: torch.Tensor | None,
        device,
        batch_size: int,
    ) -> torch.Tensor:
        recent_n = min(policy.recent_window, keep_n, seq_len)
        heavy_n = max(keep_n - recent_n, 0)
        scores = mean_attention_scores(attention, seq_len, device).expand(batch_size, -1).clone()
        if recent_n:
            scores[:, -recent_n:] = -torch.inf
        return self._combine_top_and_recent(scores, seq_len, heavy_n, recent_n, device)

    def _hybrid_indices(
        self,
        seq_len: int,
        keep_n: int,
        policy: CompressionPolicy,
        attention: torch.Tensor | None,
        device,
        batch_size: int,
    ) -> torch.Tensor:
        recent_n = min(max(policy.recent_window // 2, 1), keep_n, seq_len)
        heavy_n = max(keep_n - recent_n, 0)
        scores = mean_attention_scores(attention, seq_len, device).expand(batch_size, -1).clone()
        if recent_n:
            scores[:, -recent_n:] = -torch.inf
        return self._combine_top_and_recent(scores, seq_len, heavy_n, recent_n, device)

    def _snapkv_indices(
        self,
        seq_len: int,
        keep_n: int,
        policy: CompressionPolicy,
        attention: torch.Tensor | None,
        device,
        batch_size: int,
    ) -> torch.Tensor:
        recent_n = min(max(policy.recent_window // 4, 1), keep_n, seq_len)
        heavy_n = max(keep_n - recent_n, 0)
        scores = mean_attention_scores(attention, seq_len, device).expand(batch_size, -1).clone()
        if recent_n:
            scores[:, -recent_n:] = -torch.inf
        return self._combine_top_and_recent(scores, seq_len, heavy_n, recent_n, device)

    def _combine_top_and_recent(
        self,
        scores: torch.Tensor,
        seq_len: int,
        heavy_n: int,
        recent_n: int,
        device,
    ) -> torch.Tensor:
        per_batch = []
        for batch_idx in range(scores.shape[0]):
            parts = []
            if heavy_n > 0:
                top_idx = torch.topk(scores[batch_idx], k=heavy_n).indices
                parts.append(top_idx)
            if recent_n > 0:
                recent = torch.arange(seq_len - recent_n, seq_len, device=device, dtype=torch.long)
                parts.append(recent)
            indices = ensure_sorted_unique(torch.cat(parts))
            if indices.numel() < heavy_n + recent_n:
                missing = heavy_n + recent_n - indices.numel()
                filler = torch.arange(seq_len, device=device, dtype=torch.long)
                mask = ~torch.isin(filler, indices)
                indices = ensure_sorted_unique(torch.cat([indices, filler[mask][:missing]]))
            per_batch.append(indices[: heavy_n + recent_n])
        return stack_batch_indices(per_batch, device)
