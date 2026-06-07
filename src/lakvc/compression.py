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
    sink_tokens: int = 4

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
        target_compression_ratio: float | None = None,
    ) -> LegacyCache:
        if not cache:
            return cache

        if len(policies) != len(cache):
            raise ValueError(
                f"Expected one compression policy per cache layer; got "
                f"{len(policies)} policies for {len(cache)} layers."
            )

        # Transformers 4.41 creates one causal mask for all decoder layers.
        # Therefore the physical cache length must be uniform. Derive that
        # length from the requested global ratio so layer-adaptive policies are
        # compared at the same actual memory budget as uniform baselines.
        if target_compression_ratio is None:
            target_compression_ratio = sum(p.compression_ratio for p in policies) / len(policies)
        ratio = min(max(target_compression_ratio, 0.0), 0.95)
        reference_length = sequence_length or cache[0][0].shape[-2]
        min_tokens = max(policy.min_tokens for policy in policies)
        target_len = max(min_tokens, 1, int(round(reference_length * (1.0 - ratio))))
        target_len = min(target_len, min(key.shape[-2] for key, _ in cache))

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
            return self._recent_indices(seq_len, keep_n, policy, device, batch_size)
        if policy.strategy == "recent":
            return self._recent_indices(seq_len, keep_n, policy, device, batch_size)
        if policy.strategy == "heavy_hitter":
            return self._heavy_hitter_indices(seq_len, keep_n, policy, attention, device, batch_size)
        if policy.strategy == "hybrid":
            return self._hybrid_indices(seq_len, keep_n, policy, attention, device, batch_size)
        if policy.strategy == "snapkv":
            return self._snapkv_indices(seq_len, keep_n, policy, attention, device, batch_size)
        raise ValueError(f"Unknown compression strategy: {policy.strategy}")

    def _recent_indices(
        self,
        seq_len: int,
        keep_n: int,
        policy: CompressionPolicy,
        device,
        batch_size: int,
    ) -> torch.Tensor:
        sink_n = min(max(policy.sink_tokens, 0), keep_n, seq_len)
        recent_n = keep_n - sink_n
        parts = [torch.arange(sink_n, device=device, dtype=torch.long)]
        if recent_n:
            parts.append(torch.arange(seq_len - recent_n, seq_len, device=device, dtype=torch.long))
        idx = ensure_sorted_unique(torch.cat(parts))
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
        sink_n, heavy_n, recent_n = self._selection_counts(seq_len, keep_n, policy)
        scores = mean_attention_scores(attention, seq_len, device).expand(batch_size, -1).clone()
        if sink_n:
            scores[:, :sink_n] = -torch.inf
        if recent_n:
            scores[:, -recent_n:] = -torch.inf
        return self._combine_selected(scores, seq_len, sink_n, heavy_n, recent_n, device)

    def _hybrid_indices(
        self,
        seq_len: int,
        keep_n: int,
        policy: CompressionPolicy,
        attention: torch.Tensor | None,
        device,
        batch_size: int,
    ) -> torch.Tensor:
        sink_n, heavy_n, recent_n = self._selection_counts(seq_len, keep_n, policy)
        scores = mean_attention_scores(attention, seq_len, device).expand(batch_size, -1).clone()
        if sink_n:
            scores[:, :sink_n] = -torch.inf
        if recent_n:
            scores[:, -recent_n:] = -torch.inf
        return self._combine_selected(scores, seq_len, sink_n, heavy_n, recent_n, device)

    def _snapkv_indices(
        self,
        seq_len: int,
        keep_n: int,
        policy: CompressionPolicy,
        attention: torch.Tensor | None,
        device,
        batch_size: int,
    ) -> torch.Tensor:
        sink_n, heavy_n, recent_n = self._selection_counts(seq_len, keep_n, policy)
        scores = mean_attention_scores(attention, seq_len, device).expand(batch_size, -1).clone()
        if sink_n:
            scores[:, :sink_n] = -torch.inf
        if recent_n:
            scores[:, -recent_n:] = -torch.inf
        return self._combine_selected(scores, seq_len, sink_n, heavy_n, recent_n, device)

    @staticmethod
    def _selection_counts(
        seq_len: int,
        keep_n: int,
        policy: CompressionPolicy,
    ) -> tuple[int, int, int]:
        sink_n = min(max(policy.sink_tokens, 0), keep_n, seq_len)
        selectable_n = keep_n - sink_n
        heavy_ratio = min(max(policy.heavy_ratio, 0.0), 1.0)
        heavy_n = min(int(round(selectable_n * heavy_ratio)), selectable_n)
        recent_n = min(selectable_n - heavy_n, policy.recent_window, seq_len - sink_n)
        heavy_n = selectable_n - recent_n
        return sink_n, heavy_n, recent_n

    def _combine_selected(
        self,
        scores: torch.Tensor,
        seq_len: int,
        sink_n: int,
        heavy_n: int,
        recent_n: int,
        device,
    ) -> torch.Tensor:
        per_batch = []
        for batch_idx in range(scores.shape[0]):
            parts = []
            if sink_n > 0:
                parts.append(torch.arange(sink_n, device=device, dtype=torch.long))
            if heavy_n > 0:
                top_idx = torch.topk(scores[batch_idx], k=heavy_n).indices
                parts.append(top_idx)
            if recent_n > 0:
                recent = torch.arange(seq_len - recent_n, seq_len, device=device, dtype=torch.long)
                parts.append(recent)
            indices = ensure_sorted_unique(torch.cat(parts))
            target_n = sink_n + heavy_n + recent_n
            if indices.numel() < target_n:
                missing = target_n - indices.numel()
                filler = torch.arange(seq_len, device=device, dtype=torch.long)
                mask = ~torch.isin(filler, indices)
                indices = ensure_sorted_unique(torch.cat([indices, filler[mask][:missing]]))
            per_batch.append(indices[:target_n])
        return stack_batch_indices(per_batch, device)
