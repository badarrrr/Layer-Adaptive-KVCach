from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .cache_utils import LegacyCache, cache_memory_bytes, to_legacy_cache
from .compression import CompressionPolicy, LayerAdaptiveCompressor
from .scheduler import LayerProfile


@dataclass(frozen=True)
class PerplexityMetrics:
    perplexity: float
    token_count: int
    elapsed_seconds: float
    tokens_per_second: float
    avg_cache_bytes_before: float
    avg_cache_bytes_after: float
    peak_cache_bytes_before: int
    peak_cache_bytes_after: int
    actual_cache_compression: float


@torch.no_grad()
def incremental_perplexity(
    model,
    tokenizer,
    texts: Iterable[str],
    compressor: LayerAdaptiveCompressor | None = None,
    policies: list[CompressionPolicy] | None = None,
    max_length: int = 512,
    target_compression_ratio: float | None = None,
) -> float:
    """Teacher-forced autoregressive PPL with optional KV compression after each step."""
    return evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        compressor=compressor,
        policies=policies,
        max_length=max_length,
        target_compression_ratio=target_compression_ratio,
    ).perplexity


@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    texts: Iterable[str],
    compressor: LayerAdaptiveCompressor | None = None,
    policies: list[CompressionPolicy] | None = None,
    max_length: int = 512,
    description: str = "perplexity",
    target_compression_ratio: float | None = None,
) -> PerplexityMetrics:
    """Evaluate teacher-forced PPL and KV-cache memory on a fixed text collection."""
    total_loss = 0.0
    token_count = 0
    before_bytes = 0
    after_bytes = 0
    peak_before_bytes = 0
    peak_after_bytes = 0
    cache_bytes_per_token: float | None = None
    device = next(model.parameters()).device
    start = time.perf_counter()
    for text in tqdm(list(texts), desc=description):
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
        ids = ids.to(device)
        if ids.shape[1] < 2:
            continue

        cache: LegacyCache | None = None
        for pos in range(ids.shape[1] - 1):
            position_ids = torch.tensor([[pos]], device=device, dtype=torch.long)
            outputs = model(
                input_ids=ids[:, pos : pos + 1],
                position_ids=position_ids,
                cache_position=torch.tensor([pos], device=device, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                output_attentions=compressor is not None,
            )
            logits = outputs.logits[:, -1, :]
            target = ids[:, pos + 1]
            total_loss += float(F.cross_entropy(logits, target, reduction="sum").item())
            token_count += int(target.numel())
            cache = to_legacy_cache(outputs.past_key_values)
            current_cache_bytes = cache_memory_bytes(cache)
            if cache_bytes_per_token is None:
                cache_bytes_per_token = current_cache_bytes / max(pos + 1, 1)
            reference_cache_bytes = int(cache_bytes_per_token * (pos + 1))
            before_bytes += reference_cache_bytes
            peak_before_bytes = max(peak_before_bytes, reference_cache_bytes)
            if compressor is not None and policies is not None:
                cache = compressor.compress_cache(
                    cache,
                    outputs.attentions,
                    policies,
                    sequence_length=pos + 1,
                    target_compression_ratio=target_compression_ratio,
                )
            compressed_cache_bytes = cache_memory_bytes(cache)
            after_bytes += compressed_cache_bytes
            peak_after_bytes = max(peak_after_bytes, compressed_cache_bytes)

    elapsed_seconds = time.perf_counter() - start
    if not token_count:
        raise ValueError("No valid samples were available for perplexity evaluation.")
    perplexity = math.exp(total_loss / token_count)
    return PerplexityMetrics(
        perplexity=perplexity,
        token_count=token_count,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=token_count / max(elapsed_seconds, 1e-8),
        avg_cache_bytes_before=before_bytes / token_count,
        avg_cache_bytes_after=after_bytes / token_count,
        peak_cache_bytes_before=peak_before_bytes,
        peak_cache_bytes_after=peak_after_bytes,
        actual_cache_compression=1.0 - (after_bytes / max(before_bytes, 1)),
    )


@torch.no_grad()
def attention_overlap_redundancy(model, tokenizer, texts: Iterable[str], max_length: int = 512) -> list[float]:
    """Estimate per-layer redundancy by top-token overlap across attention heads."""
    num_layers = model.config.num_hidden_layers
    totals = [0.0 for _ in range(num_layers)]
    counts = [0 for _ in range(num_layers)]
    device = next(model.parameters()).device

    for text in tqdm(list(texts), desc="redundancy"):
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
        if ids.shape[1] < 4:
            continue
        outputs = model(input_ids=ids, output_attentions=True, use_cache=False)
        for layer_idx, attn in enumerate(outputs.attentions):
            head_scores = attn[0].mean(dim=1)
            k = max(1, int(0.1 * head_scores.shape[-1]))
            top_sets = [set(torch.topk(head_scores[h], k=k).indices.tolist()) for h in range(head_scores.shape[0])]
            if len(top_sets) < 2:
                continue
            pair_scores = []
            for i in range(len(top_sets)):
                for j in range(i + 1, len(top_sets)):
                    union = top_sets[i] | top_sets[j]
                    pair_scores.append(len(top_sets[i] & top_sets[j]) / max(len(union), 1))
            totals[layer_idx] += sum(pair_scores) / len(pair_scores)
            counts[layer_idx] += 1

    return [totals[i] / counts[i] if counts[i] else 0.0 for i in range(num_layers)]


def build_profiles_from_sensitivity(
    baseline_ppl: float,
    layer_ppls: dict[int, dict[float, float]],
    redundancies: list[float],
    max_relative_ppl_increase: float = 0.05,
) -> list[LayerProfile]:
    tested_ratios = sorted({ratio for curve in layer_ppls.values() for ratio in curve})
    layer_stats: dict[int, tuple[float, float]] = {}
    for layer_idx, curve in sorted(layer_ppls.items()):
        safe = 0.0
        rels = []
        for ratio, ppl in sorted(curve.items()):
            rel = (ppl - baseline_ppl) / max(baseline_ppl, 1e-8)
            rels.append(rel)
            if rel <= max_relative_ppl_increase:
                safe = max(safe, ratio)
        layer_stats[layer_idx] = (safe, max(rels) if rels else 0.0)

    max_steepness = max((steepness for _, steepness in layer_stats.values()), default=0.0)
    use_redundancy_fallback = max_steepness <= 1e-8 and bool(redundancies) and bool(tested_ratios)
    redundancy_floor = _percentile(redundancies, 0.10) if redundancies else 0.0
    redundancy_ceiling = _percentile(redundancies, 0.90) if redundancies else 0.0
    num_profiled_layers = max(layer_stats.keys(), default=-1) + 1

    profiles: list[LayerProfile] = []
    for layer_idx, curve in sorted(layer_ppls.items()):
        safe, steepness = layer_stats[layer_idx]
        redundancy = redundancies[layer_idx] if layer_idx < len(redundancies) else 0.0
        if use_redundancy_fallback:
            clipped_redundancy = min(max(redundancy, redundancy_floor), redundancy_ceiling)
            redundancy_rank = (clipped_redundancy - redundancy_floor) / max(
                redundancy_ceiling - redundancy_floor,
                1e-8,
            )
            depth_rank = layer_idx / max(num_profiled_layers - 1, 1)
            compressibility = 0.75 * redundancy_rank + 0.25 * depth_rank
            importance = 1.0 - compressibility
            safe = tested_ratios[0] + compressibility * (tested_ratios[-1] - tested_ratios[0])
        else:
            importance = steepness / max(max_steepness, 1e-8) if max_steepness > 0 else 0.0
            importance = min(max(importance, 0.0), 1.0)
        profiles.append(
            LayerProfile(
                layer_idx=layer_idx,
                importance=importance,
                redundancy=redundancy,
                safe_compression=safe,
            )
        )
    return profiles


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    q = min(max(q, 0.0), 1.0)
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def kv_memory_mb(cache: LegacyCache) -> float:
    return cache_memory_bytes(cache) / (1024**2)
