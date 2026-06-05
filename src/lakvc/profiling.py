from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .cache_utils import LegacyCache, cache_memory_bytes, to_legacy_cache
from .compression import CompressionPolicy, LayerAdaptiveCompressor
from .scheduler import LayerProfile


@torch.no_grad()
def incremental_perplexity(
    model,
    tokenizer,
    texts: Iterable[str],
    compressor: LayerAdaptiveCompressor | None = None,
    policies: list[CompressionPolicy] | None = None,
    max_length: int = 512,
) -> float:
    """Teacher-forced autoregressive PPL with optional KV compression after each step."""
    losses: list[float] = []
    device = next(model.parameters()).device
    for text in tqdm(list(texts), desc="perplexity"):
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
        ids = ids.to(device)
        if ids.shape[1] < 2:
            continue

        # cache: LegacyCache | None = None
        # for pos in range(ids.shape[1] - 1):
        #     position_ids = torch.tensor([[pos]], device=device, dtype=torch.long)
        #     outputs = model(
        #         input_ids=ids[:, pos : pos + 1],
        #         position_ids=position_ids,
        #         past_key_values=cache,
        #         use_cache=True,
        #         output_attentions=compressor is not None,
        #     )
        #     logits = outputs.logits[:, -1, :]
        #     target = ids[:, pos + 1]
        #     losses.append(float(F.cross_entropy(logits, target, reduction="mean").item()))
        #     cache = to_legacy_cache(outputs.past_key_values)
        #     if compressor is not None and policies is not None:
        #         cache = compressor.compress_cache(cache, outputs.attentions, policies)
        
        cache: LegacyCache | None = None
        for pos in range(ids.shape[1] - 1):
            past_len = cache[0][0].shape[-2] if cache is not None else 0
            position_ids = torch.tensor([[past_len]], device=device, dtype=torch.long)
            outputs = model(
                input_ids=ids[:, pos : pos + 1],
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                output_attentions=compressor is not None,
            )
            logits = outputs.logits[:, -1, :]
            target = ids[:, pos + 1]
            losses.append(float(F.cross_entropy(logits, target, reduction="mean").item()))
            cache = to_legacy_cache(outputs.past_key_values)
            if compressor is not None and policies is not None:
                cache = compressor.compress_cache(cache, outputs.attentions, policies)

    if not losses:
        raise ValueError("No valid samples were available for perplexity evaluation.")
    return math.exp(sum(losses) / len(losses))


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
    profiles: list[LayerProfile] = []
    for layer_idx, curve in sorted(layer_ppls.items()):
        safe = 0.0
        for ratio, ppl in sorted(curve.items()):
            rel = (ppl - baseline_ppl) / max(baseline_ppl, 1e-8)
            if rel <= max_relative_ppl_increase:
                safe = max(safe, ratio)
        steepness = max((ppl - baseline_ppl) / max(baseline_ppl, 1e-8) for ppl in curve.values())
        importance = min(max(steepness, 0.0), 1.0)
        redundancy = redundancies[layer_idx] if layer_idx < len(redundancies) else 0.0
        profiles.append(
            LayerProfile(
                layer_idx=layer_idx,
                importance=importance,
                redundancy=redundancy,
                safe_compression=safe,
            )
        )
    return profiles


def kv_memory_mb(cache: LegacyCache) -> float:
    return cache_memory_bytes(cache) / (1024**2)
