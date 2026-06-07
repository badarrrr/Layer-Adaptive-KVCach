from __future__ import annotations

import argparse

from lakvc.compression import CompressionPolicy, LayerAdaptiveCompressor
from lakvc.config import load_model_config
from lakvc.data import load_text_samples
from lakvc.modeling import load_model_and_tokenizer
from lakvc.profiling import (
    attention_overlap_redundancy,
    build_profiles_from_sensitivity,
    incremental_perplexity,
)
from lakvc.scheduler import RuntimeScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline layer profiling for adaptive KV compression.")
    parser.add_argument("--model", choices=["llama2_7b", "mistral_7b"], required=True)
    parser.add_argument("--model-config", default="configs/models.json")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--local-text-file")
    parser.add_argument(
        "--data-file",
        help="Local parquet file to load explicitly; registered under --split.",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--max-relative-ppl-increase", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_model_config(args.model_config, args.model)
    model, tokenizer = load_model_and_tokenizer(config)
    samples = load_text_samples(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        limit=args.samples,
        local_text_file=args.local_text_file,
        data_file=args.data_file,
    )

    baseline = incremental_perplexity(model, tokenizer, samples, max_length=args.max_length)
    redundancies = attention_overlap_redundancy(model, tokenizer, samples, max_length=args.max_length)
    compressor = LayerAdaptiveCompressor(
        min_tokens=args.min_tokens,
        recent_window=args.recent_window,
    )
    num_layers = model.config.num_hidden_layers
    layer_ppls: dict[int, dict[float, float]] = {}
    recent_baselines: dict[float, float] = {}

    # Transformers 4.41 uses one causal mask across all decoder layers, so the
    # compressed KV length has to be uniform. Measure the uniform recent-window
    # pruning cost once per ratio, then attribute only the target layer's extra
    # policy cost to the sensitivity curve.
    for ratio in args.ratios:
        policies = [
            CompressionPolicy(
                ratio,
                "recent",
                recent_window=args.recent_window,
                min_tokens=args.min_tokens,
                sink_tokens=args.sink_tokens,
            )
            for _ in range(num_layers)
        ]
        recent_baselines[ratio] = incremental_perplexity(
            model,
            tokenizer,
            samples,
            compressor=compressor,
            policies=policies,
            max_length=args.max_length,
            target_compression_ratio=ratio,
        )

    for layer_idx in range(num_layers):
        layer_ppls[layer_idx] = {}
        for ratio in args.ratios:
            policies = [
                CompressionPolicy(
                    ratio,
                    "recent",
                    recent_window=args.recent_window,
                    min_tokens=args.min_tokens,
                    sink_tokens=args.sink_tokens,
                )
                for _ in range(num_layers)
            ]
            policies[layer_idx] = CompressionPolicy(
                ratio,
                "heavy_hitter",
                recent_window=args.recent_window,
                min_tokens=args.min_tokens,
                sink_tokens=args.sink_tokens,
            )
            ppl = incremental_perplexity(
                model,
                tokenizer,
                samples,
                compressor=compressor,
                policies=policies,
                max_length=args.max_length,
                target_compression_ratio=ratio,
            )
            extra_policy_cost = max(0.0, ppl - recent_baselines[ratio])
            layer_ppls[layer_idx][ratio] = baseline + extra_policy_cost

    profiles = build_profiles_from_sensitivity(
        baseline_ppl=baseline,
        layer_ppls=layer_ppls,
        redundancies=redundancies,
        max_relative_ppl_increase=args.max_relative_ppl_increase,
    )
    RuntimeScheduler.save_profiles(args.output, args.model, profiles)


if __name__ == "__main__":
    main()
