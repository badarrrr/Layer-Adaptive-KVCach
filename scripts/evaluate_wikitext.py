from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

from lakvc.compression import CompressionPolicy, LayerAdaptiveCompressor
from lakvc.config import load_model_config
from lakvc.data import load_text_samples
from lakvc.modeling import load_model_and_tokenizer
from lakvc.profiling import PerplexityMetrics, evaluate_perplexity
from lakvc.scheduler import RuntimeScheduler


METHOD_LABELS = {
    "baseline": "No compression",
    "uniform_recent": "Uniform recent",
    "uniform_heavy_hitter": "Uniform heavy-hitter",
    "layer_adaptive": "Layer-adaptive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KV-cache compression methods on the WikiText test split."
    )
    parser.add_argument("--model", choices=["llama2_7b", "mistral_7b"], required=True)
    parser.add_argument("--model-config", default="configs/models.json")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--data-file")
    parser.add_argument("--local-text-file")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--candidate-multiplier", type=int, default=100)
    parser.add_argument("--seed", type=int, default=6520)
    parser.add_argument("--min-sample-tokens", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-layer-compression", type=float, default=0.5)
    parser.add_argument("--output-dir", default="outputs/wikitext_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_model_config(args.model_config, args.model)
    model, tokenizer = load_model_and_tokenizer(config)
    candidates = load_text_samples(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        limit=args.samples * args.candidate_multiplier,
        local_text_file=args.local_text_file,
        data_file=args.data_file,
    )
    random.Random(args.seed).shuffle(candidates)
    samples = _select_samples(
        tokenizer,
        candidates,
        limit=args.samples,
        min_tokens=args.min_sample_tokens,
        max_length=args.max_length,
    )
    if len(samples) < args.samples:
        raise ValueError(
            f"Only found {len(samples)} qualifying samples; requested {args.samples}. "
            "Increase --candidate-multiplier or lower --min-sample-tokens."
        )

    scheduler = RuntimeScheduler.from_json(
        args.profile,
        num_layers=model.config.num_hidden_layers,
        recent_window=args.recent_window,
        min_tokens=args.min_tokens,
    )
    compressor = LayerAdaptiveCompressor(
        min_tokens=args.min_tokens,
        recent_window=args.recent_window,
    )

    results = []
    baseline = evaluate_perplexity(
        model,
        tokenizer,
        samples,
        max_length=args.max_length,
        description="baseline",
    )
    results.append(_result_row("baseline", 0.0, baseline, baseline.perplexity))

    for ratio in args.ratios:
        methods = {
            "uniform_recent": [
                CompressionPolicy(ratio, "recent", args.recent_window, min_tokens=args.min_tokens)
                for _ in range(model.config.num_hidden_layers)
            ],
            "uniform_heavy_hitter": [
                CompressionPolicy(
                    ratio,
                    "heavy_hitter",
                    args.recent_window,
                    min_tokens=args.min_tokens,
                )
                for _ in range(model.config.num_hidden_layers)
            ],
            "layer_adaptive": _cap_policies(
                scheduler.allocate(ratio),
                args.max_layer_compression,
            ),
        }
        for method, policies in methods.items():
            metrics = evaluate_perplexity(
                model,
                tokenizer,
                samples,
                compressor=compressor,
                policies=policies,
                max_length=args.max_length,
                description=f"{method}@{ratio:.2f}",
            )
            row = _result_row(method, ratio, metrics, baseline.perplexity)
            row["mean_policy_compression"] = sum(
                policy.compression_ratio for policy in policies
            ) / len(policies)
            results.append(row)

    metadata = {
        "model": args.model,
        "profile": args.profile,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "data_file": args.data_file,
        "samples": len(samples),
        "seed": args.seed,
        "min_sample_tokens": args.min_sample_tokens,
        "max_length": args.max_length,
        "ratios": args.ratios,
        "recent_window": args.recent_window,
        "min_tokens": args.min_tokens,
        "note": (
            "Perplexity is measured on a fixed, seeded WikiText-2 test subset with "
            "context reset for each passage. The current Transformers-compatible "
            "implementation keeps a uniform KV-cache length across layers while using "
            "layer-specific token-selection policies."
        ),
    }
    (output_dir / "results.json").write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evaluated_samples.json").write_text(
        json.dumps(samples, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "results.csv", results)
    _write_markdown(output_dir / "results_table.md", metadata, results)
    _make_plots(output_dir, results, args.profile)
    print(f"Saved evaluation artifacts to {output_dir}")


def _select_samples(tokenizer, candidates, limit: int, min_tokens: int, max_length: int):
    selected = []
    for text in candidates:
        token_count = len(tokenizer(text, truncation=True, max_length=max_length).input_ids)
        if token_count >= min_tokens:
            selected.append(text)
        if len(selected) >= limit:
            break
    return selected


def _cap_policies(policies, max_layer_compression: float):
    cap = min(max(max_layer_compression, 0.0), 0.95)
    return [
        CompressionPolicy(
            compression_ratio=min(policy.compression_ratio, cap),
            strategy=policy.strategy,
            recent_window=policy.recent_window,
            heavy_ratio=policy.heavy_ratio,
            min_tokens=policy.min_tokens,
        )
        for policy in policies
    ]


def _result_row(
    method: str,
    requested_compression: float,
    metrics: PerplexityMetrics,
    baseline_ppl: float,
):
    row = asdict(metrics)
    row.update(
        {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "requested_compression": requested_compression,
            "relative_ppl_increase": metrics.perplexity / baseline_ppl - 1.0,
            "avg_cache_mb_before": metrics.avg_cache_bytes_before / (1024**2),
            "avg_cache_mb_after": metrics.avg_cache_bytes_after / (1024**2),
            "peak_cache_mb_before": metrics.peak_cache_bytes_before / (1024**2),
            "peak_cache_mb_after": metrics.peak_cache_bytes_after / (1024**2),
            "mean_policy_compression": 0.0,
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, metadata: dict, rows: list[dict]) -> None:
    lines = [
        "# WikiText-2 Test Results",
        "",
        f"- Model: `{metadata['model']}`",
        f"- Test samples: `{metadata['samples']}`",
        f"- Sampling seed: `{metadata['seed']}`",
        f"- Maximum tokens per sample: `{metadata['max_length']}`",
        "",
        "| Method | Requested | Actual | PPL | Relative PPL | Avg cache (MB) | Peak cache (MB) | Tokens/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_label']} | {row['requested_compression']:.0%} | "
            f"{row['actual_cache_compression']:.1%} | {row['perplexity']:.3f} | "
            f"{row['relative_ppl_increase']:.2%} | {row['avg_cache_mb_after']:.2f} | "
            f"{row['peak_cache_mb_after']:.2f} | {row['tokens_per_second']:.2f} |"
        )
    lines.extend(["", f"Note: {metadata['note']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_plots(output_dir: Path, rows: list[dict], profile_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "uniform_recent": "#4C78A8",
        "uniform_heavy_hitter": "#F58518",
        "layer_adaptive": "#54A24B",
    }
    compressed_rows = [row for row in rows if row["method"] != "baseline"]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, color in colors.items():
        subset = sorted(
            (row for row in compressed_rows if row["method"] == method),
            key=lambda row: row["actual_cache_compression"],
        )
        ax.plot(
            [row["actual_cache_compression"] * 100 for row in subset],
            [row["relative_ppl_increase"] * 100 for row in subset],
            marker="o",
            linewidth=2,
            label=METHOD_LABELS[method],
            color=color,
        )
    ax.set_xlabel("Actual KV-cache compression (%)")
    ax.set_ylabel("Relative perplexity increase (%)")
    ax.set_title("Quality-Memory Trade-off on WikiText-2 Test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "quality_memory_tradeoff.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, color in colors.items():
        subset = sorted(
            (row for row in compressed_rows if row["method"] == method),
            key=lambda row: row["requested_compression"],
        )
        ax.plot(
            [row["requested_compression"] * 100 for row in subset],
            [row["avg_cache_mb_after"] for row in subset],
            marker="o",
            linewidth=2,
            label=METHOD_LABELS[method],
            color=color,
        )
    baseline = next(row for row in rows if row["method"] == "baseline")
    ax.axhline(
        baseline["avg_cache_mb_after"],
        color="#777777",
        linestyle="--",
        label="No compression",
    )
    ax.set_xlabel("Requested global compression (%)")
    ax.set_ylabel("Average KV-cache size (MB)")
    ax.set_title("KV-Cache Memory Reduction on WikiText-2 Test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cache_memory_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, color in colors.items():
        subset = sorted(
            (row for row in compressed_rows if row["method"] == method),
            key=lambda row: row["actual_cache_compression"],
        )
        ax.plot(
            [row["actual_cache_compression"] * 100 for row in subset],
            [row["tokens_per_second"] for row in subset],
            marker="o",
            linewidth=2,
            label=METHOD_LABELS[method],
            color=color,
        )
    baseline = next(row for row in rows if row["method"] == "baseline")
    ax.axhline(
        baseline["tokens_per_second"],
        color="#777777",
        linestyle="--",
        label="No compression",
    )
    ax.set_xlabel("Actual KV-cache compression (%)")
    ax.set_ylabel("Teacher-forced tokens per second")
    ax.set_title("Evaluation Throughput on WikiText-2 Test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "throughput_comparison.png", dpi=300)
    plt.close(fig)

    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))["layers"]
    layers = [item["layer_idx"] for item in profile]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(layers, [item["importance"] for item in profile], label="Importance", linewidth=2)
    ax.plot(layers, [item["redundancy"] for item in profile], label="Redundancy", linewidth=2)
    ax.plot(
        layers,
        [item["safe_compression"] for item in profile],
        label="Safe compression",
        linewidth=2,
    )
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Normalized score")
    ax.set_title("Offline Layer Profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "layer_profile.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
