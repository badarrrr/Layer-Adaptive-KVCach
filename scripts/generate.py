from __future__ import annotations

import argparse
import json
from pathlib import Path

from lakvc.config import load_model_config
from lakvc.generation import generate_layer_adaptive
from lakvc.modeling import load_model_and_tokenizer
from lakvc.scheduler import RuntimeScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with layer-adaptive KV compression.")
    parser.add_argument("--model", choices=["llama2_7b", "mistral_7b"], required=True)
    parser.add_argument("--model-config", default="configs/models.json")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--global-compression", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--recent-window", type=int, default=128)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--compression-start-tokens", type=int, default=128)
    parser.add_argument("--max-layer-compression", type=float, default=0.30)
    parser.add_argument("--repetition-penalty", type=float, default=1.10)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_model_config(args.model_config, args.model)
    model, tokenizer = load_model_and_tokenizer(config)
    scheduler = RuntimeScheduler.from_json(
        args.profile,
        num_layers=model.config.num_hidden_layers,
        recent_window=args.recent_window,
        min_tokens=args.min_tokens,
    )
    result = generate_layer_adaptive(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        scheduler=scheduler,
        global_compression=args.global_compression,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        compression_start_tokens=args.compression_start_tokens,
        max_layer_compression=args.max_layer_compression,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        format_instruction=not args.raw_prompt,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
