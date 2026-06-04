from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from datasets import load_dataset


def load_text_samples(
    dataset_name: str | None,
    dataset_config: str | None,
    split: str,
    text_column: str,
    limit: int,
    local_text_file: str | None = None,
) -> list[str]:
    if local_text_file:
        lines = Path(local_text_file).read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()][:limit]

    if dataset_name is None:
        raise ValueError("Either dataset_name or local_text_file must be provided.")
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    samples = []
    for item in dataset:
        text = str(item[text_column]).strip()
        if text:
            samples.append(text)
        if len(samples) >= limit:
            break
    return samples
