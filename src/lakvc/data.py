from __future__ import annotations

from pathlib import Path

from datasets import load_dataset


def load_text_samples(
    dataset_name: str | None,
    dataset_config: str | None,
    split: str,
    text_column: str,
    limit: int,
    local_text_file: str | None = None,
    data_file: str | None = None,
) -> list[str]:
    if local_text_file:
        lines = Path(local_text_file).read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()][:limit]

    if data_file:
        dataset = load_dataset("parquet", data_files={split: data_file}, split=split)
    else:
        if dataset_name is None:
            raise ValueError("Provide dataset_name, data_file, or local_text_file.")
        dataset = load_dataset(dataset_name, dataset_config, split=split)

    samples = []
    for item in dataset:
        if text_column not in item:
            raise ValueError(
                f"Text column {text_column!r} is not present. Available columns: {list(item)}"
            )
        text = str(item[text_column]).strip()
        if text:
            samples.append(text)
        if len(samples) >= limit:
            break

    return samples
