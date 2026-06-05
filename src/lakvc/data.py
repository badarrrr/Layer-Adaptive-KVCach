from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from datasets import load_dataset


# def load_text_samples(
#     dataset_name: str | None,
#     dataset_config: str | None,
#     split: str,
#     text_column: str,
#     limit: int,
#     local_text_file: str | None = None,
# ) -> list[str]:
#     if local_text_file:
#         lines = Path(local_text_file).read_text(encoding="utf-8").splitlines()
#         return [line.strip() for line in lines if line.strip()][:limit]

#     if dataset_name is None:
#         raise ValueError("Either dataset_name or local_text_file must be provided.")
#     # dataset = load_dataset(dataset_name, dataset_config, split=split)
#     dataset = load_dataset(
#         "./datasets/wikitext",
#         "wikitext-2-raw-v1",
#         split=split
#     )
#     samples = []
#     for item in dataset:
#         text = str(item[text_column]).strip()
#         if text:
#             samples.append(text)
#         if len(samples) >= limit:
#             break
#     return samples


def load_text_samples(
    dataset_name,
    dataset_config,
    split,
    text_column,
    limit,
    local_text_file=None
):
    if local_text_file:
        lines = Path(local_text_file).read_text(encoding="utf-8").splitlines()
        return [x.strip() for x in lines if x.strip()][:limit]

    # 🚀 强制本地 parquet（完全不走 HF builder）
    data_path = f"./datasets/wikitext/wikitext-2-raw-v1/{split}-00000-of-00001.parquet"

    dataset = load_dataset(
        "parquet",
        data_files=data_path,
        split="train"
    )

    samples = []
    for item in dataset:
        text = str(item[text_column]).strip()
        if text:
            samples.append(text)
        if len(samples) >= limit:
            break

    return samples