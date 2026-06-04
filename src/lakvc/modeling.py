from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_tokenizer(config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=resolve_dtype(config.torch_dtype),
        device_map=config.device_map,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer
