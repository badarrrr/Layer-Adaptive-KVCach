from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    torch_dtype: str = "float16"
    device_map: str = "auto"
    requires_hf_auth: bool = False


def load_model_config(config_path: str | Path, model_name: str) -> ModelConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    if model_name not in raw:
        available = ", ".join(sorted(raw))
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available}")
    item = raw[model_name]
    return ModelConfig(name=model_name, **item)
