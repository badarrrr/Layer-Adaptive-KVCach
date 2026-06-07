from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .compression import CompressionPolicy


@dataclass(frozen=True)
class LayerProfile:
    layer_idx: int
    importance: float
    redundancy: float
    safe_compression: float


class RuntimeScheduler:
    """Translate a global compression budget into layer-specific policies."""

    def __init__(
        self,
        profiles: list[LayerProfile],
        num_layers: int,
        recent_window: int = 32,
        min_tokens: int = 16,
        sink_tokens: int = 4,
    ):
        self.num_layers = num_layers
        self.recent_window = recent_window
        self.min_tokens = min_tokens
        self.sink_tokens = sink_tokens
        by_idx = {p.layer_idx: p for p in profiles}
        self.profiles = [
            by_idx.get(i, self._default_profile(i, num_layers)) for i in range(num_layers)
        ]

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        num_layers: int,
        recent_window: int = 32,
        min_tokens: int = 16,
        sink_tokens: int = 4,
    ) -> "RuntimeScheduler":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        profiles = [LayerProfile(**item) for item in raw["layers"]]
        return cls(
            profiles,
            num_layers,
            recent_window=recent_window,
            min_tokens=min_tokens,
            sink_tokens=sink_tokens,
        )

    def allocate(self, global_compression: float) -> list[CompressionPolicy]:
        budget = min(max(global_compression, 0.0), 0.95) * self.num_layers
        weights = [self._compressibility(p) for p in self.profiles]
        ratios = [0.0 for _ in self.profiles]

        remaining_budget = budget
        active = set(range(self.num_layers))
        while active and remaining_budget > 1e-8:
            total_weight = sum(weights[i] for i in active)
            if total_weight <= 0:
                break
            consumed = 0.0
            saturated = []
            for i in active:
                proposed = remaining_budget * (weights[i] / total_weight)
                room = self.profiles[i].safe_compression - ratios[i]
                delta = min(proposed, max(room, 0.0))
                ratios[i] += delta
                consumed += delta
                if ratios[i] >= self.profiles[i].safe_compression - 1e-8:
                    saturated.append(i)
            remaining_budget -= consumed
            for i in saturated:
                active.remove(i)
            if consumed <= 1e-8:
                break

        return [
            CompressionPolicy(
                compression_ratio=ratios[i],
                strategy=self._strategy_for_layer(i),
                recent_window=self.recent_window,
                heavy_ratio=self._heavy_ratio_for_layer(i),
                min_tokens=self.min_tokens,
                sink_tokens=self.sink_tokens,
            )
            for i in range(self.num_layers)
        ]

    @staticmethod
    def save_profiles(path: str | Path, model_name: str, profiles: list[LayerProfile]) -> None:
        payload = {
            "model_name": model_name,
            "layers": [asdict(profile) for profile in profiles],
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _default_profile(layer_idx: int, num_layers: int) -> LayerProfile:
        depth = layer_idx / max(num_layers - 1, 1)
        return LayerProfile(
            layer_idx=layer_idx,
            importance=1.0 - 0.5 * depth,
            redundancy=0.2 + 0.7 * depth,
            safe_compression=0.15 + 0.55 * depth,
        )

    @staticmethod
    def _compressibility(profile: LayerProfile) -> float:
        return max(0.01, (1.0 - profile.importance) + profile.redundancy)

    def _strategy_for_layer(self, layer_idx: int) -> str:
        depth = layer_idx / max(self.num_layers - 1, 1)
        if depth < 0.33:
            return "heavy_hitter"
        if depth < 0.66:
            return "hybrid"
        return "snapkv"

    def _heavy_ratio_for_layer(self, layer_idx: int) -> float:
        depth = layer_idx / max(self.num_layers - 1, 1)
        return 0.35 + 0.40 * depth
