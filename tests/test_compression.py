from __future__ import annotations

import unittest

import torch

from lakvc.compression import CompressionPolicy, LayerAdaptiveCompressor


def make_cache(layers: int = 2, length: int = 100):
    layer = (
        torch.arange(length, dtype=torch.float32).reshape(1, 1, length, 1),
        torch.arange(length, dtype=torch.float32).reshape(1, 1, length, 1),
    )
    return tuple((key.clone(), value.clone()) for key, value in [layer] * layers)


class LayerAdaptiveCompressorTest(unittest.TestCase):
    def test_global_target_controls_uniform_physical_length(self):
        cache = make_cache()
        policies = [
            CompressionPolicy(0.0, "recent", min_tokens=1),
            CompressionPolicy(0.5, "heavy_hitter", min_tokens=1),
        ]

        compressed = LayerAdaptiveCompressor().compress_cache(
            cache,
            attentions=None,
            policies=policies,
            sequence_length=100,
            target_compression_ratio=0.2,
        )

        self.assertEqual([key.shape[-2] for key, _ in compressed], [80, 80])

    def test_recent_policy_preserves_sink_and_latest_tokens(self):
        policy = CompressionPolicy(
            0.5,
            "recent",
            min_tokens=1,
            sink_tokens=2,
        )

        indices = LayerAdaptiveCompressor().select_keep_indices(
            seq_len=10,
            policy=policy,
            attention=None,
            device=torch.device("cpu"),
            batch_size=1,
            keep_n=5,
        )

        self.assertEqual(indices.tolist(), [[0, 1, 7, 8, 9]])

    def test_policy_count_must_match_cache_layers(self):
        with self.assertRaisesRegex(ValueError, "one compression policy per cache layer"):
            LayerAdaptiveCompressor().compress_cache(
                make_cache(layers=2),
                attentions=None,
                policies=[CompressionPolicy(0.1, "recent")],
                target_compression_ratio=0.1,
            )


if __name__ == "__main__":
    unittest.main()
