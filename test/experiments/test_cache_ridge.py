"""Check that the three-layer fit uses all three selected layers for both roles."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/experiments'))
import torch
from cache_attention import CacheAttention
from cache_ridge import ThreeLayerRidge


class RidgeTests(unittest.TestCase):
    def test_three_layers_recover_independent_key_and_value_maps(self):
        baseline = CacheAttention(dict(source_layers=12, source_heads=1, target_layers=3,
                                       target_heads=1, head_dim=2, width=4, attention_heads=1))
        ridge = ThreeLayerRidge(baseline)
        expected_layers = [[0, 4, 8], [2, 6, 10], [3, 7, 11]]
        self.assertEqual(ridge.layers.tolist(), expected_layers)
        gen = torch.Generator().manual_seed(91)

        def pair(count):
            source = torch.randn(count, 2, 12, 2, generator=gen)
            target = torch.stack([
                source[:, :, first] + 2 * source[:, :, middle] - source[:, :, last]
                for first, middle, last in expected_layers
            ], dim=2)
            target[:, 0] += 2
            target[:, 1] *= -2
            return source, target

        train = [pair(20), pair(40)]
        data = SimpleNamespace(train=train, pair=lambda sample, stride: tuple(t[::stride] for t in sample))
        self.assertEqual(ridge.fit(data, stride=2, ridge=1e-6), 30)
        source, target = pair(10)
        torch.testing.assert_close(ridge.predict(source, True), target, atol=1e-4, rtol=1e-4)
