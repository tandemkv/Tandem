import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/experiments/kv_translation'))
from mapper import align, translate
from semantic import score_answers


class KVTranslationTests(unittest.TestCase):
    def test_alignment_uses_shared_ends_without_lookahead(self):
        source = [(0, 2), (2, 4), (4, 5), (5, 8)]
        target = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 8)]
        self.assertEqual(align(source, target), [-1, 0, 0, 2, 2, 3])
        self.assertEqual(align(source[:-1], target[:-1]), [-1, 0, 0, 2, 2])

    def test_affine_map_uses_selected_layers_and_mean_before_alignment(self):
        normalization = {'source_mean': torch.zeros(2, 3, 1), 'source_std': torch.ones(2, 3, 1),
                         'target_mean': torch.full((2, 1, 1), 5.), 'target_std': torch.full((2, 1, 1), 2.)}
        state = {'layers': torch.tensor([[0, 1, 2]]),
                 'weights': torch.tensor([[[[1.], [2.], [3.], [4.]]],
                                          [[[3.], [2.], [1.], [0.]]]])}
        source = torch.tensor([[[[1.], [2.], [3.]], [[1.], [2.], [3.]]]])
        result = translate(source, [-1, 0, 0], state, normalization, source_heads=1)
        self.assertEqual(result.flatten().tolist(), [5., 5., 41., 25., 41., 25.])
        with self.assertRaises(ValueError):
            translate(source, [1], state, normalization, source_heads=1)

    def test_semantic_agreement_requires_both_directions(self):
        row = {'id': '1', 'question': 'How many?', 'reference': '3',
               'native': {'answer': 'three'}, 'distilled': {'answer': '3'}}
        result = score_answers([row], 'distilled', score=lambda _: iter([0.8, 0.4, 0.9, 0.8]))
        self.assertEqual(result['agreement'], 0)
        self.assertEqual(result['handoff_accuracy'], 1)
        self.assertEqual(result['preserved'], 1)


if __name__ == '__main__':
    unittest.main()
