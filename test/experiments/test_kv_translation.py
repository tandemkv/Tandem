import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/experiments/kv_translation'))
from judge import annotate, validate, validation_pairs
from mapper import align, translate
from metrics import compare, entities, summarize
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
        self.assertEqual(translate(source.bfloat16(), [0], state, normalization, 1).dtype, torch.bfloat16)
        with self.assertRaises(ValueError):
            translate(source, [1], state, normalization, source_heads=1)

    def test_semantic_agreement_requires_both_directions(self):
        row = {'id': '1', 'question': 'How many?', 'reference': '3',
               'native': {'answer': 'three'}, 'distilled': {'answer': '3'}}
        result = score_answers([row], 'distilled', score=lambda _: iter([0.8, 0.4, 0.9, 0.8]))
        self.assertEqual(result['agreement'], 0)
        self.assertEqual(result['handoff_accuracy'], 1)
        self.assertEqual(result['preserved'], 1)

    def test_entities_catch_swapped_names_and_numbers(self):
        native = 'Denise Levertov was born earlier than Harvey Pekar. She was born in 1923.'
        self.assertEqual(entities(native), {'denise levertov', 'harvey pekar'})
        self.assertNotEqual(entities('Gary Dumm was born earlier than Debra Babbit.'), entities(native))
        self.assertEqual(entities('The University of Toronto, founded 1,827'), {'university of toronto', '1827'})
        self.assertEqual(entities('Yes, both are American.'), {'yes', 'american'})

    def test_compare_is_zero_for_a_perfect_copy(self):
        answer = {'answer': 'Belfast.', 'tokens': [1, 2]}
        first = torch.tensor([0.9, 0.1]).log()
        row = compare(answer, answer, first, first, 1.5, 1.5, gold='Belfast')
        self.assertEqual(row, {'exact': True, 'top1': True, 'kl': 0.0, 'nll_gap': 0.0,
                               'entities': True, 'gold_f1': 1.0})

    def test_judge_agreement_needs_matching_entities(self):
        rows = [{'question': 'Who?', 'native': {'answer': 'Denise Levertov.'}, 'conditions': {
            'distilled': {'answer': 'Gary Dumm.', 'metrics': {'entities': False}},
            'ridge': {'answer': 'Denise Levertov, a poet.', 'metrics': {'entities': True}}}}]
        annotate(rows, lambda *_: True)
        self.assertFalse(rows[0]['conditions']['distilled']['metrics']['agree'])
        self.assertTrue(rows[0]['conditions']['ridge']['metrics']['agree'])
        rows[0]['gold'] = None
        self.assertEqual(summarize(rows)['ridge']['agree']['mean'], 1.0)

    def test_validation_pairs_swap_entities_and_trim_length(self):
        rows = [{'question': 'Who?', 'native': {'answer': 'He thinks Denise Levertov. She was a poet.'}},
                {'question': 'Where?', 'native': {'answer': 'Belfast, Northern Ireland.'}}]
        pairs = validation_pairs(rows)
        self.assertIn(('Who?', rows[0]['native']['answer'], 'He thinks Denise Levertov', True), pairs)
        self.assertIn(('Who?', rows[0]['native']['answer'],
                       'He thinks Belfast. She was a poet.', False), pairs)
        self.assertEqual(validate(rows, lambda q, a, b: a == b or b in a),
                         {'same_accuracy': 1.0, 'same_pairs': 2, 'different_accuracy': 1.0,
                          'different_pairs': 2, 'misses': []})

    def test_validation_pairs_flip_yes_no_and_skip_question_entities(self):
        rows = [{'question': 'Are both bands American?', 'native': {'answer': 'Yes, both are American.'}},
                {'question': 'Was William Cammisano in a Mafia family?',
                 'native': {'answer': 'William Cammisano was in the Kansas City family.'}},
                {'question': 'Where?', 'native': {'answer': 'Belfast, Northern Ireland.'}}]
        different = [b for _, _, b, same in validation_pairs(rows) if not same]
        # The last row has no pair: its neighbour's only entity, American, is named in that question.
        self.assertEqual(different, ['No, both are American.', 'William Cammisano was in the Belfast family.'])


if __name__ == '__main__':
    unittest.main()
