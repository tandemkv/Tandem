"""Check published score semantics with synthetic logits; no model downloads."""
import unittest

import torch
from src.routing.classifier import decode


class ClassifierScoringTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "task_type_map": {"0": "Summarization", "1": "Code Generation"},
            "weights_map": {
                "creativity_scope": [2, 1, 0], "reasoning": [0, 1],
                "contextual_knowledge": [0, 1], "number_of_few_shots": [0, 1, 2, 3, 4, 5],
                "domain_knowledge": [3, 1, 2, 0], "constraint_ct": [1, 0],
            },
            "divisor_map": {
                "creativity_scope": 2, "reasoning": 1, "contextual_knowledge": 1,
                "number_of_few_shots": 1, "domain_knowledge": 3, "constraint_ct": 1,
            },
        }

    def logits(self, choose):
        result = {"task_type": torch.tensor([-100., 100.])}
        for name, weights in self.config["weights_map"].items():
            scores = torch.full((len(weights),), -100.)
            scores[weights.index(choose(weights))] = 100.
            result[name] = scores
        return result

    def test_minimum_score_and_task_label(self):
        result = decode(self.logits(min), self.config)
        self.assertEqual(result["complexity"], 0)
        self.assertEqual(result["task"], "Code Generation")
        self.assertEqual(result["task_confidence"], 1)

    def test_maximum_preserves_few_shot_count(self):
        result = decode(self.logits(max), self.config)
        self.assertEqual(result["complexity"], 1.2)
        self.assertEqual(result["dimensions"]["number_of_few_shots"], 5)
        self.assertEqual(result["dimensions"]["domain_knowledge"], 1)

    def test_expected_weighted_score_not_argmax_class(self):
        logits = self.logits(min)
        logits["creativity_scope"] = torch.zeros(3)
        # Expected creativity = mean([2, 1, 0]) / 2 = 0.5.
        self.assertEqual(decode(logits, self.config)["complexity"], 0.175)


if __name__ == "__main__":
    unittest.main()
