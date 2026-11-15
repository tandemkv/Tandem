"""Verify the opt-in demo wiring without loading model weights."""
import unittest
from unittest.mock import Mock, patch

from src.experiments.poc.demo_setup import TASK_MODELS, build_demo_runner


class DemoSetupTests(unittest.TestCase):
    @patch('src.adapters.nvidia_classifier.PromptClassifier')
    @patch('src.models.model.Model')
    def test_factory_supplies_demo_models_classifier_policy_and_mapper(self, model_class, classifier_class):
        model_class.side_effect = lambda **kwargs: Mock(model_name=kwargs['model_name'])
        mapper = Mock()
        runner = build_demo_runner(kv_cache_mapper=mapper, mode='complexity', complexity_threshold=0.7)
        self.assertEqual(runner.model_names, {
            'model_a': 'Qwen/Qwen2.5-0.5B-Instruct',
            'model_b': 'HuggingFaceTB/SmolLM2-360M-Instruct',
        })
        self.assertIs(runner.classifier_model, classifier_class.return_value)
        self.assertIs(runner.kv_cache_mapper, mapper)
        self.assertEqual(runner.task_models, TASK_MODELS)
        self.assertEqual(runner.mode, 'complexity')
        self.assertEqual(runner.complexity_threshold, 0.7)
        classifier_class.assert_called_once_with()
