import unittest
from unittest.mock import Mock, patch

from src.translation_layer.inference_runner import InferenceRunner, MODEL_NAMES


class RunnerTests(unittest.TestCase):
    @patch('src.translation_layer.inference_runner.Model')
    @patch('src.translation_layer.inference_runner.router.route')
    def test_routing_loading_reuse_and_cache(self, route, model_class):
        route.side_effect = ['smollm', 'qwen', 'smollm']
        small, large = Mock(), Mock()
        model_class.side_effect = [small, large]
        small.run.return_value = 'answer'
        cache = small.get_kv_cache.return_value
        cache.tandem_model_name = MODEL_NAMES['smollm']
        runner = InferenceRunner(mode='complexity', complexity_threshold=0.7)
        model_class.assert_not_called()
        self.assertEqual(runner.receive_prompt('hello'), ('answer', cache))
        route.assert_called_with('hello', mode='complexity', threshold=0.7, task_models=None)
        runner.receive_prompt('code')
        runner.receive_prompt('suffix', kv_cache=cache, is_suffix=True)
        self.assertEqual(model_class.call_count, 2)
        small.run.assert_called_with('suffix', kv_cache=cache, is_suffix=True, max_new_tokens=100)
        self.assertEqual(runner.current_model, 'smollm')

    @patch('src.translation_layer.inference_runner.Model')
    @patch('src.translation_layer.inference_runner.router.route')
    def test_explicit_choice_and_errors(self, route, model_class):
        runner = InferenceRunner()
        runner.receive_prompt('hello', model_choice='qwen')
        route.assert_not_called()
        for choice in ('unknown', ''):
            with self.assertRaisesRegex(ValueError, 'unknown model'):
                runner.receive_prompt('hi', model_choice=choice)
        with self.assertRaises(ValueError):
            InferenceRunner(starting_model='unknown')
        with self.assertRaises(ValueError):
            runner.receive_prompt('')
        cache = Mock(tandem_model_name=MODEL_NAMES['smollm'])
        with self.assertRaisesRegex(ValueError, 'translation is not available'):
            runner.receive_prompt('suffix', model_choice='qwen', kv_cache=cache, is_suffix=True)
        self.assertEqual(runner.current_model, 'qwen')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            runner.receive_prompt('suffix', model_choice='qwen', kv_cache=cache, cache_model='qwen')
        with self.assertRaisesRegex(ValueError, 'cache_model is required'):
            runner.receive_prompt('suffix', model_choice='qwen', kv_cache=object())
