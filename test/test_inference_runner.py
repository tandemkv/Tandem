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
        small.run.assert_called_with('suffix', kv_cache=cache, is_suffix=True, max_new_tokens=100, prompt_mode='chat')
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

    @patch('src.translation_layer.inference_runner.Model')
    def test_adapter_receives_context_and_runs_exact_target_ids(self, model_class):
        for source, target in [('qwen', 'smollm'), ('smollm', 'qwen')]:
            with self.subTest(source=source):
                model_class.reset_mock()
                target_model = model_class.return_value
                target_model.reset_mock()
                source_cache = Mock(tandem_model_name=MODEL_NAMES[source])
                source_cache.get_seq_length.return_value = 2
                translated = object()
                target_ids = [7, 8, 9]

                def translate(**kwargs):
                    model_class.assert_called_once_with(
                        model_name=MODEL_NAMES[target], device='cuda:1',
                        dtype='bfloat16', revision='pinned')
                    self.assertIs(kwargs['target_model'], target_model)
                    self.assertIs(kwargs['cache'], source_cache)
                    self.assertEqual(kwargs['source_model'], MODEL_NAMES[source])
                    self.assertEqual(kwargs['original_text'], 'prefix suffix')
                    self.assertEqual(kwargs['cached_token_ids'], [1, 2])
                    self.assertEqual(kwargs['prefix_boundary'], 6)
                    self.assertEqual(kwargs['prompt_mode'], 'plain')
                    return translated, target_ids

                runner = InferenceRunner(
                    cache_adapter=translate, prompt_mode='plain', device='cuda:1',
                    dtype='bfloat16', model_revisions={target: 'pinned'})
                result = runner.receive_prompt(
                    'prefix suffix', model_choice=target, kv_cache=source_cache,
                    original_text='prefix suffix', cached_token_ids=[1, 2],
                    prefix_boundary=6, max_new_tokens=3)
                target_model.run.assert_not_called()
                target_model.run_tokens.assert_called_once_with(
                    target_ids, kv_cache=translated, max_new_tokens=3)
                self.assertEqual(result, (target_model.run_tokens.return_value,
                                          target_model.get_kv_cache.return_value))
                self.assertEqual(runner.current_model, target)

    @patch('src.translation_layer.inference_runner.Model')
    def test_translation_rejects_missing_or_mismatched_context(self, model_class):
        adapter = Mock()
        runner = InferenceRunner(cache_adapter=adapter)
        cache = Mock(tandem_model_name=MODEL_NAMES['smollm'])
        cache.get_seq_length.return_value = 2
        for context in ({}, {'original_text': 'abc', 'prefix_boundary': 4},
                        {'original_text': 'abc', 'prefix_boundary': 1,
                         'cached_token_ids': [1]}):
            with self.assertRaises(ValueError):
                runner.receive_prompt('abc', model_choice='qwen', kv_cache=cache, **context)
        adapter.assert_not_called()
        model_class.assert_not_called()
