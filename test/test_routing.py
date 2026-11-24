"""Offline routing tests: no model downloads or ML dependencies."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock

from src.routing import router


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.classifier = Mock()
        self.task_models = {'code': 'model_a', 'rewrite': 'model_b'}

    def route(self, prompt, mode='task', **kwargs):
        return router.route(prompt, mode, classifier_model=self.classifier,
                            task_models=self.task_models, **kwargs)

    def test_task_and_complexity_can_disagree(self):
        self.classifier.return_value = {'task': 'code', 'complexity': 0.2}
        self.assertEqual(self.route('Print hello'), 'model_a')
        self.assertEqual(self.route('Print hello', mode='complexity'), 'model_b')

    def test_complex_rewrite_and_threshold_boundary(self):
        self.classifier.return_value = {'task': 'rewrite', 'complexity': 0.7}
        self.assertEqual(self.route('Rewrite'), 'model_b')
        self.assertEqual(self.route('Rewrite', mode='complexity', threshold=0.7), 'model_a')
        self.assertEqual(self.route('Rewrite', mode='complexity', threshold=0.71), 'model_b')

    def test_classifier_defines_its_own_score_scale_and_aliases(self):
        self.classifier.return_value = {'complexity': 70.0}
        for threshold, expected in [(70, 'large'), (71, 'small')]:
            self.assertEqual(router.route('Hello', mode='complexity', threshold=threshold,
                                          classifier_model=self.classifier,
                                          model_a='large', model_b='small'), expected)

    def test_classifier_failure_falls_back_but_analyze_raises(self):
        self.classifier.side_effect = RuntimeError('inference failed')
        for mode in ('task', 'complexity'):
            with self.subTest(mode=mode), self.assertLogs(level='WARNING') as logs:
                self.assertEqual(self.route('Hello', mode), 'model_a')
            self.assertIn('falling back to model_a', logs.output[0])
            self.assertIn('inference failed', logs.output[0])
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            router.analyze('Hello', classifier_model=self.classifier)

    def test_missing_dependencies_or_policy_fail_before_inference(self):
        with self.assertRaises(TypeError):
            router.route('Hello')
        with self.assertRaises(TypeError):
            router.route('Hello', mode='complexity', classifier_model=None)
        with self.assertRaisesRegex(ValueError, 'task_models is required'):
            router.route('Hello', classifier_model=self.classifier)
        self.classifier.assert_not_called()

    def test_invalid_inputs_do_not_call_classifier(self):
        for prompt in [None, 123, '', ' \n']:
            for mode in ('task', 'complexity'):
                with self.assertRaises(ValueError):
                    self.route(prompt, mode)
        with self.assertRaises(ValueError):
            self.route('Hi', 'invalid')
        for threshold in [None, '0.5', True, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                self.route('Hi', 'complexity', threshold=threshold)
        self.classifier.assert_not_called()

    def test_analyze_exposes_classifier_result(self):
        expected = {'task': 'rewrite', 'complexity': 0.1, 'task_confidence': 0.9}
        self.classifier.return_value = expected
        self.assertEqual(router.analyze('Hello', classifier_model=self.classifier), expected)

    def test_core_works_without_site_packages_or_optional_adapters(self):
        # -S removes site-packages, including torch/transformers. Run in a fresh
        # interpreter so earlier tests cannot hide an accidental optional import.
        code = '''
from types import SimpleNamespace
from src.routing.interfaces import ClassificationResult, Classifier
from src.models.interfaces import AnswerModel
from src.translation_layer.interfaces import KVCacheMapper
from src.translation_layer.conversation_manager import ConversationManager
from src.translation_layer.inference_runner import InferenceRunner

class CustomModel:
    def __init__(self, name):
        self.model_name = name
    def run(self, prompt, **kwargs):
        return 'custom answer'
    def run_tokens(self, tokens, **kwargs):
        return 'custom answer'
    def get_kv_cache(self):
        return SimpleNamespace(tandem_model_name=self.model_name)

runner = InferenceRunner(CustomModel('a'), CustomModel('b'),
                         lambda prompt: {'task': 'custom'}, None,
                         task_models={'custom': 'model_b'})
answer, cache = runner.receive_prompt('hello', conversation_text='hello', current_model='model_a')
assert answer == 'custom answer'
assert cache.tandem_model_name == 'b'
import sys
assert not any(name.startswith(('src.adapters.', 'src.models.model')) for name in sys.modules)
'''
        result = subprocess.run([sys.executable, '-S', '-c', code],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
