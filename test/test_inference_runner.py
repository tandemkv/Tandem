import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.translation_layer.inference_runner import InferenceRunner
from src.translation_layer.selection import TranslationCheckpoint


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.model_a = Mock(model_name='custom/answer-a')
        self.model_b = Mock(model_name='custom/answer-b')
        self.classifier = Mock(return_value={'task': 'Chatbot', 'complexity': 0.2})
        self.mapper = Mock()
        self.task_models = {'Chatbot': 'model_b', 'Code Generation': 'model_a'}
        self.runner = InferenceRunner(self.model_a, self.model_b, self.classifier, self.mapper,
                                      task_models=self.task_models)

    def test_conversation_text_is_required_before_routing(self):
        with self.assertRaises(TypeError):
            self.runner.receive_prompt('hello')
        for conversation_text in (None, '', '   ', 123):
            with self.subTest(conversation_text=conversation_text), self.assertRaisesRegex(
                    ValueError, 'conversation_text must be nonempty text'):
                self.runner.receive_prompt('hello', conversation_text=conversation_text,
                                           current_model='model_a')
        self.classifier.assert_not_called()
        self.model_a.run.assert_not_called()
        self.model_b.run.assert_not_called()

    def test_supplied_models_route_reuse_and_cache(self):
        runner = InferenceRunner(self.model_a, self.model_b, self.classifier, self.mapper,
                                 mode='complexity', complexity_threshold=0.7)
        self.classifier.side_effect = [{'complexity': 0.2}, {'complexity': 0.7}, {'complexity': 0.1}]
        self.model_b.run.return_value = 'answer'
        cache = self.model_b.get_kv_cache.return_value
        cache.tandem_model_name = self.model_b.model_name
        self.assertEqual(runner.receive_prompt('hello', conversation_text='hello', current_model='model_a'),
                         ('answer', cache))
        runner.receive_prompt('code', conversation_text='code', current_model='model_b')
        runner.receive_prompt('suffix', conversation_text='prefix suffix', current_model='model_b',
                              kv_cache=cache, is_suffix=True)
        self.model_a.run.assert_called_once()
        self.assertEqual(self.model_b.run.call_count, 2)
        self.model_b.run.assert_called_with('suffix', kv_cache=cache, is_suffix=True,
                                           max_new_tokens=100, prompt_mode='chat')
        self.classifier.assert_called_with('suffix')
        self.mapper.assert_not_called()

    def test_task_routing_and_custom_labels(self):
        self.classifier.return_value = {'task': 'Code Generation'}
        self.runner.receive_prompt('code', conversation_text='code', current_model='model_b')
        self.model_a.run.assert_called_once()
        runner = InferenceRunner(self.model_a, self.model_b, self.classifier, self.mapper,
                                 task_models={'custom': 'model_b'})
        self.classifier.return_value = {'task': 'custom'}
        runner.receive_prompt('hello', conversation_text='hello', current_model='model_a')
        self.model_b.run.assert_called_once()

    def test_classifier_failure_falls_back_to_supplied_model_a(self):
        self.classifier.side_effect = RuntimeError('failed')
        with self.assertLogs(level='WARNING'):
            self.runner.receive_prompt('hello', conversation_text='hello', current_model='model_a')
        self.model_a.run.assert_called_once()
        self.model_b.run.assert_not_called()

    def test_current_model_and_errors(self):
        runner = InferenceRunner(self.model_a, self.model_b, self.classifier, None,
                                 task_models=self.task_models)
        with self.assertRaises(TypeError):
            runner.receive_prompt('hello', conversation_text='hello')
        for current in ('unknown', ''):
            with self.assertRaisesRegex(ValueError, 'unknown model'):
                runner.receive_prompt('hi', conversation_text='hi', current_model=current)
        self.classifier.assert_not_called()
        self.classifier.return_value = {'task': 'Code Generation'}
        cache = Mock(tandem_model_name=self.model_b.model_name)
        with self.assertRaisesRegex(ValueError, 'translation is not available'):
            runner.receive_prompt('suffix', conversation_text='prefix suffix', current_model='model_b', kv_cache=cache)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            runner.receive_prompt('suffix', conversation_text='prefix suffix', current_model='model_a', kv_cache=cache)
        # current_model supplies provenance for an untagged external cache.
        runner.receive_prompt('suffix', conversation_text='prefix suffix', current_model='model_a',
                              kv_cache=object(), is_suffix=True)
        with patch('src.routing.router.route', return_value='unknown'):
            with self.assertRaisesRegex(ValueError, 'unknown model'):
                runner.receive_prompt('hi', conversation_text='hi', current_model='model_a')

    def test_mapper_receives_context_and_runs_exact_target_ids(self):
        for source, target, task in [('model_a', 'model_b', 'Chatbot'),
                                     ('model_b', 'model_a', 'Code Generation')]:
            with self.subTest(source=source):
                self.model_a.reset_mock()
                self.model_b.reset_mock()
                self.classifier.return_value = {'task': task}
                source_cache = Mock(tandem_model_name=self.runner.model_names[source])
                source_cache.get_seq_length.return_value = 2
                translated, target_ids = object(), [7, 8, 9]
                self.mapper.return_value = translated, target_ids
                self.runner.prompt_mode = 'plain'
                target_model = self.runner._models[target]
                result = self.runner.receive_prompt(
                    'prefix suffix', current_model=source, kv_cache=source_cache,
                    conversation_text='prefix suffix', cached_token_ids=[1, 2],
                    prefix_boundary=6, max_new_tokens=3)
                self.mapper.assert_called_with(
                    cache=source_cache, source_model=self.runner.model_names[source],
                    target_model=target_model, conversation_text='prefix suffix',
                    cached_token_ids=[1, 2], prefix_boundary=6, prompt_mode='plain')
                target_model.run.assert_not_called()
                target_model.run_tokens.assert_called_once_with(
                    target_ids, kv_cache=translated, max_new_tokens=3)
                self.assertEqual(result, (target_model.run_tokens.return_value,
                                          target_model.get_kv_cache.return_value))

    def test_translation_rejects_missing_or_mismatched_context(self):
        cache = Mock(tandem_model_name=self.model_a.model_name)
        cache.get_seq_length.return_value = 2
        registered_runner = InferenceRunner(
            self.model_a, self.model_b, self.classifier, task_models=self.task_models,
            translation_model='kvport', translation_models={'kvport': {
                (self.model_a.model_name, self.model_b.model_name):
                    TranslationCheckpoint('test-checkpoint', self.mapper),
            }})
        for name, runner in (('legacy', self.runner), ('registered', registered_runner)):
            for context in ({'conversation_text': 'abc'}, {'conversation_text': 'abc', 'prefix_boundary': 4},
                            {'conversation_text': 'abc', 'prefix_boundary': 1, 'cached_token_ids': [1]}):
                with self.subTest(mapper=name, context=context), self.assertRaises(ValueError):
                    runner.receive_prompt('abc', current_model='model_a', kv_cache=cache, **context)
        self.mapper.assert_not_called()
        self.model_b.run_tokens.assert_not_called()
        self.model_b.run.assert_not_called()

    def test_invalid_dependencies_and_task_aliases(self):
        with self.assertRaisesRegex(ValueError, 'distinct model_name'):
            InferenceRunner(self.model_a, self.model_a, self.classifier, self.mapper)
        with self.assertRaisesRegex(ValueError, 'nonempty model_name'):
            InferenceRunner(Mock(model_name=''), self.model_b, self.classifier, self.mapper)
        with self.assertRaisesRegex(TypeError, 'classifier_model must be callable'):
            InferenceRunner(self.model_a, self.model_b, None, self.mapper)
        with self.assertRaisesRegex(TypeError, 'kv_cache_mapper must be callable'):
            InferenceRunner(self.model_a, self.model_b, self.classifier, object())
        with self.assertRaisesRegex(ValueError, 'unknown model'):
            InferenceRunner(self.model_a, self.model_b, self.classifier, self.mapper,
                            task_models={'Chatbot': 'unknown'})

    def test_task_mode_requires_explicit_policy(self):
        with self.assertRaisesRegex(ValueError, 'task_models is required'):
            InferenceRunner(self.model_a, self.model_b, self.classifier, self.mapper)
        self.classifier.assert_not_called()

    def test_initialization_rejects_missing_answer_model_capabilities(self):
        for alias in ('model_a', 'model_b'):
            for method in ('run', 'run_tokens', 'get_kv_cache'):
                for invalid in (None, 'not callable'):
                    with self.subTest(alias=alias, method=method, invalid=invalid):
                        model = SimpleNamespace(model_name='custom/model', run=lambda: None,
                                                run_tokens=lambda: None, get_kv_cache=lambda: None)
                        if invalid is None:
                            delattr(model, method)
                        else:
                            setattr(model, method, invalid)
                        models = {'model_a': self.model_a, 'model_b': self.model_b, alias: model}
                        with self.assertRaisesRegex(TypeError, f'{alias} must provide callable {method}'):
                            InferenceRunner(**models, classifier_model=self.classifier,
                                            kv_cache_mapper=self.mapper, task_models=self.task_models)
        for model in (None, object(), SimpleNamespace(model_name=123)):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, 'model_a.*model_name'):
                InferenceRunner(model, self.model_b, self.classifier, self.mapper,
                                task_models=self.task_models)
        self.classifier.assert_not_called()
        self.mapper.assert_not_called()

    def test_initialization_rejects_invalid_routing_configuration(self):
        for settings, error, message in [
            ({'mode': 'invalid'}, ValueError, 'mode must be'),
            ({'task_models': [('Chatbot', 'model_a')]}, TypeError, 'task_models must be a mapping'),
            ({'task_models': {'': 'model_a'}}, ValueError, 'labels must be nonempty'),
            ({'task_models': {'Chatbot': []}}, ValueError, 'unknown model'),
            *[({'mode': 'complexity', 'complexity_threshold': value}, ValueError, 'threshold must be')
              for value in (None, '0.5', True, float('nan'), float('inf'))],
        ]:
            with self.subTest(settings=settings), self.assertRaisesRegex(error, message):
                InferenceRunner(self.model_a, self.model_b, self.classifier, self.mapper,
                                **settings)
        self.classifier.assert_not_called()
        self.mapper.assert_not_called()
