import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.translation_layer.conversation_manager import ConversationManager
from src.translation_layer.inference_runner import InferenceRunner
from src.translation_layer.interfaces import UnsupportedTranslationError


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.route_patch = patch('src.translation_layer.inference_runner.router.route',
                                 return_value='model_a')
        self.route = self.route_patch.start()
        self.addCleanup(self.route_patch.stop)
        self.classifier = Mock()
        self.runner = InferenceRunner(
            self.make_model(model_name='custom/a'), self.make_model(model_name='custom/b'),
            self.classifier, None, task_models={'custom': 'model_a'})
        self.conversation = ConversationManager(self.runner, current_model='model_a')

    def make_model(self, *, model_name, **kwargs):
        model = Mock(model_name=model_name)

        def run_tokens(tokens, *, kv_cache=None, max_new_tokens=100):
            prefix = [] if kv_cache is None else kv_cache.tandem_token_ids
            ids = prefix + tokens + [99]
            model.cache = SimpleNamespace(
                tandem_model_name=model_name, tandem_token_ids=ids,
                tandem_turn_complete=True, get_seq_length=lambda: len(ids))
            return 'answer'

        def run(prompt, **kwargs):
            kwargs.pop('is_suffix', None)
            return run_tokens([ord(c) for c in prompt], **kwargs)

        model.run.side_effect = run
        model.run_tokens.side_effect = run_tokens
        model.get_kv_cache.side_effect = lambda: model.cache
        return model

    def test_multiple_turns_keep_exact_metadata_and_route_user_message(self):
        chat = self.conversation
        self.assertEqual(chat.send('Explain gravity.', max_new_tokens=3), 'answer')
        cache, ids, prefix = chat.kv_cache, chat.cached_token_ids[:], chat.conversation_text
        self.assertEqual(chat.send('How does that affect time?'), 'answer')
        suffix = ('\n<|im_start|>user\nHow does that affect time?<|im_end|>\n'
                  '<|im_start|>assistant\n')
        self.runner._models['model_a'].run.assert_called_with(
            suffix, kv_cache=cache, is_suffix=True, max_new_tokens=100)
        self.route.assert_called_with('How does that affect time?', mode='task',
                                      threshold=0.5, task_models={'custom': 'model_a'}, classifier_model=self.classifier,
                                      model_a='model_a', model_b='model_b')
        self.assertEqual(cache.tandem_token_ids, ids)
        self.assertEqual(chat.cached_token_ids, ids + [ord(c) for c in suffix] + [99])
        self.assertEqual(chat.conversation_text, prefix + suffix + 'answer<|im_end|>')
        self.assertEqual(chat.prefix_boundary, len(chat.conversation_text))
        self.assertEqual(chat.history, [
            {'role': 'user', 'content': 'Explain gravity.'},
            {'role': 'assistant', 'content': 'answer'},
            {'role': 'user', 'content': 'How does that affect time?'},
            {'role': 'assistant', 'content': 'answer'},
        ])

    def test_unfinished_assistant_turn_is_closed(self):
        self.conversation.send('hello')
        self.conversation.kv_cache.tandem_turn_complete = False
        self.conversation.send('next')
        suffix = self.runner._models['model_a'].run.call_args.args[0]
        self.assertTrue(suffix.startswith('<|im_end|>\n<|im_start|>user\nnext'))

    def test_switches_in_both_directions_then_continues(self):
        self.route.side_effect = ['model_a', 'model_b', 'model_a', 'model_a']

        def adapter(**kwargs):
            self.assertEqual(kwargs['cached_token_ids'], self.conversation.cached_token_ids)
            self.assertEqual(kwargs['prefix_boundary'], self.conversation.prefix_boundary)
            self.assertTrue(kwargs['conversation_text'].startswith(self.conversation.conversation_text))
            # Real adapters must retain the exact IDs of the translated prefix.
            cache = SimpleNamespace(tandem_token_ids=[7, 8])
            return cache, [9, 10]

        self.runner.kv_cache_mapper = Mock(side_effect=adapter)
        for expected in ['model_a', 'model_b', 'model_a', 'model_a']:
            self.conversation.send('next')
            self.assertEqual(self.conversation.current_model, expected)
            self.assertEqual(len(self.conversation.cached_token_ids),
                             self.conversation.kv_cache.get_seq_length())
        self.assertEqual(self.runner.kv_cache_mapper.call_count, 2)

    def test_failed_send_preserves_state_and_can_retry(self):
        chat = self.conversation
        chat.send('hello')
        before = (chat.kv_cache, chat.cached_token_ids[:], chat.history[:],
                  chat.conversation_text, chat.prefix_boundary, chat.current_model)
        self.route.return_value = 'model_b'
        self.runner.kv_cache_mapper = Mock(side_effect=ValueError('token IDs do not match'))
        with self.assertRaisesRegex(ValueError, 'token IDs do not match'):
            chat.send('failed')
        self.assertEqual(before, (chat.kv_cache, chat.cached_token_ids, chat.history,
                                 chat.conversation_text, chat.prefix_boundary, chat.current_model))
        self.route.return_value = 'model_a'
        chat.send('retry')
        self.assertEqual(chat.history[-2]['content'], 'retry')
        self.assertEqual(len(chat.history), 4)

    def test_unsupported_switch_reruns_the_conversation_natively(self):
        chat = self.conversation

        def one_way_mapper(*, source_model, target_model, **kwargs):
            if source_model == self.runner.model_names['model_b']:
                raise UnsupportedTranslationError(source_model, target_model.model_name)
            return SimpleNamespace(tandem_token_ids=[7, 8]), [9, 10]

        self.runner.kv_cache_mapper = one_way_mapper
        chat.send('first')
        self.route.return_value = 'model_b'
        chat.send('supported switch')
        self.route.return_value = 'model_a'
        target = self.runner._models['model_a']
        target.reset_mock()
        full_text = (chat.conversation_text + '\n<|im_start|>user\nunsupported switch<|im_end|>\n'
                     '<|im_start|>assistant\n')
        chat.send('unsupported switch')
        target.run.assert_called_once_with(full_text, max_new_tokens=100)
        self.assertEqual(chat.current_model, 'model_a')
        self.assertEqual(chat.cached_token_ids, [ord(c) for c in full_text] + [99])
        self.assertEqual(len(chat.history), 6)

    def test_managers_share_runner_without_sharing_history(self):
        other = ConversationManager(self.runner)
        self.conversation.send('first')
        original = self.conversation.kv_cache
        other.send('independent')
        self.conversation.send('followup')
        self.assertIs(self.runner._models['model_a'].run.call_args.kwargs['kv_cache'], original)
        self.assertEqual(len(other.history), 2)

    def test_incomplete_translation_metadata_does_not_commit(self):
        chat = self.conversation
        chat.send('hello')
        original = chat.kv_cache
        self.route.return_value = 'model_b'
        self.runner.kv_cache_mapper = Mock(return_value=(
            SimpleNamespace(tandem_token_ids=[7]), [8]))
        # Simulate an adapter/model integration that cannot retain source IDs.
        target = self.make_model(model_name=self.runner.model_names['model_b'])
        self.runner._models['model_b'] = target
        target.run_tokens.side_effect = None
        target.run_tokens.return_value = 'answer'
        target.cache = SimpleNamespace(tandem_model_name=self.runner.model_names['model_b'],
                                       tandem_token_ids=None)
        with self.assertRaisesRegex(ValueError, 'exact tandem_token_ids'):
            chat.send('next')
        self.assertIs(chat.kv_cache, original)
        self.assertEqual(chat.current_model, 'model_a')
        self.assertEqual(len(chat.history), 2)

    def test_chatml_text_and_invalid_messages(self):
        for message in ('', '  ', None, 1):
            with self.assertRaises(ValueError):
                self.conversation.send(message)
        self.route.assert_not_called()
        model_a = self.runner._models['model_a']
        self.conversation.send('hello')
        hello = '<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n'
        self.assertEqual(model_a.run.call_args.args[0], hello)
        self.conversation.send('next')
        nxt = '\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n'
        self.assertEqual(model_a.run.call_args.args[0], nxt)
        self.assertEqual(self.conversation.conversation_text,
                         hello + 'answer<|im_end|>' + nxt + 'answer<|im_end|>')
        with self.assertRaisesRegex(ValueError, 'unknown model'):
            ConversationManager(self.runner, current_model='unknown')
