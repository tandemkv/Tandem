import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.translation_layer.conversation_manager import ConversationManager
from src.translation_layer.inference_runner import InferenceRunner, MODEL_NAMES


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.route_patch = patch('src.translation_layer.inference_runner.router.route',
                                 return_value='qwen')
        self.route = self.route_patch.start()
        self.addCleanup(self.route_patch.stop)
        self.model_patch = patch('src.translation_layer.inference_runner.Model')
        self.model_class = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.model_class.side_effect = self.make_model
        self.runner = InferenceRunner()
        self.conversation = ConversationManager(self.runner, current_model='qwen')

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
            kwargs.pop('is_suffix')
            kwargs.pop('prompt_mode')
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
        self.runner._models['qwen'].run.assert_called_with(
            suffix, kv_cache=cache, is_suffix=True, max_new_tokens=100, prompt_mode='chat')
        self.route.assert_called_with('How does that affect time?', mode='task',
                                      threshold=0.5, task_models=None)
        self.assertEqual(cache.tandem_token_ids, ids)
        self.assertEqual(chat.cached_token_ids, ids + [ord(c) for c in suffix] + [99])
        self.assertEqual(chat.conversation_text, prefix + suffix + 'answer')
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
        suffix = self.runner._models['qwen'].run.call_args.args[0]
        self.assertTrue(suffix.startswith('<|im_end|>\n<|im_start|>user\nnext'))

    def test_switches_in_both_directions_then_continues(self):
        self.route.side_effect = ['qwen', 'smollm', 'qwen', 'qwen']

        def adapter(**kwargs):
            self.assertEqual(kwargs['cached_token_ids'], self.conversation.cached_token_ids)
            self.assertEqual(kwargs['prefix_boundary'], self.conversation.prefix_boundary)
            self.assertTrue(kwargs['conversation_text'].startswith(self.conversation.conversation_text))
            # Real adapters must retain the exact IDs of the translated prefix.
            cache = SimpleNamespace(tandem_token_ids=[7, 8])
            return cache, [9, 10]

        self.runner.cache_adapter = Mock(side_effect=adapter)
        for expected in ['qwen', 'smollm', 'qwen', 'qwen']:
            self.conversation.send('next')
            self.assertEqual(self.conversation.current_model, expected)
            self.assertEqual(len(self.conversation.cached_token_ids),
                             self.conversation.kv_cache.get_seq_length())
        self.assertEqual(self.runner.cache_adapter.call_count, 2)

    def test_failed_send_preserves_state_and_can_retry(self):
        chat = self.conversation
        chat.send('hello')
        before = (chat.kv_cache, chat.cached_token_ids[:], chat.history[:],
                  chat.conversation_text, chat.prefix_boundary, chat.current_model)
        self.route.return_value = 'smollm'
        with self.assertRaisesRegex(ValueError, 'translation is not available'):
            chat.send('failed')
        self.assertEqual(before, (chat.kv_cache, chat.cached_token_ids, chat.history,
                                 chat.conversation_text, chat.prefix_boundary, chat.current_model))
        self.route.return_value = 'qwen'
        chat.send('retry')
        self.assertEqual(chat.history[-2]['content'], 'retry')
        self.assertEqual(len(chat.history), 4)

    def test_managers_share_runner_without_sharing_history(self):
        other = ConversationManager(self.runner)
        self.conversation.send('first')
        original = self.conversation.kv_cache
        other.send('independent')
        self.conversation.send('followup')
        self.assertIs(self.runner._models['qwen'].run.call_args.kwargs['kv_cache'], original)
        self.assertEqual(len(other.history), 2)
        self.assertEqual(self.model_class.call_count, 1)

    def test_incomplete_translation_metadata_does_not_commit(self):
        chat = self.conversation
        chat.send('hello')
        original = chat.kv_cache
        self.route.return_value = 'smollm'
        self.runner.cache_adapter = Mock(return_value=(
            SimpleNamespace(tandem_token_ids=[7]), [8]))
        # Simulate an adapter/model integration that cannot retain source IDs.
        target = self.make_model(model_name=MODEL_NAMES['smollm'])
        self.runner._models['smollm'] = target
        target.run_tokens.side_effect = None
        target.run_tokens.return_value = 'answer'
        target.cache = SimpleNamespace(tandem_model_name=MODEL_NAMES['smollm'],
                                       tandem_token_ids=None)
        with self.assertRaisesRegex(ValueError, 'exact tandem_token_ids'):
            chat.send('next')
        self.assertIs(chat.kv_cache, original)
        self.assertEqual(chat.current_model, 'qwen')
        self.assertEqual(len(chat.history), 2)

    def test_plain_mode_and_invalid_messages(self):
        self.runner.prompt_mode = 'plain'
        for message in ('', '  ', None, 1):
            with self.assertRaises(ValueError):
                self.conversation.send(message)
        self.route.assert_not_called()
        self.conversation.send('hello')
        self.conversation.send('next')
        self.assertEqual(self.runner._models['qwen'].run.call_args.args[0], '\nnext')
        with self.assertRaisesRegex(ValueError, 'unknown model'):
            ConversationManager(self.runner, current_model='unknown')
