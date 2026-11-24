"""Offline cache regression tests, plus opt-in real Instruct replay."""
import copy
import os
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.models.model import Model


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=256,
                             eos_token_id=None)
        config._attn_implementation = "eager"
        network = Qwen2ForCausalLM(config).eval()
        tokenizer = Mock()
        tokenizer.decode.side_effect = lambda ids, **kwargs: str(ids)
        tokenizer.encode.return_value = [10, 11, 12]
        with patch('src.models.model.AutoModelForCausalLM.from_pretrained', return_value=network), \
             patch('src.models.model.AutoTokenizer.from_pretrained', return_value=tokenizer):
            self.model = Model(model_name='tiny-qwen')


    def test_handoff_matches_full_history_and_preserves_branches(self):
        model = self.model
        model.run('Hello', max_new_tokens=3)
        model.tokenizer.encode.assert_called_once_with('Hello', add_special_tokens=False)
        history = [10, 11, 12] + model.generated_tokens
        cache = model.get_kv_cache()
        before = copy.deepcopy(cache)
        self.assertEqual(cache.tandem_token_ids, history)
        self.assertFalse(cache.tandem_turn_complete)
        self.assertEqual(cache.get_seq_length(), len(history))
        for suffix in ([21, 22], [31, 32, 33]):
            model.run_tokens(suffix, kv_cache=cache, max_new_tokens=4)
            cached_tokens = model.generated_tokens[:]
            cached = model.get_kv_cache()
            self.assertEqual(cached.tandem_token_ids, history + suffix + cached_tokens)
            model.run_tokens(history + suffix, max_new_tokens=4)
            self.assertEqual(cached_tokens, model.generated_tokens)
            self.assertEqual(cached.get_seq_length(), len(history) + len(suffix) + 4)
            for actual, native in zip(cached.layers, model.get_kv_cache().layers):
                torch.testing.assert_close(actual.keys, native.keys, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(actual.values, native.values, atol=1e-5, rtol=1e-5)
        self.assertEqual(cache.get_seq_length(), len(history))
        for actual, original in zip(cache.layers, before.layers):
            self.assertTrue(torch.equal(actual.keys, original.keys))
            self.assertTrue(torch.equal(actual.values, original.values))


    def test_failed_generation_preserves_input_and_previous_result(self):
        model = self.model
        model.run('Hello', max_new_tokens=2)
        cache = model.get_kv_cache()
        before = copy.deepcopy(cache)
        original_forward = model._forward
        calls = 0

        def fail_after_prefill(tokens, past):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('decode failed')
            return original_forward(tokens, past)

        with patch.object(model, '_forward', side_effect=fail_after_prefill):
            with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                model.run_tokens([21], kv_cache=cache, max_new_tokens=2)
        self.assertIs(model.get_kv_cache(), cache)
        for actual, original in zip(cache.layers, before.layers):
            self.assertTrue(torch.equal(actual.keys, original.keys))
            self.assertTrue(torch.equal(actual.values, original.values))


    def test_eos_is_included_in_cache(self):
        model = self.model
        model.run_tokens([10, 11], max_new_tokens=1)
        model.model.generation_config.eos_token_id = model.generated_tokens[0]
        model.run_tokens([10, 11], max_new_tokens=5)
        self.assertEqual(len(model.generated_tokens), 1)
        self.assertEqual(model.get_kv_cache().get_seq_length(), 3)
        self.assertTrue(model.get_kv_cache().tandem_turn_complete)


    def test_explicit_suffix_and_constructor_cache(self):
        model = self.model
        model.run('Hi', max_new_tokens=1)
        cache = model.get_kv_cache()
        model._initial_cache = cache
        model.tokenizer.encode.return_value = [21, 22]
        with self.assertRaisesRegex(ValueError, 'is_suffix'):
            model.run('followup')
        model.run('formatted suffix', is_suffix=True, max_new_tokens=1)
        self.assertIsNone(model._initial_cache)
        self.assertEqual(model.get_kv_cache().get_seq_length(), 7)
        model.run('formatted suffix', kv_cache=cache, is_suffix=True, max_new_tokens=1)
        self.assertEqual(model.get_kv_cache().get_seq_length(), 7)
        self.assertEqual(cache.get_seq_length(), 4)
        model.run('independent', max_new_tokens=1)
        self.assertEqual(model.get_kv_cache().get_seq_length(), 3)


    def test_run_encodes_text_verbatim(self):
        model = self.model
        model.run('Raw training prompt', max_new_tokens=2)
        text_tokens = model.generated_tokens[:]
        model.tokenizer.encode.assert_called_once_with(
            'Raw training prompt', add_special_tokens=False)
        model.run_tokens([10, 11, 12], max_new_tokens=2)
        self.assertEqual(text_tokens, model.generated_tokens)


    def test_conversation_manager_matches_full_token_replay(self):
        from src.translation_layer.conversation_manager import ConversationManager
        from src.translation_layer.inference_runner import InferenceRunner

        model = self.model
        model.tokenizer.encode.return_value = [21, 22]
        runner = InferenceRunner(model, Mock(model_name='other-model'),
                                 Mock(return_value={'task': 'Code Generation'}), None,
                                 task_models={'Code Generation': 'model_a'})
        conversation = ConversationManager(runner)
        with patch('src.translation_layer.inference_runner.router.route', return_value='model_a'):
            conversation.send('Explain gravity.', max_new_tokens=2)
            prefix = conversation.cached_token_ids[:]
            original = conversation.kv_cache
            conversation.send('How does that affect time?', max_new_tokens=3)
        generated = model.generated_tokens[:]
        cached = conversation.kv_cache
        model.run_tokens(prefix + [21, 22], max_new_tokens=3)
        self.assertEqual(generated, model.generated_tokens)
        self.assertEqual(conversation.cached_token_ids, prefix + [21, 22] + generated)
        self.assertEqual(original.get_seq_length(), len(prefix))
        for actual, expected in zip(cached.layers, model.cache.layers):
            torch.testing.assert_close(actual.keys, expected.keys, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(actual.values, expected.values, atol=1e-5, rtol=1e-5)


    def test_forced_switch_matches_native_target_with_identical_tokens(self):
        from src.translation_layer.inference_runner import InferenceRunner

        # A native-prefill adapter isolates the runner contract from learned
        # translation quality. Exercise both aliases with an offline tiny model.
        model_names = {'model_a': 'tiny-a', 'model_b': 'tiny-b'}
        for source, target in [('model_a', 'model_b'), ('model_b', 'model_a')]:
            with self.subTest(source=source):
                model = self.model
                model.model_name = model_names[source]
                model.run_tokens([10, 11], max_new_tokens=1)
                source_ids = [10, 11] + model.generated_tokens
                source_cache = model.get_kv_cache()
                original = copy.deepcopy(source_cache)
                model.model_name = model_names[target]
                target_prefix, target_suffix = [30, 31, 32, 33], [40, 41]

                def native_adapter(**kwargs):
                    self.assertIs(kwargs['target_model'], model)
                    with torch.no_grad():
                        cache = model._forward(target_prefix, None).past_key_values
                    cache.tandem_model_name = model_names[target]
                    cache.tandem_token_ids = target_prefix[:]
                    return cache, target_suffix

                models = {source: Mock(model_name=model_names[source]), target: model}
                runner = InferenceRunner(models['model_a'], models['model_b'], Mock(),
                                         native_adapter,
                                         task_models={'custom': target})
                with patch('src.translation_layer.inference_runner.router.route', return_value=target):
                    _, switched_cache = runner.receive_prompt(
                        'prefix suffix', current_model=source, kv_cache=source_cache,
                        conversation_text='prefix suffix', prefix_boundary=6,
                        cached_token_ids=source_ids, max_new_tokens=3)
                switched_ids = model.generated_tokens[:]
                self.assertEqual(switched_cache.tandem_token_ids,
                                 target_prefix + target_suffix + switched_ids)
                model.run_tokens(target_prefix + target_suffix, max_new_tokens=3)
                self.assertEqual(switched_ids, model.generated_tokens)
                for actual, expected in zip(switched_cache.layers, model.cache.layers):
                    torch.testing.assert_close(actual.keys, expected.keys, atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(actual.values, expected.values, atol=1e-5, rtol=1e-5)
                for actual, expected in zip(source_cache.layers, original.layers):
                    self.assertTrue(torch.equal(actual.keys, expected.keys))
                    self.assertTrue(torch.equal(actual.values, expected.values))


    def test_loading_configuration_applies_to_model_and_tokenizer(self):
        with patch('src.models.model.AutoModelForCausalLM.from_pretrained') as load, \
             patch('src.models.model.AutoTokenizer.from_pretrained') as tokenizer:
            Model(model_name='checkpoint', device='cuda:1', dtype=torch.bfloat16,
                  revision='commit')
        load.assert_called_once_with('checkpoint', torch_dtype=torch.bfloat16,
                                     revision='commit')
        load.return_value.to.assert_called_once_with('cuda:1')
        tokenizer.assert_called_once_with('checkpoint', revision='commit')


    def test_bad_cache_and_inputs(self):
        model = self.model
        model.run('Hi', max_new_tokens=1)
        cache = model.get_kv_cache()
        for kind in ('owner', 'shape', 'dtype', 'nonfinite', 'layers'):
            bad = copy.deepcopy(cache)
            if kind == 'owner':
                bad.tandem_model_name = 'other'
            elif kind == 'shape':
                bad.layers[0].keys = bad.layers[0].keys[..., :-1, :]
            elif kind == 'dtype':
                bad.layers[0].keys = bad.layers[0].keys.double()
            elif kind == 'nonfinite':
                bad.layers[0].keys[0, 0, 0, 0] = float('nan')
            else:
                bad.layers.pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                model.run_tokens([21], kv_cache=bad)
        with self.assertRaisesRegex(ValueError, 'DynamicCache'):
            model.run_tokens([21], kv_cache=object())
        for limit in (0, -1, True):
            with self.assertRaises(ValueError):
                model.run('Hi', max_new_tokens=limit)
        with self.assertRaisesRegex(ValueError, 'context window'):
            model.run_tokens([10] * 256)


@unittest.skipUnless(os.environ.get('TANDEM_REAL_MODEL_TEST') == '1', 'opt-in model weights')
class RealModelTests(unittest.TestCase):
    def test_router_to_runner_with_both_checkpoints(self):
        from src.routing import router
        from src.translation_layer.inference_runner import InferenceRunner

        classifier = Mock()
        runner = InferenceRunner(
            Model(model_name='Qwen/Qwen2.5-0.5B-Instruct'),
            Model(model_name='HuggingFaceTB/SmolLM2-360M-Instruct'), classifier, None,
            task_models={'Chatbot': 'model_b', 'Code Generation': 'model_a'})
        for task, choice in [('Chatbot', 'model_b'), ('Code Generation', 'model_a')]:
            classifier.return_value = {'task': task}
            answer, cache = runner.receive_prompt('Say hello.', conversation_text='Say hello.', current_model=choice, max_new_tokens=16)
            self.assertTrue(answer.strip())
            loaded = runner._models[choice]
            length = cache.get_seq_length()
            suffix = '\n<|im_start|>user\nSay goodbye.<|im_end|>\n<|im_start|>assistant\n'
            with patch.object(router, 'route', return_value=choice):
                answer, next_cache = runner.receive_prompt(
                    suffix, conversation_text='Say hello.' + answer + suffix,
                    current_model=choice, kv_cache=cache, is_suffix=True, max_new_tokens=8)
            self.assertTrue(answer.strip())
            self.assertIs(runner._models[choice], loaded)
            self.assertEqual(cache.get_seq_length(), length)
            self.assertGreater(next_cache.get_seq_length(), length)

    def test_instruct_conversation_replay(self):
        model = Model(model_name='Qwen/Qwen2.5-0.5B-Instruct')
        prefix = model.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'Say hello.'}], tokenize=True, add_generation_prompt=True, return_dict=False)
        model.run_tokens(prefix, max_new_tokens=16)
        history = prefix + model.generated_tokens
        cache = model.get_kv_cache()
        before = copy.deepcopy(cache)
        suffix = model.tokenizer.encode(
            '\n<|im_start|>user\nWhat did I ask you to say?<|im_end|>\n<|im_start|>assistant\n',
            add_special_tokens=False)
        model.run_tokens(suffix, kv_cache=cache, max_new_tokens=16)
        cached_tokens = model.generated_tokens[:]
        model.run_tokens(history + suffix, max_new_tokens=16)
        self.assertEqual(cached_tokens, model.generated_tokens)
        self.assertEqual(cache.get_seq_length(), len(history))
        for actual, original in zip(cache.layers, before.layers):
            self.assertTrue(torch.equal(actual.keys, original.keys))
            self.assertTrue(torch.equal(actual.values, original.values))


if __name__ == '__main__':
    unittest.main()
