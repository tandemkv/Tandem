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
        tokenizer.apply_chat_template.return_value = [10, 11, 12]
        with patch('src.models.model.AutoModelForCausalLM.from_pretrained', return_value=network), \
             patch('src.models.model.AutoTokenizer.from_pretrained', return_value=tokenizer):
            self.model = Model(model_name='tiny-qwen')

    def test_handoff_matches_full_history_and_preserves_branches(self):
        model = self.model
        model.run('Hello', max_new_tokens=3)
        model.tokenizer.apply_chat_template.assert_called_once_with(
            [{'role': 'user', 'content': 'Hello'}], tokenize=True, add_generation_prompt=True, return_dict=False)
        history = [10, 11, 12] + model.generated_tokens
        cache = model.get_kv_cache()
        before = copy.deepcopy(cache)
        self.assertEqual(cache.get_seq_length(), len(history))
        for suffix in ([21, 22], [31, 32, 33]):
            model.run_tokens(suffix, kv_cache=cache, max_new_tokens=4)
            cached_tokens = model.generated_tokens[:]
            cached = model.get_kv_cache()
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
        self.assertEqual(model.get_kv_cache().get_seq_length(), 4)

    def test_plain_text_uses_verbatim_tokens(self):
        model = self.model
        model.tokenizer.encode.return_value = [10, 11, 12]
        model.run('Raw training prompt', prompt_mode='plain', max_new_tokens=2)
        plain_tokens = model.generated_tokens[:]
        model.tokenizer.encode.assert_called_once_with(
            'Raw training prompt', add_special_tokens=False)
        model.tokenizer.apply_chat_template.assert_not_called()
        model.run_tokens([10, 11, 12], max_new_tokens=2)
        self.assertEqual(plain_tokens, model.generated_tokens)
        with self.assertRaisesRegex(ValueError, 'prompt_mode'):
            model.run('hi', prompt_mode='invalid')

    def test_forced_switch_matches_native_target_with_identical_tokens(self):
        from src.translation_layer.inference_runner import InferenceRunner, MODEL_NAMES

        # A native-prefill adapter isolates the runner contract from learned
        # translation quality. Exercise both aliases with an offline tiny model.
        for source, target in [('qwen', 'smollm'), ('smollm', 'qwen')]:
            with self.subTest(source=source):
                model = self.model
                model.model_name = MODEL_NAMES[source]
                model.run_tokens([10, 11], max_new_tokens=1)
                source_ids = [10, 11] + model.generated_tokens
                source_cache = model.get_kv_cache()
                original = copy.deepcopy(source_cache)
                model.model_name = MODEL_NAMES[target]
                target_prefix, target_suffix = [30, 31, 32, 33], [40, 41]

                def native_adapter(**kwargs):
                    self.assertIs(kwargs['target_model'], model)
                    with torch.no_grad():
                        cache = model._forward(target_prefix, None).past_key_values
                    cache.tandem_model_name = MODEL_NAMES[target]
                    return cache, target_suffix

                runner = InferenceRunner(cache_adapter=native_adapter, prompt_mode='plain')
                runner._models[target] = model
                _, switched_cache = runner.receive_prompt(
                    'prefix suffix', model_choice=target, kv_cache=source_cache,
                    original_text='prefix suffix', prefix_boundary=6,
                    cached_token_ids=source_ids, max_new_tokens=3)
                switched_ids = model.generated_tokens[:]
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

        runner = InferenceRunner()
        for task, choice in [('Chatbot', 'smollm'), ('Code Generation', 'qwen')]:
            with patch.object(router, 'analyze', return_value={'task': task}):
                answer, cache = runner.receive_prompt('Say hello.', max_new_tokens=16)
            self.assertTrue(answer.strip())
            self.assertEqual(runner.current_model, choice)
            loaded = runner._models[choice]
            length = cache.get_seq_length()
            suffix = '\n<|im_start|>user\nSay goodbye.<|im_end|>\n<|im_start|>assistant\n'
            answer, next_cache = runner.receive_prompt(
                suffix, model_choice=choice, kv_cache=cache, is_suffix=True, max_new_tokens=8)
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
