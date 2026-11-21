"""Verify the cache handoff using tiny local Llama and Qwen models."""
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from src.models.model import Model
from src.translation_layer.cache_adapter import CacheTranslationAdapter
from src.translation_layer.inference_runner import InferenceRunner
from src.translation_layer.interfaces import UnsupportedTranslationError


def _model(network, name, prefix_ids, prefix_offsets):
    tokenizer = Mock()
    tokenizer.encode.return_value = prefix_ids
    tokenizer.return_value = {
        "input_ids": prefix_ids + [9],
        "offset_mapping": prefix_offsets + [(20, 24)],
    }
    tokenizer.decode.return_value = "answer"
    with patch("src.models.model.AutoModelForCausalLM.from_pretrained", return_value=network), \
         patch("src.models.model.AutoTokenizer.from_pretrained", return_value=tokenizer):
        return Model(model_name=name, revision="fixture")


class CacheAdapterTests(unittest.TestCase):
    def test_live_handoff_matches_projected_features(self):
        torch.manual_seed(7)
        settings = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_attention_heads=4, eos_token_id=None)
        smol = _model(LlamaForCausalLM(LlamaConfig(**settings, num_hidden_layers=2,
                                                 num_key_value_heads=2)),
                      "smol-fixture", [1, 2, 3, 4], [(0, 3), (4, 7), (8, 14), (15, 19)])
        qwen_ids = [1, 2, 5, 6, 4]
        qwen = _model(Qwen2ForCausalLM(Qwen2Config(**settings, num_hidden_layers=1,
                                                  num_key_value_heads=1)),
                      "qwen-fixture", qwen_ids, [(0, 3), (4, 7), (8, 10), (10, 14), (15, 19)])
        config = dict(source_layers=2, source_heads=2, target_layers=1, target_heads=1, head_dim=4)
        mapper = Mock(config=config, target_mean=torch.zeros(2, 1, 4))
        mapper.predict.side_effect = lambda features: features[:, :, :1] * 0.5
        run = dict(source="smol", target="qwen", batch=2, config=config, models={
            "smol": dict(id=smol.model_name, revision=smol.revision, type="llama"),
            "qwen": dict(id=qwen.model_name, revision=qwen.revision, type="qwen2"),
        })
        adapter = CacheTranslationAdapter(models=(smol, qwen), translators=((mapper, run),))

        projected = {"k": [], "v": []}

        def capture(role):
            def save(_module, _inputs, output):
                projected[role].append(output[0].reshape(4, 2, 4))
            return save

        hooks = [getattr(layer.self_attn, f"{role}_proj").register_forward_hook(capture(role))
                 for layer in smol.model.model.layers for role in projected]
        source_ids = [1, 2, 3, 4]
        try:
            with torch.no_grad():
                source_cache = smol._forward(source_ids, None).past_key_values
        finally:
            for hook in hooks:
                hook.remove()
        source_keys = [layer.keys.clone() for layer in source_cache.layers]

        # Build the offline reference directly from pre-RoPE projection hooks.
        with torch.no_grad():
            features = torch.stack([torch.cat(rows, dim=1) for rows in projected.values()], dim=1)
            predictions = features[[0, 1, 1, 2, 3], :, :1] * 0.5
            keys = predictions[:, 0, 0][None, None]
            cos, sin = qwen.model.model.rotary_emb(keys, torch.arange(5)[None])
            _, keys = apply_rotary_pos_emb(keys, keys, cos, sin)
            values = predictions[:, 1, 0][None, None]

        runner = InferenceRunner(
            qwen, smol, lambda _: {"task": "question"}, adapter,
            task_models={"question": "model_a"}, prompt_mode="plain",
        )
        with patch.object(qwen, "run_tokens", wraps=qwen.run_tokens) as run_tokens:
            _, answer_cache = runner.receive_prompt(
                "when", current_model="model_b", conversation_text="the bus leaves late when",
                kv_cache=source_cache, cached_token_ids=source_ids, prefix_boundary=19,
                max_new_tokens=1,
            )
        self.assertEqual(run_tokens.call_args.args, ([9],))
        cache = run_tokens.call_args.kwargs["kv_cache"]
        torch.testing.assert_close(cache.layers[0].keys, keys, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(cache.layers[0].values, values, rtol=1e-5, atol=1e-6)
        self.assertEqual(answer_cache.tandem_token_ids, qwen_ids + [9] + qwen.generated_tokens)
        for layer, original in zip(source_cache.layers, source_keys):
            self.assertTrue(torch.equal(layer.keys, original))

        with self.assertRaises(UnsupportedTranslationError):
            adapter(cache=answer_cache, source_model=qwen.model_name, target_model=smol,
                    conversation_text="", cached_token_ids=[], prefix_boundary=0, prompt_mode="plain")
