"""Exercise the live handoff with tiny networks and a known affine map."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from src.translation_layer.cache_mapper import CacheMapper, SMOLLM, QWEN, _split
from src.models.model import Model
from src.translation_layer.inference_runner import InferenceRunner
from src.translation_layer.interfaces import UnsupportedTranslationError


class Tokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def __call__(self, text, **kwargs):
        ids, spans = [], []
        pos = 0
        while pos < len(text):
            piece = next(p for p in self.pieces if text.startswith(p, pos))
            ids.append(self.pieces.index(piece))
            spans.append((pos, pos + len(piece)))
            pos += len(piece)
        return SimpleNamespace(input_ids=ids, offset_mapping=spans)

    def encode(self, text, **kwargs):
        return self(text).input_ids

    def decode(self, ids, **kwargs):
        return str(ids)


class DistilledMapperTests(unittest.TestCase):
    def test_split_moves_unshared_tail_to_target_suffix(self):
        src = Tokenizer(["a", "b", "c"])
        tgt = Tokenizer(["a", "bc"])
        self.assertEqual(_split(src, tgt, "abc", 2, [0, 1]), ([0], [0], [1]))
        with self.assertRaisesRegex(ValueError, "exact plain-text prefix"):
            _split(src, tgt, "abc", 2, [1, 1])

    @torch.no_grad()
    def test_runner_translates_live_keys_and_only_forwards_suffix(self):
        torch.manual_seed(73)
        tokenizer = Tokenizer(["a", "b", "c", "d"])
        models = []
        for pin, config, network in ((SMOLLM, LlamaConfig, LlamaForCausalLM),
                                     (QWEN, Qwen2Config, Qwen2ForCausalLM)):
            cfg = config(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, eos_token_id=None)
            with patch("src.models.model.AutoModelForCausalLM.from_pretrained", return_value=network(cfg)), \
                 patch("src.models.model.AutoTokenizer.from_pretrained", return_value=tokenizer):
                models.append(Model(model_name=pin[0], revision=pin[1]))
        source, target = models
        weights = torch.zeros(2, 2, 9, 8)
        weights[:, :, :8] = torch.eye(8)
        checkpoint = {"layers": torch.tensor([[1], [0]]), "weights": weights}
        norms = {f"{role}_{stat}": fn(2, 2, 8) for role in ("source", "target")
                 for stat, fn in (("mean", torch.zeros), ("std", torch.ones))}
        with patch("src.translation_layer.cache_mapper._load", side_effect=[checkpoint, {"model": norms}]):
            mapper = CacheMapper([source, target])
        captured = []
        handles = [layer.self_attn.k_proj.register_forward_hook(lambda m, args, out: captured.append(out))
                   for layer in source.model.model.layers]
        try:
            cache = source._forward([0, 1, 2], None).past_key_values
        finally:
            for handle in handles:
                handle.remove()
        cache.tandem_model_name, cache.tandem_model_revision = SMOLLM
        cache.tandem_token_ids = [0, 1, 2]
        original = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
        runner = InferenceRunner(target, source, lambda _: {"task": "switch"}, mapper,
                                 task_models={"switch": "model_a"})
        with patch.object(target, "run_tokens", wraps=target.run_tokens) as run_tokens:
            _, result = runner.receive_prompt("d", conversation_text="abcd", current_model="model_b",
                                              kv_cache=cache, prefix_boundary=3,
                                              cached_token_ids=[0, 1, 2], max_new_tokens=1)
        (ids,) = run_tokens.call_args.args
        translated = run_tokens.call_args.kwargs["kv_cache"]
        self.assertEqual(ids, [3])
        self.assertEqual(translated.tandem_token_ids, [0, 1, 2])
        self.assertEqual(result.tandem_token_ids, [0, 1, 2, 3] + target.generated_tokens)
        cos, sin = target.model.model.rotary_emb(captured[0], torch.arange(3)[None])
        for layer, source_idx in zip(translated.layers, (1, 0)):
            raw = captured[source_idx]
            left, right = raw.chunk(2, dim=-1)
            expected = raw * cos + torch.cat((-right, left), dim=-1) * sin
            torch.testing.assert_close(layer.keys, expected.unsqueeze(1))
            torch.testing.assert_close(layer.values, original[source_idx][1])
        for layer, (keys, values) in zip(cache.layers, original):
            self.assertTrue(torch.equal(layer.keys, keys) and torch.equal(layer.values, values))
        with self.assertRaises(UnsupportedTranslationError):
            mapper(cache=result, source_model=QWEN[0], target_model=source, conversation_text="",
                   cached_token_ids=[], prefix_boundary=0)
