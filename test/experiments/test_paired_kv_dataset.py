# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2",
#                 "huggingface-hub==0.36.0", "certifi==2025.10.5"]
# ///
"""Offline tests; tiny random architectures, no model/data downloads."""
import copy
import importlib.util
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

spec = importlib.util.spec_from_file_location(
    "dataset", Path(__file__).resolve().parents[2] / "src/experiments/paired_kv_dataset.py")
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)


class FeatureTests(unittest.TestCase):
    def test_both_architectures_capture_pre_rope_keys_and_values(self):
        for config_class, model_class in ((Qwen2Config, Qwen2ForCausalLM),
                                          (LlamaConfig, LlamaForCausalLM)):
            with self.subTest(model=model_class.__name__):
                torch.manual_seed(9)
                config = config_class(vocab_size=64, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=2, max_position_embeddings=64)
                config._attn_implementation = "eager"
                model = model_class(config).eval()
                ids = [3, 7, 8, 2, 9, 1, 4, 6, 5]
                tensors, shape = dataset.capture(model, ids, "cpu")
                self.assertEqual(shape["width"], 16)
                with torch.inference_mode():
                    native = model(torch.tensor([ids]), use_cache=True, logits_to_keep=1)
                    positions = torch.arange(len(ids))[None]
                    cos, sin = model.model.rotary_emb(torch.zeros(1, len(ids), 32), positions)
                for i, layer in enumerate(native.past_key_values.layers):
                    key = tensors[f"layer_{i}.k"].reshape(1, len(ids), 2, 8).transpose(1, 2)
                    half = key.shape[-1] // 2
                    rotated_half = torch.cat((-key[..., half:], key[..., :half]), dim=-1)
                    rotated = key * cos[:, None] + rotated_half * sin[:, None]
                    value = tensors[f"layer_{i}.v"].reshape(1, len(ids), 2, 8).transpose(1, 2)
                    torch.testing.assert_close(rotated, layer.keys)
                    torch.testing.assert_close(value, layer.values)
                    self.assertFalse(torch.allclose(key, layer.keys))
                self.assertTrue(all(not layer.self_attn.k_proj._forward_hooks
                                    for layer in model.model.layers))

    def test_alignment_uses_preceding_shared_boundary(self):
        source = [[i, i + 1] for i in range(8)] + [[8, 10], [10, 12], [12, 14]]
        target = [[i, i + 1] for i in range(14)]
        # The receiver-only boundary at character 9 must use shared
        # boundary 8, not a proportional/index-based correspondence.
        result = dataset.align(source, target)
        self.assertEqual(result["target_to_source"][8], 7)
        self.assertEqual(result["modes"][8], "forward_fill")
        self.assertEqual(result["target_to_source"][9], 8)
        self.assertIn([8, 9], result["exact_pairs"])

    def test_character_offsets_handle_unicode_and_prefix_before_first_match(self):
        source = [[0, 3], [3, 4]] + [[i, i + 1] for i in range(4, 13)]
        target = [[0, 1], [1, 3], [3, 4]] + [[i, i + 1] for i in range(4, 13)]
        result = dataset.align(source, target)
        self.assertEqual(result["modes"][0], "before_first")
        self.assertEqual(result["target_to_source"][0], 0)
        self.assertEqual(dataset.document_key("MONTRÉAL"), dataset.document_key("Montre\u0301al"))

    def test_invalid_feature_shapes_values_and_layers_are_rejected(self):
        shape = {"layers": 1, "tokens": 8, "width": 4}
        valid = {"layer_0.k": torch.zeros(8, 4), "layer_0.v": torch.ones(8, 4)}
        dataset.validate_features(valid, shape)
        for bad in ({"layer_0.k": valid["layer_0.k"]},
                    dict(valid, **{"layer_0.k": torch.zeros(7, 4)}),
                    dict(valid, **{"layer_0.k": torch.zeros(8, 4).half()}),
                    dict(valid, **{"layer_0.k": torch.full((8, 4), float("nan"))})):
            with self.assertRaises(ValueError):
                dataset.validate_features(bad, shape)

    def test_formatter_excludes_answer_and_preserves_question_span(self):
        row = {"question": "Where?", "answer": "SECRET_LABEL",
               "context": {"title": ["Title"], "sentences": [["Full passage."]]}}
        text, (start, end) = dataset.format_prompt(row)
        self.assertNotIn("SECRET_LABEL", text)
        self.assertIn("Full passage.", text)
        self.assertEqual(text[start:end], row["question"])

    def test_manifest_rejects_corrupt_text_leakage_and_alignment(self):
        samples = []
        for i in range(2):
            text = str(i) * 100
            offsets = [[j, j + 1] for j in range(100)]
            correspondence = dataset.align(offsets, offsets)
            samples.append({"id": str(i), "text": text, "text_sha256": dataset.sha(text.encode()),
                            "documents": [str(i)], "split": "train" if i == 0 else "validation",
                            "question_span": [90, 100],
                            "question_positions": {k: list(range(90, 100)) for k in dataset.MODELS},
                            "tokens": {k: {"ids": [1] * 100, "offsets": offsets} for k in dataset.MODELS},
                            "alignment": {k: copy.deepcopy(correspondence)
                                          for k in ("qwen_to_smol", "smol_to_qwen")}})
        manifest = {"schema": 1, "source": dataset.SOURCE, "models": dataset.MODELS,
                    "feature_space": dataset.SPACE, "max_tokens": 1024, "samples": samples}
        dataset.validate_manifest(manifest)
        for mode in ("text", "leakage", "alignment", "question"):
            bad = copy.deepcopy(manifest)
            if mode == "text":
                bad["samples"][0]["text"] = "other"
            elif mode == "leakage":
                bad["samples"][1]["documents"] = ["0"]
            elif mode == "alignment":
                bad["samples"][0]["alignment"]["qwen_to_smol"]["target_to_source"][1] = 90
            else:
                bad["samples"][0]["question_positions"]["qwen"] = [1]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                dataset.validate_manifest(bad)


if __name__ == "__main__":
    unittest.main()
