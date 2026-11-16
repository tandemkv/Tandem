"""Selection checks use fake adapters: no weights, downloads, or GPU needed."""
import unittest
from dataclasses import asdict
from unittest.mock import Mock, patch
from types import ModuleType

from src.translation_layer.inference_runner import InferenceRunner
from src.translation_layer.interfaces import UnsupportedTranslationError
from src.translation_layer.selection import TranslationCheckpoint


class TranslationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.a = Mock(model_name="source/a")
        self.b = Mock(model_name="target/b")
        self.classifier = Mock(return_value={"complexity": 0.1})
        self.mapper = Mock(return_value=(object(), [7]))
        self.checkpoint = TranslationCheckpoint("weights/revision-1", self.mapper)
        self.cache = Mock(tandem_model_name="source/a")
        self.cache.get_seq_length.return_value = 1

    def runner(self, **options):
        return InferenceRunner(self.a, self.b, self.classifier, mode="complexity", **options)

    def handoff(self, runner):
        return runner.receive_prompt("new", conversation_text="old new", current_model="model_a",
                                     kv_cache=self.cache, cached_token_ids=[1], prefix_boundary=3)

    def test_default_and_explicit_selection_and_per_result_metadata(self):
        for name in ("tandem", "kvport"):
            with self.subTest(name=name):
                self.mapper.reset_mock()
                options = {} if name == "tandem" else {"translation_model": name}
                runner = self.runner(translation_models={name: {("source/a", "target/b"): self.checkpoint}},
                                     **options)
                result = self.handoff(runner)
                answer, cache = result
                self.assertEqual((answer, cache), (self.b.run_tokens.return_value, self.b.get_kv_cache.return_value))
                self.assertEqual(asdict(result.metadata), {
                    "translation_model": name, "checkpoint": "weights/revision-1",
                    "source_model": "source/a", "target_model": "target/b", "translated": True,
                })
                self.mapper.assert_called_once_with(
                    cache=self.cache, source_model="source/a", target_model=self.b,
                    conversation_text="old new", cached_token_ids=[1], prefix_boundary=3, prompt_mode="chat")
                # A later non-translated call must not overwrite this result's metadata.
                later = runner.receive_prompt("new", conversation_text="new", current_model="model_b")
                self.assertFalse(later.metadata.translated)
                self.assertIsNone(later.metadata.checkpoint)
                self.assertEqual(result.metadata.checkpoint, "weights/revision-1")

    def test_only_selected_implementation_runs(self):
        other = Mock()
        runner = self.runner(translation_model="kvport", translation_models={
            "tandem": {("source/a", "target/b"): TranslationCheckpoint("other", other)},
            "kvport": {("source/a", "target/b"): self.checkpoint},
        })
        self.handoff(runner)
        other.assert_not_called()
        self.mapper.assert_called_once()

    def test_same_model_cache_bypasses_selected_translator(self):
        runner = self.runner(translation_models={"tandem": {("source/a", "target/b"): self.checkpoint}})
        self.classifier.return_value = {"complexity": 0.9}
        result = runner.receive_prompt("new", conversation_text="old new", current_model="model_a",
                                       kv_cache=self.cache, is_suffix=True)
        self.mapper.assert_not_called()
        self.a.run.assert_called_once()
        self.assertFalse(result.metadata.translated)

    def test_demo_forwards_named_selection_without_loading_real_models(self):
        from src.experiments.poc.demo_setup import build_demo_runner

        models = ModuleType("src.models.model")
        models.Model = Mock(side_effect=[self.a, self.b])
        registrations = {"kvport": {("source/a", "target/b"): self.checkpoint}}
        with patch.dict("sys.modules", {"src.models.model": models}), patch(
                "src.experiments.poc.demo_setup.build_demo_classifier", return_value=self.classifier):
            runner = build_demo_runner(translation_model="kvport", translation_models=registrations,
                                       mode="complexity")
        result = self.handoff(runner)
        self.assertEqual(result.metadata.translation_model, "kvport")
        self.assertEqual(result.metadata.checkpoint, "weights/revision-1")

    def test_reverse_direction_selects_its_own_checkpoint(self):
        reverse = Mock(return_value=(object(), [8]))
        runner = self.runner(translation_models={"tandem": {
            ("source/a", "target/b"): self.checkpoint,
            ("target/b", "source/a"): TranslationCheckpoint("reverse-weights", reverse),
        }})
        self.classifier.return_value = {"complexity": 0.9}
        self.cache.tandem_model_name = "target/b"
        result = runner.receive_prompt("new", conversation_text="old new", current_model="model_b",
                                       kv_cache=self.cache, cached_token_ids=[1], prefix_boundary=3)
        self.assertEqual(result.metadata.checkpoint, "reverse-weights")
        reverse.assert_called_once()
        self.mapper.assert_not_called()

    def test_unsupported_direction_fails_before_adapter_or_generation(self):
        runner = self.runner(translation_models={"tandem": {("target/b", "source/a"): self.checkpoint}})
        with self.assertRaises(UnsupportedTranslationError):
            self.handoff(runner)
        self.mapper.assert_not_called()
        self.b.run_tokens.assert_not_called()
        self.b.run.assert_not_called()

    def test_unknown_unavailable_and_conflicting_selections(self):
        for options, message in [
            ({"translation_model": "typo"}, "unknown translation model"),
            ({"translation_model": "heterofold"}, "unavailable"),
            ({"translation_model": "kvport", "translation_models": {"kvport": {}}}, "unavailable"),
            ({"translation_model": "kvport", "kv_cache_mapper": self.mapper}, "legacy"),
            ({"translation_models": {}, "kv_cache_mapper": self.mapper}, "legacy"),
        ]:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                self.runner(**options)
        self.classifier.assert_not_called()

    def test_unconfigured_default_allows_native_calls_but_not_handoffs(self):
        runner = self.runner()
        result = runner.receive_prompt("new", conversation_text="new", current_model="model_a")
        self.assertEqual(result.metadata.translation_model, "tandem")
        self.assertFalse(result.metadata.translated)
        with self.assertRaisesRegex(ValueError, "translation is not available"):
            self.handoff(runner)
