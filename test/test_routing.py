"""Offline routing tests: no model downloads or target-model execution."""
import io
import unittest
from unittest.mock import patch
from src.routing import router


class RoutingTests(unittest.TestCase):
    @patch.object(router, "_classifier")
    def test_task_and_complexity_can_disagree(self, classifier):
        classifier.return_value.return_value = {"task": "Code Generation", "complexity": 0.2}
        self.assertEqual(router.route("Print hello", mode="task"), "qwen")
        self.assertEqual(router.route("Print hello", mode="complexity"), "smollm")

    @patch.object(router, "_classifier")
    def test_complex_rewrite_and_threshold_boundary(self, classifier):
        classifier.return_value.return_value = {"task": "Rewrite", "complexity": 0.7}
        self.assertEqual(router.route("Rewrite", mode="task"), "smollm")
        self.assertEqual(router.route("Rewrite", mode="complexity", threshold=0.7), "qwen")
        self.assertEqual(router.route("Rewrite", mode="complexity", threshold=0.71), "smollm")

    @patch.object(router, "_classifier")
    def test_default_and_unknown_task(self, classifier):
        classifier.return_value.return_value = {"task": "Unknown", "complexity": 0.1}
        self.assertEqual(router.route("Ambiguous"), "qwen")

    @patch.object(router, "_classifier")
    def test_classifier_loading_failures_fall_back_to_qwen(self, classifier):
        for error in (ImportError("dependency missing"), OSError("download failed"),
                      ValueError("invalid checkpoint")):
            classifier.side_effect = error
            for mode in ("task", "complexity"):
                with self.subTest(error=error, mode=mode):
                    self.assertEqual(router.route("Hello", mode), "qwen")

    @patch.object(router, "_classifier")
    def test_classifier_runtime_failure_falls_back_but_analyze_raises(self, classifier):
        classifier.return_value.side_effect = RuntimeError("inference failed")
        for mode in ("task", "complexity"):
            with self.subTest(mode=mode):
                self.assertEqual(router.route("Hello", mode), "qwen")
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            router.analyze("Hello")

    @patch.object(router, "_classifier")
    def test_invalid_inputs_do_not_load_model(self, classifier):
        for prompt in [None, 123, "", " \n"]:
            for mode in ("task", "complexity"):
                with self.assertRaises(ValueError):
                    router.route(prompt, mode)
        with self.assertRaises(ValueError):
            router.route("Hi", "invalid")
        for threshold in [None, "0.5", True, -0.1, 1.21, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                router.route("Hi", "complexity", threshold=threshold)
        classifier.assert_not_called()

    @patch.object(router, "_classifier")
    def test_analyze_exposes_classifier_result(self, classifier):
        expected = {"task": "Chatbot", "complexity": 0.1, "task_confidence": 0.9}
        classifier.return_value.return_value = expected
        self.assertEqual(router.analyze("Hello"), expected)

    @patch.object(router, "route", return_value="smollm")
    def test_cli_toggle(self, route):
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(router.main(["Hello", "--mode", "complexity", "--threshold", "0.6"]), 0)
        route.assert_called_once_with("Hello", mode="complexity", threshold=0.6)
        self.assertEqual(output.getvalue().strip(), "smollm")


if __name__ == "__main__":
    unittest.main()
