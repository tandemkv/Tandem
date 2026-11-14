"""Optional Qwen/SmolLM/NVIDIA wiring for the local proof of concept."""
from typing import Literal

from src.routing.interfaces import Classifier
from src.translation_layer.inference_runner import InferenceRunner
from src.translation_layer.interfaces import KVCacheMapper

# Experimental policy; calibrate against measured target-model performance.
TASK_MODELS = {
    "Brainstorming": "model_b",
    "Chatbot": "model_b",
    "Classification": "model_b",
    "Closed QA": "model_b",
    "Code Generation": "model_a",
    "Extraction": "model_b",
    "Open QA": "model_a",
    "Other": "model_a",
    "Rewrite": "model_b",
    "Summarization": "model_b",
    "Text Generation": "model_b",
    "Unknown": "model_a",
}


def build_demo_classifier() -> Classifier:
    from src.adapters.nvidia_classifier import PromptClassifier

    return PromptClassifier()


def build_demo_runner(*, kv_cache_mapper: KVCacheMapper | None = None,
                      mode: Literal["task", "complexity"] = "task",
                      complexity_threshold: float = 0.5) -> InferenceRunner:
    from src.models.model import Model

    return InferenceRunner(
        model_a=Model(model_name="Qwen/Qwen2.5-0.5B-Instruct"),
        model_b=Model(model_name="HuggingFaceTB/SmolLM2-360M-Instruct"),
        classifier_model=build_demo_classifier(),
        kv_cache_mapper=kv_cache_mapper,
        task_models=TASK_MODELS,
        mode=mode,
        complexity_threshold=complexity_threshold,
    )
