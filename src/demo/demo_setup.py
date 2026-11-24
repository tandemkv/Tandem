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


def build_demo_runner(*, classifier: Classifier | None = None, kv_cache_mapper: KVCacheMapper | None = None,
                      mode: Literal["task", "complexity"] = "task",
                      complexity_threshold: float = 0.5) -> InferenceRunner:
    from src.models.model import Model
    from src.translation_layer.cache_mapper import QWEN, SMOLLM, CacheMapper

    # The mapper's checkpoints were fit to these revisions.
    qwen = Model(model_name=QWEN[0], revision=QWEN[1])
    smollm = Model(model_name=SMOLLM[0], revision=SMOLLM[1])
    return InferenceRunner(
        model_a=qwen,
        model_b=smollm,
        classifier_model=classifier if classifier is not None else build_demo_classifier(),
        kv_cache_mapper=kv_cache_mapper if kv_cache_mapper is not None else CacheMapper([qwen, smollm]),
        task_models=TASK_MODELS,
        mode=mode,
        complexity_threshold=complexity_threshold,
    )
