"""Prompt -> model alias, using a supplied classifier and routing policy."""
from collections.abc import Mapping
import logging
import math

from src.routing.interfaces import ClassificationResult, Classifier


def _validate_prompt(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty text")


def analyze(prompt: str, *, classifier_model: Classifier) -> ClassificationResult:
    """Return the supplied classifier's predictions without loading a model."""
    _validate_prompt(prompt)
    return classifier_model(prompt)


def validate_configuration(*, mode, threshold, task_models, model_a="model_a", model_b="model_b"):
    """Validate routing settings before constructing a runner or invoking a classifier."""
    if mode not in ("task", "complexity"):
        raise ValueError("mode must be 'task' or 'complexity'")
    if mode == "task" and task_models is None:
        raise ValueError("task_models is required for task routing")
    if task_models is not None:
        if not isinstance(task_models, Mapping):
            raise TypeError("task_models must be a mapping of task labels to model aliases")
        for task, alias in task_models.items():
            if not isinstance(task, str) or not task.strip():
                raise ValueError("task_models labels must be nonempty strings")
            if alias not in (model_a, model_b):
                raise ValueError(f"unknown model {alias!r}; choose from {model_a}, {model_b}")
    if mode == "complexity" and (
        not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
        or not math.isfinite(threshold)
    ):
        raise ValueError("threshold must be a finite number on the classifier's score scale")


def route(prompt: str, mode: str = "task", *, classifier_model: Classifier,
          threshold: float = 0.5, task_models: Mapping[str, str] | None = None,
          model_a: str = "model_a", model_b: str = "model_b") -> str:
    """Return an answer-model alias without loading or calling an answer model.

    Task mode requires an explicit task-label-to-alias mapping. Complexity mode
    selects model_a at or above threshold, otherwise model_b. The threshold uses
    the supplied classifier's score scale. Classifier failures fall back to
    model_a; invalid configuration or missing prediction fields raise errors.
    """
    validate_configuration(mode=mode, threshold=threshold, task_models=task_models,
                           model_a=model_a, model_b=model_b)
    if not callable(classifier_model):
        raise TypeError("classifier_model must be callable")
    _validate_prompt(prompt)
    try:
        result = analyze(prompt, classifier_model=classifier_model)
    except Exception:
        logging.warning("Classifier failed; falling back to %s", model_a, exc_info=True)
        return model_a
    if mode == "task":
        return task_models[result["task"]]
    return model_a if result["complexity"] >= threshold else model_b
