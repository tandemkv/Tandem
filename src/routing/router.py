"""Prompt -> model choice, by task classification or prompt complexity."""
from functools import lru_cache
import math

# Experimental policy; calibrate against measured target-model performance.
TASK_MODELS = {
    "Brainstorming": "smollm",
    "Chatbot": "smollm",
    "Classification": "smollm",
    "Closed QA": "smollm",
    "Code Generation": "qwen",
    "Extraction": "smollm",
    "Open QA": "qwen",
    "Other": "qwen",
    "Rewrite": "smollm",
    "Summarization": "smollm",
    "Text Generation": "smollm",
    "Unknown": "qwen",
}


@lru_cache(maxsize=1)
def _classifier():
    # Import and model loading happen only on the first valid request.
    if __package__:
        from .classifier import PromptClassifier
    else:
        from classifier import PromptClassifier
    return PromptClassifier()


def analyze(prompt: str) -> dict:
    """Return task, task confidence, complexity and its component scores."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty text")
    return _classifier()(prompt)


def route(prompt: str, mode: str = "task", *, threshold: float = 0.5) -> str:
    """Return 'qwen' or 'smollm'; never load or call either target model.

    task: map NVIDIA's predicted task label through TASK_MODELS.
    complexity: choose qwen at or above threshold, otherwise smollm.
    threshold applies only to complexity mode. Both modes use one cached model.
    """
    if mode not in ("task", "complexity"):
        raise ValueError("mode must be 'task' or 'complexity'")
    if mode == "complexity" and (
        not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
        or not math.isfinite(threshold) or not 0 <= threshold <= 1.2
    ):
        raise ValueError("threshold must be a finite number between 0 and 1.2")
    result = analyze(prompt)
    if mode == "task":
        return TASK_MODELS[result["task"]]
    return "qwen" if result["complexity"] >= threshold else "smollm"


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--mode", choices=("task", "complexity"), default="task")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Complexity cutoff (default: 0.5; ignored in task mode)")
    args = parser.parse_args(argv)
    try:
        print(route(args.prompt, mode=args.mode, threshold=args.threshold))
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
