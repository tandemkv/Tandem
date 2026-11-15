"""Route a prompt with the optional NVIDIA classifier and demo policy."""
import argparse

from src.experiments.poc.demo_setup import TASK_MODELS, build_demo_classifier
from src.routing.router import route


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--mode", choices=("task", "complexity"), default="task")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Complexity cutoff (default: 0.5; ignored in task mode)")
    args = parser.parse_args(argv)

    def classify(prompt):
        # This CLI routes one prompt; defer loading until input validation passes.
        return build_demo_classifier()(prompt)

    try:
        print(route(args.prompt, mode=args.mode, threshold=args.threshold,
                    classifier_model=classify, task_models=TASK_MODELS))
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
