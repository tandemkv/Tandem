"""Chat interactively or route independent prompts in batch mode."""
import argparse

from src.translation_layer.conversation_manager import ConversationManager
from .demo_setup import build_demo_classifier, build_demo_runner


def interactive_prompts():
    print("Enter a prompt, or type /quit to end the session.")
    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt == "/quit":
            return
        if prompt:
            yield prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", nargs="*", help="prompts to run in batch mode")
    args = parser.parse_args()
    if any(not prompt.strip() for prompt in args.prompts):
        parser.error("each prompt must contain text")

    try:
        classifier = build_demo_classifier()

        def show_task(prompt):
            result = classifier(prompt)
            print(f"Task: {result['task']}")
            return result

        runner = build_demo_runner(classifier=show_task)
    except (ImportError, OSError) as error:
        parser.exit(1, f"Could not load demo dependencies or checkpoints: {error}\n")

    if not args.prompts:
        conversation = ConversationManager(runner)
        for prompt in interactive_prompts():
            try:
                answer = conversation.send(prompt, max_new_tokens=128)
            except ValueError as error:
                print(f"Could not complete turn: {error}")
                continue
            print(f"Model: {runner.model_names[conversation.current_model]}\nAnswer: {answer}")
        return

    for prompt in args.prompts:
        print(f"\nPrompt: {prompt}")
        # A fresh conversation per prompt frames it as a ChatML turn.
        conversation = ConversationManager(runner)
        answer = conversation.send(prompt, max_new_tokens=128)
        print(f"Model: {runner.model_names[conversation.current_model]}\nAnswer: {answer}")


if __name__ == "__main__":
    main()
