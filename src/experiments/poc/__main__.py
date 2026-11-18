"""Show how Tandem classifies and answers independent prompts."""
import argparse

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

    for prompt in args.prompts or interactive_prompts():
        if args.prompts:
            print(f"\nPrompt: {prompt}")
        answer, cache = runner.receive_prompt(prompt, conversation_text=prompt,
                                              current_model="model_a", max_new_tokens=64)
        print(f"Model: {cache.tandem_model_name}\nAnswer: {answer}")


if __name__ == "__main__":
    main()
