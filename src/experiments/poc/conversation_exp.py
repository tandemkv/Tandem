import logging

from src.experiments.poc.demo_setup import build_demo_runner
from src.translation_layer.conversation_manager import ConversationManager


def main():
    logging.basicConfig(level=logging.INFO)
    # Supply a mapper to switch a populated conversation.
    conversation_manager = ConversationManager(build_demo_runner())
    print(f"Model before sending any prompts {conversation_manager.current_model}")
    print(f"Prompt: Explain Gravity -> Answer: {conversation_manager.send('Explain gravity.')}")
    print(f"Prompt: Hello -> Answer: {conversation_manager.send('Hello')}")
    print(f"Model at end of session {conversation_manager.current_model}")


if __name__ == '__main__':
    main()
