import logging
logging.basicConfig(level=logging.INFO)

from src.translation_layer.conversation_manager import ConversationManager
from src.translation_layer.inference_runner import InferenceRunner

conversation_manager = ConversationManager(InferenceRunner(mode='task'))
print(f"Model before sending any prompts {conversation_manager.current_model}")
print(f"Prompt: Explain Gravity -> Answer: {conversation_manager.send("Explain gravity.")}")
print(f" Prompt: Hello -> Answer: {conversation_manager.send("Hello")}")
print(f"Model at end of session {conversation_manager.current_model}")