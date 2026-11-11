"""Per-conversation history and cache ownership for sequential inference."""
import logging

from src.translation_layer.inference_runner import InferenceRunner, MODEL_NAMES


class ConversationManager:
    """Send user messages without manually handing off inference state.

    Chat continuations use ChatML, as supported by both runner checkpoints.
    A runner may be shared by independent managers for sequential calls.
    """

    def __init__(self, runner: InferenceRunner, *, current_model: str = "qwen"):
        runner._validate_model(current_model)
        self.runner = runner
        self.current_model = current_model
        self.history: list[dict[str, str]] = []
        self.kv_cache = None
        self.cached_token_ids: list[int] = []
        self.conversation_text = ""
        self.prefix_boundary = 0

    def send(self, message: str, *, max_new_tokens: int = 100) -> str:
        """Return an answer and commit history only after successful inference.

        Model switches require the runner's cache_adapter. For continued use,
        adapters must tag translated prefixes with exact tandem_token_ids.
        """
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be nonempty text")
        suffix = message
        if self.kv_cache is not None:
            if self.runner.prompt_mode == "chat":
                end = "" if self.kv_cache.tandem_turn_complete else "<|im_end|>"
                suffix = (
                    f"{end}\n<|im_start|>user\n{message}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            else:
                suffix = "\n" + message
        text = self.conversation_text + suffix
        answer, cache = self.runner.receive_prompt(
            message, conversation_text=text, current_model=self.current_model,
            kv_cache=self.kv_cache, is_suffix=self.kv_cache is not None,
            uncached_suffix=suffix if self.kv_cache is not None else None,
            cached_token_ids=list(self.cached_token_ids),
            prefix_boundary=self.prefix_boundary, max_new_tokens=max_new_tokens,
        )
        model = next((alias for alias, name in MODEL_NAMES.items()
                      if name == getattr(cache, "tandem_model_name", None)), None)
        ids = getattr(cache, "tandem_token_ids", None)
        if (model is None or ids is None or len(ids) != cache.get_seq_length()
                or any(type(token) is not int or token < 0 for token in ids)):
            raise ValueError("conversation caches require model provenance and exact tandem_token_ids")
        self.current_model = model
        self.kv_cache = cache
        self.cached_token_ids = list(ids)
        self.conversation_text = text + answer
        self.prefix_boundary = len(self.conversation_text)
        self.history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ])
        logging.info("Current model: %s", self.current_model)
        return answer
