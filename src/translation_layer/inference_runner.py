from collections.abc import Mapping
from typing import Literal

import routing.router as router

class InferenceRunner:

    def __init__(
        self,
        task_models: Mapping[str, str] | None,
        *,
        mode: Literal["task", "complexity"] = "task",
        complexity_threshold: float = 0.5,
        starting_model: str = "qwen",
    ) -> None:
        self.task_models: Mapping[str, str] | None = task_models
        self.mode: Literal["task", "complexity"] = mode
        self.complexity_threshold: float = complexity_threshold
        # defaulting starting model to higher param model
        self.current_model: str = starting_model

    def recieve_prompt(self, prompt: str) -> None:
        model = router.route(prompt, mode=self.mode,
                     threshold=self.complexity_threshold,
                     task_models=self.task_models)

        if model != self.current_model:
            # Do translation
            self.current_model = model

        # in poc models will return full output at once
        return self._run_current_model(prompt)

    def _run_current_model(self, prompt) -> None:
        pass

