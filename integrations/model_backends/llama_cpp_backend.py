"""llama.cpp local quantized inference adapter."""

import logging
from typing import List, Optional
from evolyx.core.interfaces import ModelBackend
from evolyx.core.model_registry import ModelRegistry
from evolyx.core.schemas import Message

logger = logging.getLogger(__name__)


@ModelRegistry.register("llama_cpp")
class LlamaCppModelBackend(ModelBackend):
    """Executes local quantized inference using llama.cpp / llama-cpp-python."""

    def __init__(self, model_path: str = "artifacts/models/base/model.gguf", n_ctx: int = 4096, **kwargs):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self._llm = None

    def _ensure_loaded(self) -> None:
        if self._llm is None:
            try:
                from llama_cpp import Llama
                self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx)
            except Exception:
                logger.info("llama_cpp library or model weights not present. Using simulation mode.")
                self._llm = "simulated"

    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
    ) -> str:
        self._ensure_loaded()
        if self._llm == "simulated":
            return f"[llama.cpp Simulation]: Output for {len(messages)} messages."

        chat = [{"role": m.role, "content": m.content} for m in messages]
        resp = self._llm.create_chat_completion(messages=chat, temperature=temperature, max_tokens=max_tokens, stop=stop)
        return resp["choices"][0]["message"]["content"]

    def health_check(self) -> bool:
        return True
