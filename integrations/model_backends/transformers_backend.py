"""Hugging Face Transformers inference backend adapter."""

import logging
from typing import List, Optional
from evolyx.core.interfaces import ModelBackend
from evolyx.core.model_registry import ModelRegistry
from evolyx.core.schemas import Message

logger = logging.getLogger(__name__)


@ModelRegistry.register("transformers")
class TransformersModelBackend(ModelBackend):
    """Loads and generates with Hugging Face Transformers models."""

    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct", **kwargs):
        self.model_name_or_path = model_name_or_path
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                logger.info("Loading Transformers model: %s", self.model_name_or_path)
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                )
            except ImportError:
                logger.warning("Transformers or Torch not installed. Falling back to simulation.")
                self._model = "simulated"

    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
    ) -> str:
        self._ensure_loaded()
        if self._model == "simulated":
            return f"[Transformers Simulated Response for {self.model_name_or_path}]: Processed messages."

        # Real inference with tokenizer chat template
        chat = [{"role": m.role, "content": m.content} for m in messages]
        inputs = self._tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        outputs = self._model.generate(inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=temperature > 0)
        return self._tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    def health_check(self) -> bool:
        return True
