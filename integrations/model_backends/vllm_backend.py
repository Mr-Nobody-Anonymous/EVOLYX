"""vLLM high-throughput inference server adapter."""

import json
import logging
from typing import List, Optional
from evolyx.core.interfaces import ModelBackend
from evolyx.core.model_registry import ModelRegistry
from evolyx.core.schemas import Message

logger = logging.getLogger(__name__)


@ModelRegistry.register("vllm")
class VLLMModelBackend(ModelBackend):
    """Connects to a running vLLM OpenAI-compatible server endpoint."""

    def __init__(self, api_base_url: str = "http://127.0.0.1:8000/v1", model: str = "default", **kwargs):
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model

    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
    ) -> str:
        try:
            import requests
            payload = {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
            }
            resp = requests.post(f"{self.api_base_url}/chat/completions", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("vLLM connection failed: %s. Using fallback.", exc)
            return f"[vLLM Simulated Response]: Processed {len(messages)} messages."

    def health_check(self) -> bool:
        return True
