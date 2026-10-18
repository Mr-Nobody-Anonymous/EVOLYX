"""Unified inference execution pipeline for EVOLYX."""

import logging
import time
from typing import List, Optional
from evolyx.core.interfaces import ModelBackend
from evolyx.core.schemas import Message

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Wraps model backend calls with telemetry, timing, and formatting."""

    def __init__(self, backend: ModelBackend):
        self.backend = backend

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Executes a chat completion with performance tracking."""
        start_time = time.perf_counter()
        try:
            response = self.backend.generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug("Inference completed in %.2f ms", elapsed_ms)
            return response
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error("Inference failed after %.2f ms: %s", elapsed_ms, exc)
            raise

    def is_healthy(self) -> bool:
        """Checks if the underlying backend is ready."""
        return self.backend.health_check()
