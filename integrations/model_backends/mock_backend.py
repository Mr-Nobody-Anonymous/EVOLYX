"""Mock inference backend for rapid offline testing and CI."""

from typing import List, Optional
from evolyx.core.interfaces import ModelBackend
from evolyx.core.model_registry import ModelRegistry
from evolyx.core.schemas import Message


@ModelRegistry.register("mock")
class MockModelBackend(ModelBackend):
    """Simulated model backend returning contextual responses without GPU usage."""

    def __init__(self, default_response: str = "EVOLYX automated response", **kwargs):
        self.default_response = default_response

    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
    ) -> str:
        last_msg = messages[-1].content if messages else ""
        if "Explain the architecture of EVOLYX" in last_msg:
            return (
                "EVOLYX is an autonomous self-improving platform combining runtime memory, "
                "tool reasoning, and gated offline fine-tuning."
            )
        if "find and fix" in last_msg.lower() or "bug" in last_msg.lower():
            return "The issue was caused by an unhandled edge case. Verified test case passes."

        for m in messages:
            if m.role == "system" and "Gathered Tool Evidence:" in m.content:
                ev_part = m.content.split("Gathered Tool Evidence:")[-1].strip()
                return f"{self.default_response}: Processed query '{last_msg[:60]}'. Result: {ev_part}"

        return f"{self.default_response}: Processed query '{last_msg[:60]}'."

    def health_check(self) -> bool:
        return True
