"""Intent classification for incoming user requests."""

import json
import logging
import re
from typing import Optional
from evolyx.core.interfaces import ModelBackend
from evolyx.core.prompts import INTENT_CLASSIFICATION_PROMPT
from evolyx.core.schemas import IntentType, Message

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classifies user input into structured EVOLYX intent types."""

    def __init__(self, model: Optional[ModelBackend] = None):
        self.model = model

    def classify(self, message: str) -> IntentType:
        """Determines the intent of the message."""
        text = message.strip().lower()

        # Fast rule-based heuristics first
        if any(w in text for w in ["evolve", "self-improve", "architecture", "redesign", "propose change", "benchmark candidate"]):
            return IntentType.SYSTEM_EVOLUTION

        if any(w in text for w in ["run python", "execute", "write file", "calculate", "def ", "import "]):
            return IntentType.CODE_AND_TOOL

        if any(w in text for w in ["search for", "look up", "find docs", "retrieve", "what is the latest"]):
            return IntentType.RESEARCH_AND_RETRIEVE

        if any(w in text for w in ["step by step", "first do", "pipeline", "workflow"]):
            return IntentType.MULTI_STEP_WORKFLOW

        # If LLM model is provided, use prompt-based classification
        if self.model and len(text) > 20:
            try:
                prompt = INTENT_CLASSIFICATION_PROMPT.format(message=message)
                response = self.model.generate([Message(role="user", content=prompt)], temperature=0.0)
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    intent_str = data.get("intent")
                    for it in IntentType:
                        if it.value == intent_str:
                            return it
            except Exception as exc:
                logger.warning("LLM intent parsing failed, falling back to heuristics: %s", exc)

        return IntentType.DIRECT_ANSWER
