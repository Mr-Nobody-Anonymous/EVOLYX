"""Adversarial prompt injection defense and instruction boundary enforcement."""

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PromptInjectionGuard:
    """Detects and mitigates prompt injection, jailbreak attempts, and instruction boundary breakouts."""

    INJECTION_PATTERNS = [
        (r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|directions)\b", "instruction_override"),
        (r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions|rules)\b", "instruction_override"),
        (r"(?i)\byou\s+are\s+now\s+(in\s+developer\s+mode|unrestricted|DAN|jailbroken)\b", "persona_hijack"),
        (r"(?i)\b(system\s+override|override\s+safety\s+filters|disable\s+guardrails)\b", "safety_override"),
        (r"(?i)\b(output|print|display|reveal)\s+(your\s+)?(system\s+prompt|initial\s+instructions|secret\s+instructions)\b", "system_prompt_extraction"),
        (r"(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|\[INST\]|\[/INST\]|<s>|</s>)", "delimiter_smuggling"),
        (r"(?i)\bpretend\s+you\s+have\s+no\s+(rules|limits|guidelines)\b", "persona_hijack"),
    ]

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def check_prompt(self, text: str) -> Tuple[bool, Optional[str], float]:
        """Analyzes text for injection vectors.
        
        Returns:
            Tuple of (is_safe, attack_type, threat_score between 0.0 and 1.0)
        """
        threat_score = 0.0
        detected_types = []

        for pattern, attack_type in self.INJECTION_PATTERNS:
            if re.search(pattern, text):
                threat_score += 0.5
                detected_types.append(attack_type)

        # Check for abnormal delimiter repetitions
        if text.count("```") > 10:
            threat_score += 0.3
            detected_types.append("delimiter_flooding")

        threat_score = min(1.0, threat_score)
        is_safe = threat_score < 0.4 if self.strict_mode else threat_score < 0.7
        primary_threat = detected_types[0] if detected_types else None

        if not is_safe:
            logger.warning(
                "Adversarial prompt injection blocked. Type: %s, Threat score: %.2f",
                primary_threat,
                threat_score,
            )

        return is_safe, primary_threat, threat_score

    def sanitize(self, text: str) -> str:
        """Strips injection delimiters and control tokens."""
        for pattern, _ in self.INJECTION_PATTERNS:
            text = re.sub(pattern, "[FILTERED_INSTRUCTION_OVERRIDE]", text)
        return text
