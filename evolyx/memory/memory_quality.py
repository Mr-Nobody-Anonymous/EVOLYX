"""Memory quality filter and PII scrubbing."""

import logging
import re
from typing import Tuple
from evolyx.core.schemas import MemoryRecord, VerificationStatus

logger = logging.getLogger(__name__)


class MemoryQualityFilter:
    """Sanitizes, checks confidence, and validates memory records prior to storage."""

    def __init__(self, min_confidence: float = 0.75, require_verification: bool = False):
        self.min_confidence = min_confidence
        self.require_verification = require_verification

    def sanitize_text(self, text: str) -> str:
        """Strips API keys, emails, and sensitive patterns."""
        # Scrub email addresses
        text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_REDACTED]", text)
        # Scrub API keys / tokens (generic hex or base64 keys > 24 chars)
        text = re.sub(r"\b(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|[a-f0-9]{32,64})\b", "[SECRET_REDACTED]", text)
        # Scrub IPv4 addresses
        text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP_REDACTED]", text)
        return text

    def validate(self, record: MemoryRecord) -> Tuple[bool, str]:
        """Returns (is_valid, reason)."""
        if record.confidence < self.min_confidence:
            return False, f"Confidence {record.confidence:.2f} below required {self.min_confidence:.2f}"

        if self.require_verification and record.status != VerificationStatus.APPROVED:
            return False, f"Memory status {record.status.value} is not approved"

        if len(record.content.strip()) < 5:
            return False, "Memory content too short or empty"

        return True, "Valid"

    def process(self, record: MemoryRecord) -> Tuple[bool, MemoryRecord]:
        """Sanitizes and evaluates the memory record."""
        record.content = self.sanitize_text(record.content)
        valid, reason = self.validate(record)
        if not valid:
            logger.warning("Rejected memory record: %s", reason)
            return False, record
        return True, record
