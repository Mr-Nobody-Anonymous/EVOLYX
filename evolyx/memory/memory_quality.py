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
        """Strips API keys, private keys, credentials, emails, and sensitive PII."""
        # Scrub PEM private keys
        text = re.sub(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", "[PRIVATE_KEY_REDACTED]", text)
        # Scrub AWS access keys (e.g. AKIA...)
        text = re.sub(r"\bAKIA[0-9A-Z]{16}\b", "[AWS_KEY_REDACTED]", text)
        # Scrub JWT tokens (3 base64url segments separated by dots)
        text = re.sub(r"\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b", "[JWT_REDACTED]", text)
        # Scrub API keys / tokens (sk-..., sk-ant-..., ghp_..., generic 32-64 hex/b64 keys)
        text = re.sub(r"\b(?:sk-(?:ant-)?[a-zA-Z0-9_-]{20,}|ghp_[a-zA-Z0-9]{20,}|gho_[a-zA-Z0-9]{20,}|glpat-[a-zA-Z0-9_-]{20,}|[a-f0-9]{32,64})\b", "[SECRET_REDACTED]", text)
        # Scrub email addresses
        text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_REDACTED]", text)
        # Scrub Social Security Numbers (SSN: XXX-XX-XXXX)
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", text)
        # Scrub Credit Card numbers (16 digits in 4 groups)
        text = re.sub(r"\b(?:\d{4}[ -]?){3}\d{4}\b", "[CREDIT_CARD_REDACTED]", text)
        # Scrub IPv4 addresses
        text = re.sub(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", "[IP_REDACTED]", text)
        # Scrub phone numbers (US / standard format)
        text = re.sub(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]", text)
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
