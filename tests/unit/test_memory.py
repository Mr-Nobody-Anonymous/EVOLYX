"""Unit tests for long-term memory and quality filtering."""

import tempfile
from pathlib import Path
from evolyx.core.schemas import MemoryKind, MemoryRecord, VerificationStatus
from evolyx.memory.long_term import LongTermMemory
from evolyx.memory.memory_quality import MemoryQualityFilter


def test_memory_quality_pii_scrubbing():
    qf = MemoryQualityFilter()
    sanitized = qf.sanitize_text("Contact me at user@example.com with key sk-1234567890abcdef1234567890")
    assert "[EMAIL_REDACTED]" in sanitized
    assert "[SECRET_REDACTED]" in sanitized


def test_long_term_memory_storage_and_retrieval():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_mem.db")
        ltm = LongTermMemory(db_path=db_path)
        record = MemoryRecord(
            content="The tested parsing solution resolved the buffer issue.",
            kind=MemoryKind.VERIFIED_PROCEDURE,
            source="test_run_1",
            confidence=0.95,
            status=VerificationStatus.APPROVED,
            user_id="user_123",
        )
        mem_id = ltm.remember(record)
        assert mem_id is not None

        retrieved = ltm.retrieve(user_id="user_123", query="parsing buffer issue")
        assert len(retrieved) > 0
        assert retrieved[0].id == mem_id
