"""Unit tests for long-term memory, quality filtering, and lifecycle verification."""

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


def test_memory_lifecycle_transitions_and_superseding():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_mem.db")
        ltm = LongTermMemory(db_path=db_path)

        rec1 = MemoryRecord(
            content="Old approach: use regex parser for HTML extraction.",
            kind=MemoryKind.DOMAIN_FACT,
            source="session_1",
            confidence=0.8,
            status=VerificationStatus.OBSERVED,
            user_id="dev_user",
        )
        id1 = ltm.remember(rec1)

        rec2 = MemoryRecord(
            content="New approach: use BeautifulSoup parser for HTML extraction.",
            kind=MemoryKind.DOMAIN_FACT,
            source="session_2",
            confidence=0.95,
            status=VerificationStatus.APPROVED_FOR_TRAINING,
            user_id="dev_user",
        )
        id2 = ltm.remember(rec2)

        # Transition rec1 to candidate then mark superseded by rec2
        assert ltm.transition_status(id1, VerificationStatus.CANDIDATE, reason="Under review")
        assert ltm.mark_superseded(id1, id2, reason="Regex parser breaks on malformed HTML")

        # In retrieve, superseded memory should be excluded
        results = ltm.retrieve(user_id="dev_user", query="HTML extraction parser")
        retrieved_ids = [r.id for r in results]
        assert id1 not in retrieved_ids
        assert id2 in retrieved_ids

        # Transition rec2 to REJECTED and verify it is also excluded
        assert ltm.transition_status(id2, VerificationStatus.REJECTED, reason="Found security flaw")
        results_after_reject = ltm.retrieve(user_id="dev_user", query="HTML extraction parser")
        assert id2 not in [r.id for r in results_after_reject]


def test_get_training_approved_memories():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_mem.db")
        ltm = LongTermMemory(db_path=db_path)

        rec_candidate = MemoryRecord(
            content="Candidate unverified note",
            kind=MemoryKind.DOMAIN_FACT,
            source="session_1",
            confidence=0.8,
            status=VerificationStatus.CANDIDATE,
            user_id="user_1",
        )
        rec_approved = MemoryRecord(
            content="Verified procedure for zero-shot inference",
            kind=MemoryKind.VERIFIED_PROCEDURE,
            source="session_2",
            confidence=0.98,
            status=VerificationStatus.APPROVED_FOR_TRAINING,
            user_id="user_1",
        )
        ltm.remember(rec_candidate)
        approved_id = ltm.remember(rec_approved)

        approved_list = ltm.get_training_approved_memories()
        assert len(approved_list) == 1
        assert approved_list[0].id == approved_id
        assert approved_list[0].status == VerificationStatus.APPROVED_FOR_TRAINING
