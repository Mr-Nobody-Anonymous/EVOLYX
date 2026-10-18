"""Letta (formerly MemGPT) memory backend adapter."""

import logging
from typing import Any, Dict, List
from evolyx.core.interfaces import MemoryBackend
from evolyx.core.schemas import MemoryKind, MemoryRecord, VerificationStatus

logger = logging.getLogger(__name__)


class LettaMemoryAdapter(MemoryBackend):
    """Adapts Letta persistent agents for EVOLYX long-term state."""

    def __init__(self, endpoint: str = "http://localhost:8283"):
        self.endpoint = endpoint
        self._local_cache: Dict[str, MemoryRecord] = {}

    def remember(self, record: MemoryRecord) -> str:
        rec_id = record.id or f"letta_{len(self._local_cache)}"
        record.id = rec_id
        self._local_cache[rec_id] = record
        logger.info("Letta adapter stored memory: %s", record.content[:40])
        return rec_id

    def retrieve(
        self, user_id: str, query: str, limit: int = 5, min_confidence: float = 0.7
    ) -> List[MemoryRecord]:
        return list(self._local_cache.values())[:limit]

    def verify_memory(self, memory_id: str, approved: bool) -> bool:
        if memory_id in self._local_cache:
            self._local_cache[memory_id].status = (
                VerificationStatus.APPROVED if approved else VerificationStatus.REJECTED
            )
            return True
        return False
