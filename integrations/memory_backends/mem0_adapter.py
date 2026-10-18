"""Mem0 memory layer adapter."""

import logging
from typing import Dict, List
from evolyx.core.interfaces import MemoryBackend
from evolyx.core.schemas import MemoryRecord, VerificationStatus

logger = logging.getLogger(__name__)


class Mem0MemoryAdapter(MemoryBackend):
    """Adapts Mem0 user/agent memory graph for EVOLYX."""

    def __init__(self, api_key: str = "mock"):
        self.api_key = api_key
        self._memories: Dict[str, MemoryRecord] = {}

    def remember(self, record: MemoryRecord) -> str:
        rec_id = record.id or f"mem0_{len(self._memories)}"
        record.id = rec_id
        self._memories[rec_id] = record
        return rec_id

    def retrieve(
        self, user_id: str, query: str, limit: int = 5, min_confidence: float = 0.7
    ) -> List[MemoryRecord]:
        return list(self._memories.values())[:limit]

    def verify_memory(self, memory_id: str, approved: bool) -> bool:
        if memory_id in self._memories:
            self._memories[memory_id].status = (
                VerificationStatus.APPROVED if approved else VerificationStatus.REJECTED
            )
            return True
        return False
