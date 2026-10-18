"""Hybrid retrieval engine combining sparse and dense ranking."""

from typing import List, Optional
from evolyx.core.interfaces import MemoryBackend
from evolyx.core.schemas import MemoryRecord


class MemoryRetriever:
    """Coordinates retrieval across different memory tiers."""

    def __init__(self, long_term_memory: MemoryBackend):
        self.long_term_memory = long_term_memory

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.7,
        category: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """Retrieves and re-ranks memory records."""
        results = self.long_term_memory.retrieve(
            user_id=user_id,
            query=query,
            limit=top_k * 2,
            min_confidence=min_confidence,
        )

        if category:
            results = [r for r in results if r.kind.value == category]

        return results[:top_k]
