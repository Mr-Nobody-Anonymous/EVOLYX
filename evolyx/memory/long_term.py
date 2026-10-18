"""Long-term persistent memory store backed by SQLite."""

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from evolyx.core.interfaces import MemoryBackend
from evolyx.core.schemas import MemoryKind, MemoryRecord, VerificationStatus
from evolyx.memory.memory_quality import MemoryQualityFilter

logger = logging.getLogger(__name__)


class LongTermMemory(MemoryBackend):
    """Persistent SQLite-backed storage for verified knowledge and procedures."""

    def __init__(self, db_path: str = "artifacts/evolyx.db", quality_filter: Optional[MemoryQualityFilter] = None):
        self.db_path = db_path
        self.quality_filter = quality_filter or MemoryQualityFilter()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    content TEXT,
                    kind TEXT,
                    source TEXT,
                    confidence REAL,
                    status TEXT,
                    created_at TEXT,
                    last_verified_at TEXT,
                    metadata_json TEXT
                )
            """)
            conn.commit()

    def remember(self, record: MemoryRecord) -> str:
        """Saves a verified, sanitized memory record."""
        is_valid, record = self.quality_filter.process(record)
        if not is_valid:
            raise ValueError("Memory record rejected by quality gate.")

        record_id = record.id or str(uuid.uuid4())
        record.id = record_id

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id, user_id, content, kind, source, confidence, status, created_at, last_verified_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    record.user_id,
                    record.content,
                    record.kind.value,
                    record.source,
                    record.confidence,
                    record.status.value,
                    record.created_at,
                    record.last_verified_at,
                    json.dumps(record.metadata),
                ),
            )
            conn.commit()
        logger.info("Persisted memory %s: %s", record_id, record.content[:50])
        return record_id

    def retrieve(
        self, user_id: str, query: str, limit: int = 5, min_confidence: float = 0.7
    ) -> List[MemoryRecord]:
        """Retrieves relevant memories using keyword matching and confidence score."""
        tokens = [t.lower() for t in query.split() if len(t) > 3]

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Base query on user_id or global user
            cursor.execute(
                """
                SELECT * FROM memories 
                WHERE (user_id = ? OR user_id = 'global' OR user_id = 'default_user')
                  AND confidence >= ?
                  AND status != 'rejected'
                ORDER BY confidence DESC, created_at DESC
                LIMIT 50
                """,
                (user_id, min_confidence),
            )
            rows = cursor.fetchall()

        records: List[MemoryRecord] = []
        for r in rows:
            content = r["content"].lower()
            # Simple relevance scoring based on token overlap
            score = sum(1 for t in tokens if t in content) if tokens else 1
            if score > 0 or not tokens:
                records.append(
                    MemoryRecord(
                        id=r["id"],
                        user_id=r["user_id"],
                        content=r["content"],
                        kind=MemoryKind(r["kind"]),
                        source=r["source"],
                        confidence=r["confidence"],
                        status=VerificationStatus(r["status"]),
                        created_at=r["created_at"],
                        last_verified_at=r["last_verified_at"],
                        metadata=json.loads(r["metadata_json"] or "{}"),
                    )
                )

        return records[:limit]

    def verify_memory(self, memory_id: str, approved: bool) -> bool:
        status = VerificationStatus.APPROVED if approved else VerificationStatus.REJECTED
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE memories SET status = ? WHERE id = ?",
                (status.value, memory_id),
            )
            conn.commit()
            return cursor.rowcount > 0
