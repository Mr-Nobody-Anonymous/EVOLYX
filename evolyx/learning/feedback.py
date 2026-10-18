"""Feedback collector for human corrections and automated unit tests."""

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Captures direct human rating, explicit correction text, or test signals."""

    def __init__(self, db_path: str = "artifacts/evolyx.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    rating INTEGER, -- +1 (positive) or -1 (negative)
                    correction_text TEXT,
                    source TEXT, -- "user", "unit_test", "critic"
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def submit_feedback(
        self,
        task_id: str,
        rating: int,
        correction_text: Optional[str] = None,
        source: str = "user",
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO feedback (task_id, rating, correction_text, source) VALUES (?, ?, ?, ?)",
                (task_id, rating, correction_text, source),
            )
            # If negative feedback with correction, mark trajectory ineligible or update
            if rating < 0 and not correction_text:
                conn.execute(
                    "UPDATE trajectories SET eligible_for_training = 0 WHERE task_id = ?",
                    (task_id,),
                )
            conn.commit()
        logger.info("Feedback recorded for task %s (rating=%d, source=%s)", task_id, rating, source)
