"""Storage for agent execution trajectories and feedback signals."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional
from evolyx.core.schemas import Assessment, IntentType, TaskResult, Trajectory

logger = logging.getLogger(__name__)


class TrajectoryStore:
    """Stores full execution traces of user requests, tool calls, and evaluations."""

    def __init__(self, db_path: str = "artifacts/evolyx.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    prompt TEXT,
                    messages_json TEXT,
                    tool_calls_json TEXT,
                    result_json TEXT,
                    assessment_score REAL,
                    eligible_for_training INTEGER,
                    created_at TEXT
                )
            """)
            conn.commit()

    def record(self, trajectory: Trajectory) -> None:
        """Saves an agent execution trajectory."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trajectories (
                    task_id, user_id, prompt, messages_json, tool_calls_json,
                    result_json, assessment_score, eligible_for_training, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trajectory.task_id,
                    trajectory.user_id,
                    trajectory.prompt,
                    json.dumps([{"role": m.role, "content": m.content} for m in trajectory.messages]),
                    json.dumps([
                        {"tool": t.tool_name, "success": t.success, "output": str(t.output)}
                        for t in trajectory.tool_calls
                    ]),
                    json.dumps({
                        "answer": trajectory.result.answer,
                        "success": trajectory.result.success,
                        "intent": trajectory.result.intent.value,
                    }),
                    trajectory.assessment.score,
                    1 if trajectory.eligible_for_training else 0,
                    trajectory.created_at,
                ),
            )
            conn.commit()
        logger.info("Recorded trajectory %s (eligible: %s)", trajectory.task_id, trajectory.eligible_for_training)

    def get_training_eligible(self, limit: int = 100) -> List[dict]:
        """Fetches trajectories that passed the quality and assessment threshold."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM trajectories 
                WHERE eligible_for_training = 1
                ORDER BY assessment_score DESC, created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
