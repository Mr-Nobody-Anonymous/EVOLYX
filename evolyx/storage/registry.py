"""Model, dataset, and experiment registry with cryptographic integrity verification."""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentRegistry:
    """Tracks training runs, configuration hashes, benchmark records, and promotion lineages."""

    def __init__(self, db_path: str = "artifacts/evolyx.db", artifacts_root: str = "artifacts"):
        self.db_path = db_path
        self.artifacts_root = Path(artifacts_root)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_registry (
                    run_id TEXT PRIMARY KEY,
                    parent_version TEXT,
                    dataset_id TEXT,
                    config_hash TEXT,
                    checkpoint_checksum TEXT,
                    metrics_json TEXT,
                    benchmark_scores_json TEXT,
                    promoted INTEGER DEFAULT 0,
                    git_commit TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Computes SHA256 hex digest of any file."""
        if not file_path.exists():
            return "FILE_NOT_FOUND"
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_dict_hash(data: Dict[str, Any]) -> str:
        """Computes deterministic SHA256 of a configuration dictionary."""
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def register_run(
        self,
        run_id: str,
        parent_version: str,
        dataset_id: str,
        config: Dict[str, Any],
        checkpoint_file: Optional[Path],
        metrics: Dict[str, Any],
        benchmark_scores: Dict[str, float],
        git_commit: str = "unknown",
    ) -> Dict[str, Any]:
        """Persists complete immutable experiment metadata."""
        config_hash = self.compute_dict_hash(config)
        checksum = self.compute_sha256(checkpoint_file) if checkpoint_file else "none"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiment_registry (
                    run_id, parent_version, dataset_id, config_hash,
                    checkpoint_checksum, metrics_json, benchmark_scores_json,
                    promoted, git_commit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    run_id,
                    parent_version,
                    dataset_id,
                    config_hash,
                    checksum,
                    json.dumps(metrics),
                    json.dumps(benchmark_scores),
                    git_commit,
                ),
            )
            conn.commit()

        logger.info("Registered experiment run %s (config_hash=%s, checksum=%s)", run_id, config_hash[:8], checksum[:8])
        return {
            "run_id": run_id,
            "config_hash": config_hash,
            "checkpoint_checksum": checksum,
            "status": "registered",
        }

    def atomic_promote(self, run_id: str, version_tag: str) -> bool:
        """Atomically promotes a registered candidate to active production version."""
        pointer_file = self.artifacts_root / "models" / "ACTIVE_VERSION.txt"
        pointer_file.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE experiment_registry SET promoted = 1 WHERE run_id = ?", (run_id,))
            if cursor.rowcount == 0:
                logger.error("Failed to promote: Run ID %s not found in registry.", run_id)
                return False
            conn.commit()

        # Atomic write to ACTIVE_VERSION.txt using temp file swap
        tmp_pointer = pointer_file.with_suffix(".tmp")
        tmp_pointer.write_text(version_tag, encoding="utf-8")
        tmp_pointer.replace(pointer_file)

        logger.info("Successfully promoted run %s to %s atomically.", run_id, version_tag)
        return True

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiment_registry WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["metrics"] = json.loads(res["metrics_json"] or "{}")
            res["benchmark_scores"] = json.loads(res["benchmark_scores_json"] or "{}")
            return res

    def get_active_version(self) -> str:
        """Reads the currently active production version tag."""
        pointer_file = self.artifacts_root / "models" / "ACTIVE_VERSION.txt"
        if pointer_file.exists():
            return pointer_file.read_text(encoding="utf-8").strip()
        return "none"

    def rollback(self, target_version_tag: str, reason: str = "") -> bool:
        """Atomically rolls back the active production version to a previous version."""
        pointer_file = self.artifacts_root / "models" / "ACTIVE_VERSION.txt"
        pointer_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_pointer = pointer_file.with_suffix(".rollback.tmp")
        tmp_pointer.write_text(target_version_tag, encoding="utf-8")
        tmp_pointer.replace(pointer_file)
        logger.warning(
            "Rolled back active model pointer to '%s'. Reason: %s",
            target_version_tag,
            reason or "Unspecified",
        )
        return True
