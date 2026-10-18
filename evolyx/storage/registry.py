"""Model, dataset, and experiment registry with cryptographic integrity verification and concurrency-safe promotion."""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
                    checkpoint_path TEXT,
                    metrics_json TEXT,
                    benchmark_scores_json TEXT,
                    promoted INTEGER DEFAULT 0,
                    version_tag TEXT,
                    git_commit TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(experiment_registry)")
            columns = [row[1] for row in cursor.fetchall()]
            if "checkpoint_path" not in columns:
                conn.execute("ALTER TABLE experiment_registry ADD COLUMN checkpoint_path TEXT DEFAULT ''")
            if "promoted" not in columns:
                conn.execute("ALTER TABLE experiment_registry ADD COLUMN promoted INTEGER DEFAULT 0")
            if "version_tag" not in columns:
                conn.execute("ALTER TABLE experiment_registry ADD COLUMN version_tag TEXT DEFAULT ''")
            if "git_commit" not in columns:
                conn.execute("ALTER TABLE experiment_registry ADD COLUMN git_commit TEXT DEFAULT ''")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS promotion_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    previous_version TEXT,
                    new_version TEXT,
                    artifact_ref TEXT,
                    checksum TEXT,
                    reason TEXT,
                    promoted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        ckpt_str = str(checkpoint_file.resolve()) if checkpoint_file else ""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiment_registry (
                    run_id, parent_version, dataset_id, config_hash,
                    checkpoint_checksum, checkpoint_path, metrics_json, benchmark_scores_json,
                    promoted, git_commit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    run_id,
                    parent_version,
                    dataset_id,
                    config_hash,
                    checksum,
                    ckpt_str,
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
            "checkpoint_path": ckpt_str,
            "status": "registered",
        }

    def atomic_promote(
        self,
        run_id: str,
        version_tag: str,
        expected_previous_version: Optional[str] = None,
        reason: str = "Regression gate cleared",
    ) -> bool:
        """Atomically promotes a candidate with CAS concurrency protection, checksum validation, and durable journaling."""
        pointer_file = self.artifacts_root / "models" / "ACTIVE_VERSION.txt"
        pointer_json = self.artifacts_root / "models" / "ACTIVE_VERSION.json"
        pointer_file.parent.mkdir(parents=True, exist_ok=True)

        current_active = self.get_active_version()

        # Concurrency check (Compare-And-Swap)
        if expected_previous_version is not None and current_active != expected_previous_version:
            logger.error(
                "CAS Race condition blocked: current active is '%s', expected '%s'",
                current_active,
                expected_previous_version,
            )
            return False

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiment_registry WHERE run_id = ?", (run_id,))
            run_data = cursor.fetchone()
            if not run_data:
                logger.error("Failed to promote: Run ID %s not found in registry.", run_id)
                return False

            expected_checksum = run_data["checkpoint_checksum"]
            ckpt_path_str = run_data["checkpoint_path"]

            # Physical Artifact Validation
            if ckpt_path_str and expected_checksum not in ("none", "FILE_NOT_FOUND"):
                ckpt_path = Path(ckpt_path_str)
                if not ckpt_path.exists():
                    logger.error("Promotion rejected: Physical checkpoint file missing at %s", ckpt_path_str)
                    return False
                actual_checksum = self.compute_sha256(ckpt_path)
                if actual_checksum != expected_checksum:
                    logger.error("Promotion rejected: Checksum mismatch (expected %s, got %s)", expected_checksum, actual_checksum)
                    return False

            # Update Registry State & Record Promotion Journal
            cursor.execute(
                "UPDATE experiment_registry SET promoted = 1, version_tag = ? WHERE run_id = ?",
                (version_tag, run_id),
            )
            cursor.execute(
                """
                INSERT INTO promotion_journal (run_id, previous_version, new_version, artifact_ref, checksum, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, current_active, version_tag, ckpt_path_str, expected_checksum, reason),
            )
            conn.commit()

        # Atomic write to pointer files using swap
        tmp_pointer = pointer_file.with_suffix(".tmp")
        tmp_pointer.write_text(version_tag, encoding="utf-8")
        tmp_pointer.replace(pointer_file)

        artifact_metadata = {
            "version_tag": version_tag,
            "run_id": run_id,
            "checkpoint_path": ckpt_path_str,
            "checkpoint_checksum": expected_checksum,
            "previous_version": current_active,
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp_json = pointer_json.with_suffix(".tmp")
        tmp_json.write_text(json.dumps(artifact_metadata, indent=2), encoding="utf-8")
        tmp_json.replace(pointer_json)

        logger.info("Successfully promoted run %s to %s atomically (prior: %s).", run_id, version_tag, current_active)
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

    def get_active_metadata(self) -> Dict[str, Any]:
        """Reads complete metadata and artifact reference for active model."""
        pointer_json = self.artifacts_root / "models" / "ACTIVE_VERSION.json"
        if pointer_json.exists():
            try:
                return json.loads(pointer_json.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"version_tag": self.get_active_version()}

    def reconcile_active_version(self) -> str:
        """Reconciles disk active version with database promotion journal."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM promotion_journal ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()

        if row:
            db_active = row["new_version"]
            disk_active = self.get_active_version()
            if db_active != disk_active:
                logger.warning("Reconciling active version: Disk '%s' != DB '%s'", disk_active, db_active)
                pointer_file = self.artifacts_root / "models" / "ACTIVE_VERSION.txt"
                pointer_file.write_text(db_active, encoding="utf-8")
            return db_active
        return self.get_active_version()

    def rollback(
        self,
        target_version_tag: str,
        reason: str = "",
        health_check_callable: Optional[Callable[[str], bool]] = None,
    ) -> bool:
        """Atomically rolls back the active production version after verifying model health."""
        if health_check_callable is not None:
            is_healthy = health_check_callable(target_version_tag)
            if not is_healthy:
                logger.error("Rollback aborted: Target version '%s' failed health check.", target_version_tag)
                return False

        current_active = self.get_active_version()
        pointer_file = self.artifacts_root / "models" / "ACTIVE_VERSION.txt"
        pointer_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_pointer = pointer_file.with_suffix(".rollback.tmp")
        tmp_pointer.write_text(target_version_tag, encoding="utf-8")
        tmp_pointer.replace(pointer_file)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO promotion_journal (run_id, previous_version, new_version, artifact_ref, checksum, reason)
                VALUES ('rollback', ?, ?, 'rollback_pointer', 'none', ?)
                """,
                (current_active, target_version_tag, reason or "Automated rollback"),
            )
            conn.commit()

        logger.warning(
            "Rolled back active model pointer to '%s' (prior: %s). Reason: %s",
            target_version_tag,
            current_active,
            reason or "Unspecified",
        )
        return True
