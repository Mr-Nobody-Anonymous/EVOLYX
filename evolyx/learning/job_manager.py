"""Persistent pipeline job manager with state recovery, resumption, and failed artifact cleanup."""

import json
import logging
import shutil
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class JobStage(str, Enum):
    INITIALIZED = "initialized"
    DATASET_READY = "dataset_ready"
    TRAINING_COMPLETE = "training_complete"
    EVALUATION_COMPLETE = "evaluation_complete"
    GATE_EVALUATED = "gate_evaluated"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    FAILED = "failed"


class PipelineJobManager:
    """Manages persistent execution state for training and self-improvement jobs to support fault-tolerant resumption."""

    def __init__(self, jobs_dir: str = "artifacts/jobs"):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _job_file(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def create_job(self, job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initializes persistent record for a new training job."""
        job_data = {
            "job_id": job_id,
            "stage": JobStage.INITIALIZED.value,
            "config": config,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "artifacts": {},
            "metrics": {},
            "error": None,
        }
        self._write_job(job_id, job_data)
        return job_data

    def _write_job(self, job_id: str, data: Dict[str, Any]) -> None:
        target = self._job_file(job_id)
        tmp = target.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(target)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        target = self._job_file(job_id)
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_stage(
        self,
        job_id: str,
        stage: JobStage,
        artifacts: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically transitions job to next execution stage."""
        job_data = self.get_job(job_id) or self.create_job(job_id, {})
        job_data["stage"] = stage.value
        job_data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if artifacts:
            job_data.setdefault("artifacts", {}).update(artifacts)
        if metrics:
            job_data.setdefault("metrics", {}).update(metrics)
        if error:
            job_data["error"] = error

        self._write_job(job_id, job_data)
        logger.info("Job %s transitioned to stage '%s'", job_id, stage.value)
        return job_data

    def cleanup_failed_job(self, job_id: str, candidate_dir: Optional[str] = None) -> None:
        """Cleans up incomplete artifacts and dangling temporary directories from failed jobs."""
        self.update_stage(job_id, JobStage.FAILED, error="Execution failed or aborted.")
        if candidate_dir and Path(candidate_dir).exists():
            try:
                shutil.rmtree(candidate_dir)
                logger.info("Cleaned up orphaned candidate directory: %s", candidate_dir)
            except Exception as e:
                logger.warning("Failed to remove candidate dir %s: %s", candidate_dir, e)
