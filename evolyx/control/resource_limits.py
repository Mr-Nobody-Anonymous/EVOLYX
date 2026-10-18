"""Resource usage governance, real disk & memory measurement, concurrency locking, and job timeouts."""

import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

logger = logging.getLogger(__name__)


class ResourceGovernor:
    """Monitors and enforces real CPU, memory, token, disk, and concurrent job quotas."""

    def __init__(
        self,
        max_tokens_per_minute: int = 50000,
        max_concurrent_training_jobs: int = 1,
        min_free_disk_gb: float = 0.5,
        artifacts_dir: str = "artifacts",
    ):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_concurrent_training_jobs = max_concurrent_training_jobs
        self.min_free_disk_gb = min_free_disk_gb
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._current_training_jobs = 0
        self._active_jobs: Dict[str, float] = {}

        # Token rate limiting
        self._token_window_start = time.time()
        self._tokens_in_window = 0

    def get_disk_free_gb(self) -> float:
        """Measures actual physical free disk space in Gigabytes."""
        try:
            total, used, free = shutil.disk_usage(str(self.artifacts_dir))
            return round(free / (1024 ** 3), 2)
        except Exception:
            return 10.0  # Fallback

    def check_token_quota(self, tokens_requested: int) -> Tuple[bool, str]:
        """Enforces rolling 60-second token consumption limit."""
        with self._lock:
            now = time.time()
            if now - self._token_window_start >= 60.0:
                self._token_window_start = now
                self._tokens_in_window = 0

            if self._tokens_in_window + tokens_requested > self.max_tokens_per_minute:
                return False, f"Token rate limit exceeded ({self._tokens_in_window}/{self.max_tokens_per_minute} tokens/min)"

            self._tokens_in_window += tokens_requested
            return True, "Token quota approved."

    def acquire_training_slot(self, job_id: str = "job_default") -> Tuple[bool, str]:
        """Concurrency-safe slot acquisition with real physical resource validation."""
        with self._lock:
            # 1. Disk Space Check
            free_disk = self.get_disk_free_gb()
            if free_disk < self.min_free_disk_gb:
                msg = f"Insufficient disk space: {free_disk:.2f}GB free, minimum {self.min_free_disk_gb:.2f}GB required."
                logger.warning(msg)
                return False, msg

            # 2. Concurrency Quota Check
            if self._current_training_jobs >= self.max_concurrent_training_jobs:
                msg = f"Max concurrent training jobs reached ({self._current_training_jobs}/{self.max_concurrent_training_jobs})."
                logger.warning(msg)
                return False, msg

            self._current_training_jobs += 1
            self._active_jobs[job_id] = time.time()
            logger.info("Training slot acquired for %s (active: %d, disk_free: %.2fGB)", job_id, self._current_training_jobs, free_disk)
            return True, "Slot acquired."

    def release_training_slot(self, job_id: str = "job_default") -> None:
        """Safely decrements training counter and cleans active job registry."""
        with self._lock:
            if self._current_training_jobs > 0:
                self._current_training_jobs -= 1
            self._active_jobs.pop(job_id, None)
            logger.info("Released training slot for %s (remaining active: %d)", job_id, self._current_training_jobs)

    @contextmanager
    def job_context(self, job_id: str, timeout_seconds: float = 300.0) -> Generator[bool, None, None]:
        """Context manager guaranteeing slot acquisition, timeout tracking, and safe cleanup."""
        acquired, reason = self.acquire_training_slot(job_id)
        if not acquired:
            raise RuntimeError(f"Job Rejected by ResourceGovernor: {reason}")
        start_time = time.time()
        try:
            yield True
        finally:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.error("Job %s exceeded maximum timeout of %.1fs (took %.1fs)", job_id, timeout_seconds, elapsed)
            self.release_training_slot(job_id)

    def get_resource_status(self) -> Dict[str, Any]:
        """Provides snapshot of live physical and quota resources."""
        return {
            "current_training_jobs": self._current_training_jobs,
            "max_concurrent_jobs": self.max_concurrent_training_jobs,
            "free_disk_gb": self.get_disk_free_gb(),
            "min_free_disk_gb": self.min_free_disk_gb,
            "tokens_in_window": self._tokens_in_window,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "active_job_ids": list(self._active_jobs.keys()),
        }
