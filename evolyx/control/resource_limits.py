"""Resource usage throttling and quota controls."""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ResourceGovernor:
    """Monitors and caps CPU, memory, token, and disk usage."""

    def __init__(
        self,
        max_tokens_per_minute: int = 50000,
        max_concurrent_training_jobs: int = 1,
        min_free_disk_gb: float = 0.5,
    ):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_concurrent_training_jobs = max_concurrent_training_jobs
        self.min_free_disk_gb = min_free_disk_gb
        self._current_training_jobs = 0

    def acquire_training_slot(self) -> Tuple[bool, str]:
        if self._current_training_jobs >= self.max_concurrent_training_jobs:
            return False, "Max concurrent training jobs reached."
        self._current_training_jobs += 1
        return True, "Slot acquired."

    def release_training_slot(self) -> None:
        if self._current_training_jobs > 0:
            self._current_training_jobs -= 1
