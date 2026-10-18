"""Safe automated and manual rollback manager."""

import logging
from typing import Optional
from evolyx.training.checkpoints import CheckpointManager

logger = logging.getLogger(__name__)


class RollbackManager:
    """Restores stable previous checkpoints or configurations when anomalies occur."""

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()

    def execute_rollback(self, target_version: str, reason: str) -> bool:
        """Executes a verified rollback to a prior production checkpoint."""
        logger.warning("Initiating rollback to %s due to: %s", target_version, reason)
        try:
            self.checkpoint_manager.rollback(target_version)
            logger.info("Rollback to %s completed successfully.", target_version)
            return True
        except Exception as exc:
            logger.critical("Rollback to %s FAILED: %s", target_version, exc)
            return False
