"""Supervised fine-tuning pipeline runner."""

import logging
from typing import Any, Dict, Optional
from evolyx.core.interfaces import TrainerBackend
from evolyx.core.schemas import CandidateModel
from evolyx.training.trainer_adapter import MockTrainerAdapter

logger = logging.getLogger(__name__)


class SFTPipeline:
    """Configures and launches Supervised Fine-Tuning (SFT) jobs."""

    def __init__(self, trainer: Optional[TrainerBackend] = None):
        self.trainer = trainer or MockTrainerAdapter()

    def run_sft(
        self,
        base_model: str,
        dataset_manifest: Dict[str, Any],
        training_config: Dict[str, Any],
    ) -> CandidateModel:
        """Launches SFT run and produces a new CandidateModel."""
        train_file = dataset_manifest.get("train_file", "")
        logger.info("Starting SFT Pipeline using %s on dataset %s", base_model, train_file)

        candidate = self.trainer.train_candidate(
            base_model=base_model,
            dataset_path=train_file,
            config=training_config,
        )
        return candidate
