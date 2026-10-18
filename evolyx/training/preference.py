"""Preference optimization (DPO / ORPO) pipeline runner."""

import logging
from typing import Any, Dict, Optional
from evolyx.core.interfaces import TrainerBackend
from evolyx.core.schemas import CandidateModel
from evolyx.training.trainer_adapter import MockTrainerAdapter

logger = logging.getLogger(__name__)


class PreferencePipeline:
    """Configures Direct Preference Optimization (DPO) runs."""

    def __init__(self, trainer: Optional[TrainerBackend] = None):
        self.trainer = trainer or MockTrainerAdapter()

    def run_dpo(
        self,
        base_model: str,
        preference_dataset_path: str,
        config: Dict[str, Any],
    ) -> CandidateModel:
        logger.info("Starting DPO training on %s with base model %s", preference_dataset_path, base_model)
        return self.trainer.train_candidate(
            base_model=base_model,
            dataset_path=preference_dataset_path,
            config=config,
        )
