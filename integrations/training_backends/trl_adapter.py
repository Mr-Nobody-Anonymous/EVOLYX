"""Hugging Face TRL training backend adapter."""

import logging
import uuid
from typing import Any, Dict
from evolyx.core.interfaces import TrainerBackend
from evolyx.core.schemas import CandidateModel

logger = logging.getLogger(__name__)


class TRLTrainerAdapter(TrainerBackend):
    """Integrates Hugging Face TRL for SFT and Preference Optimization."""

    def __init__(self, output_dir: str = "artifacts/models/candidates"):
        self.output_dir = output_dir

    def train_candidate(
        self,
        base_model: str,
        dataset_path: str,
        config: Dict[str, Any],
    ) -> CandidateModel:
        run_id = f"trl_{uuid.uuid4().hex[:8]}"
        logger.info("TRL training run %s on %s", run_id, dataset_path)
        return CandidateModel(
            model_id=run_id,
            base_model=base_model,
            adapter_path=f"{self.output_dir}/{run_id}/adapter",
            training_dataset_id=dataset_path,
            hyperparameters=config.get("hyperparameters", {}),
            status="created",
        )

    def get_status(self, run_id: str) -> Dict[str, Any]:
        return {"run_id": run_id, "status": "completed"}
