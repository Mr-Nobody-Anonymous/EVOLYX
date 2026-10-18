"""Unified training adapter bridging EVOLYX to Unsloth, TRL, or PEFT."""

import logging
import uuid
from typing import Any, Dict, Optional
from evolyx.core.interfaces import TrainerBackend
from evolyx.core.schemas import CandidateModel

logger = logging.getLogger(__name__)


class MockTrainerAdapter(TrainerBackend):
    """Safe, fast simulation trainer for development and offline testing."""

    def __init__(self, output_dir: str = "artifacts/models/candidates"):
        self.output_dir = output_dir

    def train_candidate(
        self,
        base_model: str,
        dataset_path: str,
        config: Dict[str, Any],
    ) -> CandidateModel:
        run_id = f"cand_{uuid.uuid4().hex[:8]}"
        logger.info("Executing simulated fine-tuning for candidate %s on %s", run_id, dataset_path)

        candidate = CandidateModel(
            model_id=run_id,
            base_model=base_model,
            adapter_path=f"{self.output_dir}/{run_id}/adapter_model",
            training_dataset_id=dataset_path,
            hyperparameters=config.get("hyperparameters", {}),
            status="created",
        )
        return candidate

    def get_status(self, run_id: str) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "status": "completed",
            "loss": 0.342,
            "epochs_completed": 3,
        }
