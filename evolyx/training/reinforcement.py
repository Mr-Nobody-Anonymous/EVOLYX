"""Reinforcement learning (PPO / GRPO) pipeline runner."""

import logging
from typing import Any, Callable, Dict, Optional
from evolyx.core.interfaces import TrainerBackend
from evolyx.core.schemas import CandidateModel
from evolyx.training.trainer_adapter import MockTrainerAdapter

logger = logging.getLogger(__name__)


class ReinforcementPipeline:
    """Configures PPO/GRPO reinforcement learning runs with reward functions."""

    def __init__(self, trainer: Optional[TrainerBackend] = None):
        self.trainer = trainer or MockTrainerAdapter()

    def run_grpo(
        self,
        base_model: str,
        task_prompts_dataset: str,
        reward_fn: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> CandidateModel:
        logger.info("Starting GRPO multi-turn RL training with base model %s", base_model)
        return self.trainer.train_candidate(
            base_model=base_model,
            dataset_path=task_prompts_dataset,
            config=config or {},
        )
