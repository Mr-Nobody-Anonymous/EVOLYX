"""Hugging Face PEFT LoRA adapter configuration utility."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PEFTAdapterHelper:
    """Configures LoraConfig parameter matrices for base models."""

    @staticmethod
    def build_lora_config(r: int = 16, lora_alpha: int = 32, target_modules: list = None) -> Dict[str, Any]:
        return {
            "r": r,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules or ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
