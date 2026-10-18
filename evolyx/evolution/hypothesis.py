"""Hypothesis generator for autonomous self-improvement cycles."""

import logging
import uuid
from typing import Any, Dict, List
from evolyx.core.schemas import EvolutionProposal

logger = logging.getLogger(__name__)


class HypothesisGenerator:
    """Diagnoses performance bottlenecks and generates targeted improvement proposals."""

    def analyze_failures(self, failure_logs: List[Dict[str, Any]]) -> EvolutionProposal:
        """Examines failure trajectories and outputs a bounded hypothesis."""
        if not failure_logs:
            # Propose a standard Level 1 optimization
            return EvolutionProposal(
                proposal_id=f"prop_{uuid.uuid4().hex[:6]}",
                level="level1_prompt",
                hypothesis="Refining prompt instruction formatting will reduce planning ambiguity.",
                proposed_change={
                    "target_component": "evolyx.core.prompts.TASK_PLANNING_PROMPT",
                    "action": "add_few_shot_examples",
                    "parameters": {"temperature": 0.1},
                },
                baseline_version="v001",
            )

        # Check common failure kinds
        tool_failures = [f for f in failure_logs if "tool" in str(f).lower()]
        if len(tool_failures) > len(failure_logs) / 2:
            return EvolutionProposal(
                proposal_id=f"prop_{uuid.uuid4().hex[:6]}",
                level="level2_weights",
                hypothesis="Tool argument formatting has high failure rate; fine-tune adapter on tool execution trajectories.",
                proposed_change={
                    "target_component": "evolyx.training.sft",
                    "action": "train_tool_adapter",
                    "parameters": {"target_modules": ["q_proj", "v_proj"], "lora_r": 16},
                },
                baseline_version="v001",
            )

        return EvolutionProposal(
            proposal_id=f"prop_{uuid.uuid4().hex[:6]}",
            level="level1_prompt",
            hypothesis="Add explicit self-verification step to planner prompt to prevent premature conclusions.",
            proposed_change={
                "target_component": "evolyx.core.prompts",
                "action": "inject_verification_step",
            },
            baseline_version="v001",
        )
