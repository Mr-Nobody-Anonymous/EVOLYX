"""Isolated experiment execution engine."""

import logging
from typing import Any, Dict
from evolyx.core.schemas import EvolutionProposal

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Executes controlled experiments to test evolution hypotheses without touching production."""

    def run_experiment(self, proposal: EvolutionProposal, baseline_version: str) -> Dict[str, Any]:
        """Simulates running the proposal in an isolated test environment."""
        logger.info("Executing experiment for proposal %s (%s)", proposal.proposal_id, proposal.level)

        # Baseline evaluation
        baseline_score = 0.870

        # Simulated outcome based on proposal level
        if proposal.level == "level1_prompt":
            candidate_score = 0.895
        elif proposal.level == "level2_weights":
            candidate_score = 0.912
        elif proposal.level == "level3_software":
            candidate_score = 0.885
        elif proposal.level == "level4_architecture":
            candidate_score = 0.880
        else:
            candidate_score = baseline_score

        delta = candidate_score - baseline_score
        passes_gates = delta > 0.01  # Must demonstrate statistically meaningful gain

        result = {
            "proposal_id": proposal.proposal_id,
            "level": proposal.level,
            "baseline_version": baseline_version,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "delta": round(delta, 4),
            "passes_gates": passes_gates,
            "status": "completed",
        }
        logger.info("Experiment %s result: delta=%.4f (passed=%s)", proposal.proposal_id, delta, passes_gates)
        return result
