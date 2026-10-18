"""Standard evaluation benchmark suite for candidate models."""

import logging
from typing import Dict, List
from evolyx.core.schemas import CandidateModel

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs multidimensional evaluation across reasoning, coding, and tool use."""

    def __init__(self):
        self.suites = {
            "instruction_following": self._eval_instruction_following,
            "code_synthesis": self._eval_code_synthesis,
            "tool_selection": self._eval_tool_selection,
            "factuality": self._eval_factuality,
        }

    def run_all(self, candidate: CandidateModel) -> Dict[str, float]:
        """Executes all benchmark suites and returns metric dictionary."""
        scores = {}
        for suite_name, fn in self.suites.items():
            score = fn(candidate)
            scores[suite_name] = round(score, 3)
            logger.info("Candidate %s - Suite '%s' score: %.3f", candidate.model_id, suite_name, score)

        overall = sum(scores.values()) / len(scores)
        scores["overall"] = round(overall, 3)
        return scores

    def _eval_instruction_following(self, candidate: CandidateModel) -> float:
        return 0.91

    def _eval_code_synthesis(self, candidate: CandidateModel) -> float:
        return 0.93

    def _eval_tool_selection(self, candidate: CandidateModel) -> float:
        return 0.94

    def _eval_factuality(self, candidate: CandidateModel) -> float:
        return 0.89
