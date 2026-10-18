"""Release gating and candidate model promotion logic."""

import logging
from typing import Optional
from evolyx.core.schemas import CandidateModel, EvaluationReport
from evolyx.evaluation.benchmarks import BenchmarkRunner
from evolyx.evaluation.regression import RegressionChecker
from evolyx.training.checkpoints import CheckpointManager

logger = logging.getLogger(__name__)


class PromotionManager:
    """Evaluates candidates against release gates and manages promotion decisions."""

    def __init__(
        self,
        benchmark_runner: Optional[BenchmarkRunner] = None,
        regression_checker: Optional[RegressionChecker] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        min_overall_score: float = 0.88,
    ):
        self.benchmark_runner = benchmark_runner or BenchmarkRunner()
        self.regression_checker = regression_checker or RegressionChecker()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.min_overall_score = min_overall_score

    def evaluate_candidate(self, candidate: CandidateModel) -> EvaluationReport:
        """Executes full benchmark battery and generates formal evaluation report."""
        scores = self.benchmark_runner.run_all(candidate)
        has_regression, reg_msg = self.regression_checker.check_regression(scores)
        overall = scores.get("overall", 0.0)

        passed_gates = (not has_regression) and (overall >= self.min_overall_score)
        details = (
            f"Release Gate Result: {'PASSED' if passed_gates else 'FAILED'}. "
            f"Overall: {overall:.3f} (Required: {self.min_overall_score}). {reg_msg}"
        )

        report = EvaluationReport(
            model_id=candidate.model_id,
            overall_score=overall,
            metric_scores=scores,
            passed_release_gates=passed_gates,
            regression_detected=has_regression,
            details=details,
        )
        return report

    def decide_promotion(self, candidate: CandidateModel, report: EvaluationReport, version_tag: str) -> bool:
        """Promotes candidate if release gates pass."""
        if report.passed_release_gates:
            self.checkpoint_manager.promote_to_production(candidate.model_id, version_tag)
            logger.info("Candidate %s successfully PROMOTED to %s", candidate.model_id, version_tag)
            return True
        else:
            logger.warning("Candidate %s REJECTED: %s", candidate.model_id, report.details)
            return False
