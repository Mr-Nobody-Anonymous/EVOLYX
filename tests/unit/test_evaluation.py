"""Unit tests for benchmarks, regression, and promotion gates."""

import tempfile
from evolyx.core.schemas import CandidateModel
from evolyx.evaluation.benchmarks import BenchmarkRunner
from evolyx.evaluation.promotion import PromotionManager
from evolyx.evaluation.regression import RegressionChecker


def test_benchmarks_and_promotion_decision():
    cand = CandidateModel(
        model_id="cand_test_01",
        base_model="Qwen2.5-0.5B",
        adapter_path=None,
        training_dataset_id="ds_01",
        hyperparameters={},
    )
    runner = BenchmarkRunner()
    scores = runner.run_all(cand)
    assert scores["overall"] >= 0.85

    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_checker = RegressionChecker(baseline_path=f"{tmp_dir}/baseline.json")
        has_reg, _ = reg_checker.check_regression(scores)
        assert has_reg is False

        pm = PromotionManager(
            benchmark_runner=runner,
            regression_checker=reg_checker,
            min_overall_score=0.85,
        )
        report = pm.evaluate_candidate(cand)
        assert report.passed_release_gates is True
