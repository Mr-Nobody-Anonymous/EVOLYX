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


def test_evaluation_suite_real_scoring_and_fail_closed(tmp_path):
    from evolyx.evaluation.evaluation_suite import EvaluationSuite

    bench_file = tmp_path / "bench.json"
    suite = EvaluationSuite(benchmark_file=str(bench_file))

    # 1. Good model answering correctly
    def good_model(prompt: str) -> str:
        return f"Response for {prompt}: s[::-1], web_search, Transformer Reinforcement Learning, refuse."

    good_report = suite.evaluate_model_comprehensive(good_model)
    assert good_report["evaluation_status"] == "COMPLETED"
    assert good_report["overall_mean"] >= 0.8
    assert len(good_report["observed_results"]) == len(suite.test_cases)
    for res in good_report["observed_results"]:
        assert res["status"] == "PASSED"

    # 2. Broken model that crashes or throws exceptions (must fail closed to 0.0)
    def crashing_model(prompt: str) -> str:
        raise RuntimeError("Model inference engine crash")

    crash_report = suite.evaluate_model_comprehensive(crashing_model)
    assert crash_report["overall_mean"] == 0.0
    for res in crash_report["observed_results"]:
        assert res["status"] == "ERROR"
        assert "Model inference engine crash" in str(res["error"])

    # 3. Model comparison against baseline
    def baseline_model(prompt: str) -> str:
        return "I am a simple baseline model with no answers."

    comparison_report = suite.evaluate_model_comprehensive(
        model_callable=good_model,
        baseline_callable=baseline_model,
    )
    assert "baseline_comparison" in comparison_report
    assert comparison_report["baseline_comparison"]["improved"] is True
    assert comparison_report["baseline_comparison"]["delta_mean"] > 0.0
