"""Unit tests for EvaluationSuite executing 25+ benchmark battery and paired bootstrap confidence intervals."""

from pathlib import Path
from evolyx.evaluation.evaluation_suite import EvaluationSuite


def test_benchmarks_and_25_task_battery(tmp_path: Path):
    bench_file = tmp_path / "held_out.json"
    suite = EvaluationSuite(benchmark_file=str(bench_file))
    assert len(suite.test_cases) >= 20

    categories = set(c["category"] for c in suite.test_cases)
    assert {"coding", "tool_selection", "factuality", "safety", "reasoning"}.issubset(categories)


def test_paired_bootstrap_confidence_interval():
    suite = EvaluationSuite()
    values = [0.9, 0.95, 0.85, 0.92, 0.88, 0.91, 0.94]
    mean, ci_low, ci_high = suite.compute_bootstrap_confidence_interval(values, n_resamples=500)

    assert 0.80 <= ci_low <= mean
    assert mean <= ci_high <= 1.0


def test_evaluation_suite_real_scoring_and_safety_blocking(tmp_path: Path):
    bench_file = tmp_path / "bench.json"
    suite = EvaluationSuite(benchmark_file=str(bench_file))

    # Candidate answers with correct answers except safety refusal
    unsafe_candidate = lambda prompt: "s[::-1], web_search, Transformer Reinforcement Learning, os.system('rmdir /s /q C:\\Windows'), 42"
    safe_baseline = lambda prompt: "s[::-1], web_search, Transformer Reinforcement Learning, refuse, 42"

    results = suite.evaluate_model_comprehensive(
        model_callable=unsafe_candidate,
        baseline_callable=safe_baseline,
    )

    assert "baseline_comparison" in results
    b_comp = results["baseline_comparison"]
    # Because unsafe candidate executed dangerous payload, safety_preserved must be False
    assert b_comp["safety_preserved"] is False
    assert b_comp["candidate_safety_mean"] < b_comp["baseline_safety_mean"]

    # Verify per-case detailed results exist
    assert len(results["detailed_results"]) == len(suite.test_cases)
    assert all("status" in r for r in results["detailed_results"])


def test_evaluation_fail_closed_on_crashing_candidate(tmp_path: Path):
    bench_file = tmp_path / "bench.json"
    suite = EvaluationSuite(benchmark_file=str(bench_file))

    def crashing_candidate(prompt):
        raise RuntimeError("GPU out of memory or unhandled inference crash")

    results = suite.evaluate_model_comprehensive(
        model_callable=crashing_candidate,
        baseline_callable=lambda p: "Baseline response",
    )

    # Fail closed: must score 0.0 with recorded error status
    assert results["overall_mean"] == 0.0
    assert results["total_errors"] == len(suite.test_cases)
    assert all(r["status"] == "ERROR" for r in results["detailed_results"])
