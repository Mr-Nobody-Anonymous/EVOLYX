"""Unit tests for FailurePatternAnalyzer and error taxonomy classification."""

from evolyx.learning.failure_analysis import FailurePatternAnalyzer


def test_failure_pattern_clustering_and_curriculum():
    analyzer = FailurePatternAnalyzer()

    # Test error categorization
    assert analyzer.categorize_error("SyntaxError: unexpected EOF while parsing") == "syntax_error"
    assert analyzer.categorize_error("Execution timed out after 30 seconds") == "tool_timeout"
    assert analyzer.categorize_error("Permission denied: restricted action") == "permission_denied"
    assert analyzer.categorize_error("AssertionError: unsupported claim made without evidence") == "unsupported_claim"
    assert analyzer.categorize_error("AssertionError: expected x == 20") == "logic_assertion_failure"

    # Test trajectory batch analysis
    sample_trajectories = [
        {"eligible_for_training": True, "result_json": '{"status": "ok"}'},
        {"eligible_for_training": False, "result_json": '{"error": "SyntaxError: invalid syntax"}'},
        {"eligible_for_training": False, "result_json": '{"error": "SyntaxError: unexpected indent"}'},
        {"eligible_for_training": False, "result_json": '{"feedback": "unsupported claim without facts"}'},
    ]

    analysis = analyzer.analyze_trajectories(sample_trajectories)
    assert analysis["total_trajectories"] == 4
    assert analysis["failed_count"] == 3
    assert analysis["error_distribution"]["syntax_error"] == 2
    assert analysis["error_distribution"]["unsupported_claim"] == 1
    assert analysis["primary_weakness"] == "syntax_error"
    assert "strict Python syntax" in analysis["curriculum_recommendation"]
