"""Unit tests for ContinualLearningAuditor and catastrophic forgetting detection."""

from pathlib import Path
from evolyx.learning.continual_learner import ContinualLearningAuditor, ReplayBufferManager


def test_replay_buffer_manager_anchors(tmp_path: Path):
    buffer_file = tmp_path / "replay.json"
    mgr = ReplayBufferManager(buffer_file=str(buffer_file))
    assert len(mgr.replay_tasks) >= 4
    skills = set(t["skill"] for t in mgr.replay_tasks)
    assert "basic_arithmetic" in skills
    assert "adversarial_refusal" in skills


def test_continual_learning_auditor_pass_and_fail(tmp_path: Path):
    buffer_file = tmp_path / "replay.json"
    mgr = ReplayBufferManager(buffer_file=str(buffer_file))
    auditor = ContinualLearningAuditor(replay_manager=mgr)

    # 1. Candidate that preserves all skills
    good_cand = lambda p: "611, ytivargitna, def square(x): return x*x, refuse cannot"
    good_base = lambda p: "611, ytivargitna, def square(x): return x*x, refuse cannot"

    res_good = auditor.evaluate_forgetting(good_cand, good_base, max_forgetting_tolerance=0.05)
    assert res_good["retention_passed"] is True
    assert res_good["average_forgetting"] == 0.0

    # 2. Candidate that forgot arithmetic and reversed strings
    forgetful_cand = lambda p: "I forgot math and strings, but I refuse dangerous commands"
    res_bad = auditor.evaluate_forgetting(forgetful_cand, good_base, max_forgetting_tolerance=0.05)
    assert res_bad["retention_passed"] is False
    assert res_bad["average_forgetting"] > 0.05
