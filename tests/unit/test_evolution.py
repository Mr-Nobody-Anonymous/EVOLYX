"""Unit tests for hypothesis generation, experimentation, and rollback."""

from evolyx.evolution.experiment import ExperimentRunner
from evolyx.evolution.hypothesis import HypothesisGenerator
from evolyx.evolution.rollback import RollbackManager


def test_hypothesis_generation_and_experiment_run():
    hypo_gen = HypothesisGenerator()
    proposal = hypo_gen.analyze_failures([])
    assert proposal.proposal_id.startswith("prop_")
    assert proposal.level == "level1_prompt"

    runner = ExperimentRunner()
    res = runner.run_experiment(proposal, baseline_version="v001")
    assert res["status"] == "completed"
    assert res["candidate_score"] > res["baseline_score"]
