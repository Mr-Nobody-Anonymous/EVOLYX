"""Unit tests for ExperimentRegistry cryptographic verification and atomic promotions."""

import json
from pathlib import Path
from evolyx.storage.registry import ExperimentRegistry


def test_registry_registration_and_retrieval(tmp_path: Path):
    db_file = tmp_path / "reg.db"
    artifacts_dir = tmp_path / "artifacts"
    registry = ExperimentRegistry(db_path=str(db_file), artifacts_root=str(artifacts_dir))

    dummy_weights = tmp_path / "weights.pt"
    dummy_weights.write_bytes(b"dummy_neural_weights_data")

    res = registry.register_run(
        run_id="cand_test_01",
        parent_version="v001",
        dataset_id="sft_001",
        config={"lr": 0.001, "batch_size": 4},
        checkpoint_file=dummy_weights,
        metrics={"loss": 1.25},
        benchmark_scores={"overall": 0.95},
    )

    assert res["status"] == "registered"
    assert len(res["checkpoint_checksum"]) == 64
    assert len(res["config_hash"]) == 64

    # Fetch back
    record = registry.get_run("cand_test_01")
    assert record is not None
    assert record["parent_version"] == "v001"
    assert record["metrics"]["loss"] == 1.25
    assert record["benchmark_scores"]["overall"] == 0.95


def test_registry_promotion_and_rollback(tmp_path: Path):
    db_file = tmp_path / "reg.db"
    artifacts_dir = tmp_path / "artifacts"
    registry = ExperimentRegistry(db_path=str(db_file), artifacts_root=str(artifacts_dir))

    registry.register_run(
        run_id="cand_test_02",
        parent_version="v001",
        dataset_id="sft_002",
        config={},
        checkpoint_file=None,
        metrics={},
        benchmark_scores={},
    )

    # Atomic promotion
    success = registry.atomic_promote(run_id="cand_test_02", version_tag="v002")
    assert success is True
    assert registry.get_active_version() == "v002"

    # Rollback
    registry.rollback("v001", reason="Failed verification")
    assert registry.get_active_version() == "v001"
