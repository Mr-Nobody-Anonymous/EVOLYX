"""Unit tests for ExperimentRegistry cryptographic verification, CAS concurrency safety, and journaling."""

import json
from pathlib import Path
import sqlite3
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


def test_registry_promotion_cas_race_protection(tmp_path: Path):
    db_file = tmp_path / "reg.db"
    artifacts_dir = tmp_path / "artifacts"
    registry = ExperimentRegistry(db_path=str(db_file), artifacts_root=str(artifacts_dir))

    # Initialize active version to v001
    pointer_file = artifacts_dir / "models" / "ACTIVE_VERSION.txt"
    pointer_file.parent.mkdir(parents=True, exist_ok=True)
    pointer_file.write_text("v001", encoding="utf-8")

    registry.register_run(
        run_id="cand_test_cas",
        parent_version="v001",
        dataset_id="sft_001",
        config={},
        checkpoint_file=None,
        metrics={},
        benchmark_scores={},
    )

    # Attempt CAS promotion with mismatched expected_previous_version
    success_fail = registry.atomic_promote(
        run_id="cand_test_cas",
        version_tag="v002",
        expected_previous_version="v000_stale",
    )
    assert success_fail is False
    assert registry.get_active_version() == "v001"

    # CAS promotion with matching expected_previous_version
    success_ok = registry.atomic_promote(
        run_id="cand_test_cas",
        version_tag="v002",
        expected_previous_version="v001",
    )
    assert success_ok is True
    assert registry.get_active_version() == "v002"


def test_registry_checksum_and_missing_artifact_validation(tmp_path: Path):
    db_file = tmp_path / "reg.db"
    artifacts_dir = tmp_path / "artifacts"
    registry = ExperimentRegistry(db_path=str(db_file), artifacts_root=str(artifacts_dir))

    dummy_weights = tmp_path / "weights_valid.pt"
    dummy_weights.write_bytes(b"initial_valid_weights")

    registry.register_run(
        run_id="cand_test_chk",
        parent_version="v001",
        dataset_id="sft_001",
        config={},
        checkpoint_file=dummy_weights,
        metrics={},
        benchmark_scores={},
    )

    # Tamper with the checkpoint file (corrupt content)
    dummy_weights.write_bytes(b"tampered_corrupted_weights")
    promoted = registry.atomic_promote(run_id="cand_test_chk", version_tag="v002")
    assert promoted is False

    # Delete file completely
    dummy_weights.unlink()
    promoted_missing = registry.atomic_promote(run_id="cand_test_chk", version_tag="v002")
    assert promoted_missing is False


def test_registry_promotion_journal_and_reconcile(tmp_path: Path):
    db_file = tmp_path / "reg.db"
    artifacts_dir = tmp_path / "artifacts"
    registry = ExperimentRegistry(db_path=str(db_file), artifacts_root=str(artifacts_dir))

    registry.register_run(
        run_id="cand_test_jrn",
        parent_version="v001",
        dataset_id="sft_001",
        config={},
        checkpoint_file=None,
        metrics={},
        benchmark_scores={},
    )

    registry.atomic_promote(run_id="cand_test_jrn", version_tag="v002", reason="Passed automated eval")
    assert registry.get_active_version() == "v002"

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT previous_version, new_version, reason FROM promotion_journal")
        journal_entries = cursor.fetchall()
        assert len(journal_entries) == 1
        assert journal_entries[0][0] == "none"
        assert journal_entries[0][1] == "v002"
        assert journal_entries[0][2] == "Passed automated eval"

    # Simulate disk desync
    pointer_file = artifacts_dir / "models" / "ACTIVE_VERSION.txt"
    pointer_file.write_text("v001_desynced", encoding="utf-8")

    # Reconcile should restore to DB journal version v002
    reconciled = registry.reconcile_active_version()
    assert reconciled == "v002"
    assert registry.get_active_version() == "v002"


def test_registry_health_checked_rollback(tmp_path: Path):
    db_file = tmp_path / "reg.db"
    artifacts_dir = tmp_path / "artifacts"
    registry = ExperimentRegistry(db_path=str(db_file), artifacts_root=str(artifacts_dir))

    registry.register_run(
        run_id="cand_test_rb",
        parent_version="v001",
        dataset_id="sft_001",
        config={},
        checkpoint_file=None,
        metrics={},
        benchmark_scores={},
    )
    registry.atomic_promote(run_id="cand_test_rb", version_tag="v002")
    assert registry.get_active_version() == "v002"

    # Failing health check blocks rollback
    failed_rb = registry.rollback(
        target_version_tag="v001",
        reason="Test rollback",
        health_check_callable=lambda ver: False,
    )
    assert failed_rb is False
    assert registry.get_active_version() == "v002"

    # Passing health check allows rollback
    ok_rb = registry.rollback(
        target_version_tag="v001",
        reason="Test rollback confirmed",
        health_check_callable=lambda ver: True,
    )
    assert ok_rb is True
    assert registry.get_active_version() == "v001"
