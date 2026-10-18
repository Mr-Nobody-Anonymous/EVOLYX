"""Unit tests for PipelineJobManager, stage tracking, and failed artifact cleanup."""

from pathlib import Path
from evolyx.learning.job_manager import JobStage, PipelineJobManager


def test_pipeline_job_lifecycle_and_cleanup(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    mgr = PipelineJobManager(jobs_dir=str(jobs_dir))

    job_id = "test_job_101"
    job = mgr.create_job(job_id, {"epochs": 3, "backend": "adapter"})
    assert job["stage"] == JobStage.INITIALIZED.value

    # Update stage to TRAINING_COMPLETE
    updated = mgr.update_stage(
        job_id,
        JobStage.TRAINING_COMPLETE,
        artifacts={"checkpoint": "/tmp/model.pt"},
        metrics={"loss": 0.42},
    )
    assert updated["stage"] == JobStage.TRAINING_COMPLETE
    assert updated["artifacts"]["checkpoint"] == "/tmp/model.pt"
    assert updated["metrics"]["loss"] == 0.42

    # Fetch back
    fetched = mgr.get_job(job_id)
    assert fetched is not None
    assert fetched["stage"] == JobStage.TRAINING_COMPLETE

    # Create dummy candidate dir and test cleanup
    dummy_cand = tmp_path / "cand_101"
    dummy_cand.mkdir()
    (dummy_cand / "incomplete.pt").write_bytes(b"temp")

    mgr.cleanup_failed_job(job_id, str(dummy_cand))
    after_clean = mgr.get_job(job_id)
    assert after_clean["stage"] == JobStage.FAILED.value
    assert not dummy_cand.exists()
