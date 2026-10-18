"""Unit tests for trajectory storage and dataset builder."""

import tempfile
from pathlib import Path
from evolyx.core.schemas import Assessment, IntentType, TaskResult, Trajectory
from evolyx.learning.dataset_builder import DatasetBuilder
from evolyx.learning.trajectory_store import TrajectoryStore


def test_trajectory_recording_and_sft_dataset_build():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_traj.db")
        store = TrajectoryStore(db_path=db_path)

        traj = Trajectory(
            task_id="task_test_001",
            user_id="user_test",
            prompt="Write a Python binary search function.",
            messages=[],
            tool_calls=[],
            result=TaskResult(
                answer="def binary_search(arr, target): ...",
                intent=IntentType.CODE_AND_TOOL,
                success=True,
            ),
            assessment=Assessment(passed=True, score=0.95, feedback="Correct"),
            eligible_for_training=True,
        )
        store.record(traj)

        eligible = store.get_training_eligible()
        assert len(eligible) == 1

        out_dir = str(Path(tmp_dir) / "out_ds")
        builder = DatasetBuilder(trajectory_store=store, output_dir=out_dir)
        files = builder.build_sft_dataset(version="test_v1")
        assert "train_file" in files
        assert "manifest_file" in files
