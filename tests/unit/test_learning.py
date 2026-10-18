"""Unit tests for trajectory storage, dataset builder validation, and split isolation."""

import json
import tempfile
from pathlib import Path
import pytest
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


def test_dataset_builder_production_error_on_empty_store():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "empty_traj.db")
        store = TrajectoryStore(db_path=db_path)
        out_dir = str(Path(tmp_dir) / "out_ds")
        builder = DatasetBuilder(trajectory_store=store, output_dir=out_dir)

        # In production mode (demo_mode=False), raising ValueError when no verified data exists
        with pytest.raises(ValueError, match="Production dataset construction failed"):
            builder.build_sft_dataset(version="test_fail", min_approved_samples=1, demo_mode=False)


def test_dataset_builder_demo_mode_fallback():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "empty_traj.db")
        store = TrajectoryStore(db_path=db_path)
        out_dir = str(Path(tmp_dir) / "out_ds")
        builder = DatasetBuilder(trajectory_store=store, output_dir=out_dir)

        files = builder.build_sft_dataset(version="test_demo", min_approved_samples=1, demo_mode=True)
        assert Path(files["train_file"]).exists()

        with open(files["manifest_file"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["is_demo_mode"] is True
        assert manifest["total_examples"] == 1


def test_dataset_builder_disjoint_splits():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "multi_traj.db")
        store = TrajectoryStore(db_path=db_path)

        for i in range(10):
            store.record(Trajectory(
                task_id=f"task_{i:03d}",
                user_id="user_test",
                prompt=f"Task prompt number {i}",
                messages=[],
                tool_calls=[],
                result=TaskResult(
                    answer=f"Correct answer number {i}",
                    intent=IntentType.CODE_AND_TOOL,
                    success=True,
                ),
                assessment=Assessment(passed=True, score=0.95, feedback="Passed"),
                eligible_for_training=True,
            ))

        out_dir = str(Path(tmp_dir) / "out_ds")
        builder = DatasetBuilder(trajectory_store=store, output_dir=out_dir)
        files = builder.build_sft_dataset(version="split_test", val_split=0.3, demo_mode=False)

        # Inspect train and val files
        train_prompts = set()
        with open(files["train_file"], "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                train_prompts.add(item["messages"][0]["content"])

        val_prompts = set()
        with open(files["val_file"], "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                val_prompts.add(item["messages"][0]["content"])

        assert len(train_prompts) > 0
        assert len(val_prompts) > 0
        assert len(train_prompts.intersection(val_prompts)) == 0
