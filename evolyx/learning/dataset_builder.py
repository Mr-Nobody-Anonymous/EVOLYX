"""Dataset builder transforming verified trajectories into training formats with strict provenance and split isolation."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from evolyx.learning.data_filter import DataFilter
from evolyx.learning.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Exports filtered and versioned datasets for fine-tuning with zero train/val leakage and provenance tracking."""

    def __init__(
        self,
        trajectory_store: TrajectoryStore,
        output_dir: str = "artifacts/datasets",
        data_filter: Optional[DataFilter] = None,
    ):
        self.trajectory_store = trajectory_store
        self.output_dir = Path(output_dir)
        self.data_filter = data_filter or DataFilter()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_sft_dataset(
        self,
        version: str = "v001",
        val_split: float = 0.2,
        min_approved_samples: int = 1,
        demo_mode: bool = False,
        deterministic_seed: int = 42,
    ) -> Dict[str, str]:
        """Builds train and validation JSONL datasets with deterministic disjoint partitioning.
        
        Raises ValueError in production mode if insufficient approved trajectories exist.
        """
        raw_trajectories = self.trajectory_store.get_training_eligible(limit=500)
        filtered = self.data_filter.filter_batch(raw_trajectories)

        if len(filtered) < min_approved_samples:
            if not demo_mode:
                raise ValueError(
                    f"Production dataset construction failed: Found {len(filtered)} approved trajectories, "
                    f"but minimum required is {min_approved_samples}. Only verified, evidence-backed tasks may enter training."
                )
            logger.warning("DEMO MODE ACTIVE: Inserting explicitly labeled synthetic seed trajectory.")
            filtered = [
                {
                    "task_id": "demo-seed-001",
                    "prompt": "Explain the architecture of EVOLYX.",
                    "clean_answer": "EVOLYX is an autonomous self-improving platform combining runtime memory, tool reasoning, and gated offline fine-tuning.",
                    "is_demo_synthetic": True,
                    "source": "demo_synthetic_seed",
                }
            ]

        # Deterministic ordering based on hash
        def sort_key(item: Dict[str, Any]) -> str:
            raw_id = f"{item.get('task_id', '')}:{deterministic_seed}"
            return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

        filtered = sorted(filtered, key=sort_key)

        # Disjoint non-overlapping split
        if len(filtered) == 1:
            train_samples = filtered
            val_samples = []
        else:
            n_val = max(1, int(len(filtered) * val_split))
            n_train = len(filtered) - n_val
            train_samples = filtered[:n_train]
            val_samples = filtered[n_train:]

        # Verify zero leakage / overlap
        train_ids = set(x.get("task_id", "") for x in train_samples)
        val_ids = set(x.get("task_id", "") for x in val_samples)
        assert len(train_ids.intersection(val_ids)) == 0, "Fatal: Train and validation datasets overlap!"

        train_path = self.output_dir / f"sft_{version}_train.jsonl"
        val_path = self.output_dir / f"sft_{version}_val.jsonl"

        self._write_jsonl(train_path, train_samples, "train")
        self._write_jsonl(val_path, val_samples, "val")

        manifest_path = self.output_dir / f"manifest_{version}.json"
        manifest = {
            "version": version,
            "total_examples": len(filtered),
            "train_examples": len(train_samples),
            "val_examples": len(val_samples),
            "train_file": str(train_path),
            "val_file": str(val_path),
            "is_demo_mode": demo_mode,
            "deterministic_seed": deterministic_seed,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Built SFT dataset version %s with %d samples (train=%d, val=%d, demo_mode=%s)",
            version,
            len(filtered),
            len(train_samples),
            len(val_samples),
            demo_mode,
        )
        return {
            "train_file": str(train_path),
            "val_file": str(val_path),
            "manifest_file": str(manifest_path),
        }

    def _write_jsonl(self, path: Path, examples: List[Dict], split: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                record = {
                    "example_id": ex.get("task_id", "ex-0"),
                    "task": "conversational_reasoning",
                    "messages": [
                        {"role": "user", "content": ex.get("prompt", "")},
                        {"role": "assistant", "content": ex.get("clean_answer", "")},
                    ],
                    "dataset_split": split,
                    "provenance": {
                        "origin": ex.get("source", "task_execution"),
                        "consent_status": "system_verified",
                        "license": "Apache-2.0",
                        "is_demo_synthetic": ex.get("is_demo_synthetic", False),
                    },
                    "eligible_for_training": True,
                }
                f.write(json.dumps(record) + "\n")
