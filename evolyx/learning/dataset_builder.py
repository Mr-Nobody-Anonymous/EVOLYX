"""Dataset builder transforming verified trajectories into training formats."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from evolyx.learning.data_filter import DataFilter
from evolyx.learning.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Exports filtered and versioned datasets for fine-tuning with Unsloth or TRL."""

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
        val_split: float = 0.1,
    ) -> Dict[str, str]:
        """Builds train and validation JSONL datasets following the standard EVOLYX format."""
        raw_trajectories = self.trajectory_store.get_training_eligible(limit=500)
        filtered = self.data_filter.filter_batch(raw_trajectories)

        if not filtered:
            logger.warning("No eligible trajectories found to build SFT dataset.")
            # Generate a minimal seed dataset so downstream pipeline functions
            filtered = [
                {
                    "task_id": "seed-001",
                    "prompt": "Explain the architecture of EVOLYX.",
                    "clean_answer": "EVOLYX is an autonomous self-improving platform combining runtime memory, tool reasoning, and gated offline fine-tuning.",
                }
            ]

        train_path = self.output_dir / f"sft_{version}_train.jsonl"
        val_path = self.output_dir / f"sft_{version}_val.jsonl"

        split_idx = max(1, int(len(filtered) * (1 - val_split)))
        train_samples = filtered[:split_idx]
        val_samples = filtered[split_idx:] if split_idx < len(filtered) else filtered[:1]

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
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Built SFT dataset version %s with %d samples", version, len(filtered))
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
                    "eligible_for_training": True,
                }
                f.write(json.dumps(record) + "\n")
