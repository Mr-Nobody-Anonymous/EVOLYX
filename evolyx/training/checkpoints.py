"""Checkpoint and artifact lifecycle manager."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from evolyx.core.schemas import CandidateModel

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages immutable model checkpoints, candidate promotions, and rollbacks."""

    def __init__(self, root_dir: str = "artifacts/models"):
        self.root = Path(root_dir)
        self.production_dir = self.root / "production"
        self.candidates_dir = self.root / "candidates"
        self.rejected_dir = self.root / "rejected"
        for p in [self.production_dir, self.candidates_dir, self.rejected_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def register_candidate(self, candidate: CandidateModel) -> None:
        meta_file = self.candidates_dir / f"{candidate.model_id}.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(candidate.__dict__, f, indent=2)
        logger.info("Saved candidate checkpoint metadata: %s", meta_file)

    def promote_to_production(self, candidate_id: str, version_tag: str) -> Path:
        target = self.production_dir / version_tag
        target.mkdir(parents=True, exist_ok=True)

        meta_file = self.candidates_dir / f"{candidate_id}.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["status"] = "promoted"
            meta["promoted_version"] = version_tag
            with open(target / "model_manifest.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        # Update active pointer
        active_pointer = self.root / "ACTIVE_VERSION.txt"
        with open(active_pointer, "w", encoding="utf-8") as f:
            f.write(version_tag)

        logger.info("Promoted candidate %s to production version %s", candidate_id, version_tag)
        return target

    def get_active_production_version(self) -> Optional[str]:
        pointer = self.root / "ACTIVE_VERSION.txt"
        if pointer.exists():
            return pointer.read_text(encoding="utf-8").strip()
        return None

    def rollback(self, target_version: str) -> None:
        target_path = self.production_dir / target_version
        if not target_path.exists():
            raise FileNotFoundError(f"Cannot rollback: version '{target_version}' does not exist.")
        pointer = self.root / "ACTIVE_VERSION.txt"
        with open(pointer, "w", encoding="utf-8") as f:
            f.write(target_version)
        logger.warning("Rolled back active production version to %s", target_version)
