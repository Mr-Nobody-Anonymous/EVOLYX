"""Artifact store managing models, datasets, evaluations, and logs."""

from pathlib import Path
from typing import List


class ArtifactStore:
    """Manages files and directory structures in the artifacts/ tree."""

    def __init__(self, root: str = "artifacts"):
        self.root = Path(root)
        self.models = self.root / "models"
        self.datasets = self.root / "datasets"
        self.evaluations = self.root / "evaluations"
        self.checkpoints = self.root / "checkpoints"
        for p in [self.models, self.datasets, self.evaluations, self.checkpoints]:
            p.mkdir(parents=True, exist_ok=True)

    def list_candidate_models(self) -> List[str]:
        cand_dir = self.models / "candidates"
        if not cand_dir.exists():
            return []
        return [p.name for p in cand_dir.glob("*.json")]

    def list_datasets(self) -> List[str]:
        if not self.datasets.exists():
            return []
        return [p.name for p in self.datasets.glob("*.jsonl")]
