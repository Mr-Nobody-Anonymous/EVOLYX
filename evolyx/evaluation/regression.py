"""Regression testing framework ensuring candidates do not degrade baseline skills."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RegressionChecker:
    """Compares candidate benchmark scores against production baselines."""

    def __init__(self, baseline_path: str = "artifacts/evaluations/baseline_v001.json", max_drop_pct: float = 2.0):
        self.baseline_path = Path(baseline_path)
        self.max_drop_pct = max_drop_pct
        self.baseline_scores = self._load_or_create_baseline()

    def _load_or_create_baseline(self) -> Dict[str, float]:
        if self.baseline_path.exists():
            with open(self.baseline_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Default standard baseline
        default_baseline = {
            "instruction_following": 0.88,
            "code_synthesis": 0.85,
            "tool_selection": 0.90,
            "factuality": 0.85,
            "overall": 0.87,
        }
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, "w", encoding="utf-8") as f:
            json.dump(default_baseline, f, indent=2)
        return default_baseline

    def check_regression(self, candidate_scores: Dict[str, float]) -> Tuple[bool, str]:
        """Checks if candidate dropped more than max_drop_pct on any tracked metric."""
        for metric, base_score in self.baseline_scores.items():
            cand_score = candidate_scores.get(metric, 0.0)
            drop_pct = ((base_score - cand_score) / base_score) * 100.0
            if drop_pct > self.max_drop_pct:
                msg = f"Regression detected on '{metric}': dropped {drop_pct:.1f}% (baseline={base_score}, candidate={cand_score})"
                logger.warning(msg)
                return True, msg

        return False, "No regression detected across all benchmarks."
