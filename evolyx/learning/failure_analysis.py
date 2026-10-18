"""Failure pattern analyzer clustering agent execution errors and weaknesses."""

import json
import logging
from collections import Counter
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class FailurePatternAnalyzer:
    """Classifies error traces and extracts actionable curriculum priorities."""

    ERROR_CATEGORIES = {
        "syntax_error": ["syntaxerror", "invalid syntax", "unexpected indent", "parsing error"],
        "tool_timeout": ["timed out", "timeout", "did not respond", "took too long"],
        "unsupported_claim": ["unsupported", "hallucination", "low evidence", "contradiction"],
        "permission_denied": ["forbidden", "security violation", "blocked", "permission denied"],
        "logic_assertion_failure": ["assertionerror", "test failed", "expected", "got False"],
    }

    def categorize_error(self, error_message: str) -> str:
        msg = error_message.lower()
        for cat, keywords in self.ERROR_CATEGORIES.items():
            if any(k in msg for k in keywords):
                return cat
        return "unclassified_failure"

    def analyze_trajectories(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes a collection of trajectories to uncover systemic failure modes."""
        failed = [t for t in trajectories if not t.get("eligible_for_training", False)]
        total = max(1, len(trajectories))
        failure_rate = len(failed) / total

        error_counts = Counter()
        for t in failed:
            result_raw = t.get("result_json", "{}")
            res_data = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
            err_msg = res_data.get("error") or res_data.get("feedback") or "unknown"
            cat = self.categorize_error(str(err_msg))
            error_counts[cat] += 1

        top_weakness = error_counts.most_common(1)[0][0] if error_counts else "none"

        return {
            "total_trajectories": total,
            "failed_count": len(failed),
            "failure_rate": round(failure_rate, 4),
            "error_distribution": dict(error_counts),
            "primary_weakness": top_weakness,
            "curriculum_recommendation": self._get_recommendation(top_weakness),
        }

    def _get_recommendation(self, weakness: str) -> str:
        recommendations = {
            "syntax_error": "Generate focused training samples on strict Python syntax and AST compliance.",
            "tool_timeout": "Optimize step decomposition to break long tasks into shorter bounded queries.",
            "unsupported_claim": "Increase citation grounding penalty and enforce evidence-checking gates.",
            "permission_denied": "Fine-tune agent to respect filesystem and tool execution boundary rules.",
            "logic_assertion_failure": "Curate algorithmic test-driven pairs with verified unit test assertions.",
        }
        return recommendations.get(weakness, "Continue general reinforcement on diverse tasks.")
