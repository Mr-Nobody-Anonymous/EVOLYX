"""Curriculum scheduler ranking task complexity for staged learning."""

from typing import Dict, List


class CurriculumScheduler:
    """Scores difficulty and stages datasets from simple instructions to multi-step reasoning."""

    @staticmethod
    def score_difficulty(example: Dict) -> float:
        """Assigns difficulty score between 0.0 (simple) and 1.0 (complex)."""
        prompt = example.get("prompt", "")
        answer = example.get("clean_answer", "")

        # Scoring heuristics: token length, tool usage, code blocks
        length_score = min(1.0, (len(prompt) + len(answer)) / 2000.0)
        has_code = 0.3 if "```" in answer else 0.0
        multi_step = 0.2 if any(w in prompt.lower() for w in ["first", "then", "finally", "step"]) else 0.0

        return min(1.0, length_score * 0.5 + has_code + multi_step)

    def sort_by_curriculum(self, examples: List[Dict]) -> List[Dict]:
        """Sorts examples from easiest to hardest."""
        return sorted(examples, key=self.score_difficulty)
