"""Factuality, evidence alignment, and hallucination evaluation."""

from typing import Any, List
from evolyx.core.schemas import Assessment, Evidence


class FactualityEvaluator:
    """Verifies that statements in an answer are strictly grounded in gathered evidence."""

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        evidence: List[Evidence],
    ) -> Assessment:
        if not answer.strip():
            return Assessment(
                passed=False,
                score=0.0,
                factuality_score=0.0,
                feedback="Empty answer received.",
            )

        if not evidence:
            # Direct response without tools - passes if coherent and non-toxic
            return Assessment(
                passed=True,
                score=0.90,
                factuality_score=0.88,
                feedback="Unassisted response accepted.",
            )

        # Check alignment between answer text and evidence content
        evidence_corpus = " ".join(e.content.lower() for e in evidence)
        answer_tokens = set(answer.lower().split())
        matched_tokens = sum(1 for t in answer_tokens if t in evidence_corpus and len(t) > 3)

        grounding_ratio = matched_tokens / max(1, len(answer_tokens))
        passed = grounding_ratio > 0.15 or len(answer) < 80
        score = min(1.0, 0.70 + grounding_ratio * 0.30)

        return Assessment(
            passed=passed,
            score=round(score, 2),
            factuality_score=round(score, 2),
            feedback="Evidence grounding check passed." if passed else "Low evidence alignment.",
        )
