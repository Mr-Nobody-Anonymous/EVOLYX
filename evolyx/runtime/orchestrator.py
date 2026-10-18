"""Main agent runtime orchestrator for EVOLYX with fail-closed assessment and tool failure propagation."""

import logging
import time
from typing import Any, List, Optional
from evolyx.core.interfaces import EvaluatorBackend, MemoryBackend, ModelBackend
from evolyx.core.schemas import (
    Assessment,
    EvaluationStatus,
    Evidence,
    IntentType,
    MemoryKind,
    MemoryRecord,
    Message,
    Plan,
    TaskResult,
    ToolExecutionResult,
    Trajectory,
    VerificationStatus,
)
from evolyx.runtime.intent import IntentClassifier
from evolyx.runtime.planner import Planner
from evolyx.runtime.session import SessionManager
from evolyx.runtime.tool_router import ToolRouter

logger = logging.getLogger(__name__)


class EvolyxAgent:
    """The central intelligent agent integrating reasoning, memory, tools, and learning with fail-closed release gates."""

    def __init__(
        self,
        model: ModelBackend,
        memory: Optional[MemoryBackend] = None,
        planner: Optional[Planner] = None,
        tool_router: Optional[ToolRouter] = None,
        evaluator: Optional[EvaluatorBackend] = None,
        experience_store: Optional[Any] = None,
    ):
        self.model = model
        self.memory = memory
        self.tool_router = tool_router or ToolRouter()
        self.planner = planner or Planner(model=model)
        self.intent_classifier = IntentClassifier(model=model)
        self.evaluator = evaluator
        self.experience_store = experience_store
        self.sessions: dict[str, SessionManager] = {}

    def get_or_create_session(self, user_id: str) -> SessionManager:
        if user_id not in self.sessions:
            self.sessions[user_id] = SessionManager(session_id=f"sess_{user_id}", user_id=user_id)
        return self.sessions[user_id]

    def run(self, user_id: str, message: str) -> TaskResult:
        """Executes an end-to-end task cycle with strict fail-closed safety and tool failure propagation."""
        start_time = time.perf_counter()
        session = self.get_or_create_session(user_id)
        session.add_message(role="user", content=message)

        # 1. Intent Detection
        intent = self.intent_classifier.classify(message)
        logger.info("Classified user message as %s", intent.value)

        # 2. Retrieve Relevant Memories (excluding superseded and rejected records)
        memories: List[MemoryRecord] = []
        if self.memory:
            try:
                memories = self.memory.retrieve(user_id=user_id, query=message, limit=5)
            except Exception as exc:
                logger.warning("Failed to retrieve memories: %s", exc)

        # 3. Formulate Action Plan
        available_tools = self.tool_router.list_tools()
        plan: Plan = self.planner.create_plan(
            message=message,
            intent=intent,
            memories=memories,
            available_tools=available_tools,
        )

        # 4. Execute Plan Steps, Track Tool Errors & Gather Evidence
        gathered_evidence: List[Evidence] = []
        tool_calls: List[ToolExecutionResult] = []
        tool_errors: List[str] = []

        for step in plan.steps:
            if step.tool_name:
                logger.info("Executing plan step %d with tool %s", step.step_id, step.tool_name)
                exec_result = self.tool_router.execute_step(step)
                tool_calls.append(exec_result)

                if not exec_result.success:
                    err = exec_result.error or "Unknown tool execution error"
                    tool_errors.append(f"Tool {step.tool_name} failed: {err}")
                    gathered_evidence.append(
                        Evidence(
                            source=step.tool_name,
                            content=f"[TOOL_FAILURE] {err}",
                            confidence=0.0,
                        )
                    )
                else:
                    gathered_evidence.extend(exec_result.evidence)

        # 5. Generate Model Draft Response
        context_msgs: List[Message] = []
        system_intro = (
            "You are EVOLYX, an advanced autonomous, self-improving AI platform.\n"
            "Provide accurate, verified solutions backed by available evidence."
        )
        if memories:
            mems_text = "\n".join(f"- [{m.kind.value}] {m.content}" for m in memories)
            system_intro += f"\n\nRelevant Long-Term Memories:\n{mems_text}"

        if gathered_evidence:
            ev_text = "\n".join(f"[{e.source}]: {e.content}" for e in gathered_evidence)
            system_intro += f"\n\nGathered Tool Evidence:\n{ev_text}"

        context_msgs.append(Message(role="system", content=system_intro))
        context_msgs.extend(session.get_messages())

        draft_answer = self.model.generate(context_msgs)

        # 6. Objective Assessment & Self-Critique (Strict Fail-Closed Policy)
        has_tool_failure = len(tool_errors) > 0

        if self.evaluator:
            try:
                assessment = self.evaluator.evaluate_answer(
                    question=message,
                    answer=draft_answer,
                    evidence=gathered_evidence,
                )
                # If tools failed, the assessment cannot pass cleanly
                if has_tool_failure:
                    assessment.passed = False
                    assessment.score = min(assessment.score, 0.40)
                    assessment.feedback = f"Penalized for tool failures: {'; '.join(tool_errors)}"
                    assessment.status = EvaluationStatus.REJECTED
                else:
                    assessment.status = (
                        EvaluationStatus.VERIFIED if assessment.passed else EvaluationStatus.REJECTED
                    )
            except Exception as exc:
                logger.error("Evaluator assessment failed closed: %s", exc)
                assessment = Assessment(
                    passed=False,
                    score=0.0,
                    feedback=f"Evaluation Error: {exc}",
                    status=EvaluationStatus.EVALUATION_ERROR,
                )
        else:
            # Policy: Never interpret missing verification as success
            assessment = Assessment(
                passed=False,
                score=0.0,
                feedback="Unverified: draft generated without independent evaluation pass. Ineligible for training.",
                status=EvaluationStatus.UNVERIFIED,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Result success requires no tool failure and no evaluation error
        is_task_successful = (not has_tool_failure) and (assessment.status not in (EvaluationStatus.EVALUATION_ERROR, EvaluationStatus.REJECTED))

        result = TaskResult(
            answer=draft_answer,
            intent=intent,
            evidence=gathered_evidence,
            success=is_task_successful,
            latency_ms=elapsed_ms,
        )

        # 7. Record Trajectory in Learning / Experience Store
        session.add_message(role="assistant", content=draft_answer)

        # Strict eligibility: requires VERIFIED status, passed assessment, score >= 0.85, and NO tool errors
        is_training_eligible = (
            assessment.passed
            and assessment.score >= 0.85
            and assessment.status == EvaluationStatus.VERIFIED
            and not has_tool_failure
        )

        if self.experience_store:
            try:
                trajectory = Trajectory(
                    task_id=f"task_{int(time.time() * 1000)}",
                    user_id=user_id,
                    prompt=message,
                    messages=session.get_messages(),
                    tool_calls=tool_calls,
                    result=result,
                    assessment=assessment,
                    eligible_for_training=is_training_eligible,
                )
                self.experience_store.record(trajectory)
            except Exception as exc:
                logger.warning("Failed to record trajectory in experience store: %s", exc)

        # 8. Store Verified Procedures in Memory (strictly requires VERIFIED training-eligible status)
        if self.memory and is_training_eligible and gathered_evidence:
            try:
                new_mem = MemoryRecord(
                    content=f"Verified resolution for '{message[:80]}...': {draft_answer[:200]}...",
                    kind=MemoryKind.VERIFIED_PROCEDURE,
                    source=f"task_{intent.value}",
                    confidence=assessment.score,
                    status=VerificationStatus.APPROVED_FOR_TRAINING,
                    user_id=user_id,
                )
                self.memory.remember(new_mem)
            except Exception as exc:
                logger.warning("Failed to store verified memory: %s", exc)

        return result
