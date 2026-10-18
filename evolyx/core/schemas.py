"""Core domain schemas and data structures for EVOLYX."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class IntentType(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    RESEARCH_AND_RETRIEVE = "research_and_retrieve"
    CODE_AND_TOOL = "code_and_tool"
    MULTI_STEP_WORKFLOW = "multi_step_workflow"
    SYSTEM_EVOLUTION = "system_evolution"


class MemoryKind(str, Enum):
    VERIFIED_PROCEDURE = "verified_procedure"
    USER_PREFERENCE = "user_preference"
    DOMAIN_FACT = "domain_fact"
    SYSTEM_LESSON = "system_lesson"


class EvaluationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    EVALUATION_ERROR = "evaluation_error"
    REJECTED = "rejected"


class VerificationStatus(str, Enum):
    OBSERVED = "observed"
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    APPROVED_FOR_TRAINING = "approved_for_training"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    # Legacy aliases
    PENDING = "pending"
    APPROVED = "approved"
    CONTRADICTED = "contradicted"


@dataclass
class Message:
    role: str  # "user", "assistant", "system", "tool"
    content: str
    name: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class PlanStep:
    step_id: int
    description: str
    tool_name: Optional[str] = None
    tool_arguments: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)
    completed: bool = False
    result: Optional[str] = None


@dataclass
class Plan:
    intent: IntentType
    reasoning: str
    steps: List[PlanStep] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class Evidence:
    source: str
    content: str
    verification_method: str = "automated_assertion"
    confidence: float = 1.0


@dataclass
class ToolExecutionResult:
    tool_name: str
    success: bool
    output: Any
    evidence: List[Evidence] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class Assessment:
    passed: bool
    score: float
    feedback: str
    status: EvaluationStatus = EvaluationStatus.UNVERIFIED
    factuality_score: float = 1.0
    regression_risk: float = 0.0


@dataclass
class TaskResult:
    answer: str
    intent: IntentType
    evidence: List[Evidence] = field(default_factory=list)
    success: bool = True
    learning_candidate_id: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class MemoryRecord:
    content: str
    kind: MemoryKind
    source: str
    confidence: float
    status: VerificationStatus
    user_id: str = "default_user"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_verified_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    task_id: str
    user_id: str
    prompt: str
    messages: List[Message]
    tool_calls: List[ToolExecutionResult]
    result: TaskResult
    assessment: Assessment
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    eligible_for_training: bool = False


@dataclass
class CandidateModel:
    model_id: str
    base_model: str
    adapter_path: Optional[str]
    training_dataset_id: str
    hyperparameters: Dict[str, Any]
    status: str = "created"  # created, evaluating, promoted, rejected
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class EvaluationReport:
    model_id: str
    overall_score: float
    metric_scores: Dict[str, float]
    passed_release_gates: bool
    regression_detected: bool
    details: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class EvolutionProposal:
    proposal_id: str
    level: str  # "level1_prompt", "level2_weights", "level3_software", "level4_architecture"
    hypothesis: str
    proposed_change: Dict[str, Any]
    baseline_version: str
    approved: bool = False
    status: str = "pending"  # pending, testing, promoted, rejected, rolled_back
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
