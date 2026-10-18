"""Core interfaces and typing protocols for EVOLYX components."""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from evolyx.core.schemas import (
    Assessment,
    CandidateModel,
    EvaluationReport,
    MemoryRecord,
    Message,
    Plan,
    TaskResult,
    ToolExecutionResult,
)


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol for LLM inference backends (Transformers, vLLM, llama.cpp, Mock)."""

    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generates completion text for a list of chat messages."""
        ...

    def health_check(self) -> bool:
        """Returns True if the backend is initialized and healthy."""
        ...


@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol for long-term and short-term memory backends (SQLite, Letta, Mem0)."""

    def remember(
        self,
        record: MemoryRecord,
    ) -> str:
        """Stores a memory record and returns the assigned memory ID."""
        ...

    def retrieve(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.7,
    ) -> List[MemoryRecord]:
        """Retrieves relevant memory records matching query."""
        ...

    def verify_memory(self, memory_id: str, approved: bool) -> bool:
        """Updates the verification status of a recorded memory."""
        ...


@runtime_checkable
class ToolBackend(Protocol):
    """Protocol for tools that can be called by the agent runtime."""

    @property
    def name(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def schema(self) -> Dict[str, Any]:
        ...

    def execute(self, **kwargs: Any) -> ToolExecutionResult:
        ...


@runtime_checkable
class TrainerBackend(Protocol):
    """Protocol for fine-tuning and training candidate models (Unsloth, TRL, PEFT)."""

    def train_candidate(
        self,
        base_model: str,
        dataset_path: str,
        config: Dict[str, Any],
    ) -> CandidateModel:
        """Launches a controlled training run and returns candidate metadata."""
        ...

    def get_status(self, run_id: str) -> Dict[str, Any]:
        """Queries the status and progress of a training run."""
        ...


@runtime_checkable
class EvaluatorBackend(Protocol):
    """Protocol for evaluating candidate models and checking release gates."""

    def evaluate(
        self,
        candidate: CandidateModel,
        baseline_model: Optional[str] = None,
    ) -> EvaluationReport:
        """Evaluates candidate performance against benchmark test suites."""
        ...

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        evidence: List[Any],
    ) -> Assessment:
        """Evaluates an individual answer for factual correctness and adherence."""
        ...


@runtime_checkable
class AgentBackend(Protocol):
    """Protocol for high-level agent runtime orchestrator."""

    def run(self, user_id: str, message: str) -> TaskResult:
        """Executes an agent task cycle and returns verified result."""
        ...
