"""EVOLYX Command Line Interface (CLI)."""

import argparse
import sys
from evolyx.core.model_registry import ModelRegistry
from evolyx.evaluation.factuality import FactualityEvaluator
from evolyx.evolution.experiment import ExperimentRunner
from evolyx.evolution.hypothesis import HypothesisGenerator
from evolyx.learning.trajectory_store import TrajectoryStore
from evolyx.memory.long_term import LongTermMemory
from evolyx.runtime.orchestrator import EvolyxAgent
from evolyx.runtime.tool_router import ToolRouter
from evolyx.tools.document_search import DocumentSearchTool
from evolyx.tools.python_executor import PythonExecutorTool
from evolyx.tools.web_search import WebSearchTool

# Ensure default backend is loaded
import integrations.model_backends.mock_backend  # noqa: F401


def build_default_agent() -> EvolyxAgent:
    model = ModelRegistry.get_instance("mock")
    memory = LongTermMemory(db_path="artifacts/evolyx.db")
    tool_router = ToolRouter()
    tool_router.register_tool(WebSearchTool())
    tool_router.register_tool(DocumentSearchTool())
    tool_router.register_tool(PythonExecutorTool())
    evaluator = FactualityEvaluator()
    experience_store = TrajectoryStore(db_path="artifacts/evolyx.db")

    return EvolyxAgent(
        model=model,
        memory=memory,
        tool_router=tool_router,
        evaluator=evaluator,
        experience_store=experience_store,
    )


def cmd_task(agent: EvolyxAgent, prompt: str) -> None:
    print(f"\n[EVOLYX] Running task: '{prompt}'")
    result = agent.run(user_id="cli_user", message=prompt)
    print("\n--- TASK RESULT ---")
    print(f"Intent:  {result.intent.value}")
    print(f"Success: {result.success}")
    print(f"Latency: {result.latency_ms:.1f} ms")
    if result.evidence:
        print(f"Evidence items: {len(result.evidence)}")
        for e in result.evidence:
            print(f"  [{e.source}] {e.content[:80]}...")
    print(f"\nAnswer:\n{result.answer}\n")


def cmd_chat(agent: EvolyxAgent) -> None:
    print("==================================================")
    print("  EVOLYX Interactive Research Terminal (Type 'exit' to quit)")
    print("==================================================")
    while True:
        try:
            user_input = input("\nuser> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", ":q"]:
                print("Exiting EVOLYX.")
                break
            result = agent.run(user_id="cli_user", message=user_input)
            print(f"\nevolyx [{result.intent.value}]> {result.answer}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def cmd_evolve() -> None:
    print("\n[EVOLYX EVOLUTION ENGINE] Initiating Autonomous Evolution Cycle...")
    hypo_gen = HypothesisGenerator()
    runner = ExperimentRunner()

    proposal = hypo_gen.analyze_failures([])
    print(f"1. Generated Proposal: {proposal.proposal_id} ({proposal.level})")
    print(f"   Hypothesis: {proposal.hypothesis}")

    print("2. Running isolated experiment against baseline...")
    exp_res = runner.run_experiment(proposal, baseline_version="v001")
    print(f"   Baseline Score:  {exp_res['baseline_score']:.3f}")
    print(f"   Candidate Score: {exp_res['candidate_score']:.3f}")
    print(f"   Score Delta:     {exp_res['delta']:+.4f}")
    print(f"   Passed Gates:    {exp_res['passes_gates']}")
    print("Cycle complete.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="EVOLYX AI Research Platform CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Chat command
    subparsers.add_parser("chat", help="Start interactive conversational session")

    # Task command
    task_parser = subparsers.add_parser("task", help="Execute a single goal or query")
    task_parser.add_argument("prompt", type=str, help="The instruction or question")

    # Evolve command
    subparsers.add_parser("evolve", help="Run an autonomous evolution experiment cycle")

    args = parser.parse_args()
    agent = build_default_agent()

    if args.command == "chat":
        cmd_chat(agent)
    elif args.command == "task":
        cmd_task(agent, args.prompt)
    elif args.command == "evolve":
        cmd_evolve()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
