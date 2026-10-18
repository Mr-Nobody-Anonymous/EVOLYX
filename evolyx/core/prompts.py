"""Standardized and versioned prompt templates for EVOLYX."""

INTENT_CLASSIFICATION_PROMPT = """You are the EVOLYX Intent Analyzer.
Classify the user's intent into exactly ONE of the following categories:
- direct_answer: A straightforward conversational question or creative request that needs no tools.
- research_and_retrieve: Involves facts, external documentation, or domain knowledge lookup.
- code_and_tool: Requires executing code, manipulating files, or calculating programmatic answers.
- multi_step_workflow: Complex objective requiring a structured step-by-step plan.
- system_evolution: Request to inspect, self-redesign, improve, or evaluate EVOLYX itself.

Respond strictly in valid JSON format:
{
  "intent": "<category>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}

User Message:
{message}
"""

TASK_PLANNING_PROMPT = """You are the EVOLYX Strategic Planner.
Given the user goal and available context/memories, decompose the task into a robust, minimal sequence of actionable steps.

Available Tools:
{available_tools}

Relevant Memories:
{memories}

User Request:
{message}

Respond strictly in valid JSON format:
{
  "intent": "{intent}",
  "reasoning": "<strategic overview>",
  "steps": [
    {
      "step_id": 1,
      "description": "<what this step achieves>",
      "tool_name": "<name of tool or null>",
      "tool_arguments": { "<arg_name>": "<arg_value>" },
      "depends_on": []
    }
  ]
}
"""

SELF_EVALUATION_PROMPT = """You are the EVOLYX Objective Evaluator.
Critically evaluate the proposed answer against the user's request and the gathered evidence.
Ensure that:
1. Every factual statement is supported by the evidence.
2. The answer satisfies all user constraints.
3. No hallucinations or dangerous advice are present.

User Request: {question}
Gathered Evidence: {evidence}
Candidate Answer: {answer}

Respond strictly in valid JSON format:
{
  "passed": <true or false>,
  "score": <float between 0.0 and 1.0>,
  "factuality_score": <float between 0.0 and 1.0>,
  "feedback": "<concise constructive feedback>"
}
"""

EVOLUTION_HYPOTHESIS_PROMPT = """You are the EVOLYX Evolution Architect.
Analyze recent failure patterns and low-scoring trajectories from the task history.
Propose a single bounded hypothesis to improve system performance.

Failure Patterns:
{failure_logs}

Levels of Evolution:
- level1_prompt: Modify instructions, tool routing guidance, or system prompts.
- level2_weights: Propose training data filtering, LoRA fine-tuning, or preference updates.
- level3_software: Propose bug fix or refactoring in EVOLYX source code with unit tests.
- level4_architecture: Bounded structural exploration (e.g. adapter rank, expert routing).

Respond strictly in valid JSON format:
{
  "level": "<level>",
  "hypothesis": "<why this change will improve the system>",
  "proposed_change": {
    "target_component": "<module or file>",
    "action": "<action description>",
    "parameters": {}
  },
  "expected_gain": "<metric to improve>",
  "test_plan": "<how to measure the result>"
}
"""
