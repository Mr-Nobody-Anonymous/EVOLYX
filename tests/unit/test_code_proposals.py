"""Unit tests for CodeProposalEngine diff generation, AST validation, and sandboxing."""

import pytest
from pathlib import Path
from evolyx.evolution.code_proposals import CodeProposalEngine


def test_code_proposal_ast_and_diff():
    engine = CodeProposalEngine()

    # 1. AST check for safe code
    safe_code = "def calculate_mean(values):\n    return sum(values) / len(values)\n"
    is_safe, msg = engine.validate_ast(safe_code)
    assert is_safe is True

    # 2. Unified diff generation
    orig_code = "def calculate_mean(values):\n    return 0\n"
    diff = engine.generate_unified_diff(orig_code, safe_code, "math_utils.py")
    assert "--- a/math_utils.py" in diff
    assert "+++ b/math_utils.py" in diff
    assert "+    return sum(values) / len(values)" in diff

    # 3. Invalid syntax should be caught by AST check
    bad_syntax = "def broken(:"
    is_safe_bad, msg_bad = engine.validate_ast(bad_syntax)
    assert is_safe_bad is False
    assert "Syntax Error" in msg_bad
