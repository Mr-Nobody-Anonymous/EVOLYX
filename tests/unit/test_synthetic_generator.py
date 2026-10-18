"""Unit tests for VerifiedSyntheticGenerator execution-checked data admission."""

from evolyx.learning.synthetic_generator import VerifiedSyntheticGenerator


def test_synthetic_data_generation_and_sandboxed_filtering():
    generator = VerifiedSyntheticGenerator()

    # 1. Valid code and assertions should be approved
    valid_task = "Write a function to square numbers."
    valid_code = "def square(n):\n    return n * n"
    valid_test = "assert square(4) == 16\nassert square(-3) == 9"

    res_valid = generator.generate_and_verify_coding_example(
        task_prompt=valid_task,
        solution_code=valid_code,
        test_assertions=valid_test,
    )
    assert res_valid is not None
    assert res_valid["status"] == "PASSED"
    assert res_valid["eligible_for_training"] is True
    assert res_valid["verified_solution"] == valid_code

    # 2. Buggy code that fails assertion should be rejected (returns None)
    buggy_code = "def square(n):\n    return n + n"
    res_buggy = generator.generate_and_verify_coding_example(
        task_prompt=valid_task,
        solution_code=buggy_code,
        test_assertions=valid_test,
    )
    assert res_buggy is None
