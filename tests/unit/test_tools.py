"""Unit tests for tool execution and security sandboxing."""

from evolyx.tools.python_executor import PythonExecutorTool
from evolyx.tools.tool_permissions import ToolPermissionManager
from evolyx.tools.web_search import WebSearchTool


def test_python_executor_safe_code():
    tool = PythonExecutorTool()
    res = tool.execute(code="x = 20 + 22\nprint('Answer is', x)")
    assert res.success is True
    assert "Answer is 42" in str(res.output)


def test_python_executor_blocks_forbidden_calls():
    tool = PythonExecutorTool()
    res = tool.execute(code="import subprocess\nsubprocess.run(['dir'])")
    assert res.success is False
    assert "Security Violation" in str(res.error)


def test_web_search_tool():
    tool = WebSearchTool()
    res = tool.execute(query="continuous reinforcement learning")
    assert res.success is True
    assert len(res.evidence) > 0


def test_python_executor_subprocess_timeout_and_runtime_error():
    tool = PythonExecutorTool()

    # 1. Timeout enforcement in disposable subprocess
    timeout_code = "import time\ntime.sleep(2)"
    res_timeout = tool.execute(code=timeout_code, timeout=0.2)
    assert res_timeout.success is False
    assert "Execution Timeout" in str(res_timeout.error)

    # 2. Runtime exception caught cleanly from subprocess stderr
    div_zero_code = "x = 1 / 0"
    res_err = tool.execute(code=div_zero_code)
    assert res_err.success is False
    assert "ZeroDivisionError" in str(res_err.error)
