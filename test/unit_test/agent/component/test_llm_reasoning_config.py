"""Test LLM parameter serialization without loading the service runtime."""

import ast
from pathlib import Path

import pytest


def _parameter_class():
    # LLMParam only needs its base constructor for these pure configuration
    # methods. Load the owning class without importing databases and workers.
    source = Path(__file__).resolve().parents[4] / "agent/component/llm.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "LLMParam")
    namespace = {"ComponentParamBase": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 - checked-in parameter class only
    return namespace["LLMParam"]


@pytest.mark.p1
def test_agents_keep_independent_reasoning_efforts():
    parameter = _parameter_class()
    planner = parameter()
    planner.reasoning_effort = "high"
    writer = parameter()
    writer.reasoning_effort = "low"
    assert planner.gen_conf()["reasoning_effort"] == "high"
    assert writer.gen_conf()["reasoning_effort"] == "low"
    writer.reasoning_effort = "default"
    assert "reasoning_effort" not in writer.gen_conf()
    assert "reasoning_effort" not in parameter().gen_conf()
