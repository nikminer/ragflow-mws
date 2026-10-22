"""Request-level reasoning overrides preserve stored model settings."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


def _generation_params():
    source = Path(__file__).resolve().parents[5] / "api/apps/restful_apis/_generation_params.py"
    spec = spec_from_file_location("generation_params", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.p1
@pytest.mark.parametrize("effort", ["none", "high", "default"])
def test_request_reasoning_overrides_chat_without_mutating_saved_settings(effort):
    params = _generation_params()
    request = {"reasoning_effort": effort, "reasoning": 3, "temperature": 0.1}
    saved = {"reasoning_effort": "medium", "temperature": 0.5}
    dialog = SimpleNamespace(llm_setting=saved)
    config = params.pop_generation_config(request)
    params.merge_generation_config(dialog, config)
    assert config["reasoning_effort"] == effort
    assert dialog.llm_setting["reasoning_effort"] == effort
    assert saved["reasoning_effort"] == "medium"
    assert request == {"reasoning": 3}
