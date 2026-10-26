"""MWS reasoning controls and streaming contracts, without external services."""

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.llm import chat_model
from rag.llm.chat_model import LENGTH_NOTIFICATION_EN, MWSChat

pytestmark = pytest.mark.p1
PROJECT_URL = "https://gpt.mwsapis.ru/projects/test-project"


@asynccontextmanager
async def _context(value):
    yield value


def _mock_stream(monkeypatch, events):
    async def lines():
        for event in events:
            yield b"data: " + (event.encode() if isinstance(event, str) else json.dumps(event).encode()) + b"\n\n"

    response = SimpleNamespace(status=200, content=lines())
    session = MagicMock()
    session.post.return_value = _context(response)
    monkeypatch.setattr(chat_model.aiohttp, "ClientSession", lambda **kwargs: _context(session))
    return session


def _chunk(delta=None, finish_reason=None):
    return {"choices": [{"delta": delta or {}, "finish_reason": finish_reason}]}


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "max"])
@pytest.mark.parametrize("stream", [False, True])
def test_reasoning_effort_survives_request_cleaning(effort, stream):
    model = MWSChat("token", "custom-deployment", PROJECT_URL)
    config = {"reasoning_effort": effort, "max_tokens": 128, "temperature": 0, "thinking": "enabled"}
    original = deepcopy(config)
    body = model._request_body([{"role": "user", "content": "Hi"}], config, stream=stream)
    assert body["reasoning_effort"] == effort
    assert body["max_completion_tokens"] == 128
    assert "thinking" not in body
    assert config == original


@pytest.mark.parametrize("effort", [None, "default"])
def test_default_effort_leaves_model_default_unchanged(effort):
    model = MWSChat("token", "gpt-oss-120b", PROJECT_URL)
    assert "reasoning_effort" not in model._clean_conf({"reasoning_effort": effort})


@pytest.mark.parametrize("effort", ["", "ultra", "HIGH", 3, True, {}])
def test_invalid_effort_is_rejected(effort):
    model = MWSChat("token", "gpt-oss-120b", PROJECT_URL)
    with pytest.raises(ValueError, match="reasoning_effort"):
        model._clean_conf({"reasoning_effort": effort})


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("with_reasoning", [False, True])
async def test_stream_separates_reasoning_and_answer_and_reports_usage(monkeypatch, field, with_reasoning):
    session = _mock_stream(
        monkeypatch,
        [
            _chunk({field: "First thought"}),
            _chunk(),
            _chunk({field: " then more", "content": "Answer"}),
            _chunk(finish_reason="stop"),
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}},
            "[DONE]",
        ],
    )
    model = MWSChat("token", "gpt-oss-120b", PROJECT_URL)
    output = [
        item
        async for item in model._async_chat_streamly(
            [{"role": "user", "content": "Hi"}],
            {"reasoning_effort": "high"},
            with_reasoning=with_reasoning,
        )
    ]
    text = [value for value, _ in output if value]
    assert text == (["<think>", "First thought", " then more", "</think>", "Answer"] if with_reasoning else ["Answer"])
    assert output[-1] == ("", 18)
    assert sum(tokens for _, tokens in output) == 18
    assert model.last_usage == {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
    body = session.post.call_args.kwargs["json"]
    assert body["reasoning_effort"] == "high"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_stream_yields_first_content_before_reading_later_events(monkeypatch):
    consumed = []

    async def lines():
        consumed.append("first")
        yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n'
        consumed.append("last")
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'

    response = SimpleNamespace(status=200, content=lines())
    session = MagicMock()
    session.post.return_value = _context(response)
    monkeypatch.setattr(chat_model.aiohttp, "ClientSession", lambda **kwargs: _context(session))
    model = MWSChat("token", "gpt-oss-120b", PROJECT_URL)
    stream = model._async_chat_streamly([{"role": "user", "content": "Hi"}], {})
    assert await anext(stream) == ("first", 0)
    assert consumed == ["first"]
    remaining = [item async for item in stream]
    assert remaining[-1][1] > 0


@pytest.mark.asyncio
async def test_reasoning_only_length_stop_is_closed_and_notified(monkeypatch):
    _mock_stream(monkeypatch, [_chunk({"reasoning": "Thinking"}), _chunk(finish_reason="length"), "[DONE]"])
    model = MWSChat("token", "gpt-oss-120b", PROJECT_URL)
    output = [item async for item in model._async_chat_streamly([{"role": "user", "content": "Hi"}], {})]
    assert [text for text, _ in output if text] == ["<think>", "Thinking", "</think>", LENGTH_NOTIFICATION_EN]
    assert output[-1][1] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events, message",
    [
        ([_chunk({"content": "Partial"})], "ended before"),
        ([{"error": {"message": "Quota exceeded"}}], "Quota exceeded"),
    ],
)
async def test_stream_does_not_silently_accept_errors(monkeypatch, events, message):
    _mock_stream(monkeypatch, events)
    model = MWSChat("token", "gpt-oss-120b", PROJECT_URL)
    with pytest.raises(RuntimeError, match=message):
        _ = [item async for item in model._async_chat_streamly([{"role": "user", "content": "Hi"}], {})]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
async def test_nonstreaming_reasoning_is_preserved(field):
    model = MWSChat("token", "gpt-oss-120b", PROJECT_URL)
    model._post_json = AsyncMock(
        return_value={
            "choices": [{"message": {"content": "Answer", field: "Thought"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
    )
    assert await model._async_chat([{"role": "user", "content": "Hi"}], {}) == ("<think>Thought</think>Answer", 3)
    assert await model._async_chat([{"role": "user", "content": "Hi"}], {}, with_reasoning=False) == ("Answer", 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_bound_tools_keep_mws_controls_without_other_provider_extras(stream):
    model = MWSChat("token", "qwen3-6-35b-a3b", PROJECT_URL)
    model.tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    model.max_retries = 0
    model.max_rounds = 0
    delta = SimpleNamespace(content="Answer", reasoning_content="Thought", reasoning=None, tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=delta, delta=delta, finish_reason="stop")], usage=None)

    async def chunks():
        yield response

    create = AsyncMock(return_value=chunks() if stream else response)
    model.async_client.chat.completions.create = create
    config = {"reasoning_effort": "medium", "max_tokens": 128}
    if stream:
        _ = [item async for item in model.async_chat_streamly_with_tools("", [{"role": "user", "content": "Hi"}], config)]
    else:
        await model.async_chat_with_tools("", [{"role": "user", "content": "Hi"}], config)
    request = create.await_args.kwargs
    assert request["reasoning_effort"] == "medium"
    assert request["max_completion_tokens"] == 128
    assert "extra_body" not in request
    assert "enable_thinking" not in request
    assert (request.get("stream_options") == {"include_usage": True}) is stream


@pytest.mark.asyncio
async def test_stream_closes_reasoning_before_reporting_an_upstream_error(monkeypatch):
    _mock_stream(monkeypatch, [_chunk({"reasoning": "Thinking"}), {"error": {"message": "Quota exceeded"}}])
    model = MWSChat("token", "gpt-oss-120b", PROJECT_URL)
    output = []
    with pytest.raises(RuntimeError, match="Quota exceeded"):
        async for item in model._async_chat_streamly([{"role": "user", "content": "Hi"}], {}):
            output.append(item[0])
    assert output == ["<think>", "Thinking", "</think>"]


def _mock_unclosed_stream(monkeypatch, events):
    state = {"closed": False}

    async def lines():
        for event in events:
            yield b"data: " + json.dumps(event).encode() + b"\n\n"
        await asyncio.Event().wait()

    @asynccontextmanager
    async def response_context():
        try:
            yield SimpleNamespace(status=200, content=lines())
        finally:
            state["closed"] = True

    session = MagicMock()
    session.post.return_value = response_context()
    monkeypatch.setattr(chat_model.aiohttp, "ClientSession", lambda **kwargs: _context(session))
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("reported_usage", [False, True])
async def test_terminal_stream_does_not_wait_for_provider_to_close(monkeypatch, reported_usage):
    monkeypatch.setattr(chat_model, "_MWS_STREAM_USAGE_GRACE_SECONDS", 0.02, raising=False)
    events = [_chunk({"reasoning": "Plan", "content": "Answer"}, finish_reason="stop")]
    if reported_usage:
        events.append({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}})
    state = _mock_unclosed_stream(monkeypatch, events)
    model = MWSChat("token", "custom-deployment", PROJECT_URL)

    async def collect():
        return [item async for item in model._async_chat_streamly([{"role": "user", "content": "Hi"}], {})]

    output = await asyncio.wait_for(collect(), timeout=0.5)
    assert [text for text, _ in output if text] == ["<think>", "Plan", "</think>", "Answer"]
    assert state["closed"]
    if reported_usage:
        assert output[-1][1] == 18
    assert output[-1][1] > 0


@pytest.mark.asyncio
async def test_unfinished_stream_has_a_deadline_without_component_timeout_flag(monkeypatch):
    monkeypatch.delenv("ENABLE_TIMEOUT_ASSERTION", raising=False)
    state = _mock_unclosed_stream(monkeypatch, [_chunk({"reasoning": "Plan"})])
    model = MWSChat("token", "custom-deployment", PROJECT_URL)
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0.03")

    async def collect():
        return [item async for item in model._async_chat_streamly([{"role": "user", "content": "Hi"}], {})]

    with pytest.raises(TimeoutError, match="MWS"):
        await asyncio.wait_for(collect(), timeout=0.5)
    assert state["closed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("max_rounds", [0, 1])
async def test_tools_stream_closes_and_preserves_reasoning_including_final_fallback(monkeypatch, max_rounds):
    state = {"closed": False}

    async def chunks():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Answer", reasoning_content="Plan", reasoning=None, tool_calls=None), finish_reason="stop")], usage=None)
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=7, completion_tokens=11, total_tokens=18))
        await asyncio.Event().wait()

    class Stream:
        def __aiter__(self):
            return chunks()

        async def close(self):
            state["closed"] = True

    model = MWSChat("token", "custom-deployment", PROJECT_URL)
    model.tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    model.max_retries = 0
    model.max_rounds = max_rounds
    model.async_client.chat.completions.create = AsyncMock(return_value=Stream())

    async def collect():
        return [item async for item in model.async_chat_streamly_with_tools("", [{"role": "user", "content": "Hi"}], {})]

    output = await asyncio.wait_for(collect(), timeout=0.5)
    assert [item for item in output if isinstance(item, str) and item] == ["<think>", "Plan", "</think>", "Answer"]
    assert output[-1] == 18
    assert state["closed"]


@pytest.mark.asyncio
async def test_tool_call_chunk_does_not_discard_reasoning_or_content():
    model = MWSChat("token", "custom-deployment", PROJECT_URL)
    model.tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    model.max_retries = 0
    model.max_rounds = 1
    model.terminal_tools = {"lookup"}
    model.toolcall_session = SimpleNamespace(tool_call_async=AsyncMock(return_value="Tool result"))
    tool_call = SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(name="lookup", arguments="{}"))

    async def chunks():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Preface", reasoning_content="Plan", tool_calls=[tool_call]), finish_reason="tool_calls")], usage=None)

    model.async_client.chat.completions.create = AsyncMock(return_value=chunks())
    output = [item async for item in model.async_chat_streamly_with_tools("", [{"role": "user", "content": "Hi"}], {})]
    assert "Plan" in output
    assert "Preface" in output
    assert "Tool result" in output


@pytest.mark.asyncio
async def test_next_request_runs_after_unclosed_first_response(monkeypatch):
    state = {"active": False, "closed": 0}

    @asynccontextmanager
    async def response_context():
        assert not state["active"]
        state["active"] = True

        async def lines():
            yield b'data: {"choices":[{"delta":{"content":"Answer"},"finish_reason":"stop"}]}\n'
            yield b'data: {"choices":[],"usage":{"total_tokens":7}}\n'
            await asyncio.Event().wait()

        try:
            yield SimpleNamespace(status=200, content=lines())
        finally:
            state["active"] = False
            state["closed"] += 1

    session = MagicMock()
    session.post.side_effect = lambda *args, **kwargs: response_context()
    monkeypatch.setattr(chat_model.aiohttp, "ClientSession", lambda **kwargs: _context(session))
    model = MWSChat("token", "custom-deployment", PROJECT_URL)

    async def collect():
        for _ in range(2):
            output = [item async for item in model._async_chat_streamly([{"role": "user", "content": "Hi"}], {})]
            assert output == [("Answer", 0), ("", 7)]

    await asyncio.wait_for(collect(), timeout=0.5)
    assert state["closed"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_tools_request_deadline_includes_waiting_for_first_response(monkeypatch, stream):
    model = MWSChat("token", "custom-deployment", PROJECT_URL)
    model.tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    model.max_rounds = 1
    model.max_retries = 0
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0.03")

    async def create(**kwargs):
        await asyncio.Event().wait()

    model.async_client.chat.completions.create = create

    async def collect():
        if stream:
            return [item async for item in model.async_chat_streamly_with_tools("", [{"role": "user", "content": "Hi"}], {})]
        return await model.async_chat_with_tools("", [{"role": "user", "content": "Hi"}], {})

    result = await asyncio.wait_for(collect(), timeout=0.5)
    assert any(isinstance(item, str) and "timed out" in item and "MWS" in item for item in result)


@pytest.mark.asyncio
async def test_cancelled_tools_stream_is_closed():
    entered = asyncio.Event()
    closed = asyncio.Event()

    class Stream:
        async def __aiter__(self):
            entered.set()
            await asyncio.Event().wait()
            yield

        async def close(self):
            closed.set()

    model = MWSChat("token", "custom-deployment", PROJECT_URL)
    model.tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    model.max_rounds = 1
    model.max_retries = 0
    model.async_client.chat.completions.create = AsyncMock(return_value=Stream())

    async def collect():
        return [item async for item in model.async_chat_streamly_with_tools("", [{"role": "user", "content": "Hi"}], {})]

    task = asyncio.create_task(collect())
    await asyncio.wait_for(entered.wait(), timeout=0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
