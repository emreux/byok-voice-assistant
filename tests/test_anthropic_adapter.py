"""What the Anthropic adapter must turn its vendor's shapes into (phase 4.5,
17 Sep 2026).

No network, and no key: the fake client mimics the two things the adapter
calls - `messages.stream` and `models.list` - and the events it yields are
real `anthropic.types` objects, so a field this suite reads is a field the
SDK actually has. The refusals are the SDK's own exception classes, built
the way the SDK builds them.

What every adapter has to do - text in order, a call never half-built, a
refusal as one of the two exceptions - is the contract, tested once for
each adapter in `test_llm_adapters.py`. This file is what this API does
differently: the system prompt as a cached block, tool calls as content
blocks that stop, results as `tool_result` blocks in a user message, the
token counts in two halves - plus the `build` function at the bottom that
lets the contract suite drive it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import (
    InputJSONDelta,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)
from anthropic.types import Message as WireMessage
from anthropic.types import ModelInfo as WireModel
from anthropic.types import Usage as WireUsage
from anthropic.types.raw_message_delta_event import Delta as StopDelta

from assistant.llm.anthropic_adapter import KEY_PLACEHOLDER, AnthropicAdapter
from assistant.llm.base import (
    AuthenticationError,
    Delta,
    LLMProvider,
    Message,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
)
from tests.contract import (
    COMPLAINT,
    Adapter,
    Calls,
    Nothing,
    Refuses,
    Says,
    Spends,
    Starts,
    Step,
)

URL = "https://api.anthropic.com/v1/messages"


# --------------------------------------------------------------------------
# The events, as the SDK parses them off the wire
# --------------------------------------------------------------------------


def started(input_tokens: int = 0, cached: int = 0, written: int = 0) -> RawMessageStartEvent:
    """`message_start`: the input half of the bill, and an empty message."""
    return RawMessageStartEvent(
        type="message_start",
        message=WireMessage(
            id="msg_1",
            type="message",
            role="assistant",
            model="claude-x",
            content=[],
            stop_reason=None,
            stop_sequence=None,
            usage=WireUsage(
                input_tokens=input_tokens,
                output_tokens=0,
                cache_read_input_tokens=cached,
                cache_creation_input_tokens=written,
            ),
        ),
    )


def text_block(index: int = 0) -> RawContentBlockStartEvent:
    return RawContentBlockStartEvent(
        type="content_block_start", index=index, content_block=TextBlock(type="text", text="")
    )


def text(piece: str, index: int = 0) -> RawContentBlockDeltaEvent:
    return RawContentBlockDeltaEvent(
        type="content_block_delta", index=index, delta=TextDelta(type="text_delta", text=piece)
    )


def tool_block(index: int, *, call_id: str, name: str) -> RawContentBlockStartEvent:
    return RawContentBlockStartEvent(
        type="content_block_start",
        index=index,
        content_block=ToolUseBlock(type="tool_use", id=call_id, name=name, input={}),
    )


def arguments(index: int, partial_json: str) -> RawContentBlockDeltaEvent:
    return RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=index,
        delta=InputJSONDelta(type="input_json_delta", partial_json=partial_json),
    )


def stopped(index: int) -> RawContentBlockStopEvent:
    return RawContentBlockStopEvent(type="content_block_stop", index=index)


def ended(
    reason: str | None = "end_turn",
    *,
    output: int = 0,
    input_tokens: int | None = None,
    cached: int | None = None,
) -> RawMessageDeltaEvent:
    """`message_delta`: why it stopped and the output half of the bill. Since
    the 1.x SDK the delta may repeat the input too."""
    return RawMessageDeltaEvent(
        type="message_delta",
        delta=StopDelta(stop_reason=reason, stop_sequence=None),  # type: ignore[arg-type]
        usage=MessageDeltaUsage(
            output_tokens=output, input_tokens=input_tokens, cache_read_input_tokens=cached
        ),
    )


class FakeEvents:
    """What `async with client.messages.stream(...)` hands back: iterated
    for its events, closed on the way out."""

    def __init__(self, events: list[Any], error: Exception | None, error_after: int) -> None:
        self._events = events
        self._error = error
        self._error_after = error_after
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[Any]:
        for number, event in enumerate(self._events):
            if self._error is not None and number == self._error_after:
                raise self._error
            yield event
        if self._error is not None:
            raise self._error


class FakeStreamManager:
    def __init__(self, events: FakeEvents, error: Exception | None, error_after: int) -> None:
        self._events = events
        self._error = error
        self._error_after = error_after

    async def __aenter__(self) -> FakeEvents:
        # The request is made on entry: a refusal outright comes here.
        if self._error is not None and not self._error_after:
            raise self._error
        return self._events

    async def __aexit__(self, *_: object) -> None:
        self._events.closed = True


class FakeMessages:
    def __init__(
        self, events: list[Any] | None = None, error: Exception | None = None, error_after: int = 0
    ) -> None:
        self.events = events or []
        self.error = error
        self.error_after = error_after
        self.sent: dict[str, Any] = {}
        self.streams: list[FakeEvents] = []

    def stream(self, **request: Any) -> FakeStreamManager:
        self.sent = request
        events = FakeEvents(self.events, self.error, self.error_after)
        self.streams.append(events)
        return FakeStreamManager(events, self.error, self.error_after)


class FakeModels:
    def __init__(
        self, models: list[WireModel] | None = None, error: Exception | None = None
    ) -> None:
        self.models = models or []
        self.error = error

    def list(self) -> AsyncIterator[WireModel]:
        return self._pages()

    async def _pages(self) -> AsyncIterator[WireModel]:
        if self.error is not None:
            raise self.error
        for model in self.models:
            yield model


def adapter_for(
    messages: FakeMessages | None = None, models: FakeModels | None = None
) -> AnthropicAdapter:
    client = SimpleNamespace(messages=messages or FakeMessages(), models=models or FakeModels())
    return AnthropicAdapter(api_key="unused", client=client)


async def collect(adapter: AnthropicAdapter, **kwargs: Any) -> list[Delta]:
    defaults: dict[str, Any] = {
        "messages": [Message.user("merhaba")],
        "tools": [],
        "model": "claude-x",
    }
    defaults.update(kwargs)
    return [
        delta
        async for delta in adapter.stream(
            defaults["messages"],
            defaults["tools"],
            model=defaults["model"],
            temperature=defaults.get("temperature"),
        )
    ]


def calls_in(deltas: list[Delta]) -> list[ToolCall]:
    return [d.tool_call for d in deltas if d.tool_call is not None]


def refusal(status: int, message: str) -> anthropic.APIStatusError:
    """A real SDK error, of the class the SDK raises for that status."""
    request = httpx2.Request("POST", URL)
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    response = httpx2.Response(status, request=request, json=body)
    kinds: dict[int, type[anthropic.APIStatusError]] = {
        401: anthropic.AuthenticationError,
        403: anthropic.PermissionDeniedError,
        404: anthropic.NotFoundError,
        429: anthropic.RateLimitError,
    }
    kind = kinds.get(status, anthropic.InternalServerError)
    return kind(message, response=response, body=body)


def unreachable() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx2.Request("POST", URL))


# --------------------------------------------------------------------------
# Tool calls: a block that starts, streams its arguments, and stops
# --------------------------------------------------------------------------


async def test_a_call_is_emitted_once_and_whole_when_its_block_stops() -> None:
    adapter = adapter_for(
        FakeMessages(
            [
                started(12),
                tool_block(0, call_id="toolu_1", name="open_app"),
                arguments(0, '{"na'),
                arguments(0, 'me": "no'),
                arguments(0, 'tepad"}'),
                stopped(0),
                ended("tool_use", output=5),
            ]
        )
    )

    deltas = await collect(adapter)

    calls = calls_in(deltas)
    assert len(calls) == 1
    assert (calls[0].id, calls[0].name) == ("toolu_1", "open_app")
    assert dict(calls[0].arguments) == {"name": "notepad"}
    # Emitted at the stop, before the message ends - and not before.
    assert [type(d.tool_call or d.finish_reason).__name__ for d in deltas] == ["ToolCall", "str"]


async def test_a_prefix_that_happens_to_parse_is_not_a_finished_call() -> None:
    """`{}` is valid JSON. The text that follows the pieces proves the call
    waited for its block to stop."""
    adapter = adapter_for(
        FakeMessages(
            [
                tool_block(0, call_id="toolu_1", name="send_email"),
                arguments(0, "{}"),
                text_block(1),
                text("Sending.", index=1),
                stopped(1),
                stopped(0),
                ended("tool_use"),
            ]
        )
    )

    deltas = await collect(adapter)

    kinds = [type(d.text or d.tool_call or d.finish_reason).__name__ for d in deltas]
    assert kinds == ["str", "ToolCall", "str"]
    assert dict(calls_in(deltas)[0].arguments) == {}


async def test_two_calls_in_one_message_are_told_apart_by_block_index() -> None:
    adapter = adapter_for(
        FakeMessages(
            [
                tool_block(0, call_id="a", name="open_app"),
                tool_block(1, call_id="b", name="get_weather"),
                arguments(0, '{"name": "x"}'),
                arguments(1, '{"place": "y"}'),
                stopped(1),
                stopped(0),
                ended("tool_use"),
            ]
        )
    )

    calls = calls_in(await collect(adapter))

    assert [(c.id, c.name, dict(c.arguments)) for c in calls] == [
        ("b", "get_weather", {"place": "y"}),
        ("a", "open_app", {"name": "x"}),
    ]


async def test_a_call_whose_block_never_stopped_is_emitted_at_the_end() -> None:
    adapter = adapter_for(
        FakeMessages([tool_block(0, call_id="a", name="open_app"), arguments(0, '{"name": "x"}')])
    )

    calls = calls_in(await collect(adapter))

    assert [(c.name, dict(c.arguments)) for c in calls] == [("open_app", {"name": "x"})]


async def test_arguments_that_never_became_json_are_dropped_and_logged() -> None:
    from loguru import logger

    adapter = adapter_for(
        FakeMessages(
            [tool_block(0, call_id="a", name="open_app"), arguments(0, '{"name": '), stopped(0)]
        )
    )
    warned: list[str] = []
    sink = logger.add(lambda message: warned.append(str(message)), level="WARNING")
    try:
        deltas = await collect(adapter)
    finally:
        logger.remove(sink)

    assert calls_in(deltas) == []
    assert any("open_app" in line and "never became JSON" in line for line in warned)


# --------------------------------------------------------------------------
# The bill: input at the start, output at the end, reported once
# --------------------------------------------------------------------------


async def test_the_counts_come_in_two_halves_and_leave_as_one() -> None:
    adapter = adapter_for(
        FakeMessages([started(100, cached=40, written=10), text("hi"), ended(output=7)])
    )

    deltas = await collect(adapter)

    spent = [d.usage for d in deltas if d.usage is not None]
    # 100 fresh + 40 read + 10 written is the whole prompt; 40 was cached.
    assert spent == [Usage(input_tokens=150, output_tokens=7, cached_tokens=40)]


async def test_a_delta_that_repeats_the_input_is_believed() -> None:
    adapter = adapter_for(
        FakeMessages([started(100), text("hi"), ended(output=7, input_tokens=120, cached=20)])
    )

    spent = [d.usage for d in await collect(adapter) if d.usage is not None]

    assert spent == [Usage(input_tokens=140, output_tokens=7, cached_tokens=20)]


async def test_the_stop_reason_is_the_vendors_own_word() -> None:
    adapter = adapter_for(FakeMessages([text("hi"), ended("max_tokens")]))

    deltas = await collect(adapter)

    assert [d.finish_reason for d in deltas if d.finish_reason] == ["max_tokens"]


# --------------------------------------------------------------------------
# Out: what is sent
# --------------------------------------------------------------------------


async def test_the_system_prompt_is_a_cached_block_and_not_a_message() -> None:
    messages = FakeMessages([text("hi")])
    adapter = adapter_for(messages)

    await collect(adapter, messages=[Message.system("Be brief."), Message.user("merhaba")])

    assert messages.sent["system"] == [
        {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}
    ]
    assert messages.sent["messages"] == [{"role": "user", "content": "merhaba"}]
    assert messages.sent["max_tokens"] == 4096
    assert "tools" not in messages.sent and "temperature" not in messages.sent


async def test_a_conversation_with_tools_takes_the_apis_shapes() -> None:
    """An assistant turn is blocks; two results that answer two calls made
    together go back in one user message."""
    messages = FakeMessages([text("hi")])
    adapter = adapter_for(messages)
    first = ToolCall(id="a", name="open_app", arguments={"name": "x"})
    second = ToolCall(id="b", name="get_weather", arguments={"place": "y"})
    tool = ToolSpec(name="open_app", description="Opens.", parameters={"type": "object"})

    await collect(
        adapter,
        messages=[
            Message.user("aç"),
            Message.assistant("Opening.", (first, second)),
            Message.tool_result(first, "opened"),
            Message.tool_result(second, "sunny"),
            Message.assistant("Done."),
        ],
        tools=[tool],
        temperature=0.2,
    )

    assert messages.sent["messages"] == [
        {"role": "user", "content": "aç"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Opening."},
                {"type": "tool_use", "id": "a", "name": "open_app", "input": {"name": "x"}},
                {"type": "tool_use", "id": "b", "name": "get_weather", "input": {"place": "y"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "opened"},
                {"type": "tool_result", "tool_use_id": "b", "content": "sunny"},
            ],
        },
        {"role": "assistant", "content": "Done."},
    ]
    assert messages.sent["tools"] == [
        {"name": "open_app", "description": "Opens.", "input_schema": {"type": "object"}}
    ]
    assert messages.sent["temperature"] == 0.2


async def test_the_stream_is_closed_however_it_ends() -> None:
    messages = FakeMessages([text("hi")])
    adapter = adapter_for(messages)

    stream = adapter.stream([Message.user("x")], [], model="m")
    await anext(stream)
    await stream.aclose()

    assert messages.streams[0].closed


def test_a_missing_key_is_sent_as_a_placeholder_not_refused_here() -> None:
    adapter = AnthropicAdapter(api_key="")

    assert adapter._client.api_key == KEY_PLACEHOLDER


def test_it_announces_prompt_caching() -> None:
    assert AnthropicAdapter(api_key="k").capabilities == frozenset({"prompt_caching"})


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
async def test_a_refused_key_is_named_as_one(status: int) -> None:
    adapter = adapter_for(FakeMessages(error=refusal(status, "invalid x-api-key")))

    with pytest.raises(AuthenticationError, match="invalid x-api-key"):
        await collect(adapter)


async def test_an_overloaded_api_is_not_a_key_problem() -> None:
    adapter = adapter_for(FakeMessages(error=refusal(529, "Overloaded")))

    with pytest.raises(ProviderError, match="529") as raised:
        await collect(adapter)

    assert not isinstance(raised.value, AuthenticationError)


async def test_a_transport_error_mid_stream_is_a_refusal() -> None:
    adapter = adapter_for(
        FakeMessages([text("half")], error=httpx2.ReadError("gone"), error_after=1)
    )

    with pytest.raises(ProviderError, match="ReadError"):
        await collect(adapter)


async def test_models_are_listed_by_id_and_display_name() -> None:
    adapter = adapter_for(
        models=FakeModels(
            [
                WireModel(
                    id="claude-opus-5", display_name="Claude Opus 5", created_at=NOW, type="model"
                )
            ]
        )
    )

    listed = await adapter.list_models()

    assert [(m.id, m.display_name, m.supports_tools) for m in listed] == [
        ("claude-opus-5", "Claude Opus 5", None)
    ]


NOW = datetime(2026, 9, 17, tzinfo=UTC)


# --------------------------------------------------------------------------
# The contract suite's way in
# --------------------------------------------------------------------------


def _events(script: Sequence[Step]) -> list[Any]:
    """The script, in the events this API produces.

    `Starts` opens a tool block with the first half of the arguments of
    the `Calls` that finishes it; that `Calls` is the second half and the
    stop. A `Calls` on its own is a block that opens, streams and stops.
    """
    events: list[Any] = []
    index_of: dict[str, int] = {}
    opened: set[str] = set()
    for step in script:
        if isinstance(step, Nothing):
            events.append(text_block(99))
        elif isinstance(step, Says):
            events.append(text(step.text))
        elif isinstance(step, Spends):
            events.append(
                ended(None, output=step.output, input_tokens=step.input, cached=step.cached)
            )
        elif isinstance(step, Starts | Calls):
            index = index_of.setdefault(step.id, 10 + len(index_of))
            whole = json.dumps(_finished_arguments(script, step), ensure_ascii=False)
            half = len(whole) // 2
            if isinstance(step, Starts):
                opened.add(step.id)
                events.append(tool_block(index, call_id=step.id, name=step.name))
                events.append(arguments(index, whole[:half]))
            elif step.id in opened:
                events.append(arguments(index, whole[half:]))
                events.append(stopped(index))
            else:
                events.append(tool_block(index, call_id=step.id, name=step.name))
                events.append(arguments(index, whole))
                events.append(stopped(index))
        else:
            events.append(ended("end_turn"))
    return events


def _finished_arguments(script: Sequence[Step], step: Starts | Calls) -> dict[str, Any]:
    for later in script:
        if isinstance(later, Calls) and later.id == step.id:
            return dict(later.arguments)
    return dict(step.arguments)


def _how_it_refuses(refuses: Refuses | None) -> Exception | None:
    if refuses is None:
        return None
    if refuses is Refuses.THE_KEY:
        return refusal(401, "invalid x-api-key")
    if refuses is Refuses.THE_NETWORK:
        return unreachable()
    return refusal(529, COMPLAINT)


def build(
    *script: Step,
    refuses: Refuses | None = None,
    after: int = 0,
    models: Sequence[tuple[str, str]] = (),
) -> LLMProvider:
    """What `contract.Build` asks for, answered in the Messages API's shapes."""
    error = _how_it_refuses(refuses)
    return adapter_for(
        FakeMessages(events=_events(script), error=error, error_after=after),
        FakeModels(
            models=[
                WireModel(id=model_id, display_name=name, created_at=NOW, type="model")
                for model_id, name in models
            ],
            error=error,
        ),
    )


ANTHROPIC = Adapter(name="anthropic", build=build)
