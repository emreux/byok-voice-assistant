"""What the OpenAI-compatible adapter must turn its vendor's shapes into.

No network. The fake client mimics the two things the adapter calls -
`chat.completions.create` and `models.list` - but the chunks it yields are
real `openai.types` objects, so a field this suite reads is a field the SDK
actually has, and the refusals are the SDK's own exception classes built
the way the SDK builds them.

What is *not* here is anything every adapter has to do. Text arriving in
order, a tool call never arriving half-built, a refusal reaching the caller
as one of the two exceptions the application knows - all of those are the
contract, and they are tested once for every adapter in
`test_llm_adapters.py`. This file is only what this API does differently:
tool arguments streamed in slices and gathered by index, the token counts
on a final chunk with no choices, the shapes of the messages, the
placeholder key for a server that wants none - plus the `build` function at
the bottom that lets the contract suite drive it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any, Literal

import httpx2
import openai
import pytest
from openai.types import CompletionUsage, Model
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import PromptTokensDetails

from assistant.llm.base import (
    AuthenticationError,
    Delta,
    LLMProvider,
    Message,
    ProviderError,
    ToolCall,
    ToolSpec,
)
from assistant.llm.openai_compat_adapter import KEY_PLACEHOLDER, OpenAICompatAdapter, _client
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

URL = "http://fake.test/v1"


# --------------------------------------------------------------------------
# The chunks, as the SDK parses them off the wire
# --------------------------------------------------------------------------


def chunk(
    *,
    text: str | None = None,
    role: Literal["assistant"] | None = None,
    slices: Sequence[ChoiceDeltaToolCall] = (),
    finish: str | None = None,
    usage: CompletionUsage | None = None,
    with_choice: bool = True,
) -> ChatCompletionChunk:
    """One chunk. `with_choice=False` is the final, usage-only chunk, or a
    keep-alive: no choices at all."""
    choices = []
    if with_choice:
        delta = ChoiceDelta(content=text, role=role, tool_calls=list(slices) or None)
        choices.append(Choice(index=0, delta=delta, finish_reason=finish))  # type: ignore[arg-type]
    return ChatCompletionChunk(
        id="chatcmpl-x",
        choices=choices,
        created=0,
        model="m",
        object="chat.completion.chunk",
        usage=usage,
    )


def text_chunk(text: str) -> ChatCompletionChunk:
    return chunk(text=text)


def slice_chunk(
    index: int,
    *,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> ChatCompletionChunk:
    """One slice of one tool call: the first carries the id and the name,
    the later ones only more of the arguments."""
    function = ChoiceDeltaToolCallFunction(name=name, arguments=arguments)
    piece = ChoiceDeltaToolCall(index=index, id=call_id, type="function", function=function)
    return chunk(slices=[piece])


def finish_chunk(reason: str = "stop") -> ChatCompletionChunk:
    return chunk(finish=reason)


def usage_chunk(prompt: int, output: int, cached: int = 0) -> ChatCompletionChunk:
    """The last chunk `stream_options.include_usage` adds: no choices, only counts."""
    counted = CompletionUsage(
        prompt_tokens=prompt,
        completion_tokens=output,
        total_tokens=prompt + output,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached),
    )
    return chunk(usage=counted, with_choice=False)


class FakeStream:
    """What `create(stream=True)` hands back: iterated for its chunks, and
    closed - as the SDK's `AsyncStream` is - by leaving an `async with`."""

    def __init__(
        self, chunks: list[ChatCompletionChunk], error: Exception | None, error_after: int
    ) -> None:
        self._chunks = chunks
        self._error = error
        self._error_after = error_after
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        for number, piece in enumerate(self._chunks):
            if self._error is not None and number == self._error_after:
                raise self._error
            yield piece
        if self._error is not None:
            raise self._error

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *_: object) -> None:
        self.closed = True


class FakeCompletions:
    """Stands in for `client.chat.completions`, recording what the adapter sent."""

    def __init__(
        self,
        chunks: list[ChatCompletionChunk] | None = None,
        error: Exception | None = None,
        error_after: int = 0,
    ) -> None:
        self.chunks = chunks or []
        self.error = error
        # How many chunks arrive before the error does. Zero is the request
        # being refused outright; anything else is a stream that dies part way.
        self.error_after = error_after
        self.sent: dict[str, Any] = {}
        self.streams: list[FakeStream] = []

    async def create(self, **request: Any) -> FakeStream:
        self.sent = request
        if self.error is not None and not self.error_after:
            raise self.error

        stream = FakeStream(self.chunks, self.error, self.error_after)
        self.streams.append(stream)
        return stream


class FakeModels:
    """Stands in for `client.models`; `list()` is iterated, not awaited, as
    the SDK's paginator is."""

    def __init__(self, models: list[Model] | None = None, error: Exception | None = None) -> None:
        self.models = models or []
        self.error = error

    def list(self) -> AsyncIterator[Model]:
        return self._pages()

    async def _pages(self) -> AsyncIterator[Model]:
        if self.error is not None:
            raise self.error
        for model in self.models:
            yield model


def adapter_for(
    completions: FakeCompletions | None = None, models: FakeModels | None = None
) -> OpenAICompatAdapter:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions or FakeCompletions()),
        models=models or FakeModels(),
    )
    return OpenAICompatAdapter(api_key="unused", base_url=URL, client=client)


async def collect(adapter: OpenAICompatAdapter, **kwargs: Any) -> list[Delta]:
    defaults: dict[str, Any] = {
        "messages": [Message.user("merhaba")],
        "tools": [],
        "model": "gpt-x",
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


# --------------------------------------------------------------------------
# Tool calls arrive in slices, and leave whole
# --------------------------------------------------------------------------


async def test_a_call_streamed_in_slices_is_emitted_once_and_whole() -> None:
    """The shape of section 3.2's trap: `{"na`, `me": "no`, `tepad"}`. One
    `ToolCall`, with the arguments parsed, when the stream says it is done."""
    adapter = adapter_for(
        FakeCompletions(
            [
                slice_chunk(0, call_id="call_1", name="open_app", arguments='{"na'),
                slice_chunk(0, arguments='me": "no'),
                slice_chunk(0, arguments='tepad"}'),
                finish_chunk("tool_calls"),
            ]
        )
    )

    calls = calls_in(await collect(adapter))

    assert len(calls) == 1
    assert (calls[0].id, calls[0].name) == ("call_1", "open_app")
    assert dict(calls[0].arguments) == {"name": "notepad"}


async def test_a_prefix_that_happens_to_parse_is_not_a_finished_call() -> None:
    """`{}` is valid JSON and so is `{"name": "no"}`; neither is the call.
    Emitting on "it parses" would hand the gate an action with half its
    arguments. The text after the slices proves the call waited: it comes
    out after the text, at the finish."""
    adapter = adapter_for(
        FakeCompletions(
            [
                slice_chunk(0, call_id="c1", name="send_email", arguments="{}"),
                slice_chunk(0, arguments=""),
                text_chunk("Sending."),
                finish_chunk("tool_calls"),
            ]
        )
    )

    deltas = await collect(adapter)

    assert [d.text for d in deltas if d.text] == ["Sending."]
    assert [type(d.text or d.tool_call).__name__ for d in deltas][:2] == ["str", "ToolCall"]


async def test_two_calls_streamed_together_are_told_apart_by_index() -> None:
    """The model asks for two things at once and the server interleaves
    their slices; each is gathered under its own index."""
    adapter = adapter_for(
        FakeCompletions(
            [
                slice_chunk(0, call_id="c1", name="open_app", arguments='{"name": "'),
                slice_chunk(1, call_id="c2", name="open_url", arguments='{"url": "'),
                slice_chunk(0, arguments='notepad"}'),
                slice_chunk(1, arguments='https://a.test"}'),
                finish_chunk("tool_calls"),
            ]
        )
    )

    calls = calls_in(await collect(adapter))

    assert [(c.id, c.name, dict(c.arguments)) for c in calls] == [
        ("c1", "open_app", {"name": "notepad"}),
        ("c2", "open_url", {"url": "https://a.test"}),
    ]


async def test_a_call_whose_arguments_never_become_json_is_dropped() -> None:
    """The stream ended with a call cut in half. Half an action is not
    handed to the gate; nothing is, and the log says why."""
    adapter = adapter_for(
        FakeCompletions(
            [
                slice_chunk(0, call_id="c1", name="send_email", arguments='{"to": "a@'),
                finish_chunk("length"),
            ]
        )
    )

    deltas = await collect(adapter)

    assert calls_in(deltas) == []
    assert [d.finish_reason for d in deltas if d.finish_reason] == ["length"]


async def test_a_call_with_no_arguments_at_all_is_a_call_with_none() -> None:
    adapter = adapter_for(
        FakeCompletions([slice_chunk(0, call_id="c1", name="get_current_time"), finish_chunk()])
    )

    [call] = calls_in(await collect(adapter))

    assert dict(call.arguments) == {}


async def test_a_call_the_server_ended_without_a_finish_reason_still_arrives() -> None:
    """Some servers close the stream without saying why. What was streamed
    of the call is all there will be, so it is emitted at the end."""
    adapter = adapter_for(
        FakeCompletions([slice_chunk(0, call_id="c1", name="open_app", arguments='{"name": "x"}')])
    )

    [call] = calls_in(await collect(adapter))

    assert dict(call.arguments) == {"name": "x"}


async def test_arguments_that_are_json_but_not_an_object_are_dropped() -> None:
    adapter = adapter_for(
        FakeCompletions(
            [slice_chunk(0, call_id="c1", name="x", arguments="[1, 2]"), finish_chunk()]
        )
    )

    assert calls_in(await collect(adapter)) == []


# --------------------------------------------------------------------------
# What the turn cost, and how it ended
# --------------------------------------------------------------------------


async def test_the_token_counts_come_on_the_final_chunk_with_no_choices() -> None:
    adapter = adapter_for(
        FakeCompletions([text_chunk("hi"), finish_chunk(), usage_chunk(prompt=7, output=2)])
    )

    usage = [d.usage for d in await collect(adapter) if d.usage is not None]

    assert len(usage) == 1
    assert (usage[0].input_tokens, usage[0].output_tokens) == (7, 2)


async def test_the_cached_tokens_are_read_from_the_prompt_details() -> None:
    """OpenAI reports its cache hits under `prompt_tokens_details`; a cache
    hit is a discount the bill of section 6 should show."""
    adapter = adapter_for(FakeCompletions([text_chunk("hi"), usage_chunk(7, 2, cached=5)]))

    [usage] = [d.usage for d in await collect(adapter) if d.usage is not None]

    assert usage.cached_tokens == 5


async def test_a_keep_alive_with_neither_choices_nor_counts_is_nothing() -> None:
    adapter = adapter_for(FakeCompletions([chunk(with_choice=False), text_chunk("hi")]))

    deltas = await collect(adapter)

    assert [d.text for d in deltas] == ["hi"]


async def test_the_chunk_that_only_announces_the_role_is_nothing() -> None:
    adapter = adapter_for(FakeCompletions([chunk(role="assistant"), text_chunk("hi")]))

    assert [d.text for d in await collect(adapter)] == ["hi"]


async def test_the_sdk_s_stream_is_closed_once_it_has_been_read() -> None:
    """Left to the garbage collector, the connection underneath is closed at
    the shutdown of the event loop, with a complaint on stderr and a
    connection held until then (measured 2026-09-10)."""
    completions = FakeCompletions([text_chunk("hi"), finish_chunk()])

    await collect(adapter_for(completions))

    assert [stream.closed for stream in completions.streams] == [True]


async def test_the_sdk_s_stream_is_closed_when_the_reader_walks_away() -> None:
    """A consumer that stops after the first delta - a turn cancelled by a
    key press - leaves nothing open either."""
    completions = FakeCompletions([text_chunk("one"), text_chunk("two"), finish_chunk()])
    adapter = adapter_for(completions)

    stream = adapter.stream([Message.user("m")], [], model="m")
    assert isinstance(stream, AsyncGenerator)
    first = await anext(stream)
    await stream.aclose()

    assert first.text == "one"
    assert [stream.closed for stream in completions.streams] == [True]


async def test_the_sdk_s_stream_is_closed_when_it_dies_part_way() -> None:
    completions = FakeCompletions(
        [text_chunk("Tür")], error=httpx2.ReadError("connection closed"), error_after=1
    )

    with pytest.raises(ProviderError):
        await collect(adapter_for(completions))

    assert [stream.closed for stream in completions.streams] == [True]


async def test_the_reason_generation_stopped_passes_through_in_the_server_s_word() -> None:
    adapter = adapter_for(FakeCompletions([text_chunk("hi"), finish_chunk("length")]))

    reasons = [d.finish_reason for d in await collect(adapter) if d.finish_reason]

    assert reasons == ["length"]


# --------------------------------------------------------------------------
# What is sent
# --------------------------------------------------------------------------


async def test_the_system_prompt_is_a_message_like_any_other() -> None:
    completions = FakeCompletions([text_chunk("ok")])
    adapter = adapter_for(completions)

    await collect(adapter, messages=[Message.system("Be brief."), Message.user("merhaba")])

    assert completions.sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "merhaba"},
    ]


async def test_an_earlier_call_goes_back_on_the_assistant_s_own_message() -> None:
    """The arguments go back as JSON text, the way they were streamed."""
    completions = FakeCompletions([text_chunk("ok")])
    adapter = adapter_for(completions)
    call = ToolCall(id="c1", name="open_app", arguments={"name": "Not Defteri"})

    await collect(adapter, messages=[Message.assistant("Açıyorum.", tool_calls=(call,))])

    [sent] = completions.sent["messages"]
    assert sent["role"] == "assistant"
    assert sent["content"] == "Açıyorum."
    [resent] = sent["tool_calls"]
    assert (resent["id"], resent["type"], resent["function"]["name"]) == (
        "c1",
        "function",
        "open_app",
    )
    assert json.loads(resent["function"]["arguments"]) == {"name": "Not Defteri"}


async def test_a_call_with_nothing_said_beside_it_carries_no_content() -> None:
    """Some servers refuse an empty content beside tool calls; it travels
    only when there is some."""
    completions = FakeCompletions([text_chunk("ok")])
    adapter = adapter_for(completions)
    call = ToolCall(id="c1", name="open_app", arguments={})

    await collect(adapter, messages=[Message.assistant(tool_calls=(call,))])

    assert "content" not in completions.sent["messages"][0]


async def test_a_tool_result_goes_back_under_the_role_tool_with_the_call_s_id() -> None:
    completions = FakeCompletions([text_chunk("ok")])
    adapter = adapter_for(completions)
    call = ToolCall(id="c1", name="open_app", arguments={})

    await collect(adapter, messages=[Message.tool_result(call, "opened")])

    assert completions.sent["messages"] == [
        {"role": "tool", "tool_call_id": "c1", "content": "opened"}
    ]


async def test_tools_are_offered_as_function_objects() -> None:
    completions = FakeCompletions([text_chunk("ok")])
    adapter = adapter_for(completions)
    spec = ToolSpec(
        name="open_app",
        description="Opens an application.",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}},
    )

    await collect(adapter, tools=[spec])

    assert completions.sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "open_app",
                "description": "Opens an application.",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        }
    ]


async def test_no_tools_key_is_sent_when_there_are_no_tools() -> None:
    """Some servers refuse an empty list where OpenAI ignores one."""
    completions = FakeCompletions([text_chunk("ok")])
    adapter = adapter_for(completions)

    await collect(adapter)

    assert "tools" not in completions.sent


async def test_the_request_streams_and_asks_for_the_counts_and_carries_the_ceiling() -> None:
    completions = FakeCompletions([text_chunk("ok")])
    adapter = adapter_for(completions)

    await collect(adapter)

    sent = completions.sent
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["max_tokens"] == 4096
    assert sent["model"] == "gpt-x"


async def test_the_temperature_travels_only_when_one_was_asked_for() -> None:
    completions = FakeCompletions([text_chunk("ok")])
    adapter = adapter_for(completions)

    await collect(adapter)
    assert "temperature" not in completions.sent

    await collect(adapter, temperature=0.2)
    assert completions.sent["temperature"] == 0.2


# --------------------------------------------------------------------------
# The key, the address and the model list
# --------------------------------------------------------------------------


def test_an_empty_key_becomes_the_placeholder_the_sdk_insists_on() -> None:
    """Ollama has no key to give and the SDK refuses to be built without
    one; a given key is kept as it is."""
    assert _client("", URL).api_key == KEY_PLACEHOLDER
    assert _client("sk-real", URL).api_key == "sk-real"


def test_the_address_is_the_catalogue_s() -> None:
    assert str(_client("k", "https://api.groq.com/openai/v1").base_url).startswith(
        "https://api.groq.com/openai/v1"
    )


async def test_the_models_are_listed_by_id_with_nothing_known_about_them() -> None:
    """The list endpoint says nothing about what a model can do, so the
    probe of 2.6 is left to find out; and OpenAI's has no display name, so
    the id is shown."""
    listed = await adapter_for(
        models=FakeModels([Model(id="llama-3.3-70b", created=0, object="model", owned_by="x")])
    ).list_models()

    assert [(m.id, m.display_name, m.supports_tools) for m in listed] == [
        ("llama-3.3-70b", "llama-3.3-70b", None)
    ]


async def test_a_server_that_names_its_models_is_shown_the_name() -> None:
    """OpenRouter adds a `name` beside the id; the SDK keeps a field it does
    not know as an extra, and the wizard's menu is better for it."""
    named = _named("google/gemini-2.5-flash", "Google: Gemini 2.5 Flash")

    [listed] = await adapter_for(models=FakeModels([named])).list_models()

    assert (listed.id, listed.display_name) == (
        "google/gemini-2.5-flash",
        "Google: Gemini 2.5 Flash",
    )


# --------------------------------------------------------------------------
# What a refusal is turned into
# --------------------------------------------------------------------------


def refusal(status: int, message: str) -> openai.APIStatusError:
    """A real SDK error, of the class the SDK raises for that status."""
    request = httpx2.Request("POST", f"{URL}/chat/completions")
    body = {"error": {"message": message, "type": "invalid_request_error"}}
    response = httpx2.Response(status, request=request, json=body)
    kinds: dict[int, type[openai.APIStatusError]] = {
        401: openai.AuthenticationError,
        403: openai.PermissionDeniedError,
        404: openai.NotFoundError,
        429: openai.RateLimitError,
    }
    kind = kinds.get(status, openai.InternalServerError)
    return kind(message, response=response, body=body)


def unreachable() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx2.Request("POST", f"{URL}/chat/completions"))


@pytest.mark.parametrize("status", [401, 403])
async def test_a_refused_key_is_named_as_one(status: int) -> None:
    adapter = adapter_for(FakeCompletions(error=refusal(status, "Invalid API Key")))

    with pytest.raises(AuthenticationError, match="Invalid API Key"):
        await collect(adapter)


async def test_a_rate_limit_is_not_a_key_problem() -> None:
    """Groq's free tier says 429 several times an hour. "Renew your key" would
    send the user to replace a key that is fine."""
    adapter = adapter_for(FakeCompletions(error=refusal(429, "Rate limit reached")))

    with pytest.raises(ProviderError, match="429") as raised:
        await collect(adapter)

    assert not isinstance(raised.value, AuthenticationError)


async def test_a_model_the_server_does_not_have_is_a_refusal_with_its_words() -> None:
    adapter = adapter_for(FakeCompletions(error=refusal(404, "The model `gpt-9` does not exist")))

    with pytest.raises(ProviderError, match="gpt-9"):
        await collect(adapter)


async def test_a_network_that_is_not_there_is_a_refusal() -> None:
    with pytest.raises(ProviderError) as raised:
        await collect(adapter_for(FakeCompletions(error=unreachable())))

    assert not isinstance(raised.value, AuthenticationError)


async def test_a_stream_the_transport_drops_is_a_refusal_and_not_the_transport_s_own() -> None:
    """The SDK translates the transport's errors on the way in and not once
    the stream is being read; there they arrive as `httpx2`'s own, which
    nothing above this layer would catch."""
    adapter = adapter_for(
        FakeCompletions(
            [text_chunk("Tür")], error=httpx2.ReadError("connection closed"), error_after=1
        )
    )

    with pytest.raises(ProviderError, match="ReadError"):
        await collect(adapter)


async def test_a_key_refused_while_listing_models_does_not_validate() -> None:
    adapter = adapter_for(models=FakeModels(error=refusal(401, "Invalid API Key")))

    assert await adapter.validate_credentials() is False


async def test_a_failure_of_an_unexpected_shape_is_not_dressed_up_as_a_bad_key() -> None:
    """Whatever is neither the SDK's nor the transport's is a bug, and a bug
    dressed up as a bad key is one nobody fixes."""
    adapter = adapter_for(models=FakeModels(error=RuntimeError("something nobody foresaw")))

    with pytest.raises(RuntimeError):
        await adapter.validate_credentials()


# --------------------------------------------------------------------------
# How the contract suite drives this adapter
#
# `test_llm_adapters.py` scripts a provider without naming one; this is where
# that script becomes the chat completions API. A call the script starts
# and later finishes is one call streamed in two slices, which is how this
# API really sends one.
# --------------------------------------------------------------------------


def _chunks(script: Sequence[Step]) -> list[ChatCompletionChunk]:
    """The script, in the chunks this API produces.

    `Starts` is the first slice of a call - the id, the name and the first
    half of the arguments of the `Calls` that finishes it; that `Calls` is
    then the second half. A `Calls` on its own is one slice with everything.
    """
    chunks: list[ChatCompletionChunk] = []
    index_of: dict[str, int] = {}
    started: set[str] = set()

    for step in script:
        if isinstance(step, Nothing):
            chunks.append(chunk(role="assistant"))
        elif isinstance(step, Says):
            chunks.append(text_chunk(step.text))
        elif isinstance(step, Spends):
            chunks.append(usage_chunk(step.input, step.output, step.cached))
        elif isinstance(step, Starts | Calls):
            index = index_of.setdefault(step.id, len(index_of))
            whole = json.dumps(_finished_arguments(script, step), ensure_ascii=False)
            half = len(whole) // 2
            if isinstance(step, Starts):
                started.add(step.id)
                chunks.append(
                    slice_chunk(index, call_id=step.id, name=step.name, arguments=whole[:half])
                )
            elif step.id in started:
                chunks.append(slice_chunk(index, arguments=whole[half:]))
            else:
                chunks.append(slice_chunk(index, call_id=step.id, name=step.name, arguments=whole))
        else:
            chunks.append(finish_chunk("stop"))
    return chunks


def _finished_arguments(script: Sequence[Step], step: Starts | Calls) -> dict[str, Any]:
    """The arguments the call ends up with: those of the `Calls` that
    finishes a `Starts`, or its own."""
    for later in script:
        if isinstance(later, Calls) and later.id == step.id:
            return dict(later.arguments)
    return dict(step.arguments)


def _how_it_refuses(refuses: Refuses | None) -> Exception | None:
    if refuses is None:
        return None
    if refuses is Refuses.THE_KEY:
        return refusal(401, "Incorrect API key provided")
    if refuses is Refuses.THE_NETWORK:
        return unreachable()
    return refusal(503, COMPLAINT)


def build(
    *script: Step,
    refuses: Refuses | None = None,
    after: int = 0,
    models: Sequence[tuple[str, str]] = (),
) -> LLMProvider:
    """What `contract.Build` asks for, answered in the chat completions API's shapes.

    The display name travels the way OpenRouter sends one: as a `name` field
    the SDK does not know and keeps as an extra."""
    error = _how_it_refuses(refuses)
    return adapter_for(
        FakeCompletions(chunks=_chunks(script), error=error, error_after=after),
        FakeModels(models=[_named(model_id, name) for model_id, name in models], error=error),
    )


def _named(model_id: str, name: str) -> Model:
    """A model with the `name` OpenRouter adds - a field the SDK does not
    declare and keeps as an extra, exactly as it does off the wire."""
    return Model.model_validate(
        {"id": model_id, "created": 0, "object": "model", "owned_by": "x", "name": name}
    )


OPENAI_COMPAT = Adapter(name="openai_compat", build=build)
