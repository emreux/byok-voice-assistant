"""What the Gemini adapter must turn its vendor's shapes into.

No network. The fake client mimics the one method the adapter calls, but the
chunks it yields are real `google.genai.types` objects, so a field this suite
reads is a field the SDK actually has. A hand-rolled stub would let the tests
pass while the adapter reads an attribute that does not exist.

What is *not* here is anything every adapter has to do. Text arriving in
order, a tool call never arriving half-built, a refusal reaching the caller as
one of the two exceptions the application knows - all of those are the
contract, and they are tested once for every adapter in `test_llm_adapters.py`.
This file is only what Gemini does differently, plus the `build` function at
the bottom that lets the contract suite drive it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from google.genai import errors, types

from assistant.llm.base import (
    AuthenticationError,
    Delta,
    LLMProvider,
    Message,
    ToolSpec,
)
from assistant.llm.gemini_adapter import GeminiAdapter
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


def text_chunk(
    text: str, *, prompt: int | None = None, output: int | None = None
) -> types.GenerateContentResponse:
    """A chunk of the answer. Gemini attaches a running token total to each one."""
    running = None
    if prompt is not None:
        running = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt, candidates_token_count=output, cached_content_token_count=0
        )
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))
        ],
        usage_metadata=running,
    )


def final_chunk(
    *,
    reason: types.FinishReason = types.FinishReason.STOP,
    prompt: int | None = None,
    output: int = 0,
) -> types.GenerateContentResponse:
    """The end of the stream, with token counts only if the caller asks for
    them - a provider that reported none must not be read as reporting zero."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[]), finish_reason=reason)
        ],
        usage_metadata=None
        if prompt is None
        else types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt, candidates_token_count=output, cached_content_token_count=0
        ),
    )


def call_chunk(
    name: str, args: dict[str, Any], *, call_id: str = "c1", still_streaming: bool = False
) -> types.GenerateContentResponse:
    function_call = types.FunctionCall(
        id=call_id, name=name, args=args, will_continue=still_streaming or None
    )
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(function_call=function_call)])
            )
        ]
    )


class FakeModels:
    """Stands in for `client.aio.models`, recording what the adapter sent."""

    def __init__(
        self,
        chunks: list[types.GenerateContentResponse] | None = None,
        models: list[types.Model] | None = None,
        error: Exception | None = None,
        error_after: int = 0,
    ) -> None:
        self.chunks = chunks or []
        self.models = models or []
        self.error = error
        # How many chunks arrive before the error does. Zero is the request
        # being refused outright; anything else is a stream that dies part way.
        self.error_after = error_after
        self.sent: dict[str, Any] = {}

    async def generate_content_stream(
        self, *, model: str, contents: Any, config: Any
    ) -> AsyncIterator[types.GenerateContentResponse]:
        self.sent = {"model": model, "contents": contents, "config": config}
        if self.error is not None and not self.error_after:
            raise self.error

        async def chunks() -> AsyncIterator[types.GenerateContentResponse]:
            for number, chunk in enumerate(self.chunks):
                if self.error is not None and number == self.error_after:
                    raise self.error
                yield chunk
            if self.error is not None:
                raise self.error

        return chunks()

    async def list(self) -> Any:
        if self.error is not None:
            raise self.error

        class Pager:
            def __init__(self, models: list[types.Model]) -> None:
                self._models = models

            async def __aiter__(self) -> AsyncIterator[types.Model]:
                for model in self._models:
                    yield model

        return Pager(self.models)


def adapter_for(models: FakeModels) -> GeminiAdapter:
    client = type("FakeClient", (), {"aio": type("Aio", (), {"models": models})()})()
    return GeminiAdapter(api_key="unused", client=client)


async def collect(adapter: GeminiAdapter, **kwargs: Any) -> list[Delta]:
    defaults: dict[str, Any] = {
        "messages": [Message.user("merhaba")],
        "tools": [],
        "model": "gemini-x",
    }
    defaults.update(kwargs)
    return [
        delta
        async for delta in adapter.stream(
            defaults["messages"], defaults["tools"], model=defaults["model"]
        )
    ]


async def test_the_token_counts_are_reported_once_not_per_chunk() -> None:
    """Every Gemini chunk repeats a running total, so a consumer that added
    them up would report - and charge for - several times the real usage."""
    adapter = adapter_for(
        FakeModels(
            [
                text_chunk("One,", prompt=7, output=2),
                text_chunk(" two, three", prompt=7, output=20),
                final_chunk(prompt=7, output=40),
            ]
        )
    )

    usage = [d.usage for d in await collect(adapter) if d.usage is not None]

    assert len(usage) == 1
    assert (usage[0].input_tokens, usage[0].output_tokens) == (7, 40)


async def test_the_totals_survive_a_stream_that_stops_reporting_them() -> None:
    """The last chunk carrying counts is the authority, not the last chunk."""
    adapter = adapter_for(FakeModels([text_chunk("hi", prompt=7, output=40), text_chunk("!")]))

    usage = [d.usage for d in await collect(adapter) if d.usage is not None]

    assert [(u.input_tokens, u.output_tokens) for u in usage] == [(7, 40)]


async def test_the_reason_generation_stopped_is_reported() -> None:
    adapter = adapter_for(FakeModels([final_chunk(reason=types.FinishReason.MAX_TOKENS)]))

    reasons = [d.finish_reason for d in await collect(adapter) if d.finish_reason]

    assert reasons == ["MAX_TOKENS"]


async def test_the_system_message_is_lifted_out_of_the_conversation() -> None:
    models = FakeModels([text_chunk("ok")])
    adapter = adapter_for(models)

    await collect(adapter, messages=[Message.system("Be brief."), Message.user("merhaba")])

    assert models.sent["config"].system_instruction == "Be brief."
    assert len(models.sent["contents"]) == 1
    assert models.sent["contents"][0].role == "user"


async def test_the_assistant_speaks_under_the_role_gemini_expects() -> None:
    models = FakeModels([text_chunk("ok")])
    adapter = adapter_for(models)

    await collect(
        adapter,
        messages=[Message.user("merhaba"), Message.assistant("selam"), Message.user("naber")],
    )

    assert [c.role for c in models.sent["contents"]] == ["user", "model", "user"]


async def test_a_tool_result_goes_back_as_a_function_response() -> None:
    models = FakeModels([text_chunk("ok")])
    adapter = adapter_for(models)

    await collect(adapter, messages=[Message.tool_result("c1", "22 degrees")])

    part = models.sent["contents"][0].parts[0]
    assert part.function_response is not None
    assert part.function_response.response == {"result": "22 degrees"}


async def test_tools_are_offered_as_function_declarations() -> None:
    models = FakeModels([text_chunk("ok")])
    adapter = adapter_for(models)
    spec = ToolSpec(
        name="open_app",
        description="Opens an application.",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}},
    )

    await collect(adapter, tools=[spec])

    declared = models.sent["config"].tools[0].function_declarations[0]
    assert declared.name == "open_app"


async def test_no_tool_block_is_sent_when_there_are_no_tools() -> None:
    models = FakeModels([text_chunk("ok")])
    adapter = adapter_for(models)

    await collect(adapter)

    assert not models.sent["config"].tools


async def test_only_models_that_can_generate_text_are_offered() -> None:
    models = FakeModels(
        models=[
            types.Model(
                name="models/gemini-x",
                display_name="Gemini X",
                input_token_limit=1000,
                supported_actions=["generateContent"],
            ),
            types.Model(
                name="models/embed-1", display_name="Embed 1", supported_actions=["embedContent"]
            ),
        ]
    )

    listed = await adapter_for(models).list_models()

    assert [m.id for m in listed] == ["gemini-x"]
    assert listed[0].context_window == 1000


async def test_a_key_that_fails_in_an_unexpected_way_is_still_just_a_no() -> None:
    """Not every refusal arrives as an SDK error - a proxy, a DNS failure and a
    closed socket all reach this as something else entirely. The setup command
    needs one answer, and it is the same answer."""
    adapter = adapter_for(FakeModels(error=RuntimeError("API key not valid")))

    assert await adapter.validate_credentials() is False


@pytest.mark.parametrize("role", ["user", "assistant"])
async def test_every_message_reaches_the_provider(role: str) -> None:
    models = FakeModels([text_chunk("ok")])
    adapter = adapter_for(models)
    message = Message.user("a") if role == "user" else Message.assistant("a")

    await collect(adapter, messages=[message])

    assert len(models.sent["contents"]) == 1


# --------------------------------------------------------------------------
# What a refusal is turned into
# --------------------------------------------------------------------------


def refusal(code: int, message: str, status: str) -> errors.APIError:
    """A real SDK error, built the way the SDK builds one from a response."""
    kind = errors.ClientError if code < 500 else errors.ServerError
    return kind(code, {"error": {"code": code, "message": message, "status": status}})


async def test_a_refused_key_is_named_as_one_rather_than_left_to_the_caller() -> None:
    """Section 3.2: the turn is cancelled and the user is told to renew the
    key. Nothing above this layer may import an SDK to find that out."""
    adapter = adapter_for(FakeModels(error=refusal(403, "Permission denied", "PERMISSION_DENIED")))

    with pytest.raises(AuthenticationError):
        await collect(adapter)


async def test_the_shape_gemini_actually_refuses_a_bad_key_in_is_recognised() -> None:
    """Gemini answers a mistyped or revoked key with 400 INVALID_ARGUMENT and
    not with 401 - exactly the sort of vendor detail an adapter exists for."""
    bad_key = refusal(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT")
    adapter = adapter_for(FakeModels(error=bad_key))

    with pytest.raises(AuthenticationError):
        await collect(adapter)


# --------------------------------------------------------------------------
# How the contract suite drives this adapter
#
# `test_llm_adapters.py` scripts a provider without naming one; this is where
# that script becomes Gemini. Everything vendor-shaped about the contract suite
# is in these three functions, which is the same rule the adapter itself lives
# by (section 3.2).
# --------------------------------------------------------------------------


def _chunk(step: Step) -> types.GenerateContentResponse:
    """One step of a scripted provider, in the shapes Gemini produces."""
    if isinstance(step, Nothing):
        return types.GenerateContentResponse(candidates=[])
    if isinstance(step, Says):
        return text_chunk(step.text)
    if isinstance(step, Calls):
        return call_chunk(step.name, dict(step.arguments), call_id=step.id)
    if isinstance(step, Starts):
        return call_chunk(step.name, dict(step.arguments), call_id=step.id, still_streaming=True)
    if isinstance(step, Spends):
        return types.GenerateContentResponse(
            candidates=[],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=step.input,
                candidates_token_count=step.output,
                cached_content_token_count=step.cached,
            ),
        )
    return final_chunk()


def _how_it_refuses(refuses: Refuses | None) -> Exception | None:
    if refuses is None:
        return None
    if refuses is Refuses.THE_KEY:
        # The shape it really uses, rather than the 401 everybody expects.
        return refusal(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT")
    return refusal(503, COMPLAINT, "UNAVAILABLE")


def build(
    *script: Step,
    refuses: Refuses | None = None,
    after: int = 0,
    models: Sequence[tuple[str, str]] = (),
) -> LLMProvider:
    """What `contract.Build` asks for, answered in Gemini's own shapes."""
    return adapter_for(
        FakeModels(
            chunks=[_chunk(step) for step in script],
            models=[
                types.Model(
                    name=f"models/{model_id}",
                    display_name=name,
                    supported_actions=["generateContent"],
                )
                for model_id, name in models
            ],
            error=_how_it_refuses(refuses),
            error_after=after,
        )
    )


GEMINI = Adapter(name="gemini", build=build)
