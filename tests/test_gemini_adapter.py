"""What the Gemini adapter must turn its vendor's shapes into.

No network. The fake client mimics the one method the adapter calls, but the
chunks it yields are real `google.genai.types` objects, so a field this suite
reads is a field the SDK actually has. A hand-rolled stub would let the tests
pass while the adapter reads an attribute that does not exist.

Phase 1 calls the adapter with an empty tool list, so the tool-call tests
describe a contract that only pays off in phase 2. They are here now because
the buffering rule - never emit a half-built call - is easy to get right while
writing the loop and expensive to retrofit once the permission gate depends on
it (design.md section 3.2).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from google.genai import types

from assistant.llm.base import Delta, LLMProvider, Message, ToolSpec
from assistant.llm.gemini_adapter import GeminiAdapter


def text_chunk(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))
        ]
    )


def final_chunk(
    *, reason: types.FinishReason = types.FinishReason.STOP, prompt: int = 0, output: int = 0
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[]), finish_reason=reason)
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
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
    ) -> None:
        self.chunks = chunks or []
        self.models = models or []
        self.error = error
        self.sent: dict[str, Any] = {}

    async def generate_content_stream(
        self, *, model: str, contents: Any, config: Any
    ) -> AsyncIterator[types.GenerateContentResponse]:
        self.sent = {"model": model, "contents": contents, "config": config}
        if self.error is not None:
            raise self.error

        async def chunks() -> AsyncIterator[types.GenerateContentResponse]:
            for chunk in self.chunks:
                yield chunk

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


def test_the_adapter_satisfies_the_protocol() -> None:
    provider: LLMProvider = adapter_for(FakeModels())

    assert isinstance(provider, LLMProvider)
    assert provider.id == "gemini"


async def test_text_arrives_in_the_order_the_provider_sent_it() -> None:
    adapter = adapter_for(FakeModels([text_chunk("Mer"), text_chunk("haba")]))

    deltas = await collect(adapter)

    assert [d.text for d in deltas] == ["Mer", "haba"]


async def test_the_closing_chunk_carries_the_token_counts() -> None:
    adapter = adapter_for(FakeModels([text_chunk("hi"), final_chunk(prompt=12, output=5)]))

    usage = [d.usage for d in await collect(adapter) if d.usage is not None]

    assert len(usage) == 1
    assert (usage[0].input_tokens, usage[0].output_tokens) == (12, 5)


async def test_the_reason_generation_stopped_is_reported() -> None:
    adapter = adapter_for(FakeModels([final_chunk(reason=types.FinishReason.MAX_TOKENS)]))

    reasons = [d.finish_reason for d in await collect(adapter) if d.finish_reason]

    assert reasons == ["MAX_TOKENS"]


async def test_a_complete_tool_call_is_handed_over_whole() -> None:
    adapter = adapter_for(FakeModels([call_chunk("open_app", {"name": "notepad"})]))

    calls = [d.tool_call for d in await collect(adapter) if d.tool_call is not None]

    assert len(calls) == 1
    assert (calls[0].name, dict(calls[0].arguments)) == ("open_app", {"name": "notepad"})


async def test_a_tool_call_still_being_streamed_is_withheld() -> None:
    """The permission gate cannot judge an action it can only see the start of."""
    adapter = adapter_for(
        FakeModels(
            [
                call_chunk("send_email", {"to": "a@b"}, still_streaming=True),
                call_chunk("send_email", {"to": "a@b.com", "body": "hi"}),
            ]
        )
    )

    calls = [d.tool_call for d in await collect(adapter) if d.tool_call is not None]

    assert len(calls) == 1
    assert dict(calls[0].arguments) == {"to": "a@b.com", "body": "hi"}


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


async def test_a_rejected_key_is_reported_rather_than_raised() -> None:
    adapter = adapter_for(FakeModels(error=RuntimeError("API key not valid")))

    assert await adapter.validate_credentials() is False


async def test_a_working_key_validates() -> None:
    models = FakeModels(
        models=[types.Model(name="models/gemini-x", supported_actions=["generateContent"])]
    )

    assert await adapter_for(models).validate_credentials() is True


def test_the_adapter_announces_what_it_can_do() -> None:
    adapter = adapter_for(FakeModels())

    assert isinstance(adapter.capabilities, frozenset)


async def test_an_empty_chunk_does_not_become_an_empty_delta() -> None:
    """Providers send chunks that only advance their own state; they are not output."""
    adapter = adapter_for(
        FakeModels([types.GenerateContentResponse(candidates=[]), text_chunk("hi")])
    )

    deltas = await collect(adapter)

    assert [d.text for d in deltas] == ["hi"]


@pytest.mark.parametrize("role", ["user", "assistant"])
async def test_every_message_reaches_the_provider(role: str) -> None:
    models = FakeModels([text_chunk("ok")])
    adapter = adapter_for(models)
    message = Message.user("a") if role == "user" else Message.assistant("a")

    await collect(adapter, messages=[message])

    assert len(models.sent["contents"]) == 1
