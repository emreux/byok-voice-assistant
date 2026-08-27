"""Contract tests for the provider layer (design.md section 3.2).

Every adapter has to pass this file unchanged. Phase 1 ships one adapter, so
for now the suite proves the shape of the contract itself: that an async
generator satisfies `LLMProvider`, that a stream is consumed without awaiting
the call, and that no provider SDK leaks into the protocol module.

The `def stream` signature is the subtle part. Adapters implement it with
`async def ... yield`, which returns an `AsyncIterator` when called, with no
`await`. Declaring it `async def` in the protocol would force callers to await
first and no adapter would satisfy the type. `test_stream_is_consumed_without
_awaiting_the_call` is what keeps somebody from "fixing" that signature.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from assistant.llm.base import (
    Delta,
    LLMProvider,
    Message,
    ModelInfo,
    ToolCall,
    ToolSpec,
    Usage,
)

PROVIDER_SDKS = frozenset({"google", "openai", "anthropic", "litellm"})


class FakeProvider:
    """A minimal provider that satisfies the protocol without any network call.

    Adapter tests reuse this as the reference implementation: whatever a real
    adapter does differently, it must still look like this from the outside.
    """

    id = "fake"

    def __init__(self, deltas: list[Delta] | None = None) -> None:
        self.deltas = deltas if deltas is not None else [Delta(text="hello")]
        self.calls: list[dict[str, object]] = []

    async def validate_credentials(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="fake-1", display_name="Fake 1")]

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        self.calls.append({"messages": messages, "tools": tools, "model": model})
        for delta in self.deltas:
            yield delta


def test_an_async_generator_implementation_satisfies_the_protocol() -> None:
    # The annotation is the real assertion: mypy --strict checks it structurally,
    # which is stricter than isinstance, since a runtime protocol check only looks
    # for the names and ignores every signature.
    provider: LLMProvider = FakeProvider()

    assert isinstance(provider, LLMProvider)


async def test_stream_is_consumed_without_awaiting_the_call() -> None:
    provider = FakeProvider([Delta(text="Mer"), Delta(text="haba")])

    chunks = [delta.text async for delta in provider.stream([], [], model="fake-1")]

    assert chunks == ["Mer", "haba"]


def test_the_protocol_declares_stream_without_async_def() -> None:
    """An `async def` here would make every adapter fail to type check."""
    assert not inspect.iscoroutinefunction(LLMProvider.stream)


def test_a_tool_result_needs_the_call_it_answers() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        Message(role="tool", content="42")


def test_only_an_assistant_message_carries_tool_calls() -> None:
    call = ToolCall(id="1", name="open_app", arguments={"name": "notepad"})

    with pytest.raises(ValueError, match="assistant"):
        Message(role="user", content="open notepad", tool_calls=(call,))


def test_a_message_cannot_be_edited_after_it_is_built() -> None:
    message = Message(role="user", content="merhaba")

    with pytest.raises(AttributeError):
        message.content = "something else"  # type: ignore[misc]


def test_an_empty_delta_carries_nothing() -> None:
    delta = Delta()

    assert delta.text is None
    assert delta.tool_call is None
    assert delta.finish_reason is None
    assert delta.usage is None


def test_token_counts_start_at_zero() -> None:
    usage = Usage()

    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens) == (0, 0, 0)


def test_the_protocol_module_imports_no_provider_sdk() -> None:
    """agent/, tools/ and policy.py read this module; a vendor import here leaks everywhere."""
    source = Path(inspect.getfile(Delta)).read_text(encoding="utf-8")
    imported: set[str] = set()

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    assert not imported & PROVIDER_SDKS
