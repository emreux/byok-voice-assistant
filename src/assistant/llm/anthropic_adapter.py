"""Anthropic's Messages API, translated into the protocol (design.md section
3.2, phase 4.5; 17 Sep 2026).

This is the only file in the project that imports `anthropic`. The third
adapter, and the one section 10's claim was waiting for: one loop, one
gate, three vendors. It was written against the SDK's own types and the
contract suite, not against a live key - the owner develops with a Google
key until release (section 12, decision 22) - and the README says so.

What this adapter absorbs is the shape of a message. Anthropic has no
system role: the system prompt is a field of the request, and it is sent
as a block marked `cache_control` so that the frozen prefix of
`agent/prompts.py` is read from the cache on every request after the
first - the one vendor-only feature this adapter announces through
`capabilities`, and the reason the loop keeps the prompt byte for byte the
same. An assistant turn is a list of blocks, text and `tool_use` side by
side; a tool's result goes back as a `tool_result` block inside a *user*
message, and results that answer calls made together travel together in
one message, as the API asks.

A tool call streams as a `content_block_start` naming the tool, then
`input_json_delta` pieces of its arguments, then `content_block_stop`. The
stop is what makes the arguments whole, and it is the moment - the only
one - a `ToolCall` is emitted (section 3.9: never half-built). The token
counts come in two halves, the input at `message_start` and the output at
`message_delta`, and are reported once, together, at the end, as the
protocol has every adapter do. `input_tokens` on the wire excludes what
the cache served; the protocol's `input_tokens` is the whole prompt and
`cached_tokens` the part of it that was cached, so the two are added here.

Refusals arrive as the SDK's own exceptions - a 401 or 403 for the key,
any other status for the request, the transport's for a network that is
not there - and each is turned into one of the two the application knows.
The transport is `httpx2`, as it is under the OpenAI SDK, and an error in
the middle of a stream is its own.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2
from loguru import logger

from assistant.llm.base import (
    AuthenticationError,
    Delta,
    Message,
    ModelInfo,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = ["AnthropicAdapter"]

# Sent when the registry has no key: the SDK refuses to be built without
# one, and the refusal that comes back is then the API's, in words.
KEY_PLACEHOLDER = "missing"


@dataclass
class _Gathering:
    """One tool call whose argument pieces are still arriving, by block index."""

    id: str
    name: str
    pieces: list[str] = field(default_factory=list)


class AnthropicAdapter:
    """Speaks the Messages API."""

    id = "anthropic"

    # The frozen system prompt is sent with a cache breakpoint; nothing in
    # the loop reads this yet, but it is what the loop would ask.
    capabilities: frozenset[str] = frozenset({"prompt_caching"})

    def __init__(self, api_key: str, *, client: Any | None = None) -> None:
        self._client = (
            client
            if client is not None
            else anthropic.AsyncAnthropic(api_key=api_key or KEY_PLACEHOLDER)
        )

    async def validate_credentials(self) -> bool:
        """Asks for the model list; a key that cannot list models cannot chat either."""
        try:
            await self.list_models()
        except AuthenticationError:
            return False
        return True

    async def list_models(self) -> list[ModelInfo]:
        """Every model the key can reach. The list says nothing about tools,
        so `supports_tools` is left for the probe of 2.6 to answer."""
        try:
            models = [model async for model in self._client.models.list()]
        except anthropic.APIError as refusal:
            raise _refused(refusal) from refusal
        except httpx2.HTTPError as failure:
            raise _unreachable(failure) from failure
        return [
            ModelInfo(id=str(model.id), display_name=str(model.display_name), supports_tools=None)
            for model in models
        ]

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _to_messages(messages),
        }
        system = "\n\n".join(m.content for m in messages if m.role == "system" and m.content)
        if system:
            # One block, marked: the prefix that never changes is the one
            # worth caching (architecture guide section 2).
            request["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if tools:
            request["tools"] = [_declare(tool) for tool in tools]
        if temperature is not None:
            request["temperature"] = temperature

        gathering: dict[int, _Gathering] = {}
        finish_reason: str | None = None
        usage: Usage | None = None

        try:
            # The request is made on entry, and the SDK's stream - and the
            # connection under it - is closed on the way out, however the
            # way out is taken.
            async with self._client.messages.stream(**request) as events:
                async for raw in events:
                    # The SDK's stream yields its own helper events beside
                    # the wire's; only the wire's are read, by their `type`.
                    event: Any = raw
                    kind = getattr(event, "type", None)
                    if kind == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta" and delta.text:
                            yield Delta(text=delta.text)
                        elif delta.type == "input_json_delta" and event.index in gathering:
                            gathering[event.index].pieces.append(delta.partial_json)
                    elif kind == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            gathering[event.index] = _Gathering(id=block.id, name=block.name)
                    elif kind == "content_block_stop":
                        call = _finished(gathering, event.index)
                        if call is not None:
                            yield Delta(tool_call=call)
                    elif kind == "message_start":
                        usage = _usage_at_start(event.message.usage)
                    elif kind == "message_delta":
                        if event.delta.stop_reason is not None:
                            finish_reason = str(event.delta.stop_reason)
                        usage = _usage_at_delta(event.usage, usage)
        except anthropic.APIError as refusal:
            raise _refused(refusal) from refusal
        except httpx2.HTTPError as failure:
            raise _unreachable(failure) from failure

        # A stream that ended without stopping its blocks still ended, and
        # what it streamed of a call is all there will be.
        for index in sorted(gathering):
            call = _finished(gathering, index)
            if call is not None:
                yield Delta(tool_call=call)

        if finish_reason is not None or usage is not None:
            yield Delta(finish_reason=finish_reason, usage=usage)


def _unreachable(failure: httpx2.HTTPError) -> ProviderError:
    return ProviderError(f"the server could not be reached ({type(failure).__name__}): {failure}")


def _refused(error: anthropic.APIError) -> ProviderError:
    """One of the SDK's refusals as one of the two the application knows.

    A 401 or a 403 is the key. Any other status is the request: a 429
    from the rate limit, a 404 for a model name, a 529 when the API is
    overloaded. A connection error is the request too.
    """
    if isinstance(error, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return AuthenticationError(f"the key was refused ({error.status_code}): {error.message}")
    if isinstance(error, anthropic.APIStatusError):
        return ProviderError(
            f"the server refused the request ({error.status_code}): {error.message}"
        )
    if isinstance(error, anthropic.APIConnectionError):
        return ProviderError(f"the server could not be reached ({type(error).__name__}): {error}")
    return ProviderError(f"the server failed the request: {error}")


# --------------------------------------------------------------------------
# Out: the protocol's messages and tools, in the API's shapes
# --------------------------------------------------------------------------


def _to_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """The conversation without its system prompt, with tool results that
    answer calls made together gathered into one user message."""
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
            }
            if out and out[-1].get("_results"):
                out[-1]["content"].append(result)
            else:
                out.append({"role": "user", "content": [result], "_results": True})
            continue
        if message.role == "assistant" and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": dict(call.arguments),
                }
                for call in message.tool_calls
            )
            out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": message.role, "content": message.content})
    for entry in out:
        entry.pop("_results", None)
    return out


def _declare(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": dict(tool.parameters),
    }


# --------------------------------------------------------------------------
# In: one event at a time
# --------------------------------------------------------------------------


def _finished(gathering: dict[int, _Gathering], index: int) -> ToolCall | None:
    """The call gathered under `index`, whole, or `None` when the block was
    not a tool call - or its arguments never became JSON, in which case the
    call is dropped and the log says so."""
    call = gathering.pop(index, None)
    if call is None:
        return None
    text = "".join(call.pieces)
    try:
        arguments = json.loads(text) if text.strip() else {}
    except ValueError:
        logger.warning(
            "the call to {name} was dropped: its arguments never became JSON: {text!r}",
            name=call.name,
            text=text,
        )
        return None
    if not isinstance(arguments, dict):
        logger.warning(
            "the call to {name} carried {text!r} for arguments", name=call.name, text=text
        )
        return None
    return ToolCall(id=call.id, name=call.name, arguments=arguments)


def _usage_at_start(counted: Any) -> Usage:
    """The input half of the bill. On the wire `input_tokens` leaves out
    what the cache served or wrote; the protocol counts the whole prompt."""
    cached = int(getattr(counted, "cache_read_input_tokens", 0) or 0)
    written = int(getattr(counted, "cache_creation_input_tokens", 0) or 0)
    return Usage(
        input_tokens=int(counted.input_tokens or 0) + cached + written,
        output_tokens=int(counted.output_tokens or 0),
        cached_tokens=cached,
    )


def _usage_at_delta(counted: Any, so_far: Usage | None) -> Usage | None:
    """The output half, and the whole when the delta carries the input too.
    A delta that counts nothing leaves the bill as it was."""
    if counted is None:
        return so_far
    if getattr(counted, "input_tokens", None) is not None:
        return _usage_at_start(counted)
    if so_far is None or not counted.output_tokens:
        # Nothing counted, or an output of zero: the API counts what it
        # has written so far, and a delta that says nothing is not a bill.
        return so_far
    return Usage(
        input_tokens=so_far.input_tokens,
        output_tokens=int(counted.output_tokens),
        cached_tokens=so_far.cached_tokens,
    )
