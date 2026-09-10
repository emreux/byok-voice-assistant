"""Any OpenAI-compatible server, translated into the protocol (design.md section 3.2).

This is the only file in the project that imports `openai`. The chat
completions API that SDK speaks is the shape most of the market settled on:
OpenAI itself, OpenRouter, Groq, DeepSeek, xAI, Mistral, Together, a local
Ollama or LM Studio, and anything else that takes `Authorization: Bearer`
and a `base_url`. One adapter, one SDK, fifteen providers; the only thing
that differs between them is the address and the key, and both come from
`providers.toml` through the registry.

What this adapter absorbs is mostly the streaming of tool calls. OpenAI
sends the *arguments* of a call in slices - `{"na`, `me": "no`, `tepad"}` -
each chunk carrying the index of the call it belongs to, the first one also
the id and the name. The protocol promises a `ToolCall` is never half-built
(the permission gate of section 3.9 cannot judge an action it can only see
the beginning of), so the slices are gathered here by index and each call is
emitted once, whole, when the stream says generation is finished - and not
one moment earlier. The trap is that `json.loads` sometimes *succeeds* on a
prefix (`{}` is valid JSON, and so is `{"name": "no"}`): "it parses" is not
"it is complete", "the provider said it stopped" is.

The token counts come once, on a final chunk with no choices, and only when
`stream_options.include_usage` asks for them; the finish reason comes on the
chunk before. The two are held back and reported together at the end, as
the protocol has every adapter do. The system prompt is a message like any
other, the assistant's earlier calls go back as `tool_calls` on its own
message, and a result goes back under the role `tool` with the id of the
call it answers.

An empty key is sent as a placeholder: the SDK refuses to be built without
one, and a local Ollama has none to give. Refusals arrive as the SDK's own
exceptions - a 401 or 403 for the key, any other status for the request,
the transport's for a network that is not there - and each is turned into
one of the two the application knows. The transport is `httpx2`; an error
in the middle of a stream is its own and not the SDK's, and is caught as
such, as the Gemini adapter catches `httpx`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx2
import openai
from loguru import logger
from openai.types.chat import ChatCompletionChunk

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

__all__ = ["KEY_PLACEHOLDER", "OpenAICompatAdapter"]

# What is sent as the key when there is none. The SDK refuses to be built
# without one; a local Ollama or LM Studio ignores whatever it is given.
KEY_PLACEHOLDER = "none"


@dataclass
class _Gathering:
    """One tool call whose slices are still arriving, kept by its index."""

    id: str = ""
    name: str = ""
    arguments: list[str] = field(default_factory=list)


class OpenAICompatAdapter:
    """Speaks the chat completions API to whatever server `base_url` names."""

    id = "openai_compat"

    # OpenAI's `reasoning_effort`, DeepSeek's reasoning content and the
    # rest stay here until something in the agent loop reads them. Nothing
    # does yet, so nothing is announced.
    capabilities: frozenset[str] = frozenset()

    def __init__(self, api_key: str, *, base_url: str, client: Any | None = None) -> None:
        self._client = client if client is not None else _client(api_key, base_url)

    async def validate_credentials(self) -> bool:
        """Asks for the model list; a key that cannot list models cannot chat either.

        Only a refused key is `False`; a server that could not be reached,
        or that refused the request for its own reasons, raises instead -
        the answer to that is not "paste another key".
        """
        try:
            await self.list_models()
        except AuthenticationError:
            return False
        return True

    async def list_models(self) -> list[ModelInfo]:
        """Every model the server lists, by id.

        The list endpoint says nothing about what a model can do - OpenAI's
        includes its speech and embedding models - so `supports_tools` is
        left unknown for the probe of 2.6 to answer. OpenAI's API has no
        display name, so the id is shown; OpenRouter adds a `name` field
        of its own ("Google: Gemini 2.5 Flash"), which the SDK keeps as an
        extra, and that is shown when it is there.
        """
        try:
            models = [model async for model in self._client.models.list()]
        except openai.APIError as refusal:
            raise _refused(refusal) from refusal
        except httpx2.HTTPError as failure:
            raise _unreachable(failure) from failure

        return [
            ModelInfo(id=model.id, display_name=_display_name(model), supports_tools=None)
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
            "messages": [_to_message(m) for m in messages],
            "stream": True,
            # The token counts arrive on one last chunk, and only when asked
            # for. Asked for, so that the bill of section 6 has numbers.
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens,
        }
        if tools:
            # Left out rather than sent empty: some servers refuse `[]`.
            request["tools"] = [_declare(t) for t in tools]
        if temperature is not None:
            request["temperature"] = temperature

        # `create` is a coroutine that returns the stream, so it is awaited
        # once and iterated after - not awaited per chunk.
        try:
            chunks = await self._client.chat.completions.create(**request)
        except openai.APIError as refusal:
            raise _refused(refusal) from refusal
        except httpx2.HTTPError as failure:
            raise _unreachable(failure) from failure

        gathering: dict[int, _Gathering] = {}
        finish_reason: str | None = None
        usage: Usage | None = None

        try:
            async for chunk in chunks:
                for delta in _text(chunk):
                    yield delta
                _gather(chunk, gathering)

                reason = _finish_reason(chunk)
                if reason is not None:
                    finish_reason = reason
                    # Generation is over, so every call is whole: this is
                    # the moment they are emitted, and the only one.
                    for call in _finished(gathering):
                        yield Delta(tool_call=call)
                counted = _usage(chunk)
                if counted is not None:
                    usage = counted
        except openai.APIError as refusal:
            # Half an answer had already been yielded. The turn is abandoned
            # either way, and `agent/core.py` throws the half away with it.
            raise _refused(refusal) from refusal
        except httpx2.HTTPError as failure:
            raise _unreachable(failure) from failure

        # A server that ended the stream without saying why still ended it,
        # and what it streamed of a call is all there will be.
        for call in _finished(gathering):
            yield Delta(tool_call=call)

        if finish_reason is not None or usage is not None:
            yield Delta(finish_reason=finish_reason, usage=usage)


def _client(api_key: str, base_url: str) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key=api_key or KEY_PLACEHOLDER, base_url=base_url)


def _unreachable(failure: httpx2.HTTPError) -> ProviderError:
    """The transport failed in the middle of a stream, before the SDK could
    dress it up: a socket that closed, a read that timed out. The SDK
    translates these itself on the way in, but not once the stream is being
    read, and nothing above this layer would catch the transport's own."""
    return ProviderError(f"the server could not be reached ({type(failure).__name__}): {failure}")


def _refused(error: openai.APIError) -> ProviderError:
    """Turns one of the SDK's refusals into one of the two the application knows.

    A 401 or a 403 is the key - mistyped, revoked, out of credit, or a
    server that wanted one and got the placeholder. Any other status is
    the request: a 429 from a free tier, a 404 for a model name, a 503.
    A connection error is the request too, for the transport's reasons.
    """
    if isinstance(error, openai.AuthenticationError | openai.PermissionDeniedError):
        return AuthenticationError(f"the key was refused ({error.status_code}): {error.message}")
    if isinstance(error, openai.APIStatusError):
        return ProviderError(
            f"the server refused the request ({error.status_code}): {error.message}"
        )
    if isinstance(error, openai.APIConnectionError):
        return ProviderError(f"the server could not be reached ({type(error).__name__}): {error}")
    return ProviderError(f"the server failed the request: {error}")


# --------------------------------------------------------------------------
# Out: the protocol's messages and tools, in the API's shapes
# --------------------------------------------------------------------------


def _to_message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}

    if message.role == "assistant" and message.tool_calls:
        out: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        # As one string, the way it was streamed: the API
                        # carries arguments as JSON text, not as an object.
                        "arguments": json.dumps(dict(call.arguments), ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ],
        }
        # Content beside the calls is optional and some servers refuse an
        # empty one, so it travels only when there is some.
        if message.content:
            out["content"] = message.content
        return out

    return {"role": message.role, "content": message.content}


def _declare(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        },
    }


# --------------------------------------------------------------------------
# In: one chunk at a time
# --------------------------------------------------------------------------


def _display_name(model: Any) -> str:
    named = getattr(model, "name", None)
    return named if isinstance(named, str) and named.strip() else str(model.id)


def _text(chunk: ChatCompletionChunk) -> list[Delta]:
    """The text a chunk carries, if any. A chunk announcing the role, or one
    carrying only the token counts, has none and produces nothing."""
    return [Delta(text=choice.delta.content) for choice in chunk.choices if choice.delta.content]


def _gather(chunk: ChatCompletionChunk, gathering: dict[int, _Gathering]) -> None:
    """Adds whatever slices of tool calls this chunk carries to the ones
    being gathered, by index. The first slice of a call brings its id and
    name; the later ones bring only more of the arguments."""
    for choice in chunk.choices:
        for slice_ in choice.delta.tool_calls or []:
            call = gathering.setdefault(slice_.index, _Gathering())
            if slice_.id:
                call.id = slice_.id
            if slice_.function is not None:
                if slice_.function.name:
                    call.name = slice_.function.name
                if slice_.function.arguments:
                    call.arguments.append(slice_.function.arguments)


def _finished(gathering: dict[int, _Gathering]) -> list[ToolCall]:
    """Every gathered call as one whole `ToolCall`, in the order the server
    numbered them; the gathering is emptied on the way.

    Arguments that do not parse as JSON are a call the server never
    finished - the stream died, or the model was cut off mid-call. What
    the gate would be handed is half an action, so the call is dropped
    and the log says so; the protocol's promise is kept by not emitting.
    """
    calls: list[ToolCall] = []
    for index in sorted(gathering):
        call = gathering.pop(index)
        text = "".join(call.arguments)
        try:
            arguments = json.loads(text) if text.strip() else {}
        except ValueError:
            logger.warning(
                "the call to {name} was dropped: its arguments never became JSON: {text!r}",
                name=call.name,
                text=text,
            )
            continue
        if not isinstance(arguments, dict):
            logger.warning(
                "the call to {name} carried {text!r} for arguments", name=call.name, text=text
            )
            continue
        calls.append(ToolCall(id=call.id, name=call.name, arguments=arguments))
    return calls


def _finish_reason(chunk: ChatCompletionChunk) -> str | None:
    for choice in chunk.choices:
        if choice.finish_reason is not None:
            return str(choice.finish_reason)
    return None


def _usage(chunk: ChatCompletionChunk) -> Usage | None:
    counted = chunk.usage
    if counted is None:
        return None
    details = counted.prompt_tokens_details
    return Usage(
        input_tokens=counted.prompt_tokens or 0,
        output_tokens=counted.completion_tokens or 0,
        cached_tokens=(details.cached_tokens or 0) if details is not None else 0,
    )
