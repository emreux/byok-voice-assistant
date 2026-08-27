"""Google Gemini, translated into the protocol (design.md section 3.2).

This is the only file in the project that imports `google.genai`. Everything
Gemini does differently is absorbed here: the system prompt is a separate
config field rather than a message, the assistant speaks under the role
"model", tool results travel as function responses, and a streamed tool call
can arrive in pieces.

That last one is the reason this adapter is more than a field rename. Gemini
marks an unfinished call with `will_continue`, and the protocol promises a
`ToolCall` is never half-built - the permission gate of section 3.9 cannot
judge an action it can only see the beginning of. So unfinished calls are
dropped and only the completed one is emitted.

Phase 1 calls this with an empty tool list and no tool ever comes back. The
translation is written now anyway, because phase 2 turns tools on and a
protocol that was only ever exercised without them would not be a protocol.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types

from assistant.llm.base import Delta, Message, ModelInfo, ToolCall, ToolSpec, Usage

_TEXT_ACTION = "generateContent"


class GeminiAdapter:
    """Speaks to Google Gemini with an AI Studio key."""

    id = "gemini"

    # Gemini caches long prompts on its own terms and exposes no knob for it, so
    # there is nothing here for the agent loop to switch on yet. The attribute
    # exists so every adapter answers the same question the same way.
    capabilities: frozenset[str] = frozenset()

    def __init__(self, api_key: str, *, client: Any | None = None) -> None:
        self._client = client if client is not None else genai.Client(api_key=api_key)

    async def validate_credentials(self) -> bool:
        """Asks for the model list; a key that cannot list models cannot chat either."""
        try:
            await self.list_models()
        except Exception:
            # Providers reject a bad key in several shapes - a 400, a 403, a
            # transport error. The caller only needs to know it did not work,
            # and the setup command says so in the user's language.
            return False
        return True

    async def list_models(self) -> list[ModelInfo]:
        pager = await self._client.aio.models.list()
        models: list[ModelInfo] = []

        async for model in pager:
            if _TEXT_ACTION not in (model.supported_actions or []):
                continue
            name = (model.name or "").removeprefix("models/")
            models.append(
                ModelInfo(
                    id=name,
                    display_name=model.display_name or name,
                    context_window=model.input_token_limit,
                    supports_tools=None,
                )
            )
        return models

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        system_instruction, contents = _split_system_prompt(messages)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=[types.Tool(function_declarations=[_declare(t) for t in tools])]
            if tools
            else None,
            # Without this the SDK offers to run tools for us, which would put a
            # second execution path next to the permission gate.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        # `generate_content_stream` is a coroutine that returns the iterator, so
        # it is awaited once and iterated after - not awaited per chunk.
        chunks = await self._client.aio.models.generate_content_stream(
            model=model, contents=contents, config=config
        )

        # Gemini repeats a running token total on every chunk and the count
        # only grows, so the last one seen is the total for the request. They
        # are held back and emitted once: a consumer that added up every
        # `Delta.usage` would report - and bill - several times the real usage.
        finish_reason: str | None = None
        usage: Usage | None = None

        async for chunk in chunks:
            for delta in _translate(chunk):
                yield delta

            candidates = chunk.candidates or []
            if candidates and candidates[0].finish_reason is not None:
                finish_reason = candidates[0].finish_reason.value
            running_total = _usage(chunk.usage_metadata)
            if running_total is not None:
                usage = running_total

        if finish_reason is not None or usage is not None:
            yield Delta(finish_reason=finish_reason, usage=usage)


def _split_system_prompt(messages: list[Message]) -> tuple[str | None, list[types.Content]]:
    """Gemini takes the system prompt as configuration, not as a first message."""
    system_parts = [m.content for m in messages if m.role == "system" and m.content]
    contents = [_to_content(m) for m in messages if m.role != "system"]
    return ("\n\n".join(system_parts) or None), contents


def _to_content(message: Message) -> types.Content:
    if message.role == "tool":
        return types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=message.tool_call_id,
                        # Gemini wants an object here; the protocol carries the
                        # result as text, so it travels under a single key.
                        response={"result": message.content},
                    )
                )
            ],
        )

    parts: list[types.Part] = []
    if message.content:
        parts.append(types.Part(text=message.content))
    parts.extend(
        types.Part(
            function_call=types.FunctionCall(id=call.id, name=call.name, args=dict(call.arguments))
        )
        for call in message.tool_calls
    )

    return types.Content(role="model" if message.role == "assistant" else "user", parts=parts)


def _declare(tool: ToolSpec) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description,
        parameters_json_schema=dict(tool.parameters),
    )


def _translate(chunk: types.GenerateContentResponse) -> list[Delta]:
    """Turns one provider chunk into the protocol chunks it carries.

    Text and tool calls only. The finish reason and the token counts are
    summarised once at the end of the stream, in `stream` itself.
    """
    deltas: list[Delta] = []

    if chunk.text:
        deltas.append(Delta(text=chunk.text))

    for call in chunk.function_calls or []:
        if call.will_continue:
            # Arguments are still arriving. Emitting now would hand the gate an
            # action with half its arguments.
            continue
        deltas.append(
            Delta(
                tool_call=ToolCall(
                    id=call.id or "", name=call.name or "", arguments=call.args or {}
                )
            )
        )

    return deltas


def _usage(metadata: types.GenerateContentResponseUsageMetadata | None) -> Usage | None:
    if metadata is None:
        return None
    return Usage(
        input_tokens=metadata.prompt_token_count or 0,
        output_tokens=metadata.candidates_token_count or 0,
        cached_tokens=metadata.cached_content_token_count or 0,
    )
