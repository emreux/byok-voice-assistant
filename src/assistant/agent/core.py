"""The agent loop (design.md item 1.9 and 2.1c, architecture guide section 1).

The whole product is arranged around a handful of lines: put what the user
said at the end of the recent conversation, send it to the model, and if the
model asks for a tool, run it through the gate of section 3.9, put the result
back in and ask again. It is a loop because one sentence can take several
requests - "yarın Ahmet'i aramayı hatırlat" is the time, then the reminder,
then the answer - and the answer is whatever the model says once it stops
asking. Phase 1 ran this once per turn, with nothing to go round for; phase
2.1c gave it its `while`.

Four decisions here are worth more than the code that implements them.

**A turn is not a message.** The window of section 3.7 keeps the last twelve
*turns*, where a turn is what the user said plus everything that followed it.
Counting messages instead would be simpler and would eventually cut between a
tool call and the result that answers it - a shape every provider rejects. The
cut is at what the user said, so a tool call and its result stay together.

**Nothing is remembered until the answer is whole.** The conversation is
replaced in one assignment at the end of the turn. A dropped connection, the
sixty second `THINKING` timeout of section 3.1, a cancelled turn: all of them
leave the history exactly as it was, so the user repeats themselves and gets a
clean turn rather than one built on half of a previous one.

**The gate is handed in, not imported.** This file never sees `policy.py`. The
composition root builds the gate and passes it; a test passes a fake; and the
loop cannot be talked into running a tool any other way, because it has no
other way (invariant 1). Offering tools without a gate is refused outright.
So is the one who answers a tool's question: `confirm` comes with each turn
from `app.py`, because the microphone that hears the yes lives there and the
gate that needs it is built first - neither can be built holding the other.

**The system prompt is not part of the conversation.** It is prepended to each
request instead of living in the history, which keeps it out of reach of the
window and byte-identical from turn to turn - the one thing prompt caching
needs (architecture guide section 2).

The limit on tool calls below is the first of the limits of section 3.11 and
moves to `limits.py` with the rest of them in 2.4. It belongs beside this file
rather than inside an adapter: written in an adapter it would be written three
times, and one of the three would be forgotten.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from assistant.agent.prompts import SYSTEM_PROMPT
from assistant.llm.base import LLMProvider, Message, ToolCall, ToolSpec, Usage
from assistant.tools.registry import ToolRegistry

__all__ = [
    "MAX_TOOL_CALLS",
    "TOOL_LIMIT_REACHED",
    "WINDOW_TURNS",
    "Agent",
    "Answer",
    "Confirm",
    "Dispatch",
    "decline",
    "window",
]

# How many turns the model is allowed to remember (section 3.7). Cost grows
# roughly with the square of a conversation's length, because every turn resends
# every turn before it; this is the ceiling that stops it.
WINDOW_TURNS = 12

# How many tool calls one turn may make before the model is made to answer with
# what it has (section 3.11). A model that keeps asking is a model in a loop,
# and every round is a request paid for. Moves to `limits.py` in 2.4.
MAX_TOOL_CALLS = 8

# What a call over the limit is answered with - as a tool result, in the
# model's own channel, rather than as a user message it might argue with.
TOOL_LIMIT_REACHED = "Tool limit reached; answer with what you have."


# Asks the user a question out loud and answers yes or no. Who actually asks
# is decided by whoever calls `reply`: the state machine hands over the
# microphone, a test hands over a fake, and the gate never learns which.
Confirm = Callable[[str], Awaitable[bool]]


async def decline(question: str) -> bool:
    """Nobody to ask means no - never a quiet yes."""
    return False


class Dispatch(Protocol):
    """The gate, as the loop sees it: one call in, the words for the model out.

    `turn_id` names the turn in `tool_audit` (section 3.9) and `confirm` is
    whoever can ask the user a question; the loop carries both from `app.py`
    to the gate and reads neither.
    """

    async def __call__(self, call: ToolCall, *, turn_id: str, confirm: Confirm) -> str: ...


@dataclass(frozen=True, slots=True)
class Answer:
    """What one turn produced: the words, what they cost, and why it ended.

    `usage` is the total over every request the turn made - a turn that ran
    a tool made at least two - and is what item 1.11 writes to the log and
    section 6 later bills from. `finish_reason` is that of the last request,
    kept so that an answer that ended can be told apart from one cut off at
    the token limit. Nothing reads it yet: the sentence by sentence speech of
    2.8 is where a cut-off answer has to be said out loud.
    """

    text: str
    usage: Usage
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class _Reply:
    """What one request came back with, before the loop decides what to do."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    finish_reason: str | None


class Agent:
    """One conversation with one model. The turns live in memory and die with it.

    Not thread safe and not re-entrant: the state machine of section 3.1 is in
    exactly one state at a time, so a second turn cannot start while the first
    is still running.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str,
        system_prompt: str = SYSTEM_PROMPT,
        tools: ToolRegistry | None = None,
        dispatch: Dispatch | None = None,
    ) -> None:
        if tools is not None and dispatch is None:
            # A tool offered without a gate is a tool that would run without
            # one, which is the one thing section 3.9 forbids.
            raise ValueError("tools were offered without a gate to run them through")
        self._provider = provider
        self._model = model
        self._system = Message.system(system_prompt)
        self._tools = tools
        self._dispatch = dispatch
        self._history: list[Message] = []

    async def reply(self, said: str, *, turn_id: str = "", confirm: Confirm = decline) -> Answer:
        """Answers one thing the user said, running whatever tools it takes,
        and remembers having done so.

        `confirm` is who a tool that wants a yes asks. Left out, the answer
        is no: a tool that needs asking about is a tool that does not run
        until somebody can be asked.
        """
        conversation = window([*self._history, Message.user(said)])
        spent = Usage()
        dispatched = 0

        while True:
            # Past the limit the model is offered nothing, so the only thing
            # left for it to do is answer.
            offered = self._offered() if dispatched < MAX_TOOL_CALLS else []
            got = await self._ask(conversation, offered)
            spent = spent + got.usage

            # Nothing asked for, no gate to ask it of, or a call made after
            # being offered nothing: the turn ends with the words there are.
            if not got.tool_calls or self._dispatch is None or not offered:
                break

            conversation.append(Message.assistant(got.text, got.tool_calls))
            for call in got.tool_calls:
                if dispatched < MAX_TOOL_CALLS:
                    result = await self._dispatch(call, turn_id=turn_id, confirm=confirm)
                    dispatched += 1
                else:
                    result = TOOL_LIMIT_REACHED
                conversation.append(Message.tool_result(call, result))

        # A model that produced no text said nothing, and a message with no
        # content is refused outright by several providers - so a silent turn
        # is not written down at all rather than poisoning every turn after
        # it. When it did speak, everything in between - the calls and their
        # results - is remembered with it, so the model knows what it did.
        if got.text:
            self._history = [*conversation, Message.assistant(got.text)]
        return Answer(text=got.text, usage=spent, finish_reason=got.finish_reason)

    def _offered(self) -> list[ToolSpec]:
        return [] if self._tools is None else self._tools.specs()

    async def _ask(self, conversation: list[Message], tools: list[ToolSpec]) -> _Reply:
        """Runs one request to the end and gathers the stream into one reply.

        The whole reply is waited for before anything is spoken (section 4
        says so and budgets for it); sentence-by-sentence speech is 2.8, and
        `tts.base` already has the regrouping it needs.
        """
        spoken: list[str] = []
        calls: list[ToolCall] = []
        usage = Usage()
        finish_reason: str | None = None

        async for delta in self._provider.stream(
            [self._system, *conversation], tools, model=self._model
        ):
            if delta.text:
                spoken.append(delta.text)
            if delta.tool_call is not None:
                calls.append(delta.tool_call)
            # Both arrive at most once per stream and each adapter promises it -
            # a running total added up would bill the request several times over.
            if delta.usage is not None:
                usage = delta.usage
            if delta.finish_reason is not None:
                finish_reason = delta.finish_reason

        return _Reply(
            text="".join(spoken), tool_calls=tuple(calls), usage=usage, finish_reason=finish_reason
        )


def window(conversation: Sequence[Message], *, turns: int = WINDOW_TURNS) -> list[Message]:
    """The last `turns` turns of a conversation, each of them whole.

    A turn starts where the user spoke and ends where the next one starts, so
    everything the assistant did in between - a tool call and the result that
    answers it - is kept or dropped together.
    """
    starts = [index for index, message in enumerate(conversation) if message.role == "user"]
    if len(starts) <= turns:
        return list(conversation)
    return list(conversation[starts[-turns] :])
