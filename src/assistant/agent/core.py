"""The agent loop (design.md item 1.9, architecture guide section 1).

The whole product is arranged around a handful of lines: put what the user said
at the end of the recent conversation, send it to the model, keep the text that
comes back. Phase 2 adds the other branch - the model asks for a tool, the gate
of section 3.9 decides, the result goes back in and the loop goes round again -
and that branch is why this is a loop rather than a function call. Phase 1 runs
it once per turn because there is nothing yet for it to go round for.

Three decisions here are worth more than the code that implements them.

**A turn is not a message.** The window of section 3.7 keeps the last twelve
*turns*, where a turn is what the user said plus everything that followed it.
Counting messages instead would be simpler and would eventually cut between a
tool call and the result that answers it - a shape every provider rejects. The
cut is at what the user said, so it is already right for phase 2.

**Nothing is remembered until the answer is whole.** The conversation is
replaced in one assignment at the end of the turn. A dropped connection, the
sixty second `THINKING` timeout of section 3.1, a cancelled turn: all of them
leave the history exactly as it was, so the user repeats themselves and gets a
clean turn rather than one built on half of a previous one.

**The system prompt is not part of the conversation.** It is prepended to each
request instead of living in the history, which keeps it out of reach of the
window and byte-identical from turn to turn - the one thing prompt caching
needs (architecture guide section 2).

Limits - tool calls per turn, output tokens, spend - are section 3.11 and land
in `limits.py` in phase 2. They belong beside this file rather than inside an
adapter: written in an adapter they would be written three times, and one of
the three would be forgotten.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from assistant.agent.prompts import SYSTEM_PROMPT
from assistant.llm.base import LLMProvider, Message, ToolSpec, Usage

__all__ = ["WINDOW_TURNS", "Agent", "Answer", "window"]

# How many turns the model is allowed to remember (section 3.7). Cost grows
# roughly with the square of a conversation's length, because every turn resends
# every turn before it; this is the ceiling that stops it.
WINDOW_TURNS = 12


@dataclass(frozen=True, slots=True)
class Answer:
    """What one turn produced: the words, what they cost, and why it ended.

    `usage` is what item 1.11 writes to the log and section 6 later bills from.
    `finish_reason` is the difference between an answer that ended and one that
    was cut off at the token limit, which is a thing the user is told.
    """

    text: str
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
    ) -> None:
        self._provider = provider
        self._model = model
        self._system = Message.system(system_prompt)
        self._history: list[Message] = []

    async def reply(self, said: str) -> Answer:
        """Answers one thing the user said, and remembers having done so."""
        asked = window([*self._history, Message.user(said)])
        answer = await self._collect(asked)

        # A model that produced no text said nothing, and a message with no
        # content is refused outright by several providers - so a silent turn is
        # not written down at all rather than poisoning every turn after it.
        if answer.text:
            self._history = [*asked, Message.assistant(answer.text)]
        return answer

    async def _collect(self, conversation: list[Message]) -> Answer:
        """Runs one request to the end and gathers the stream into one answer.

        Phase 1 waits for the whole reply before speaking (section 4 says so and
        budgets for it); sentence-by-sentence speech is phase 2.8, and `tts.base`
        already has the regrouping it needs.
        """
        # The permission gate of section 3.9 does not exist yet. A tool offered
        # before the gate is a tool that runs without one, so phase 1 offers the
        # model nothing to call.
        no_tools: list[ToolSpec] = []

        spoken: list[str] = []
        usage = Usage()
        finish_reason: str | None = None

        async for delta in self._provider.stream(
            [self._system, *conversation], no_tools, model=self._model
        ):
            if delta.text:
                spoken.append(delta.text)
            # Both arrive at most once per stream and each adapter promises it -
            # a running total added up would bill the turn several times over.
            if delta.usage is not None:
                usage = delta.usage
            if delta.finish_reason is not None:
                finish_reason = delta.finish_reason

        return Answer(text="".join(spoken), usage=usage, finish_reason=finish_reason)


def window(conversation: Sequence[Message], *, turns: int = WINDOW_TURNS) -> list[Message]:
    """The last `turns` turns of a conversation, each of them whole.

    A turn starts where the user spoke and ends where the next one starts, so
    everything the assistant did in between - in phase 2 a tool call and the
    result that answers it - is kept or dropped together.
    """
    starts = [index for index, message in enumerate(conversation) if message.role == "user"]
    if len(starts) <= turns:
        return list(conversation)
    return list(conversation[starts[-turns] :])
