"""The agent loop, with no tools in it yet (design.md section 8, item 1.9).

Phase 1 proves the loop end to end: what the user said goes to the model with
the last twelve turns for company, and the text that comes back is the answer.
The tool branch - the `while` of the architecture guide, the permission gate of
section 3.9 - arrives in phase 2, and these tests are written so that adding it
changes none of them.

Three claims here are worth the file on their own.

**A turn is not a message.** The window keeps twelve *turns*, and a turn is the
user's message plus everything the assistant did about it. Counting messages
would be simpler and wrong: one day the cut lands between a tool call and the
result that answers it, and every provider rejects that.

**The prefix never moves.** The system prompt is frozen - no clock, no date -
because a provider that caches a prompt only does so while the bytes match
(architecture guide section 2). A cache that stops hitting reports nothing; it
just costs more.

**A turn that failed did not happen.** History is written once the answer is
whole, so a dropped connection, the sixty second `THINKING` timeout of section
3.1 or a cancelled turn leaves the conversation exactly as it was.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from assistant.agent import prompts
from assistant.agent.core import WINDOW_TURNS, Agent, window
from assistant.agent.prompts import BREVITY, LANGUAGE_RULE, PERSONALITY, SYSTEM_PROMPT
from assistant.llm.base import Delta, Message, ModelInfo, ToolCall, ToolSpec, Usage

MODEL = "scripted-1"


@dataclass(frozen=True, slots=True)
class Request:
    """One thing the agent asked of a provider."""

    messages: list[Message]
    tools: list[ToolSpec]
    model: str

    @property
    def turns(self) -> list[Message]:
        """Everything after the system prompt - the conversation itself."""
        return self.messages[1:]


Step = Delta | BaseException | Callable[[], None] | float


class ScriptedProvider:
    """A provider that answers from a script and remembers what it was asked.

    One entry per turn, each a sequence of steps the stream takes in order: a
    `Delta` to yield, an exception to raise where it stands - which is how a
    connection that dies halfway through an answer is written down - a number
    of seconds to spend not answering, or something to do while the model is
    supposedly writing.
    """

    id = "scripted"

    def __init__(self, *turns: Sequence[Step]) -> None:
        self._turns = list(turns)
        self.calls: list[Request] = []

    async def validate_credentials(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=MODEL, display_name="Scripted 1")]

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        self.calls.append(Request(messages=list(messages), tools=list(tools), model=model))
        turn = self._turns.pop(0) if self._turns else [Delta(text="Tamam.")]

        for item in turn:
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, int | float):
                await asyncio.sleep(item)
            elif callable(item):
                item()
            else:
                yield item


def answers(count: int) -> list[list[Delta]]:
    """A script of `count` turns, each a one word answer."""
    return [[Delta(text=f"cevap {number}")] for number in range(count)]


# --------------------------------------------------------------------------
# One turn
# --------------------------------------------------------------------------


async def test_what_the_user_said_is_what_the_model_is_asked() -> None:
    provider = ScriptedProvider([Delta(text="Saat üç.")])

    answer = await Agent(provider, model=MODEL).reply("Saat kaç?")

    assert answer.text == "Saat üç."
    assert provider.calls[-1].turns == [Message.user("Saat kaç?")]
    assert provider.calls[-1].model == MODEL


async def test_the_answer_is_every_fragment_the_model_streamed() -> None:
    """An empty delta is legal - providers send chunks that only advance their
    own state - and must not end the answer or land in it."""
    provider = ScriptedProvider([Delta(text="Mer"), Delta(), Delta(text="haba.")])

    answer = await Agent(provider, model=MODEL).reply("selam")

    assert answer.text == "Merhaba."


async def test_the_token_counts_of_the_turn_are_carried_out_of_it() -> None:
    """Item 1.11 logs them per turn, and the cost report of section 6 is built
    from that log."""
    spent = Usage(input_tokens=812, output_tokens=97)
    provider = ScriptedProvider([Delta(text="Tamam."), Delta(usage=spent)])

    answer = await Agent(provider, model=MODEL).reply("selam")

    assert answer.usage == spent


async def test_a_provider_that_counts_no_tokens_still_gives_an_answer() -> None:
    provider = ScriptedProvider([Delta(text="Tamam.")])

    answer = await Agent(provider, model=MODEL).reply("selam")

    assert (answer.text, answer.usage) == ("Tamam.", Usage())


async def test_the_reason_the_model_stopped_survives_the_turn() -> None:
    """A reply cut off at the token limit is not a reply that ended; the user
    is told so in phase 2, and cannot be if the loop drops this."""
    provider = ScriptedProvider(
        [Delta(text="Uzun cevabın başı"), Delta(finish_reason="MAX_TOKENS")]
    )

    answer = await Agent(provider, model=MODEL).reply("anlat")

    assert answer.finish_reason == "MAX_TOKENS"


async def test_phase_one_offers_the_model_no_tools() -> None:
    """The permission gate of section 3.9 does not exist yet, and a tool
    offered before the gate is a tool that would run without one."""
    provider = ScriptedProvider([Delta(text="Tamam.")])

    await Agent(provider, model=MODEL).reply("not al")

    assert provider.calls[-1].tools == []


# --------------------------------------------------------------------------
# What the model is allowed to remember
# --------------------------------------------------------------------------


async def test_the_model_is_told_what_was_already_said() -> None:
    provider = ScriptedProvider([Delta(text="Merhaba.")], [Delta(text="Ahmet.")])
    conversation = Agent(provider, model=MODEL)

    await conversation.reply("selam")
    await conversation.reply("adım neydi?")

    assert provider.calls[-1].turns == [
        Message.user("selam"),
        Message.assistant("Merhaba."),
        Message.user("adım neydi?"),
    ]


async def test_only_the_last_twelve_turns_are_still_in_the_room() -> None:
    """Section 3.7: the window is what keeps a long conversation from growing
    roughly quadratically in cost."""
    provider = ScriptedProvider(*answers(WINDOW_TURNS + 1))
    conversation = Agent(provider, model=MODEL)

    for number in range(WINDOW_TURNS + 1):
        await conversation.reply(f"soru {number}")

    asked = [message.content for message in provider.calls[-1].turns if message.role == "user"]
    assert len(asked) == WINDOW_TURNS
    assert asked[0] == "soru 1"


def test_the_window_is_the_twelve_turns_section_3_7_asks_for() -> None:
    """A number the design arrived at rather than one the code picked: twelve
    turns is what section 3.7 budgets for against the cost of resending them."""
    assert WINDOW_TURNS == 12


def test_a_turn_is_not_a_message() -> None:
    """Phase 2 puts a tool call and its result inside a single turn. The cut
    has to be at what the user said, or it eventually orphans a tool result
    from the call it answers - which every provider rejects.
    """
    call = ToolCall(id="c1", name="get_current_time", arguments={})
    conversation = [
        Message.user("saat kaç?"),
        Message.assistant(tool_calls=(call,)),
        Message.tool_result("c1", "15:04"),
        Message.assistant("Üçü dört geçiyor."),
        Message.user("teşekkürler"),
        Message.assistant("Rica ederim."),
    ]

    assert window(conversation, turns=1) == conversation[4:]


def test_a_conversation_shorter_than_the_window_is_left_whole() -> None:
    conversation = [Message.user("selam"), Message.assistant("Merhaba.")]

    assert window(conversation, turns=WINDOW_TURNS) == conversation


async def test_the_system_prompt_is_no_turn_and_cannot_fall_out_of_the_window() -> None:
    """It is prepended to every request rather than kept in the conversation,
    so there is no length of chat that can push it out."""
    provider = ScriptedProvider(*answers(WINDOW_TURNS * 2))
    conversation = Agent(provider, model=MODEL, system_prompt="You are an assistant.")

    for number in range(WINDOW_TURNS * 2):
        await conversation.reply(f"soru {number}")

    assert provider.calls[-1].messages[0] == Message.system("You are an assistant.")


async def test_every_request_starts_with_the_very_same_bytes() -> None:
    """A cached prefix is a prefix that matches byte for byte. Nothing reports
    the moment it stops - the bill just grows (architecture guide section 2)."""
    provider = ScriptedProvider(*answers(3))
    conversation = Agent(provider, model=MODEL)

    for number in range(3):
        await conversation.reply(f"soru {number}")

    assert [call.messages[0] for call in provider.calls] == [Message.system(SYSTEM_PROMPT)] * 3


# --------------------------------------------------------------------------
# When the turn goes wrong
# --------------------------------------------------------------------------


async def test_a_turn_that_failed_did_not_happen() -> None:
    """The user hears "I could not connect" (item 1.10) and says it again. What
    they must not be answered from is half a turn nobody heard."""
    provider = ScriptedProvider(
        [Delta(text="Yarım "), ConnectionError("connection reset")],
        [Delta(text="Merhaba.")],
    )
    conversation = Agent(provider, model=MODEL)

    with pytest.raises(ConnectionError):
        await conversation.reply("selam")
    await conversation.reply("selam?")

    assert provider.calls[-1].turns == [Message.user("selam?")]


async def test_a_cancelled_turn_leaves_nothing_behind() -> None:
    """`THINKING` gives up after sixty seconds (section 3.1 rule 5), which
    cancels the turn from outside. Cancellation is not an error to swallow."""
    provider = ScriptedProvider(
        [Delta(text="Düşün"), asyncio.CancelledError()],
        [Delta(text="Buyur.")],
    )
    conversation = Agent(provider, model=MODEL)

    with pytest.raises(asyncio.CancelledError):
        await conversation.reply("uzun bir soru")
    await conversation.reply("selam")

    assert provider.calls[-1].turns == [Message.user("selam")]


async def test_a_model_that_said_nothing_is_not_remembered_as_having_spoken() -> None:
    """An empty assistant message is not merely noise in the window: several
    providers refuse a message with no content at all, so one silent turn would
    break every turn after it."""
    provider = ScriptedProvider([Delta(finish_reason="SAFETY")], [Delta(text="Merhaba.")])
    conversation = Agent(provider, model=MODEL)

    answer = await conversation.reply("...")
    await conversation.reply("selam")

    assert answer.text == ""
    assert provider.calls[-1].turns == [Message.user("selam")]


# --------------------------------------------------------------------------
# The system prompt
# --------------------------------------------------------------------------


def test_the_prompt_is_made_of_the_rules_it_names() -> None:
    """Each rule is a separate constant so that it can be read, argued with and
    replaced on its own; the prompt is the three of them and nothing else."""
    assert "\n\n".join((PERSONALITY, BREVITY, LANGUAGE_RULE)) == SYSTEM_PROMPT


def test_the_prompt_pins_no_language() -> None:
    """Section 3.12: the reply mirrors whatever language the user just used.
    Naming one here would quietly beat the rule that says so."""
    said = SYSTEM_PROMPT.casefold()

    assert not [name for name in ("turkish", "türkçe", "english", "german") if name in said]


def test_the_prompt_carries_no_clock_and_no_calendar() -> None:
    """A date, a time or a count is a digit, and a digit in the prefix is a
    cache that stops hitting. The date comes from a tool in phase 2."""
    assert not [character for character in SYSTEM_PROMPT if character.isdigit()]


def test_nothing_in_the_prompt_module_can_change_between_requests() -> None:
    """Frozen means nothing computes it. Imports are where that stops being
    true - `datetime`, `locale`, a settings read - so there are none."""
    source = Path(inspect.getfile(prompts)).read_text(encoding="utf-8")
    imported: set[str] = set()

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"__future__"}
