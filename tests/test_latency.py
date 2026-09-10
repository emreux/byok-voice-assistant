"""Our own code's share of a turn (design.md section 4, 2.8).

With a recogniser, a model and a voice that answer at once, what is left of
a turn is ours: the state machine, the loop, the gate, the audit row, the
sentence cutting, the filler's clock. Section 4 wanted that measured in CI
against the previous `main`, as a relative regression. CI was cancelled on
2026-08-27, so this is the next best thing: a loose absolute bound, run
locally with the rest of the suite. The bound is an order of magnitude
above the measured value on purpose - a busy machine is not a regression,
a turn whose own overhead grew tenfold is - and it is the decision recorded
as design.md section 12, item 23, until the owner says otherwise.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterator

import pytest

from assistant.agent.core import Agent, Confirm
from assistant.agent.policy import dispatch
from assistant.app import Assistant
from assistant.llm.base import Delta, ToolCall
from assistant.store.db import open_database
from assistant.store.repos import AuditRepo
from assistant.tools.registry import ToolRegistry, tool
from tests.test_agent_loop import ScriptedProvider
from tests.test_app import (
    TURKISH,
    FakeCapture,
    FakeSpeaker,
    FakeSTT,
    FakeTTS,
    speech,
)

# What one turn of our own code may cost, from the recording being handed
# over to the first buffer reaching the speaker. Measured 2026-09-10 on the
# development machine: about 2 ms at p95. The bound is loose by design.
BOUND_MS = 50.0
TURNS = 20


@tool(risk="safe")
async def clock() -> str:
    """Tells the time."""
    return "15:04"


@pytest.fixture
def audit() -> Iterator[AuditRepo]:
    connection = open_database(":memory:")
    yield AuditRepo(connection)
    connection.close()


def turns_with_a_tool(count: int) -> ScriptedProvider:
    """`count` turns, each a tool round through the real gate and then a
    three-sentence answer streamed in pieces - the shape of the turn
    section 4 budgets for."""
    script = []
    for number in range(count):
        script.append([Delta(tool_call=ToolCall(id=f"c{number}", name="clock", arguments={}))])
        script.append(
            [
                Delta(text="Saat üçü dört geçiyor. "),
                Delta(text="Öğleden sonra "),
                Delta(text="olduk. "),
                Delta(text="Başka bir şey?"),
            ]
        )
    return ScriptedProvider(*script)


@pytest.mark.latency
async def test_a_turn_s_own_overhead_stays_under_the_bound(audit: AuditRepo) -> None:
    registry = ToolRegistry([clock])

    async def gate(call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        return await dispatch(
            call, turn_id=turn_id, registry=registry, confirm=confirm, audit=audit
        )

    assistant = Assistant(
        capture=FakeCapture(),
        stt=FakeSTT(),
        agent=Agent(turns_with_a_tool(TURNS), model="fake-1", tools=registry, dispatch=gate),
        tts=FakeTTS(),
        speaker=FakeSpeaker(),
        locale=TURKISH,
        dispatch=gate,
    )
    await assistant.begin()

    first_sounds: list[float] = []
    for _ in range(TURNS):
        turn = await assistant.turn(speech())
        assert turn.failure is None
        assert turn.tool_calls == 1
        assert turn.first_sound_ms is not None
        first_sounds.append(turn.first_sound_ms)

    p95 = statistics.quantiles(first_sounds, n=100, method="inclusive")[94]
    assert p95 < BOUND_MS, f"our own share of a turn is {p95:.1f} ms at p95"


def test_the_bound_is_the_fifty_milliseconds_of_section_3_1() -> None:
    """The same number rule 4 of section 3.1 gives anything awaited in
    `app.py`: a turn's own overhead over it is a turn that blocks the loop."""
    assert BOUND_MS == 50.0
