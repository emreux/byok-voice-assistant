"""The first tool: the time, which the frozen prompt cannot know (2.1c).

`get_current_time` is the smallest tool there is and the one the design
needs first - the system prompt carries no clock so its bytes never change,
which means the model has to be able to ask. What is tested is the shape of
the answer, with the clock pinned, and that the tool is what the registry
says a tool is: `safe`, no parameters, a description the model can act on.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone

import pytest

from assistant.tools import system
from assistant.tools.system import get_current_time

TURKEY = timezone(timedelta(hours=3), "Turkey Standard Time")


def test_it_is_a_safe_tool_with_nothing_to_fill_in() -> None:
    assert get_current_time.risk == "safe"
    assert get_current_time.spec.name == "get_current_time"
    assert get_current_time.spec.parameters["properties"] == {}
    assert get_current_time.spec.parameters["required"] == []


def test_the_description_tells_the_model_when_to_call_it() -> None:
    """Architecture guide section 10: the description is what decides whether
    the model reaches for the tool, so it names the occasion."""
    said = get_current_time.spec.description.casefold()

    assert "date" in said
    assert "time" in said


async def test_the_answer_is_the_date_the_time_the_weekday_and_the_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(system, "_now", lambda: datetime(2026, 9, 9, 14, 3, tzinfo=TURKEY))

    assert await get_current_time.run() == "2026-09-09T14:03+03:00 Wednesday, Turkey Standard Time"


async def test_the_zone_is_whatever_the_clock_says_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system, "_now", lambda: datetime(2026, 1, 1, 0, 0, tzinfo=UTC))

    assert await get_current_time.run() == "2026-01-01T00:00+00:00 Thursday, UTC"


async def test_the_real_clock_answers_in_the_same_shape() -> None:
    """Local time with its offset, then a weekday; the zone's name is the
    machine's own and is not checked."""
    said = await get_current_time.run()

    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}[+-]\d{2}:\d{2} [A-Z][a-z]+day, .+$", said)
