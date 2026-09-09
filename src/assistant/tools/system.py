"""What the assistant can do on the machine itself (design.md section 3.6).

Phase 2.1c opens this file with the smallest tool there is. `get_current_time`
touches nothing, and it answers the one question the frozen system prompt of
`agent/prompts.py` cannot: the prompt carries no clock so that its bytes never
change and the provider's cache keeps hitting (architecture guide section 2),
so knowledge that changes has to come from a tool. This is the first of them;
`open_app`, `open_url` and `open_settings` follow in 2.2.

What a tool returns is addressed to the model, not the user, so it is written
the way a model reads best - unambiguous, in one line - and in English, like
everything else that never reaches the speaker. How it is said out loud, and
in which language, is the model's job (section 3.12).
"""

from __future__ import annotations

from datetime import datetime

from assistant.tools.registry import tool

__all__ = ["get_current_time"]


def _now() -> datetime:
    """The local time with its zone attached. Kept apart so a test can pin it."""
    return datetime.now().astimezone()


@tool(risk="safe")
async def get_current_time() -> str:
    """Returns the current local date, time, weekday and time zone. Call it before
    answering anything that depends on today's date or the time of day."""
    now = _now()
    # ISO for the date and time, because every model reads it without
    # ambiguity; the weekday and the zone by name, because "yarın" and "bu
    # akşam" are questions about those.
    return f"{now.isoformat(timespec='minutes')} {now:%A}, {now.tzname()}"
