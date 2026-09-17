"""The announce queue is read between turns and nowhere else (design.md
section 3.1, invariant 5; 17 Sep 2026).

`test_scheduler.py` proves what goes on the queue. This file proves what
the state machine does with it: an announcement waiting when the machine
is `IDLE` is said, with the microphone deaf; one that arrives while a
turn is under way is said after the turn; listening switched off does not
silence it; a voice cuts it off like an answer.

The queue itself is three lines over `asyncio.Queue` and is tested in
passing.
"""

from __future__ import annotations

import asyncio

import pytest

from assistant.announce.queue import Announcement, AnnounceQueue
from assistant.app import State
from assistant.stt.base import Audio
from tests.test_app import FakeCapture, FakeSpeaker, FakeTTS, StopError, assistant_with, speech


class QuietCapture(FakeCapture):
    """A microphone that, once its utterances are spent, waits for ever
    rather than ending the run - the room after the user has gone quiet."""

    async def utterance(self) -> Audio:
        if self._waiting:
            return self._waiting.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def until(condition: object, *, tries: int = 400) -> None:
    """Waits, briefly, for `condition()` to become true."""
    for _ in range(tries):
        if condition():  # type: ignore[operator]
            return
        await asyncio.sleep(0.005)
    raise AssertionError("the condition did not come true in time")


async def running(coroutine: object) -> asyncio.Task[None]:
    task = asyncio.create_task(coroutine)  # type: ignore[arg-type]
    await asyncio.sleep(0)
    return task


async def stopped(task: asyncio.Task[None]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --------------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------------


async def test_the_queue_hands_announcements_out_in_order() -> None:
    queue = AnnounceQueue()
    assert queue.empty() and len(queue) == 0

    queue.put(Announcement(text="bir", reminder_id=1))
    queue.put(Announcement(text="iki"))

    assert len(queue) == 2
    assert await queue.get() == Announcement(text="bir", reminder_id=1)
    assert await queue.get() == Announcement(text="iki", reminder_id=None)
    assert queue.empty()


# --------------------------------------------------------------------------
# The state machine
# --------------------------------------------------------------------------


async def test_an_announcement_is_said_when_the_assistant_is_idle() -> None:
    queue = AnnounceQueue()
    capture = QuietCapture()
    speaker = FakeSpeaker()
    tts = FakeTTS()
    states: list[State] = []
    assistant = assistant_with(
        capture=capture, speaker=speaker, tts=tts, on_state=states.append, announcements=queue
    )
    queue.put(Announcement(text="Dişçi randevusu", reminder_id=3))

    task = await running(assistant.run())
    try:
        await until(lambda: speaker.heard == "Dişçi randevusu")
    finally:
        await stopped(task)

    assert states == [State.IDLE, State.ANNOUNCING, State.IDLE]
    assert tts.said == ["Dişçi randevusu"]
    # Deaf while it was said, listening again after: the microphone would
    # otherwise hear the reminder and answer it.
    assert capture.switches == [True, False]
    assert not capture.deaf


async def test_an_announcement_waits_for_the_turn_under_way() -> None:
    """Both are ready at the start; the sentence is answered first, and
    the reminder follows the answer rather than talking over it."""
    queue = AnnounceQueue()
    capture = QuietCapture(speech())
    speaker = FakeSpeaker()
    tts = FakeTTS()
    states: list[State] = []
    assistant = assistant_with(
        capture=capture, speaker=speaker, tts=tts, on_state=states.append, announcements=queue
    )
    queue.put(Announcement(text="Su iç"))

    task = await running(assistant.run())
    try:
        await until(lambda: speaker.heard.endswith("Su iç"))
    finally:
        await stopped(task)

    assert tts.said == ["Üç.", "Su iç"]
    assert states == [
        State.IDLE,
        State.TRANSCRIBING,
        State.THINKING,
        State.SPEAKING,
        State.IDLE,
        State.ANNOUNCING,
        State.IDLE,
    ]


async def test_an_announcement_that_arrives_later_is_said_then() -> None:
    queue = AnnounceQueue()
    capture = QuietCapture()
    speaker = FakeSpeaker()
    assistant = assistant_with(capture=capture, speaker=speaker, announcements=queue)

    task = await running(assistant.run())
    try:
        await asyncio.sleep(0.02)
        assert speaker.heard == ""
        queue.put(Announcement(text="Şimdi"))
        await until(lambda: speaker.heard == "Şimdi")
    finally:
        await stopped(task)


async def test_listening_switched_off_does_not_silence_a_reminder() -> None:
    queue = AnnounceQueue()
    capture = QuietCapture()
    speaker = FakeSpeaker()
    assistant = assistant_with(capture=capture, speaker=speaker, announcements=queue)

    task = await running(assistant.run())
    try:
        capture.switch_off()
        queue.put(Announcement(text="Dişçi"))
        await until(lambda: speaker.heard == "Dişçi")
    finally:
        await stopped(task)

    assert assistant.state is State.IDLE


async def test_a_voice_cuts_an_announcement_off_like_an_answer() -> None:
    queue = AnnounceQueue()
    capture = QuietCapture()
    speaker = FakeSpeaker(on_play=lambda: capture.speak())
    assistant = assistant_with(capture=capture, speaker=speaker, announcements=queue)
    queue.put(Announcement(text="Uzun bir hatırlatma"))

    task = await running(assistant.run())
    try:
        await until(lambda: speaker.stopped)
    finally:
        await stopped(task)

    assert assistant.state is State.LISTENING


async def test_without_a_queue_nothing_changes() -> None:
    """Most tests and a run with no scheduler: `run` behaves as it did."""
    capture = FakeCapture(speech())
    speaker = FakeSpeaker()
    assistant = assistant_with(capture=capture, speaker=speaker)

    with pytest.raises(StopError):
        await assistant.run()

    assert speaker.heard == "Üç."
    assert not capture.started
