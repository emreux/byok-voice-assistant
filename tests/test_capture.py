"""Push to talk: what gets recorded, and what does not.

Two threads that are not the event loop reach into this module - the keyboard
listener and PortAudio's callback - so the tests drive it the same way, and
three specific traps are pinned:

* audio that arrives while no key is held belongs to nobody,
* the buffer PortAudio hands over is reused, so it has to be copied,
* the stream is opened at the rate the speech layer declares, not a plausible
  looking one. That last mistake has already been made once in this project.

No microphone and no keyboard are touched here. The devices are behind two
small interfaces; the real ones are exercised through a fake `sounddevice`
module, which is also what proves the import stays deferred.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from assistant.audio.capture import (
    CHUNK_FRAMES,
    DEFAULT_HOTKEY,
    KeyCombination,
    PushToTalk,
    SystemHotkey,
    SystemMicrophone,
)
from assistant.stt.base import SAMPLE_RATE, Audio

CTRL, ALT, SPACE, SHIFT = "ctrl", "alt", "space", "shift"


def combination() -> KeyCombination:
    return KeyCombination(frozenset({CTRL, ALT, SPACE}))


def tone(value: float, frames: int = 4) -> Audio:
    return np.full(frames, value, dtype=np.float32)


class FakeHotkey:
    """A key combination nobody has to press."""

    def __init__(self) -> None:
        self.watching = False
        self._on_press: Any = None
        self._on_release: Any = None

    def watch(self, *, on_press: Any, on_release: Any) -> None:
        self._on_press, self._on_release = on_press, on_release
        self.watching = True

    def stop(self) -> None:
        self.watching = False

    def press(self) -> None:
        self._on_press()

    def release(self) -> None:
        self._on_release()


class FakeMicrophone:
    """An input stream that hears exactly what a test tells it to."""

    def __init__(self) -> None:
        self.opened = False
        self._on_chunk: Any = None

    def open(self, on_chunk: Any) -> None:
        self._on_chunk = on_chunk
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def hear(self, chunk: Audio) -> None:
        self._on_chunk(chunk)


def fake_sounddevice() -> tuple[SimpleNamespace, list[SimpleNamespace]]:
    """Stands in for the module, and records how the stream was opened."""
    streams: list[SimpleNamespace] = []

    def input_stream(**options: Any) -> SimpleNamespace:
        stream = SimpleNamespace(options=options, started=False, stopped=False, closed=False)
        stream.start = lambda: setattr(stream, "started", True)
        stream.stop = lambda: setattr(stream, "stopped", True)
        stream.close = lambda: setattr(stream, "closed", True)
        streams.append(stream)
        return stream

    return SimpleNamespace(InputStream=input_stream), streams


# --------------------------------------------------------------------------
# The key combination - no keyboard involved
# --------------------------------------------------------------------------


def test_the_combination_fires_once_all_of_it_is_down() -> None:
    keys = combination()

    assert keys.press(CTRL) is False
    assert keys.press(ALT) is False
    assert keys.press(SPACE) is True


def test_the_order_the_keys_go_down_does_not_matter() -> None:
    keys = combination()

    assert [keys.press(SPACE), keys.press(ALT), keys.press(CTRL)] == [False, False, True]


def test_holding_a_key_does_not_start_a_second_recording() -> None:
    """Windows repeats key-down events while a key is held; each repeat would
    otherwise look like a new press."""
    keys = combination()
    keys.press(CTRL)
    keys.press(ALT)
    keys.press(SPACE)

    assert keys.press(SPACE) is False
    assert keys.press(CTRL) is False


def test_a_key_outside_the_combination_starts_nothing() -> None:
    keys = combination()
    keys.press(CTRL)
    keys.press(ALT)

    assert keys.press(SHIFT) is False


def test_letting_go_of_any_one_key_ends_it() -> None:
    keys = combination()
    keys.press(CTRL)
    keys.press(ALT)
    keys.press(SPACE)

    assert keys.release(ALT) is True


def test_letting_go_of_an_unrelated_key_does_not_end_it() -> None:
    """Typing while holding the combination must not cut the recording."""
    keys = combination()
    keys.press(CTRL)
    keys.press(ALT)
    keys.press(SPACE)

    assert keys.release(SHIFT) is False


def test_a_key_already_held_does_not_prevent_the_combination() -> None:
    """Somebody holding shift, or a game holding a key down, must not make the
    assistant unreachable."""
    keys = combination()
    keys.press(SHIFT)
    keys.press(CTRL)
    keys.press(ALT)

    assert keys.press(SPACE) is True


def test_the_combination_can_be_used_again() -> None:
    keys = combination()
    for key in (CTRL, ALT, SPACE):
        keys.press(key)
    keys.release(SPACE)

    assert keys.press(SPACE) is True


def test_a_release_that_never_had_a_press_is_harmless() -> None:
    """The combination may be completed while another window had focus."""
    keys = combination()

    assert keys.release(SPACE) is False


def test_letting_go_twice_only_ends_it_once() -> None:
    keys = combination()
    for key in (CTRL, ALT, SPACE):
        keys.press(key)
    keys.release(SPACE)

    assert keys.release(CTRL) is False


# --------------------------------------------------------------------------
# What ends up in an utterance
# --------------------------------------------------------------------------


async def test_what_was_said_while_the_key_was_down_is_what_comes_back() -> None:
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        hotkey.press()
        microphone.hear(tone(0.1))
        microphone.hear(tone(0.2))
        hotkey.release()

        pcm = await talk.utterance()

    assert np.array_equal(pcm, np.concatenate([tone(0.1), tone(0.2)]))
    assert pcm.dtype == np.float32


async def test_a_room_that_was_never_asked_to_listen_is_not_recorded() -> None:
    """The microphone is open all the time; that is not the same as recording."""
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        microphone.hear(tone(0.9))  # nobody pressed anything
        hotkey.press()
        microphone.hear(tone(0.1))
        hotkey.release()

        pcm = await talk.utterance()

    assert np.array_equal(pcm, tone(0.1))


async def test_audio_that_arrives_after_the_release_belongs_to_no_utterance() -> None:
    """PortAudio's thread and the keyboard's thread do not agree on an order,
    so a chunk can turn up just after the key came up."""
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        hotkey.press()
        microphone.hear(tone(0.1))
        hotkey.release()
        microphone.hear(tone(0.9))

        first = await talk.utterance()

        hotkey.press()
        microphone.hear(tone(0.2))
        hotkey.release()
        second = await talk.utterance()

    assert np.array_equal(first, tone(0.1))
    assert np.array_equal(second, tone(0.2))


async def test_idle_audio_never_reaches_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The microphone is open all day.

    Handing every block to the loop while nobody is talking wakes it ten times
    a second to throw the block away again. The two tests either side of this
    one show the block does not end up in an utterance; this one shows it does
    not cross the thread boundary in the first place - a claim with no public
    face, which is why it reaches for the method behind it.
    """
    hotkey, microphone = FakeHotkey(), FakeMicrophone()
    talk = PushToTalk(hotkey=hotkey, microphone=microphone)
    crossed: list[Audio] = []
    monkeypatch.setattr(talk, "_keep", crossed.append)

    with talk:
        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

    assert crossed == []


async def test_a_block_that_crosses_the_release_joins_no_turn_at_all() -> None:
    """The real race, reproduced.

    PortAudio's thread reads the recording flag an instant before the keyboard
    thread clears it, so its block is handed to the loop *after* the take was
    closed. Driving that from the fake microphone is impossible - the flag is
    read inside it - so the block is delivered the way PortAudio's thread would
    have, one step too late.
    """
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        hotkey.press()
        microphone.hear(tone(0.1))
        hotkey.release()
        await asyncio.sleep(0)  # the loop begins, keeps and ends the take
        talk._keep(tone(0.9))  # the block that was already in PortAudio's hands

        first = await talk.utterance()

        hotkey.press()
        microphone.hear(tone(0.2))
        hotkey.release()
        second = await talk.utterance()

    assert np.array_equal(first, tone(0.1))
    assert np.array_equal(second, tone(0.2))


async def test_a_press_with_nothing_said_is_an_empty_utterance() -> None:
    """A stray keypress is still a turn; what to do about it is the state
    machine's decision, not this module's."""
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        hotkey.press()
        hotkey.release()

        pcm = await talk.utterance()

    assert len(pcm) == 0
    assert pcm.dtype == np.float32


async def test_an_utterance_survives_until_somebody_asks_for_it() -> None:
    """The key can be released while the assistant is still speaking; the
    recording is not thrown away because nobody was waiting yet."""
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        hotkey.press()
        microphone.hear(tone(0.3))
        hotkey.release()
        await asyncio.sleep(0)  # the assistant was busy with something else

        pcm = await talk.utterance()

    assert np.array_equal(pcm, tone(0.3))


async def test_a_release_nobody_pressed_produces_no_utterance() -> None:
    """The combination can be completed in another application and released
    over ours; there is no recording to hand over."""
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        hotkey.release()

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(talk.utterance(), timeout=0.05)


async def test_two_presses_are_two_utterances() -> None:
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        for level in (0.1, 0.2):
            hotkey.press()
            microphone.hear(tone(level))
            hotkey.release()

        first, second = await talk.utterance(), await talk.utterance()

    assert float(first[0]) == pytest.approx(0.1)
    assert float(second[0]) == pytest.approx(0.2)


async def test_audio_recorded_on_another_thread_still_arrives() -> None:
    """PortAudio calls back on its own thread; the bridge has to be one that
    asyncio tolerates from the outside."""
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        hotkey.press()
        recorder = threading.Thread(target=lambda: microphone.hear(tone(0.4)))
        recorder.start()
        recorder.join()
        hotkey.release()

        pcm = await talk.utterance()

    assert np.array_equal(pcm, tone(0.4))


# --------------------------------------------------------------------------
# Devices are opened and closed
# --------------------------------------------------------------------------


async def test_starting_watches_the_keyboard_and_opens_the_microphone() -> None:
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone):
        assert (hotkey.watching, microphone.opened) == (True, True)


async def test_leaving_closes_both_even_after_a_failure() -> None:
    """A microphone left open is a light that stays on and a device another
    application cannot have."""
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with pytest.raises(RuntimeError), PushToTalk(hotkey=hotkey, microphone=microphone):
        raise RuntimeError("the turn failed")

    assert (hotkey.watching, microphone.opened) == (False, False)


# --------------------------------------------------------------------------
# The real devices
# --------------------------------------------------------------------------


def test_the_stream_is_opened_at_the_rate_the_speech_layer_declares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plausible looking sample rate is a silent bug: the model assumes the
    buffer is at 16 kHz and simply hears everything at the wrong speed."""
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    SystemMicrophone().open(lambda chunk: None)

    assert streams[0].options["samplerate"] == SAMPLE_RATE
    assert streams[0].options["channels"] == 1
    assert streams[0].options["dtype"] == "float32"
    assert streams[0].started is True
    # The block being filled when the key comes up never arrives, so the block
    # size is also how much of the end of a sentence can be lost.
    assert streams[0].options["blocksize"] == CHUNK_FRAMES
    assert CHUNK_FRAMES / SAMPLE_RATE <= 0.02


def test_closing_the_microphone_releases_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    microphone = SystemMicrophone()
    microphone.open(lambda chunk: None)
    microphone.close()

    assert (streams[0].stopped, streams[0].closed) == (True, True)


def test_the_buffer_portaudio_hands_over_is_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    """PortAudio writes the next block into the same array. Keeping a
    reference instead of a copy turns the recording into the last block,
    repeated."""
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    heard: list[Audio] = []

    SystemMicrophone().open(heard.append)
    callback = streams[0].options["callback"]

    reused = np.zeros((3, 1), dtype=np.float32)
    reused[:] = 0.1
    callback(reused, 3, None, None)
    reused[:] = 0.9  # exactly what PortAudio does next
    callback(reused, 3, None, None)

    assert [float(chunk[0]) for chunk in heard] == [pytest.approx(0.1), pytest.approx(0.9)]
    assert heard[0].shape == (3,), "the stream is mono; the channel axis is dropped"


def test_the_default_combination_is_three_keys_the_system_knows() -> None:
    """A typo in the combination string would only be found by pressing it."""
    keys = SystemHotkey().keys

    assert len(keys) == 3
    assert DEFAULT_HOTKEY == "<ctrl>+<alt>+<space>"


def test_a_combination_that_makes_no_sense_is_refused_at_once() -> None:
    with pytest.raises(ValueError, match="hotkey"):
        SystemHotkey("<ctrl>+<nonsense>")


# --------------------------------------------------------------------------
# Telling the state machine that the key went down
# --------------------------------------------------------------------------


async def test_the_moment_recording_starts_is_announced() -> None:
    """`app.py` stops speaking on this. Waiting for the finished utterance
    instead would mean the assistant talks over the user until they let go of
    the key - and that the microphone records its own voice doing it."""
    hotkey, microphone = FakeHotkey(), FakeMicrophone()
    started: list[str] = []

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        talk.on_listening = lambda: started.append("now")
        hotkey.press()
        await asyncio.sleep(0)

    assert started == ["now"]


async def test_the_announcement_arrives_where_asyncio_can_be_touched() -> None:
    """It is posted from the keyboard's thread and runs on the event loop,
    which is what makes it safe for the listener to stop the speaker."""
    hotkey, microphone = FakeHotkey(), FakeMicrophone()
    loops: list[object] = []

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        talk.on_listening = lambda: loops.append(asyncio.get_running_loop())
        hotkey.press()
        await asyncio.sleep(0)

    assert loops == [asyncio.get_running_loop()]


async def test_recording_works_whether_or_not_anybody_listens_for_the_press() -> None:
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        hotkey.press()
        microphone.hear(tone(0.1))
        hotkey.release()

        pcm = await talk.utterance()

    assert np.array_equal(pcm, tone(0.1))
