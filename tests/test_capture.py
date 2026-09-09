"""Both ways of being listened to: what gets recorded, and what does not.

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
from loguru import logger

from assistant.audio.capture import (
    CHUNK_FRAMES,
    DEFAULT_HOTKEY,
    DEFAULT_TOGGLE_HOTKEY,
    ECHO_TAIL_SECONDS,
    HandsFree,
    KeyCombination,
    MicrophoneUnavailableError,
    PushToTalk,
    SystemHotkey,
    SystemMicrophone,
    device_choice,
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


def fake_sounddevice(
    *,
    host_api: str = "MME",
    native_rate: int = SAMPLE_RATE,
    accepts: set[int] | None = None,
) -> tuple[SimpleNamespace, list[SimpleNamespace]]:
    """Stands in for the module, and records how the stream was opened.

    One input device, behind `host_api`, whose own rate is `native_rate`.
    `accepts` is the set of rates it opens at - `None` for any, the way MME
    behaves; `{48_000}` for a device that refuses 16 kHz, the way WASAPI and
    kernel streaming did on 2026-09-05. A WASAPI device told to convert opens
    at any rate. A device called `nope` is refused the way `sounddevice`
    refuses a name that matches nothing: a `ValueError` that quotes the name.
    """
    streams: list[SimpleNamespace] = []

    class PortAudioError(Exception):
        pass

    class WasapiSettings:
        def __init__(self, *, auto_convert: bool = False) -> None:
            self.auto_convert = auto_convert

    def query_devices(device: Any = None, kind: str | None = None) -> dict[str, Any]:
        if device == "nope":
            raise ValueError("No input device matching 'nope'")
        return {
            "index": 0,
            "name": "Microphone Array",
            "hostapi": 0,
            "max_input_channels": 2,
            "default_samplerate": float(native_rate),
        }

    def query_hostapis(index: int | None = None) -> dict[str, Any]:
        return {"name": host_api}

    def input_stream(**options: Any) -> SimpleNamespace:
        if options.get("device") == "nope":
            raise ValueError("No input device matching 'nope'")
        extra = options.get("extra_settings")
        converts = isinstance(extra, WasapiSettings) and extra.auto_convert
        if accepts is not None and options["samplerate"] not in accepts and not converts:
            raise PortAudioError("Error opening InputStream: Invalid sample rate", -9997)
        stream = SimpleNamespace(options=options, started=False, stopped=False, closed=False)
        stream.start = lambda: setattr(stream, "started", True)
        stream.stop = lambda: setattr(stream, "stopped", True)
        stream.close = lambda: setattr(stream, "closed", True)
        streams.append(stream)
        return stream

    module = SimpleNamespace(
        InputStream=input_stream,
        PortAudioError=PortAudioError,
        WasapiSettings=WasapiSettings,
        query_devices=query_devices,
        query_hostapis=query_hostapis,
    )
    return module, streams


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


def test_a_microphone_that_cannot_be_opened_fails_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name that matches no device, or a device another program is holding:
    the user can fix either, so `assistant run` says which in a sentence."""
    module, _ = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    with pytest.raises(MicrophoneUnavailableError, match="nope"):
        SystemMicrophone(device="nope").open(lambda chunk: None)


def test_a_wasapi_device_is_asked_to_convert_the_rate_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PortAudio runs a shared WASAPI stream at the mixer's rate - 48 kHz on
    the target machine - and refuses 16 kHz unless told to convert. Measured
    2026-09-05: `--device "Microphone Array WASAPI"` died on exactly this."""
    module, streams = fake_sounddevice(
        host_api="Windows WASAPI", native_rate=48_000, accepts={48_000}
    )
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    microphone = SystemMicrophone(device="Microphone Array WASAPI")
    microphone.open(lambda chunk: None)

    assert streams[0].options["samplerate"] == SAMPLE_RATE
    assert streams[0].options["extra_settings"].auto_convert is True
    assert microphone.rate == SAMPLE_RATE


def test_other_host_apis_are_not_handed_wasapi_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """PortAudio rejects a stream whose host-specific settings belong to
    another host API, so the setting goes only where it is understood."""
    module, streams = fake_sounddevice(host_api="MME")
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    SystemMicrophone().open(lambda chunk: None)

    assert streams[0].options.get("extra_settings") is None


def test_a_device_that_will_not_run_at_16_khz_is_opened_at_its_own_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernel streaming has no converter: `Microphone Array 1` runs at 48 kHz
    or not at all (2026-09-05, "Invalid device"). The stream is opened at the
    rate the device offers, still in 20 ms blocks, and brought to 16 kHz here."""
    module, streams = fake_sounddevice(
        host_api="Windows WDM-KS", native_rate=48_000, accepts={48_000}
    )
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    microphone = SystemMicrophone(device="Microphone Array 1")
    microphone.open(lambda chunk: None)

    assert streams[0].options["samplerate"] == 48_000
    assert streams[0].options["blocksize"] == 960
    assert microphone.rate == 48_000


def test_blocks_from_a_48_khz_device_arrive_at_16_khz(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the state machine receives is what it always received: 16 kHz,
    mono, twenty milliseconds at a time."""
    module, streams = fake_sounddevice(
        host_api="Windows WDM-KS", native_rate=48_000, accepts={48_000}
    )
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    heard: list[Audio] = []

    SystemMicrophone(device="Microphone Array 1").open(heard.append)
    callback = streams[0].options["callback"]
    block = np.full((960, 1), 0.25, dtype=np.float32)
    for _ in range(10):
        callback(block, 960, None, None)

    assert [len(chunk) for chunk in heard] == [CHUNK_FRAMES] * 10
    assert float(heard[-1].mean()) == pytest.approx(0.25, abs=0.01)


def test_a_device_that_runs_at_16_khz_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    microphone = SystemMicrophone()
    microphone.open(lambda chunk: None)

    assert microphone.rate == SAMPLE_RATE
    assert streams[0].options["samplerate"] == SAMPLE_RATE


def test_a_device_that_refuses_every_rate_is_still_a_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = fake_sounddevice(host_api="Windows WDM-KS", native_rate=48_000, accepts=set())
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    with pytest.raises(MicrophoneUnavailableError, match="could not be opened"):
        SystemMicrophone(device="Microphone Array 1").open(lambda chunk: None)


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


def overflowing(microphone: SystemMicrophone, callback: Any, times: int) -> list[str]:
    """Drives the callback with `times` blocks the driver marked as overrun,
    and returns what was logged about them."""
    lines: list[str] = []
    sink = logger.add(lines.append, format="{message}")
    try:
        block = np.zeros((3, 1), dtype=np.float32)
        for _ in range(times):
            callback(block, 3, None, SimpleNamespace(input_overflow=True))
        callback(block, 3, None, None)
    finally:
        logger.remove(sink)
    return [line for line in lines if "overflow" in line]


def test_blocks_the_driver_dropped_are_counted_and_the_first_is_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PortAudio reports an overrun in `status`; until now it was thrown away
    with the block. In hands-free mode that is a hole in the sentence with no
    trace of it anywhere."""
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    microphone = SystemMicrophone()
    microphone.open(lambda chunk: None)

    logged = overflowing(microphone, streams[0].options["callback"], times=2)

    assert microphone.overflows == 2
    assert len(logged) == 1


def test_dropped_blocks_are_written_down_rarely_after_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One line per dropped block, on PortAudio's own thread, would be the
    next cause of dropped blocks. The first, the tenth and the hundredth are
    enough to tell a bad minute from a bad microphone."""
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    microphone = SystemMicrophone()
    microphone.open(lambda chunk: None)

    logged = overflowing(microphone, streams[0].options["callback"], times=100)

    assert microphone.overflows == 100
    assert len(logged) == 3
    assert "100" in logged[-1]


def test_a_block_that_arrived_whole_is_not_an_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    microphone = SystemMicrophone()
    microphone.open(lambda chunk: None)

    logged = overflowing(microphone, streams[0].options["callback"], times=0)

    assert microphone.overflows == 0
    assert logged == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", None),
        (None, None),
        ("   ", None),
        ("9", 9),
        ("  12 ", 12),
        ("Microphone Array WASAPI", "Microphone Array WASAPI"),
        (" Microphone Array 1 ", "Microphone Array 1"),
    ],
)
def test_a_device_choice_is_an_index_a_name_or_nothing(text: str | None, expected: object) -> None:
    """`sounddevice` takes an int as an index and a str as words to match. A
    digit string handed to it as a str matches no name and fails - which is
    exactly how `bench_mic.py --device 12` never worked (2026-09-05)."""
    assert device_choice(text) == expected


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


# --------------------------------------------------------------------------
# Hands free: the key that is pressed once instead of held
# --------------------------------------------------------------------------


class FakeEndpoint:
    """A detector with no patience: a loud block is a sentence, a quiet one
    ends it. Where the real thresholds sit is `test_vad.py`'s business, and
    these tests are about which blocks reach a detector at all."""

    LOUD = 0.5

    def __init__(self) -> None:
        self.speaking = False
        self.heard: list[Audio] = []
        self.resets = 0
        self._take: list[Audio] = []

    def feed(self, chunk: Audio) -> list[Audio]:
        self.heard.append(chunk)
        if float(chunk[0]) >= self.LOUD:
            self.speaking = True
            self._take.append(chunk)
            return []
        if not self._take:
            return []

        take, self._take = self._take, []
        self.speaking = False
        return [np.concatenate(take)]

    def reset(self) -> None:
        self.resets += 1
        self.speaking = False
        self._take = []


def wired(**extra: Any) -> tuple[HandsFree, FakeHotkey, FakeHotkey, FakeMicrophone, FakeEndpoint]:
    """A hands-free capture with no keyboard, no microphone and no model."""
    hotkey, toggle = FakeHotkey(), FakeHotkey()
    microphone, endpoint = FakeMicrophone(), FakeEndpoint()
    talk = HandsFree(
        hotkey=hotkey, microphone=microphone, toggle=toggle, endpoint=endpoint, **extra
    )
    return talk, toggle, hotkey, microphone, endpoint


async def test_nothing_is_listened_to_until_the_toggle_is_pressed() -> None:
    """The mode is off when the program starts. A microphone that begins live
    is one the user never agreed to."""
    talk, _, _, microphone, endpoint = wired()

    with talk:
        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

        assert talk.listening is False
        assert endpoint.heard == []


async def test_what_was_said_after_the_toggle_comes_back_without_any_key() -> None:
    talk, toggle, _, microphone, _ = wired()

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        microphone.hear(tone(0.9))
        microphone.hear(tone(0.8))
        microphone.hear(tone(0.1))  # the sentence ended

        pcm = await talk.utterance()

    assert np.array_equal(pcm, np.concatenate([tone(0.9), tone(0.8)]))
    assert pcm.dtype == np.float32


async def test_the_toggle_turns_it_off_again() -> None:
    talk, toggle, _, microphone, endpoint = wired()

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        toggle.press()
        await asyncio.sleep(0)

        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

    assert talk.listening is False
    assert endpoint.heard == []


async def test_switching_the_mode_is_said_out_where_it_can_be_shown() -> None:
    """The status line is the only thing on screen that answers whether the
    microphone is live, and a mode nobody can see the state of is one left on
    in a room with other people in it."""
    seen: list[bool] = []
    talk, toggle, _, _, _ = wired(on_mode=seen.append)

    with talk:
        toggle.press()
        toggle.press()
        await asyncio.sleep(0)

    assert seen == [True, False]


async def test_switching_the_mode_starts_a_new_stream() -> None:
    """Whatever the detector was in the middle of belongs to before the switch."""
    talk, toggle, _, _, endpoint = wired()

    with talk:
        toggle.press()
        await asyncio.sleep(0)

    assert endpoint.resets == 1


async def test_the_moment_the_detector_hears_a_voice_is_announced() -> None:
    """The hands-free equivalent of the key going down: `app.py` learns from
    it that whatever it was doing has been overtaken."""
    started: list[str] = []
    talk, toggle, _, microphone, _ = wired(on_listening=lambda: started.append("now"))

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        microphone.hear(tone(0.1))
        await asyncio.sleep(0)
        assert started == []

        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

    assert started == ["now"]


async def test_a_sentence_is_announced_once_and_not_once_a_block() -> None:
    started: list[str] = []
    talk, toggle, _, microphone, _ = wired(on_listening=lambda: started.append("now"))

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        for _ in range(3):
            microphone.hear(tone(0.9))
        await asyncio.sleep(0)

    assert started == ["now"]


# --------------------------------------------------------------------------
# The key still works, and still wins
# --------------------------------------------------------------------------


async def test_holding_the_key_records_even_while_hands_free_is_on() -> None:
    """The detector will be wrong about some room, and the key that is never
    wrong has to be under the user's thumb when it is."""
    talk, toggle, hotkey, microphone, endpoint = wired()

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        hotkey.press()
        microphone.hear(tone(0.9))
        hotkey.release()

        pcm = await talk.utterance()

    assert np.array_equal(pcm, tone(0.9))
    assert endpoint.heard == [], "the block was recorded twice over"


async def test_holding_the_key_throws_away_what_the_detector_had_collected() -> None:
    """Half a sentence the room started is not part of what the user is about
    to say deliberately."""
    talk, toggle, hotkey, microphone, endpoint = wired()

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        microphone.hear(tone(0.9))
        await asyncio.sleep(0)
        before = endpoint.resets

        hotkey.press()
        await asyncio.sleep(0)

    assert endpoint.resets == before + 1


async def test_the_key_still_works_when_hands_free_was_never_switched_on() -> None:
    talk, _, hotkey, microphone, _ = wired()

    with talk:
        hotkey.press()
        microphone.hear(tone(0.2))
        hotkey.release()

        pcm = await talk.utterance()

    assert np.array_equal(pcm, tone(0.2))


# --------------------------------------------------------------------------
# Not hearing the assistant's own voice
# --------------------------------------------------------------------------


async def test_nothing_is_listened_to_while_the_assistant_speaks() -> None:
    """Without this the detector hears the answer come out of the speakers,
    takes it for a question, and answers it - once per API call."""
    talk, toggle, _, microphone, endpoint = wired()

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        talk.mute()

        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

    assert endpoint.heard == []


async def test_the_room_still_repeating_the_answer_is_not_a_question() -> None:
    """The sound card holds some of the answer and the room holds the rest;
    both arrive after the speaker has been told to stop."""
    talk, toggle, _, microphone, endpoint = wired()
    tail = round(ECHO_TAIL_SECONDS * SAMPLE_RATE)

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        talk.mute()
        talk.unmute()

        microphone.hear(tone(0.9, frames=tail))
        await asyncio.sleep(0)
        assert endpoint.heard == [], "the assistant's own echo reached the detector"

        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

    assert len(endpoint.heard) == 1


async def test_listening_again_starts_a_new_stream() -> None:
    """The frames either side of an answer are not neighbours."""
    talk, toggle, _, _, endpoint = wired()

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        before = endpoint.resets
        talk.mute()
        talk.unmute()

    assert endpoint.resets == before + 1


async def test_push_to_talk_is_never_deafened() -> None:
    """A key pressed while the assistant is talking is the user interrupting
    it, and that is the one thing that has to keep working."""
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        talk.mute()
        hotkey.press()
        microphone.hear(tone(0.1))
        hotkey.release()

        pcm = await talk.utterance()

    assert np.array_equal(pcm, tone(0.1))


# --------------------------------------------------------------------------
# The devices, again - there are two keyboards to let go of now
# --------------------------------------------------------------------------


async def test_starting_watches_both_keys() -> None:
    talk, toggle, hotkey, microphone, _ = wired()

    with talk:
        assert (hotkey.watching, toggle.watching, microphone.opened) == (True, True, True)


async def test_leaving_lets_go_of_both_keys_and_the_microphone() -> None:
    talk, toggle, hotkey, microphone, _ = wired()

    with pytest.raises(RuntimeError), talk:
        raise RuntimeError("the turn failed")

    assert (hotkey.watching, toggle.watching, microphone.opened) == (False, False, False)
    assert talk.listening is False


def test_the_two_combinations_are_not_the_same_keys() -> None:
    """One is held and one is pressed; the same combination for both would make
    every hands-free switch a recording as well."""
    assert DEFAULT_TOGGLE_HOTKEY != DEFAULT_HOTKEY
    assert SystemHotkey(DEFAULT_TOGGLE_HOTKEY).keys != SystemHotkey(DEFAULT_HOTKEY).keys


# --------------------------------------------------------------------------
# The confirmation window (2.3): listening for an answer, key or no key
# --------------------------------------------------------------------------


async def opened(talk: PushToTalk, seconds: float = 1.0) -> asyncio.Task[Audio | None]:
    """`listen_for`, running, with its window already open."""
    window = asyncio.create_task(talk.listen_for(seconds))
    await asyncio.sleep(0)
    return window


async def test_a_press_inside_the_window_is_the_answer() -> None:
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        window = await opened(talk)
        hotkey.press()
        microphone.hear(tone(0.3))
        hotkey.release()

        answer = await window

    assert answer is not None
    assert np.array_equal(answer, tone(0.3))


async def test_a_window_nobody_answers_in_closes_with_nothing() -> None:
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        assert await talk.listen_for(0.02) is None


async def test_a_press_inside_the_window_is_not_announced() -> None:
    """The state machine would take it for a new question and withdraw the
    one it was asking (`app.py`); inside the window the press is the answer."""
    hotkey, microphone = FakeHotkey(), FakeMicrophone()
    started: list[str] = []

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        talk.on_listening = lambda: started.append("now")
        window = await opened(talk)
        hotkey.press()
        microphone.hear(tone(0.3))
        hotkey.release()
        await window

    assert started == []


async def test_the_answer_does_not_come_back_as_a_question_as_well() -> None:
    hotkey, microphone = FakeHotkey(), FakeMicrophone()

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        window = await opened(talk)
        hotkey.press()
        hotkey.release()
        await window

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(talk.utterance(), timeout=0.05)


async def test_once_the_window_is_closed_a_press_is_a_question_again() -> None:
    hotkey, microphone = FakeHotkey(), FakeMicrophone()
    started: list[str] = []

    with PushToTalk(hotkey=hotkey, microphone=microphone) as talk:
        talk.on_listening = lambda: started.append("now")
        await talk.listen_for(0.02)
        hotkey.press()
        microphone.hear(tone(0.4))
        hotkey.release()

        pcm = await talk.utterance()

    assert started == ["now"]
    assert np.array_equal(pcm, tone(0.4))


async def test_with_hands_free_off_the_window_still_hears_a_sentence() -> None:
    """The assistant asked, so the answer is heard without any key - in the
    one mode where nothing else is - and the mode is as it was afterwards."""
    started: list[str] = []
    talk, _, _, microphone, endpoint = wired(on_listening=lambda: started.append("now"))

    with talk:
        window = await opened(talk)
        microphone.hear(tone(0.9))
        microphone.hear(tone(0.8))
        microphone.hear(tone(0.1))  # the sentence ended

        answer = await window

        assert talk.listening is False
        microphone.hear(tone(0.9))  # the window is closed: nobody is listening
        await asyncio.sleep(0)

    assert answer is not None
    assert np.array_equal(answer, np.concatenate([tone(0.9), tone(0.8)]))
    assert started == []
    assert len(endpoint.heard) == 3


async def test_with_hands_free_on_the_answer_does_not_become_a_question() -> None:
    started: list[str] = []
    talk, toggle, _, microphone, _ = wired(on_listening=lambda: started.append("now"))

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        window = await opened(talk)
        microphone.hear(tone(0.9))
        microphone.hear(tone(0.1))

        answer = await window

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(talk.utterance(), timeout=0.05)

    assert answer is not None
    assert started == []


async def test_after_the_window_hands_free_listens_for_questions_again() -> None:
    started: list[str] = []
    talk, toggle, _, microphone, _ = wired(on_listening=lambda: started.append("now"))

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        assert await talk.listen_for(0.02) is None

        microphone.hear(tone(0.9))
        microphone.hear(tone(0.1))
        pcm = await talk.utterance()

    assert started == ["now"]
    assert np.array_equal(pcm, tone(0.9))


async def test_the_key_answers_inside_the_window_too() -> None:
    """The key always works (design.md section 3.5), and inside the window
    what it records is the answer - and is not shown to the detector."""
    talk, toggle, hotkey, microphone, endpoint = wired()

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        window = await opened(talk)
        hotkey.press()
        microphone.hear(tone(0.9))
        hotkey.release()

        answer = await window

    assert answer is not None
    assert np.array_equal(answer, tone(0.9))
    assert endpoint.heard == []


async def test_the_room_repeating_the_question_is_not_an_answer() -> None:
    """`unmute` leaves the echo tail behind, and the window counts it down
    before it listens - the question coming back off the walls is not a yes."""
    talk, _, _, microphone, endpoint = wired()
    tail = round(ECHO_TAIL_SECONDS * SAMPLE_RATE)

    with talk:
        talk.mute()
        talk.unmute()
        window = await opened(talk)
        microphone.hear(tone(0.9, frames=tail))
        await asyncio.sleep(0)
        assert endpoint.heard == [], "the assistant's own question reached the detector"

        microphone.hear(tone(0.9))
        microphone.hear(tone(0.1))
        answer = await window

    assert answer is not None
    assert np.array_equal(answer, tone(0.9))


async def test_the_window_starts_a_fresh_sentence_and_leaves_none_behind() -> None:
    """Whatever the detector had half collected before the question is not
    the answer, and whatever it had when time ran out is not the next question."""
    talk, _, _, _, endpoint = wired()

    with talk:
        before = endpoint.resets
        await talk.listen_for(0.02)

    assert endpoint.resets == before + 2
