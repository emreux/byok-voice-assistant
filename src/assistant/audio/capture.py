"""Two ways to be listened to: hold a key, or press one and stop holding keys.

`PushToTalk` is the first and the one that always works - the user says where
the sentence ends by letting go, which costs nothing, cuts nobody off mid-word
and cannot be confused by a television. `HandsFree` adds the second key of item
1.5b: press it once and the microphone stays live, with `audio/vad.py` deciding
where each sentence begins and ends. It is the same class plus a trigger, so
the key still works while hands-free is on, and it is still there on the day
the detector is wrong about a room.

Three threads meet in this file and only one of them owns the state.

* **PortAudio's callback thread** delivers the microphone blocks. The array it
  passes is its own scratch buffer and is overwritten on the next callback, so
  it is copied before it goes anywhere.
* **The keyboard listener's thread** reports the hotkey. It sets a plain
  `threading.Event` so that the audio callback starts keeping blocks in the
  same instant, without waiting for the event loop to get around to it.
* **The event loop** owns the recording itself. Both other threads reach it
  only through `call_soon_threadsafe`; touching an asyncio object from outside
  the loop is a race that shows up as a hang weeks later.

The microphone stays open for as long as the assistant runs - opening a
PortAudio stream takes long enough to swallow the first syllable. Open is not
the same as recording: a block that arrives while no key is held is dropped in
the callback and never reaches memory. Hands-free is exactly the decision to
stop dropping them - fifty blocks a second do cross into the event loop while
it is on, each costing a fraction of a millisecond, and none at all while it
is off.

The confirmation window of phase 2.3 is the third way to be listened to.
`listen_for` takes the next sentence - by the key, or by the detector whether
or not hands-free is on - for a few seconds, and hands it back to whoever
asked instead of queueing it as a question. What is said inside it is an
answer, so the key going down there is not announced: the state machine
would otherwise take the user's "yes" for a new question and withdraw the
one it was asking.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
from loguru import logger

from assistant.audio.resample import Resampler
from assistant.audio.vad import Endpoint, Segmenter
from assistant.stt.base import SAMPLE_RATE, Audio

__all__ = [
    "CHUNK_FRAMES",
    "DEFAULT_HOTKEY",
    "DEFAULT_TOGGLE_HOTKEY",
    "ECHO_TAIL_SECONDS",
    "HandsFree",
    "Hotkey",
    "KeyCombination",
    "Microphone",
    "MicrophoneUnavailableError",
    "PushToTalk",
    "SystemHotkey",
    "SystemMicrophone",
    "device_choice",
]

# `pynput`'s own spelling. Phase 4 moves it into `config.toml`; until then it
# is one constant rather than a string repeated in three places.
DEFAULT_HOTKEY = "<ctrl>+<alt>+<space>"

# The other key: it turns hands-free on and off rather than being held. A
# letter and not a second space, so that a key held down by a game or a chat
# application cannot be the one that opens the microphone by accident.
DEFAULT_TOGGLE_HOTKEY = "<ctrl>+<alt>+h"

# How long the microphone stays deaf after the assistant stops speaking. The
# sound card holds some of the answer and the room holds the rest of it; both
# arrive after the speaker has been told to stop, and the detector would hear
# the assistant finishing its own sentence and answer it. This is not echo
# cancellation - that is phase 5.3, and it is what barge-in needs.
ECHO_TAIL_SECONDS = 0.25

# 20 ms per block. The block being filled when the key comes up is still in
# PortAudio's hands and never arrives, so this is also how much of the end of
# the sentence can be lost: measured at 100 ms blocks, two seconds of holding
# produced 1.90 s of audio. Twenty milliseconds is below a syllable, and fifty
# callbacks a second is nothing.
CHUNK_FRAMES = SAMPLE_RATE // 50

OnChunk = Callable[[Audio], None]
OnEvent = Callable[[], None]
OnMode = Callable[[bool], None]


@runtime_checkable
class Hotkey(Protocol):
    """A key combination watched everywhere, not just in our own window."""

    def watch(self, *, on_press: OnEvent, on_release: OnEvent) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class Microphone(Protocol):
    """An input stream that hands over one block at a time."""

    def open(self, on_chunk: OnChunk) -> None: ...

    def close(self) -> None: ...


class KeyCombination:
    """Which of the wanted keys are down, and when that last changed.

    Separate from the keyboard for two reasons. It is the only part with rules
    worth stating - a held key repeats, an unrelated key must not interfere -
    and it is the only part that can be tested without pressing anything.
    """

    def __init__(self, keys: frozenset[Any]) -> None:
        self._wanted = keys
        self._down: set[Any] = set()
        self._complete = False

    def press(self, key: Any) -> bool:
        """Records a key going down. True if this completed the combination."""
        if key in self._wanted:
            self._down.add(key)

        # Windows repeats key-down events while a key is held; without this
        # guard every repeat would start another recording.
        if self._complete or self._down != self._wanted:
            return False
        self._complete = True
        return True

    def release(self, key: Any) -> bool:
        """Records a key going up. True if this broke a complete combination."""
        self._down.discard(key)

        if not self._complete or self._down == self._wanted:
            return False
        self._complete = False
        return True


class PushToTalk:
    """Turns key presses and microphone blocks into whole utterances."""

    def __init__(
        self,
        *,
        hotkey: Hotkey | None = None,
        microphone: Microphone | None = None,
        on_listening: OnEvent | None = None,
    ) -> None:
        self._hotkey = hotkey if hotkey is not None else SystemHotkey()
        self._microphone = microphone if microphone is not None else SystemMicrophone()

        # Called on the event loop the moment recording starts, before there is
        # anything to transcribe. The state machine stops the speaker from it
        # (item 1.10): an assistant that waited for the finished utterance
        # would talk over the user, into the microphone that is recording them.
        # Public because whoever builds this is rarely whoever listens to it.
        self.on_listening = on_listening

        # Read by the audio callback on PortAudio's thread, written by the
        # keyboard listener on its own. A plain event, deliberately: going
        # through the loop would lose the blocks recorded in between.
        self._recording = threading.Event()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._take: list[Audio] | None = None
        self._finished: asyncio.Queue[list[Audio]] = asyncio.Queue()

        # The window of `listen_for`, while one is open: the next take goes
        # here instead of into `_finished`, and its press is not announced.
        self._window: asyncio.Future[list[Audio]] | None = None

    def start(self) -> None:
        """Opens the microphone and starts watching the keyboard."""
        self._loop = asyncio.get_running_loop()
        self._microphone.open(self._heard)
        self._hotkey.watch(on_press=self._pressed, on_release=self._released)

    def stop(self) -> None:
        self._recording.clear()
        self._hotkey.stop()
        self._microphone.close()

    def mute(self) -> None:
        """Nothing, and deliberately so.

        The state machine deafens the microphone while it is speaking, which is
        what stops `HandsFree` from hearing the assistant's own voice and
        answering it. Push to talk has no such problem and must not pretend to:
        a key pressed while the assistant is talking is the user interrupting
        it, and that is the one thing that has to keep working.
        """

    def unmute(self) -> None:
        """The other half of `mute`, and just as empty."""

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    async def utterance(self) -> Audio:
        """Waits for the next completed recording and returns it as one buffer."""
        return _as_audio(await self._finished.get())

    async def listen_for(self, seconds: float) -> Audio | None:
        """The next recording, if the key is pressed within `seconds`; else `None`.

        The confirmation window of `app.py` (section 3.1 rule 2). With no
        detector the only way to answer is the key, so a press while the
        window is open is the answer and not a new question: `on_listening`
        is not called for it, and what it recorded comes back here rather
        than through `utterance`. The window closes when time is up or on
        the first recording, whichever comes first.
        """
        self._window = asyncio.get_running_loop().create_future()
        try:
            chunks = await asyncio.wait_for(self._window, seconds)
        except TimeoutError:
            return None
        finally:
            self._window = None
        return _as_audio(chunks)

    # ----------------------------------------------------------------------
    # Called from other threads. Nothing here touches asyncio directly.
    # ----------------------------------------------------------------------

    def _pressed(self) -> None:
        self._recording.set()
        self._to_loop(self._begin)

    def _released(self) -> None:
        self._recording.clear()
        self._to_loop(self._end)

    def _heard(self, chunk: Audio) -> None:
        if self._recording.is_set():
            self._to_loop(self._keep, chunk)

    def _to_loop(self, work: Callable[..., None], *args: Any) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(work, *args)

    # ----------------------------------------------------------------------
    # Called on the event loop, in the order the threads above submitted them.
    # ----------------------------------------------------------------------

    def _begin(self) -> None:
        self._take = []
        # Inside the window a press is the answer being given, not a new
        # question: the state machine is not told, so it does not withdraw.
        if self._window is None and self.on_listening is not None:
            self.on_listening()

    def _keep(self, chunk: Audio) -> None:
        # `None` means the recording already ended: the audio thread saw the
        # flag a moment before the keyboard thread cleared it. That block was
        # recorded, but it belongs to a turn that is over.
        if self._take is not None:
            self._take.append(chunk)

    def _end(self) -> None:
        take, self._take = self._take, None
        if take is None:
            return
        if self._window is not None and not self._window.done():
            self._window.set_result(take)
        else:
            self._finished.put_nowait(take)


class HandsFree(PushToTalk):
    """Push to talk, and a second key that leaves the microphone listening.

    Everything the parent does still works. That is the point rather than an
    implementation detail: the detector will be wrong about some room, and the
    key that is never wrong has to be under the user's thumb when it is
    (design.md section 3.5 says the same thing about the wake word of phase
    5.2 - push to talk always works).

    Three rules are worth stating, because each is about something *not*
    happening.

    **The toggle does not interrupt an answer.** Pressing it arms the mode; it
    does not stop the assistant mid-sentence, because stopping it would mean
    claiming the user is talking when they have only pressed a key, and the
    state machine reads that claim as a question being withdrawn. The push to
    talk key still interrupts, and interrupting by voice is phase 5.3.

    **The microphone is deaf while the assistant speaks.** `mute` is called by
    the state machine, and without it the detector hears the answer coming out
    of the speakers, the assistant answers itself, and every round of that
    costs an API call. `ECHO_TAIL_SECONDS` covers what the sound card and the
    room hand over after the speaker has already been told to stop.

    **The key wins.** A block that arrives while the combination is held goes
    to the parent's recording and is never shown to the detector, so holding
    the key inside hands-free mode is one deliberate sentence rather than two
    overlapping ones.
    """

    def __init__(
        self,
        *,
        hotkey: Hotkey | None = None,
        microphone: Microphone | None = None,
        toggle: Hotkey | None = None,
        endpoint: Segmenter | None = None,
        on_listening: OnEvent | None = None,
        on_mode: OnMode | None = None,
    ) -> None:
        super().__init__(hotkey=hotkey, microphone=microphone, on_listening=on_listening)
        self._toggle = toggle if toggle is not None else SystemHotkey(DEFAULT_TOGGLE_HOTKEY)
        self._endpoint = endpoint if endpoint is not None else Endpoint()

        # Called on the event loop when the mode is switched. The status line
        # is the only thing that says whether the microphone is live, and a
        # mode nobody can see the state of is a mode nobody trusts.
        self.on_mode = on_mode

        # Read by the audio callback on PortAudio's thread and written by the
        # keyboard's, exactly like `_recording` above. `_muted` is written by
        # the event loop instead, and is an event for the same reason: it has
        # to be visible to the audio thread in the same instant.
        self._on = threading.Event()
        self._muted = threading.Event()

        # Set while `listen_for` is open, so that the detector is shown blocks
        # even with hands-free off. Read on the audio thread like the others.
        self._answering = threading.Event()

        # Counted down on the event loop, in samples rather than blocks so that
        # a block of any size costs what it actually holds.
        self._deaf_samples = 0

    @property
    def listening(self) -> bool:
        """Whether the microphone is live without anybody holding a key."""
        return self._on.is_set()

    def start(self) -> None:
        super().start()
        self._toggle.watch(on_press=self._toggled, on_release=_nothing)

    def stop(self) -> None:
        self._on.clear()
        self._toggle.stop()
        super().stop()

    def mute(self) -> None:
        """Stops listening while the assistant speaks. Called by `app.py`."""
        self._muted.set()

    def unmute(self) -> None:
        """Listens again, once the room has stopped repeating the answer.

        The order matters: the tail and the reset are in place before the flag
        is cleared, so no block of the assistant's own voice can arrive between
        the two and be treated as the beginning of a question.
        """
        self._deaf_samples = round(ECHO_TAIL_SECONDS * SAMPLE_RATE)
        self._endpoint.reset()
        self._muted.clear()

    # ----------------------------------------------------------------------
    # Called from other threads. Nothing here touches asyncio directly.
    # ----------------------------------------------------------------------

    def _heard(self, chunk: Audio) -> None:
        if self._recording.is_set():
            # The key is held: this block belongs to that recording, and the
            # detector is not shown it.
            self._to_loop(self._keep, chunk)
        elif self._answering.is_set() or (self._on.is_set() and not self._muted.is_set()):
            self._to_loop(self._examine, chunk)

    def _pressed(self) -> None:
        # Whatever the detector had half collected is not part of what the user
        # is about to say deliberately.
        super()._pressed()
        self._to_loop(self._endpoint.reset)

    async def listen_for(self, seconds: float) -> Audio | None:
        """One sentence within `seconds`, by the detector or the key; else `None`.

        Opened whether or not hands-free is on: the assistant asked a
        question, and hearing the answer is the point of asking. The echo
        tail `unmute` left behind still counts down first - the room
        repeating the question is not a yes. The detector starts afresh on
        either side of the window, and what it hears inside is the answer:
        not announced, and never a question.
        """
        self._endpoint.reset()
        self._answering.set()
        try:
            return await super().listen_for(seconds)
        finally:
            self._answering.clear()
            self._endpoint.reset()

    def _toggled(self) -> None:
        if self._on.is_set():
            self._on.clear()
        else:
            self._on.set()

        # Which way it went travels with the message rather than being read
        # again on the other side. Two presses in quick succession are two
        # messages that both run after both flips, and a listener that read the
        # flag would be told the mode changed to whatever it is *now*, twice.
        self._to_loop(self._switched, self._on.is_set())

    # ----------------------------------------------------------------------
    # Called on the event loop, in the order the threads above submitted them.
    # ----------------------------------------------------------------------

    def _switched(self, listening: bool) -> None:
        self._deaf_samples = 0
        self._endpoint.reset()
        if self.on_mode is not None:
            self.on_mode(listening)

    def _examine(self, chunk: Audio) -> None:
        """One block through the detector, and a sentence out when one ended."""
        if self._deaf_samples > 0:
            self._deaf_samples -= len(chunk)
            return

        started = self._endpoint.speaking
        finished = self._endpoint.feed(chunk)

        if self._window is not None:
            # The sentence answers the question just asked: it goes to whoever
            # asked, and the state machine is not told a new question began.
            for utterance in finished:
                if not self._window.done():
                    self._window.set_result([utterance])
            return

        # Announced before the sentence is handed over, and for the same reason
        # the parent announces the key going down: this is the moment the state
        # machine learns that whatever it was doing has been overtaken.
        if not started and self._endpoint.speaking and self.on_listening is not None:
            self.on_listening()

        for utterance in finished:
            self._finished.put_nowait([utterance])


def _as_audio(chunks: list[Audio]) -> Audio:
    """The blocks of one take as a single buffer."""
    if not chunks:
        # A stray press. Still a turn - what to make of it belongs to the
        # state machine, not here.
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _nothing() -> None:
    """A key coming up means nothing to a key that toggles."""


class SystemHotkey:
    """The real keyboard, through `pynput`."""

    def __init__(self, combination: str = DEFAULT_HOTKEY) -> None:
        # Deferred, and untyped: `pynput` ships no stubs, which is why the
        # listener is held as `Any`.
        from pynput import keyboard  # type: ignore[import-untyped]

        try:
            parsed = keyboard.HotKey.parse(combination)
        except ValueError as error:
            raise ValueError(f"{combination!r} is not a hotkey: {error}") from error

        self.keys = frozenset(parsed)
        self._listener: Any = None

    def watch(self, *, on_press: OnEvent, on_release: OnEvent) -> None:
        from pynput import keyboard

        combination = KeyCombination(self.keys)

        # A real press reports `ctrl_l`, while the combination is written as
        # `ctrl`. `canonical` is what makes the two the same key.
        def pressed(key: Any) -> None:
            if combination.press(self._listener.canonical(key)):
                on_press()

        def released(key: Any) -> None:
            if combination.release(self._listener.canonical(key)):
                on_release()

        self._listener = keyboard.Listener(on_press=pressed, on_release=released)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


def device_choice(text: str | None) -> int | str | None:
    """A device setting as `sounddevice` wants it: an index, words, or the default.

    An int is an index into its device list; a str is words matched, in order,
    against "<device name>, <host API>". A digit string handed over as a str
    matches no name at all - which is how `bench_mic.py --device 12` never
    found a device. Empty, or nothing, is the system default.
    """
    chosen = (text or "").strip()
    if not chosen:
        return None
    return int(chosen) if chosen.isdigit() else chosen


class MicrophoneUnavailableError(RuntimeError):
    """The input device could not be opened: nothing matched the name, or
    PortAudio refused it. Fixable by the user, so named for `run`."""


# When a dropped block is worth a line in the log: the first, so that one bad
# minute is on record, then rarely enough that a bad microphone cannot fill
# the file. Between these the count is kept and nothing is written.
_REPORT_OVERFLOWS_AT = frozenset({1, 10, 100})
_REPORT_OVERFLOWS_EVERY = 1000


class SystemMicrophone:
    """The real microphone, through `sounddevice`.

    Asked for 16 kHz mono, which is what everything downstream expects. Not
    every path to a microphone will give it: PortAudio's WASAPI runs a shared
    stream at the mixer's rate unless told to convert, and kernel streaming
    has no converter at all - measured 2026-09-05, the internal array's two
    rawer entries refused 16 kHz while its MME entry, with Windows resampling
    in between, accepted it. So a WASAPI device is told to convert, and a
    device that still refuses is opened at its own rate and brought to 16 kHz
    here, block by block (`audio/resample.py`). `rate` says which happened.
    """

    def __init__(self, *, device: int | str | None = None) -> None:
        self._device = device
        self._stream: Any = None
        self._on_chunk: OnChunk | None = None
        self._resampler: Resampler | None = None
        # The rate the device actually runs at once opened: `SAMPLE_RATE`
        # unless it would not, in which case the blocks are resampled.
        self.rate: int | None = None
        # Blocks the driver dropped before this saw them. In hands-free mode
        # each is a hole in a sentence that nothing else would notice.
        self.overflows = 0

    def open(self, on_chunk: OnChunk) -> None:
        # Imported here rather than at module scope: it loads PortAudio's
        # native library, which `assistant setup` has no use for. It ships no
        # type information either.
        import sounddevice  # type: ignore[import-untyped]

        self._on_chunk = on_chunk
        try:
            self._stream = self._open_stream(sounddevice)
            self._stream.start()
        except (ValueError, sounddevice.PortAudioError) as failure:
            # `ValueError` is a name that matched no device. `PortAudioError`
            # is a device that exists and would not open: held by another
            # program, or unplugged since the list was made.
            self.close()
            which = (
                "the default microphone"
                if self._device is None
                else f"the microphone {self._device!r}"
            )
            raise MicrophoneUnavailableError(f"{which} could not be opened: {failure}") from failure

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _open_stream(self, sounddevice: Any) -> Any:
        """16 kHz if the device will run at it; its own rate, resampled, if not."""
        facts = sounddevice.query_devices(self._device, "input")
        host = str(sounddevice.query_hostapis(facts["hostapi"])["name"])
        native = round(float(facts["default_samplerate"]))
        # Only WASAPI understands this, and PortAudio refuses a stream whose
        # host-specific settings belong to another host API.
        extra = sounddevice.WasapiSettings(auto_convert=True) if "WASAPI" in host else None

        try:
            stream = self._stream_at(sounddevice, SAMPLE_RATE, extra)
        except sounddevice.PortAudioError:
            if native == SAMPLE_RATE:
                raise
            # Whatever the refusal was, the one thing left to try is the rate
            # the device says it runs at. If that fails too, that error is
            # the one worth reading, and it is the one that propagates.
            stream = self._stream_at(sounddevice, native, extra)
            self._resampler = Resampler(native, SAMPLE_RATE)
            self.rate = native
            logger.info(
                "microphone {device!r} runs at {rate} Hz; resampling to {target}",
                device=self._device,
                rate=native,
                target=SAMPLE_RATE,
            )
        else:
            self._resampler = None
            self.rate = SAMPLE_RATE
        return stream

    def _stream_at(self, sounddevice: Any, rate: int, extra: Any) -> Any:
        return sounddevice.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            # Twenty milliseconds at whatever rate the device runs at, so a
            # resampled block is still the block size everything else counts on.
            blocksize=round(CHUNK_FRAMES * rate / SAMPLE_RATE),
            device=self._device,
            callback=self._block,
            extra_settings=extra,
        )

    def _block(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio's callback thread. Whatever this does, it does it quickly."""
        if status and status.input_overflow:
            # Counted always, written rarely: a log line per block, on this
            # thread, would be the next thing to cause an overflow.
            self.overflows += 1
            count = self.overflows
            if count in _REPORT_OVERFLOWS_AT or count % _REPORT_OVERFLOWS_EVERY == 0:
                logger.warning("microphone overflow: {count} blocks dropped so far", count=count)
        if self._on_chunk is not None:
            # One channel, and a copy: `indata` is PortAudio's own buffer and
            # holds the next block by the time anyone reads this one.
            chunk = indata[:, 0].copy()
            if self._resampler is not None:
                chunk = self._resampler.push(chunk)
            self._on_chunk(chunk)
