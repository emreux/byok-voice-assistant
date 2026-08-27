"""Push to talk: record while the hotkey is held (design.md section 8, item 1.5).

Phase 1 has no voice activity detection on purpose. The user says where the
sentence ends by letting go of the key, which costs nothing and cuts nobody
off mid-word; live endpointing arrives in phase 2.9 with a measured threshold
behind it. Everything here is written so that turning it on later replaces one
class and leaves `app.py` alone.

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
the callback and never reaches memory.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from assistant.stt.base import SAMPLE_RATE, Audio

__all__ = [
    "CHUNK_FRAMES",
    "DEFAULT_HOTKEY",
    "Hotkey",
    "KeyCombination",
    "Microphone",
    "PushToTalk",
    "SystemHotkey",
    "SystemMicrophone",
]

# `pynput`'s own spelling. Phase 4 moves it into `config.toml`; until then it
# is one constant rather than a string repeated in three places.
DEFAULT_HOTKEY = "<ctrl>+<alt>+<space>"

# 20 ms per block. The block being filled when the key comes up is still in
# PortAudio's hands and never arrives, so this is also how much of the end of
# the sentence can be lost: measured at 100 ms blocks, two seconds of holding
# produced 1.90 s of audio. Twenty milliseconds is below a syllable, and fifty
# callbacks a second is nothing.
CHUNK_FRAMES = SAMPLE_RATE // 50

OnChunk = Callable[[Audio], None]
OnEvent = Callable[[], None]


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
    ) -> None:
        self._hotkey = hotkey if hotkey is not None else SystemHotkey()
        self._microphone = microphone if microphone is not None else SystemMicrophone()

        # Read by the audio callback on PortAudio's thread, written by the
        # keyboard listener on its own. A plain event, deliberately: going
        # through the loop would lose the blocks recorded in between.
        self._recording = threading.Event()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._take: list[Audio] | None = None
        self._finished: asyncio.Queue[list[Audio]] = asyncio.Queue()

    def start(self) -> None:
        """Opens the microphone and starts watching the keyboard."""
        self._loop = asyncio.get_running_loop()
        self._microphone.open(self._heard)
        self._hotkey.watch(on_press=self._pressed, on_release=self._released)

    def stop(self) -> None:
        self._recording.clear()
        self._hotkey.stop()
        self._microphone.close()

    def __enter__(self) -> PushToTalk:
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
        chunks = await self._finished.get()
        if not chunks:
            # A stray press. Still a turn - what to make of it belongs to the
            # state machine, not here.
            return np.empty(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

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

    def _keep(self, chunk: Audio) -> None:
        # `None` means the recording already ended: the audio thread saw the
        # flag a moment before the keyboard thread cleared it. That block was
        # recorded, but it belongs to a turn that is over.
        if self._take is not None:
            self._take.append(chunk)

    def _end(self) -> None:
        take, self._take = self._take, None
        if take is not None:
            self._finished.put_nowait(take)


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


class SystemMicrophone:
    """The real microphone, through `sounddevice`."""

    def __init__(self, *, device: int | str | None = None) -> None:
        self._device = device
        self._stream: Any = None
        self._on_chunk: OnChunk | None = None

    def open(self, on_chunk: OnChunk) -> None:
        # Imported here rather than at module scope: it loads PortAudio's
        # native library, which `assistant setup` has no use for. It ships no
        # type information either.
        import sounddevice  # type: ignore[import-untyped]

        self._on_chunk = on_chunk
        self._stream = sounddevice.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_FRAMES,
            device=self._device,
            callback=self._block,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _block(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio's callback thread. Whatever this does, it does it quickly.

        `status` carries the overrun flags; reporting them needs the logging
        setup, which arrives with the state machine in item 1.10.
        """
        if self._on_chunk is not None:
            # One channel, and a copy: `indata` is PortAudio's own buffer and
            # holds the next block by the time anyone reads this one.
            self._on_chunk(indata[:, 0].copy())
