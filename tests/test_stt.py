"""Speech to text: the protocol, and the local model behind it.

Two claims carry this module. The first is the 50 ms rule of section 3.1: the
model saturates four cores for seconds at a time, so if it ran on the event
loop the scheduler, the announce queue and audio capture would all stop with
it. `test_transcribing_leaves_the_event_loop_free` is the test that notices.

The second is that a provider which cannot stream still answers the streaming
call (section 3.4). The interface exists from day one so that phase 5's
partial transcripts are not a protocol change; until then the default body
collects the utterance and returns one final result.

Nothing here loads a real model - a `small` int8 load takes minutes on a cold
cache and seconds on a warm one. The model is injected, and the real thing is
proved by `scripts/smoke_stt.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from assistant.stt.base import SAMPLE_RATE, Audio, STTProvider, Transcript, buffered_stream
from assistant.stt.local_whisper import LocalWhisper


@dataclass
class FakeSegment:
    """What `faster_whisper` yields: text plus the numbers behind it."""

    text: str
    avg_logprob: float = -0.1
    tokens: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class FakeInfo:
    language: str = "tr"
    language_probability: float = 0.99


class FakeModel:
    """Stands in for `WhisperModel`, including its laziness.

    The real `transcribe` returns a generator and does the work while it is
    consumed, so the delay lives in the generator too: a caller that returns
    the generator without draining it would look fast here and block the event
    loop in production.
    """

    def __init__(
        self,
        *,
        texts: tuple[str, ...] = (" Merhaba,", " nasilsin?"),
        segments: tuple[FakeSegment, ...] | None = None,
        language: str = "tr",
        avg_logprob: float = -0.1,
        delay: float = 0.0,
    ) -> None:
        self.heard = (
            segments
            if segments is not None
            else tuple(FakeSegment(text=text, avg_logprob=avg_logprob) for text in texts)
        )
        self.language = language
        self.delay = delay
        self.calls: list[tuple[Audio, dict[str, Any]]] = []

    def transcribe(self, audio: Audio, **options: Any) -> tuple[Iterator[FakeSegment], FakeInfo]:
        self.calls.append((audio, options))
        return self._segments(), FakeInfo(language=self.language)

    def _segments(self) -> Iterator[FakeSegment]:
        if self.delay:
            time.sleep(self.delay)
        yield from self.heard


def silence(seconds: float = 0.5) -> Audio:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


async def chunks_of(*buffers: Audio) -> AsyncIterator[Audio]:
    for buffer in buffers:
        yield buffer


def whisper(**kwargs: Any) -> tuple[LocalWhisper, FakeModel]:
    model = FakeModel(**kwargs)
    return LocalWhisper(build=lambda: model), model


# --------------------------------------------------------------------------
# The transcript
# --------------------------------------------------------------------------


async def test_the_transcript_is_the_segments_joined() -> None:
    stt, _ = whisper()

    transcript = await stt.transcribe(silence())

    assert transcript.text == "Merhaba, nasilsin?"
    assert transcript.is_final is True


async def test_the_language_is_the_one_the_model_reports() -> None:
    """Section 3.12: the conversation language is whatever was spoken, and the
    agent is told which that was."""
    stt, _ = whisper(language="en")

    assert (await stt.transcribe(silence())).language == "en"


async def test_the_hint_is_given_to_the_model_as_the_language() -> None:
    """Phase 1 runs in `fixed` mode: the locale's language goes in as a hint."""
    stt, model = whisper()

    await stt.transcribe(silence(), hint="tr")

    assert model.calls[0][1]["language"] == "tr"


async def test_without_a_hint_the_model_detects_the_language_itself() -> None:
    stt, model = whisper()

    await stt.transcribe(silence())

    assert model.calls[0][1]["language"] is None


async def test_the_audio_reaches_the_model_untouched() -> None:
    stt, model = whisper()
    spoken = silence(0.25)

    await stt.transcribe(spoken)

    heard, _ = model.calls[0]
    assert heard.dtype == np.float32
    assert len(heard) == len(spoken)


async def test_silence_is_an_empty_transcript_not_an_error() -> None:
    """The user pressed the key and said nothing; that is a turn to drop, not
    a crash to report."""
    stt, _ = whisper(texts=())

    transcript = await stt.transcribe(silence())

    assert transcript.text == ""
    assert transcript.confidence is None


async def test_the_confidence_is_the_average_token_probability() -> None:
    """Whisper reports a log probability per segment; a bare `avg_logprob` of
    -0.7 means nothing to the rest of the application."""
    stt, _ = whisper(avg_logprob=np.log(0.5))

    transcript = await stt.transcribe(silence())

    assert transcript.confidence is not None
    assert transcript.confidence == pytest.approx(0.5)


async def test_a_long_confident_segment_outweighs_a_short_unsure_one() -> None:
    """Nine sure tokens and one unsure one is a good transcript, not half a
    good one - so the segments are weighted by how much was said in them."""
    stt, _ = whisper(
        segments=(
            FakeSegment(text=" uzun ve net bir cumle", avg_logprob=math.log(0.9), tokens=[0] * 9),
            FakeSegment(text=" ha", avg_logprob=math.log(0.1), tokens=[0]),
        )
    )

    transcript = await stt.transcribe(silence())

    # Weighted by tokens: 0.722. An unweighted mean of the two segments would
    # be 0.300 - the one-token stumble would count as half the sentence.
    per_token = math.exp((9 * math.log(0.9) + math.log(0.1)) / 10)
    assert transcript.confidence == pytest.approx(per_token)
    assert transcript.confidence == pytest.approx(0.722, abs=0.001)


# --------------------------------------------------------------------------
# The 50 ms rule (section 3.1)
# --------------------------------------------------------------------------


async def test_transcribing_leaves_the_event_loop_free() -> None:
    """The one thing this module must not do.

    Transcription is seconds of four-core work. Run on the loop, it stops the
    scheduler, the announce queue and audio capture until it finishes.
    """
    stt, _ = whisper(delay=0.3)
    ticks = 0

    async def clock() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticking = asyncio.create_task(clock())
    await stt.transcribe(silence())
    ticking.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ticking

    assert ticks >= 5, "the event loop was blocked while the model worked"


async def test_loading_the_model_leaves_the_event_loop_free() -> None:
    """A `small` int8 load reads about a gigabyte; that blocks too."""
    built_on: list[str] = []

    def build() -> FakeModel:
        built_on.append(threading.current_thread().name)
        return FakeModel()

    await LocalWhisper(build=build).load()

    assert built_on and built_on[0] != threading.main_thread().name


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


async def test_nothing_is_loaded_until_something_is_said() -> None:
    """Constructing the provider must stay cheap: `assistant --help` and the
    setup wizard both import this module and neither needs a model."""
    built = 0

    def build() -> FakeModel:
        nonlocal built
        built += 1
        return FakeModel()

    LocalWhisper(build=build)

    assert built == 0


async def test_the_model_is_loaded_once_however_often_it_is_used() -> None:
    built = 0

    def build() -> FakeModel:
        nonlocal built
        built += 1
        return FakeModel()

    stt = LocalWhisper(build=build)
    await stt.load()
    await stt.transcribe(silence())
    await stt.transcribe(silence())

    assert built == 1


async def test_two_turns_at_once_do_not_load_two_models() -> None:
    """Two loads at the same time would read two gigabytes into memory."""
    built = 0

    def build() -> FakeModel:
        nonlocal built
        built += 1
        time.sleep(0.05)  # long enough for the other thread to arrive
        return FakeModel()

    stt = LocalWhisper(build=build)
    await asyncio.gather(stt.transcribe(silence()), stt.transcribe(silence()))

    assert built == 1


# --------------------------------------------------------------------------
# Streaming, for a provider that cannot stream
# --------------------------------------------------------------------------


async def test_a_provider_that_cannot_stream_still_answers_the_stream_call() -> None:
    stt, _ = whisper()

    results = [t async for t in stt.transcribe_stream(chunks_of(silence(), silence()))]

    assert len(results) == 1
    assert results[0].is_final is True


async def test_the_whole_utterance_reaches_the_model_in_one_piece() -> None:
    """Whisper has no use for a fragment; the pieces are joined in order."""
    stt, model = whisper()
    first, second = np.full(3, 0.1, dtype=np.float32), np.full(2, 0.2, dtype=np.float32)

    async for _ in stt.transcribe_stream(chunks_of(first, second)):
        pass

    heard, _ = model.calls[0]
    assert np.array_equal(heard, np.concatenate([first, second]))


async def test_a_stream_that_carried_no_audio_is_not_an_error() -> None:
    """The key was pressed and released before a single frame arrived."""
    stt, model = whisper(texts=())

    results = [t async for t in stt.transcribe_stream(chunks_of())]

    assert [t.text for t in results] == [""]
    assert len(model.calls[0][0]) == 0


async def test_the_hint_survives_the_stream() -> None:
    stt, model = whisper()

    async for _ in stt.transcribe_stream(chunks_of(silence()), hint="en"):
        pass

    assert model.calls[0][1]["language"] == "en"


async def test_the_default_body_is_shared_not_copied() -> None:
    """`buffered_stream` is what every non-streaming provider uses; writing it
    again per provider is how one of them ends up wrong."""

    class Deaf:
        id = "deaf"
        supports_streaming = False

        async def transcribe(self, pcm: Audio, *, hint: str | None = None) -> Transcript:
            return Transcript(text=f"{len(pcm)} samples", language=hint or "")

        def transcribe_stream(
            self, pcm_chunks: AsyncIterator[Audio], *, hint: str | None = None
        ) -> AsyncIterator[Transcript]:
            return buffered_stream(self, pcm_chunks, hint=hint)

    results = [t async for t in Deaf().transcribe_stream(chunks_of(silence(0.1)))]

    assert [t.text for t in results] == ["1600 samples"]


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------


def test_the_local_model_is_an_stt_provider() -> None:
    stt, _ = whisper()

    assert isinstance(stt, STTProvider)
    assert stt.supports_streaming is False
