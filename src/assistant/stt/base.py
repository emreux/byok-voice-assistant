"""What every speech-to-text engine is reduced to (design.md section 3.4).

The audio itself is the same everywhere - one channel, 16 kHz, float32 in
[-1, 1] - so what the providers actually disagree about is the shape of the
answer and whether it arrives all at once. Both are settled here.

**The streaming call exists from the first day, and phase 1 does not stream.**
Section 4 lists partial transcripts as the optimisation that removes about a
second from the felt latency. Had the protocol started with `transcribe` alone,
that optimisation would have been a protocol change - and a protocol that
changes breaks every call site at once. So `transcribe_stream` is declared now,
and a provider that cannot stream answers it with `buffered_stream` below:
collect the utterance, return one final result. The call sites written in phase
1 keep working unchanged when a streaming provider arrives in phase 3.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["SAMPLE_RATE", "Audio", "STTProvider", "Transcript", "buffered_stream"]

# What Whisper is trained on and what every other engine accepts. `audio/`
# captures at this rate so that nothing in the pipeline has to resample.
SAMPLE_RATE = 16_000

Audio = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Transcript:
    """What was heard, and how sure the engine is about it.

    `language` is filled in by the provider rather than by the caller. It is
    what the user actually spoke, which is not necessarily the language the
    assistant was configured for - and section 3.12 lets those differ on
    purpose, because the reply mirrors the speaker.
    """

    text: str
    is_final: bool = True
    confidence: float | None = None
    language: str = ""


@runtime_checkable
class STTProvider(Protocol):
    """A speech-to-text engine, local or hosted."""

    id: str
    supports_streaming: bool

    async def transcribe(self, pcm: Audio, *, hint: str | None = None) -> Transcript:
        """Turns a whole utterance into one final transcript.

        `hint` is the language the audio is expected to be in, as an ISO 639-1
        code. It is a hint and not a setting: section 3.12's `fixed` mode
        passes the locale's language, `auto` passes nothing, and the provider
        reports back in `Transcript.language` what it actually heard.
        """
        ...

    def transcribe_stream(
        self, pcm_chunks: AsyncIterator[Audio], *, hint: str | None = None
    ) -> AsyncIterator[Transcript]:
        """Yields partial transcripts as the audio arrives, then one final one.

        Declared `def` rather than `async def` for the same reason as
        `LLMProvider.stream`: implementations are async generators, and
        `async def` would type this as a coroutine returning an iterator.
        """
        ...


async def buffered_stream(
    provider: STTProvider,
    pcm_chunks: AsyncIterator[Audio],
    *,
    hint: str | None = None,
) -> AsyncIterator[Transcript]:
    """The `transcribe_stream` body of a provider that cannot stream.

    Shared rather than copied into each provider: three near-identical
    implementations is how one of them ends up subtly different from the other
    two, and the difference is only found in production.
    """
    collected = [chunk async for chunk in pcm_chunks]
    joined = np.concatenate(collected) if collected else np.empty(0, dtype=np.float32)
    pcm: Audio = joined.astype(np.float32, copy=False)

    yield await provider.transcribe(pcm, hint=hint)
