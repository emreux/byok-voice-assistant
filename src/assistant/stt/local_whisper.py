"""Whisper on the CPU, off the event loop (design.md section 3.4).

The target machine has no discrete GPU, so this runs `faster-whisper` with int8
weights on four physical cores. That is seconds of work per utterance, and the
single most important thing in this file is where those seconds are spent:
**in a worker thread, never on the event loop.** Rule 4 of section 3.1 gives
anything awaited in `app.py` a 50 ms budget; a transcription on the loop would
freeze the reminder scheduler, the announce queue and audio capture together
until it finished. CTranslate2 releases the GIL, so a thread is enough and a
second process is not needed.

The model is loaded on first use, also in a thread - about a gigabyte of
weights - and `load()` exists so `app.py` can pay that cost at startup instead
of inside the user's first sentence.

**The recording is filtered for speech before it is decoded.** `vad_filter`
runs the same Silero network `audio/vad.py` uses over the whole recording and
hands the decoder only what it kept. Measured on this machine (2026-09-05): a
recording of a quiet room produced a subtitle credit without the filter and no
segment at all with it, while a real sentence came back unchanged. What is
left after that is judged by the decoder's own `no_speech_prob` per segment -
a hallucinated credit over the trailing silence scores 0.9 next to a real
sentence at 0.05 - and reported to the state machine as one number.

One thing this deliberately does not do yet: it does not pass `initial_prompt`.
Section 3.4's vocabulary trick arrives with the tools of phase 2, which are
what put names into it.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from typing import Any

from assistant.stt.base import NO_SPEECH_CEILING, Audio, Transcript, buffered_stream

__all__ = ["LocalWhisper"]

# Measured on the target machine (section 3.4): `small` int8 transcribes a
# short sentence in about 2.5 s and the translation is faithful.
DEFAULT_MODEL_SIZE = "small"

# Physical cores, not hyperthreads. The remaining capacity is what keeps audio
# playback and the state machine responsive while the model works.
DEFAULT_CPU_THREADS = 4

# `WhisperModel`, kept as `Any` so this module has no import-time dependency on
# the vendor package - see `_load_whisper`.
Model = Any
ModelFactory = Callable[[], Model]


class LocalWhisper:
    """`faster-whisper`, int8, on the CPU."""

    id = "local_whisper"

    # Phase 5 windows the audio and turns this on; until then the caller gets
    # `buffered_stream`, which is honest about producing nothing until the end.
    supports_streaming = False

    def __init__(
        self,
        *,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = DEFAULT_CPU_THREADS,
        build: ModelFactory | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._build = build if build is not None else self._load_whisper
        self._model: Model | None = None
        # Held while the model is built. Two turns starting at once would
        # otherwise read the weights into memory twice.
        self._loading = threading.Lock()

    async def load(self) -> None:
        """Loads the model now, so the first sentence does not wait for it."""
        await asyncio.to_thread(self._model_now)

    async def transcribe(self, pcm: Audio, *, hint: str | None = None) -> Transcript:
        return await asyncio.to_thread(self._transcribe_now, pcm, hint)

    def transcribe_stream(
        self, pcm_chunks: AsyncIterator[Audio], *, hint: str | None = None
    ) -> AsyncIterator[Transcript]:
        # Whisper has no use for a fragment: it needs the whole utterance to
        # decide what the first word was.
        return buffered_stream(self, pcm_chunks, hint=hint)

    # ----------------------------------------------------------------------
    # Everything below this line runs in a worker thread.
    # ----------------------------------------------------------------------

    def _model_now(self) -> Model:
        with self._loading:
            if self._model is None:
                self._model = self._build()
            return self._model

    def _transcribe_now(self, pcm: Audio, hint: str | None) -> Transcript:
        # `language=None` is what asks Whisper to detect the language itself.
        # `vad_filter` strips what its own detector calls silence before the
        # decoder sees it, so a recording of nothing decodes to nothing.
        segments, info = self._model_now().transcribe(pcm, language=hint, vad_filter=True)

        # The generator is lazy: the inference happens here, inside the thread.
        # Handing it back undrained would move the work onto the event loop.
        decoded = list(segments)

        # A segment the decoder itself marks as probably-not-speech is what it
        # produces over a stretch of noise the filter let through. The words
        # are dropped; the number is kept, so the caller learns why.
        heard = [segment for segment in decoded if segment.no_speech_prob < NO_SPEECH_CEILING]

        return Transcript(
            # Each segment already begins with its own separating space, so
            # joining with another one would double every gap.
            text="".join(segment.text for segment in heard).strip(),
            is_final=True,
            confidence=_confidence(heard),
            language=info.language or "",
            no_speech_probability=_no_speech(heard, decoded, kept_seconds=info.duration_after_vad),
        )

    def _load_whisper(self) -> Model:
        # Imported here rather than at module scope: it pulls in CTranslate2
        # and its native libraries, and `assistant --help` has no use for them.
        # The package ships no type information, which is why `Model` is `Any`.
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        return WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
            cpu_threads=self._cpu_threads,
        )


def _no_speech(heard: Sequence[Any], decoded: Sequence[Any], *, kept_seconds: float) -> float:
    """How likely it is that nothing was said, as one number for the caller.

    The lowest `no_speech_prob` among the segments that survived, because one
    segment of real speech means somebody spoke. When none survived, the
    lowest among those that were dropped: still over the ceiling, and the
    engine's own verdict rather than ours. When the decoder produced nothing
    at all, the filter decides - it kept no audio, so there was nothing to
    say, or it kept some and the decoder made nothing of it, which is speech
    the user deserves to be told was not understood.
    """
    if heard:
        return float(min(segment.no_speech_prob for segment in heard))
    if decoded:
        return float(min(segment.no_speech_prob for segment in decoded))
    return 1.0 if kept_seconds <= 0 else 0.0


def _confidence(segments: Iterable[Any]) -> float | None:
    """Turns Whisper's log probabilities into one number between 0 and 1.

    `avg_logprob` is the mean log probability of the tokens in a segment, which
    says nothing to the rest of the application on its own. Weighting each
    segment by how many tokens it holds and exponentiating gives the geometric
    mean probability per token.

    It is a measure of the decoder's doubt, not of whether anything was said,
    and it is biased by length: measured 2026-09-05, a correct "Merhaba." alone
    scored 0.48 and a recording of silence 0.54. It is reported for the log and
    the screen and is not what any decision rests on - `Transcript` says which
    number is.
    """
    weighted = 0.0
    tokens = 0
    for segment in segments:
        count = len(segment.tokens)
        weighted += segment.avg_logprob * count
        tokens += count

    if tokens == 0:
        return None
    return math.exp(weighted / tokens)
