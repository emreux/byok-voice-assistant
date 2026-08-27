"""Proves the speech-to-text path end to end without a microphone.

Speaks a known sentence into a WAV file with the Windows speech engine, feeds
that file to the local model, and prints what came back with the time it took.
It is a wiring check, not a benchmark: synthesised speech is far easier than a
real room, so the numbers here are a floor, not a budget. The real measurement
is `scripts/bench_stt.py` in phase 2.8 (design.md section 11).

    uv run python scripts/smoke_stt.py
    uv run python scripts/smoke_stt.py --language tr --text "Merhaba, nasilsin?"
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

DEFAULT_TEXT = "The assistant is listening and the model is loaded."
DEFAULT_LANGUAGE = "en"
MODEL_SIZE = "small"

# SPSF_16kHz16BitMono. The enumeration counts in pairs, so 22 is *22 kHz*
# 16 bit mono - a plausible looking number that writes the wrong file.
SAPI_AUDIO_FORMAT = 18
FILE_MODE_WRITE = 3


def synthesize(text: str, destination: Path) -> None:
    """Writes `text` to `destination` as a WAV file using the Windows voice."""
    import win32com.client

    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    stream.Format.Type = SAPI_AUDIO_FORMAT
    stream.Open(str(destination), FILE_MODE_WRITE)
    try:
        engine = win32com.client.Dispatch("SAPI.SpVoice")
        engine.AudioOutputStream = stream
        engine.Speak(text)
    finally:
        stream.Close()


def transcribe(path: Path, language: str) -> tuple[str, float]:
    """Returns the transcript and how long the transcription took."""
    from faster_whisper import WhisperModel

    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)

    started = time.perf_counter()
    segments, _info = model.transcribe(str(path), language=language)
    # The generator is lazy: inference happens while it is consumed, so the
    # timer has to wrap the consumption, not the call above.
    text = " ".join(segment.text for segment in segments).strip()
    return text, time.perf_counter() - started


def main() -> int:
    """Synthesises one sentence, transcribes it, and prints both."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        wav = Path(directory) / "sample.wav"

        started = time.perf_counter()
        synthesize(arguments.text, wav)
        synthesis_seconds = time.perf_counter() - started
        size_kb = wav.stat().st_size / 1024

        print(f"spoken   : {arguments.text!r}")
        print(f"written  : {size_kb:.0f} kB in {synthesis_seconds:.2f} s")

        transcript, seconds = transcribe(wav, arguments.language)

    print(f"heard    : {transcript!r}")
    print(f"took     : {seconds:.2f} s  (model '{MODEL_SIZE}', int8, 4 cpu threads)")

    if not transcript:
        print("\nNothing came back - the model did not hear the file.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
