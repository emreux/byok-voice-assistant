"""Measures the local recogniser on the recorded fixtures (design.md section 11, 2.8).

Which Whisper size, how long it takes, how many words it gets wrong: the
table that ADR-001 rests on, and the one number section 8 decides by -
`small` at p95 over 1.2 s, or over 15% of words wrong, and cloud STT is
brought forward in phase 3.

    uv run python scripts/bench_stt.py                      # tiny, base, small, medium
    uv run python scripts/bench_stt.py --sizes small base   # some of them
    uv run python scripts/bench_stt.py --fixtures some/dir  # recordings elsewhere
    uv run python scripts/bench_stt.py --language en

Every `.wav` in `fixtures/audio/` that has a `.txt` beside it saying what was
said is transcribed the way the assistant transcribes - `LocalWhisper`, int8,
four threads, the pack's vocabulary hint - and what came back is scored
against the text with `jiwer`. Both sides are folded the same way first:
case, accents and punctuation off, the way search folds (`store/normalize.py`),
so that "Fatura." and "fatura" are one word and the rate measures hearing
rather than spelling. The first fixture is read once untimed per size: the
first call pays for things that are not transcription.

The recordings are personal and stay on this machine (`.gitignore`); only
the texts are in the repository. The numbers depend on the recordings: a
sentence read to the assistant's own voice is easier than one said across a
room, and a table made from synthesised speech is a floor, not a budget.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistant import locales
from assistant.audio.resample import Resampler
from assistant.config import is_configured, load_settings
from assistant.store.normalize import normalize_search
from assistant.stt.base import SAMPLE_RATE, Audio
from assistant.stt.local_whisper import LocalWhisper

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
SIZES = ("tiny", "base", "small", "medium")

# The decision gate of design.md section 8, 2.8, for the size the assistant
# ships with (`stt/local_whisper.py`).
DECIDING_SIZE = "small"
P95_LIMIT_SECONDS = 1.2
WER_LIMIT = 0.15

# What is not a word for the purpose of scoring: punctuation, and anything
# else that is not a letter, a digit or a space once the text is folded.
_NOT_A_WORD = re.compile(r"[^a-z0-9\s]+")


@dataclass
class Fixture:
    """One recording and what was said in it."""

    wav: Path
    reference: str
    pcm: Audio
    seconds: float


@dataclass
class Result:
    """What one size made of every fixture."""

    size: str
    load_seconds: float
    durations: list[float] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    @property
    def wer(self) -> float:
        import jiwer

        return float(jiwer.wer(self.references, self.hypotheses))

    @property
    def cer(self) -> float:
        import jiwer

        return float(jiwer.cer(self.references, self.hypotheses))


def fold(text: str) -> str:
    """Case, accents and punctuation off; one space between words.

    The same folding search uses, so that the recogniser's spelling of a
    proper noun and the reference's are one word; then punctuation, which
    nobody says out loud.
    """
    return " ".join(_NOT_A_WORD.sub(" ", normalize_search(text)).split())


def percentile(values: list[float], share: float) -> float:
    """Linearly interpolated: with twenty values, p95 falls between the
    nineteenth and the twentieth rather than being the maximum."""
    if len(values) < 2:
        return values[0] if values else 0.0
    return statistics.quantiles(values, n=100, method="inclusive")[round(share * 100) - 1]


def load_fixtures(directory: Path) -> list[Fixture]:
    """Every recording with a text beside it, at the assistant's sample rate."""
    fixtures: list[Fixture] = []
    for wav in sorted(directory.glob("*.wav")):
        text = wav.with_suffix(".txt")
        if not text.is_file():
            print(f"  skipped {wav.name}: no {text.name} beside it")
            continue
        fixtures.append(
            Fixture(
                wav=wav,
                reference=text.read_text(encoding="utf-8").strip(),
                pcm=(pcm := read_wav(wav)),
                seconds=len(pcm) / SAMPLE_RATE,
            )
        )
    return fixtures


def read_wav(path: Path) -> Audio:
    """Mono float32 at 16 kHz, however the file was written."""
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    mono: Audio = np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32)
    if rate != SAMPLE_RATE:
        mono = Resampler(int(rate), SAMPLE_RATE).push(mono)
    return mono


def stt_language() -> tuple[str, str]:
    """The language to expect and the words to expect, from the pack the
    assistant would use - the same chain `assistant run` follows."""
    settings = load_settings()
    code = settings.locale.code if is_configured() else locales.system_code()
    pack = locales.load(code)
    return pack.stt_language, pack.stt_vocabulary


async def measure(size: str, fixtures: list[Fixture], *, language: str, hint: str) -> Result:
    speech = LocalWhisper(model_size=size, vocabulary=[hint])
    started = time.perf_counter()
    await speech.load()
    result = Result(size=size, load_seconds=time.perf_counter() - started)
    print(f"\n[{size}] loaded in {result.load_seconds:.1f} s")

    # The first call pays for things that are not transcription; it is
    # made once and not counted.
    await speech.transcribe(fixtures[0].pcm, hint=language)

    for fixture in fixtures:
        started = time.perf_counter()
        heard = await speech.transcribe(fixture.pcm, hint=language)
        took = time.perf_counter() - started
        result.durations.append(took)
        result.hypotheses.append(fold(heard.text))
        result.references.append(fold(fixture.reference))
        print(
            f"[{size}] {fixture.wav.name}: {took:.2f} s for {fixture.seconds:.1f} s -> {heard.text}"
        )

    return result


def report(results: list[Result], fixtures: list[Fixture]) -> None:
    seconds = [fixture.seconds for fixture in fixtures]
    print(
        f"\n{len(fixtures)} fixtures, {statistics.mean(seconds):.1f} s of speech each on average "
        f"({min(seconds):.1f}-{max(seconds):.1f} s)"
    )
    print(f"{'size':<8} {'load':>6} {'p50':>7} {'p95':>7} {'WER':>7} {'CER':>7}")
    for result in results:
        print(
            f"{result.size:<8} {result.load_seconds:>5.1f}s "
            f"{percentile(result.durations, 0.5):>6.2f}s "
            f"{percentile(result.durations, 0.95):>6.2f}s "
            f"{result.wer:>6.1%} {result.cer:>6.1%}"
        )

    deciding = next((result for result in results if result.size == DECIDING_SIZE), None)
    if deciding is None:
        return
    p95 = percentile(deciding.durations, 0.95)
    slow = p95 > P95_LIMIT_SECONDS
    wrong = deciding.wer > WER_LIMIT
    print(
        f"\nThe gate of section 8 (2.8) for `{DECIDING_SIZE}`: p95 {p95:.2f} s against "
        f"{P95_LIMIT_SECONDS} s, WER {deciding.wer:.1%} against {WER_LIMIT:.0%}."
    )
    if slow or wrong:
        print("  Over the line: cloud STT is brought forward in phase 3 (ADR-001).")
    else:
        print("  Under both: local Whisper stays the default (ADR-001).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", nargs="+", default=list(SIZES), help="which model sizes")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES, help="where the .wav files are")
    parser.add_argument("--language", default=None, help="ISO 639-1; default: the pack's")
    args = parser.parse_args(argv)

    language, hint = stt_language()
    if args.language:
        language = args.language

    fixtures = load_fixtures(args.fixtures)
    if not fixtures:
        print(f"nothing to measure: no .wav with a .txt beside it in {args.fixtures}")
        print("record some with scripts/bench_mic.py - see fixtures/audio/README.md")
        return 2

    print(f"language {language!r}, vocabulary hint {hint!r}")
    results = [
        asyncio.run(measure(size, fixtures, language=language, hint=hint)) for size in args.sizes
    ]
    report(results, fixtures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
