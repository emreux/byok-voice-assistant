"""What the microphone actually hears, and from how far (design.md item 2.9).

Hands-free listening rests on two things this machine has never been asked
about. Whether the microphone picks a sentence up from where the user sits -
the risk table calls insufficient range likely, and no amount of tuning fixes
a signal that is not there - and whether the assistant's own voice comes back
into the microphone loudly enough to be taken for a question.

    uv run python scripts/bench_mic.py --all      # all of it, in order
    uv run python scripts/bench_mic.py --at "2 m" # one distance
    uv run python scripts/bench_mic.py --quiet    # the room alone
    uv run python scripts/bench_mic.py --echo     # what the speakers put back

`--all` is the one to run. It walks through the room, one distance, another
distance and the echo, waits for you between them, loads Whisper once, and
prints the four side by side at the end - which is the only way any of these
numbers mean anything.

**A level is not an answer.** The detector saying "speech" and Whisper reading
the words are two different claims, and it is the second one that decides
whether the assistant is usable from across the room. So a take is transcribed
as well as measured, unless `--no-read` says otherwise, and the transcript is
judged against the same confidence floor `app.py` keeps.

**Nothing heard is not the same as nothing to hear.** `--echo` asks the
detector, not the average level: an answer is a few seconds of sound inside a
window that also holds the silence around it, and a mean over the whole window
buries the part that matters. If no frame reads as speech, the speakers were
not audible and the run measured nothing - saying so is the point, because a
silent take reported as a pass closes a risk that was never opened.

**The keypress is not part of the take.** Every recording starts a moment after
you have finished pressing anything: a room measured while somebody is typing
into it is not a room floor, and the first attempt at this measured one.

Nothing is written to disk and no audio leaves the machine.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from assistant.app import Heard, hear
from assistant.audio.capture import ECHO_TAIL_SECONDS, SystemMicrophone
from assistant.audio.vad import FRAME_SAMPLES, SPEECH_THRESHOLD, SileroVAD
from assistant.stt.base import NO_SPEECH_CEILING, SAMPLE_RATE, Audio, Transcript

if TYPE_CHECKING:
    from assistant.stt.local_whisper import LocalWhisper

SPOKEN = "Bir, iki, üç. Bu cümle mikrofonun ne duyduğunu ölçmek için okunuyor."

# Long enough for the keyboard to stop and the room to settle, short enough
# that nobody wonders whether the program has hung.
SETTLE_SECONDS = 1.0


@dataclass
class Take:
    """One recording and everything measured about it."""

    label: str
    pcm: Audio
    peak: float = 0.0
    rms: float = 0.0
    loudest: float = 0.0
    speaking: int = 0
    frames: int = 0
    transcript: str = ""
    confidence: float | None = None
    no_speech: float | None = None
    read: bool = False
    heard: list[float] = field(default_factory=list)

    @property
    def verdict(self) -> Heard:
        """What `app.py` would have made of this take - the same function, not a copy."""
        return hear(
            Transcript(
                text=self.transcript,
                confidence=self.confidence,
                no_speech_probability=self.no_speech,
            )
        )

    @property
    def answered(self) -> bool:
        """Whether `app.py` would have answered this turn rather than dropping it."""
        return bool(self.verdict.text)


def record(seconds: float, *, device: int | str | None = None) -> Audio:
    """Everything the microphone hears for `seconds`, at the rate phase 1 uses."""
    blocks: list[Audio] = []
    microphone = SystemMicrophone(device=device)
    microphone.open(blocks.append)
    try:
        time.sleep(seconds)
    finally:
        microphone.close()
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float32)


def loudness(pcm: Audio) -> tuple[float, float]:
    """Peak and RMS, the two numbers a level is ever described by."""
    if not len(pcm):
        return 0.0, 0.0
    return float(np.max(np.abs(pcm))), float(np.sqrt(np.mean(np.square(pcm))))


def probabilities(pcm: Audio) -> list[float]:
    """What the detector made of each 32 ms frame, in order."""
    detector = SileroVAD()
    return [
        detector.probability(pcm[at : at + FRAME_SAMPLES])
        for at in range(0, len(pcm) - FRAME_SAMPLES + 1, FRAME_SAMPLES)
    ]


def measure(pcm: Audio, *, label: str) -> Take:
    """One line per second, then what the detector made of the whole take."""
    peak, rms = loudness(pcm)
    heard = probabilities(pcm)
    take = Take(
        label=label,
        pcm=pcm,
        peak=peak,
        rms=rms,
        loudest=max(heard, default=0.0),
        speaking=sum(1 for one in heard if one >= SPEECH_THRESHOLD),
        frames=len(heard),
        heard=heard,
    )

    print(f"\n{label} - {len(pcm) / SAMPLE_RATE:.1f} s")
    print(f"{'second':>7}  {'peak':>7}  {'rms':>8}  {'loudest frame':>14}")
    for second in range(0, len(pcm), SAMPLE_RATE):
        window = pcm[second : second + SAMPLE_RATE]
        window_peak, window_rms = loudness(window)
        frames = probabilities(window)
        print(
            f"{second // SAMPLE_RATE:>7}  {window_peak:>7.3f}  "
            f"{window_rms:>8.5f}  {max(frames, default=0.0):>14.3f}"
        )

    print(f"\n  peak {take.peak:.3f}   rms {take.rms:.5f}")
    print(f"  frames called speech: {take.speaking} of {take.frames}")
    print(f"  loudest frame: {take.loudest:.3f} (threshold {SPEECH_THRESHOLD})")
    return take


def verdict(take: Take, *, quiet: bool) -> None:
    """What the levels mean for the detector. Not what they mean for Whisper."""
    if quiet:
        if take.loudest >= SPEECH_THRESHOLD:
            print("\n  The empty room already reads as speech. Hands-free would open")
            print("  turns nobody asked for; raise SPEECH_THRESHOLD or move the microphone.")
        else:
            print(f"\n  The empty room reads {take.loudest:.3f} against a threshold of")
            print(f"  {SPEECH_THRESHOLD}. Nothing here would open a turn.")
        return

    if take.speaking == 0:
        print("\n  Nothing in this take reads as speech. From this distance hands-free")
        print("  will not trigger at all - this is a microphone question, not a threshold one.")
    elif take.loudest < 0.8:
        print("\n  It triggers, but with no margin. Expect missed sentences from here.")
    else:
        print("\n  The detector is comfortable from this distance.")


async def whisper() -> LocalWhisper:
    """The recogniser, loaded once however many takes are read back."""
    from assistant.stt.local_whisper import LocalWhisper

    speech = LocalWhisper()
    await speech.load()
    return speech


async def read_back(take: Take, *, language: str, speech: LocalWhisper | None = None) -> None:
    """What Whisper makes of the take - the claim the levels cannot make.

    A signal the detector is sure about is not a signal the recogniser can
    read. Section 11's risk is "the microphone does not reach", and reaching
    means the words come back, not that something was loud enough to notice.
    """
    if speech is None:
        print("\n  Reading it back (Whisper is loading, this takes a few seconds)...")
        speech = await whisper()
    else:
        print("\n  Reading it back...")

    heard = await speech.transcribe(take.pcm, hint=language)
    take.transcript = heard.text.strip()
    take.confidence = heard.confidence
    take.no_speech = heard.no_speech_probability
    take.read = True

    confidence = "-" if heard.confidence is None else f"{heard.confidence:.2f}"
    no_speech = "-" if take.no_speech is None else f"{take.no_speech:.2f}"
    print(f'  transcript: "{take.transcript}"' if take.transcript else "  transcript: (nothing)")
    language_heard = heard.language or "-"
    print(f"  confidence: {confidence}   no-speech: {no_speech}   language: {language_heard}")

    # What `app.py` would do, said out loud rather than left for the reader to
    # work out. The decision is the engine's no-speech estimate against the
    # ceiling, never the confidence - that number is shown for the microphone
    # comparison, because a run of low ones is how a bad path is recognised.
    verdict = take.verdict
    if verdict.text:
        print("  The assistant would answer this turn.")
    elif verdict.missed:
        print("  Speech was heard and no words came of it: the assistant would say it")
        print("  did not understand. Not usable from here.")
    else:
        print(f"  The engine calls this silence (no-speech {no_speech} against the ceiling")
        print(f"  of {NO_SPEECH_CEILING}): the assistant would say nothing at all.")


def stt_language() -> str:
    """The language to expect, from the settings if there are any.

    The same chain `assistant run` uses. A script that hardcoded one would be
    the language constant section 3.12 exists to prevent.
    """
    from assistant import locales
    from assistant.config import is_configured, load_settings

    settings = load_settings()
    code = settings.locale.code if is_configured() else locales.system_code()
    return locales.load(code).stt_language


def ready(message: str) -> None:
    """Waits for the user, then lets the room go quiet before recording."""
    input(f"\n{message}\n  Press Enter when you are in place...")
    print(f"  Settling for {SETTLE_SECONDS:.0f} second, then recording. Go.")
    time.sleep(SETTLE_SECONDS)


async def echo(*, device: int | str | None = None, floor: Take | None = None) -> Take | None:
    """How much of the assistant's own voice comes back into the microphone.

    The tail is what `ECHO_TAIL_SECONDS` has to cover: the sound card holds
    part of the answer and the room holds the rest, and both arrive after the
    speaker has already been told it is finished.
    """
    from assistant.audio.player import SystemSpeaker
    from assistant.tts.sapi import SapiTTS

    tts = SapiTTS()
    voices = await tts.list_voices(None)
    if not voices:
        print("No speech voice is installed; there is nothing to echo.", file=sys.stderr)
        return None

    # The room before anything is said, as the control. Without it a take of
    # silence and a take with the speakers turned off are the same number.
    if floor is None:
        ready("The room first: say nothing, and do not type, for two seconds.")
        floor = measure(record(2.0, device=device), label="the room, as the control")
    if floor.loudest >= SPEECH_THRESHOLD:
        print("\n  Something was making a noise during the control. Nothing after it would")
        print("  mean anything - run it again and let those seconds pass in silence.")
        return None

    ready("Now the echo. Turn the speakers up to the volume you listen at, and say nothing.")
    blocks: list[Audio] = []
    microphone = SystemMicrophone(device=device)
    microphone.open(blocks.append)
    try:
        print(f"  Speaking, and listening to itself: {SPOKEN}")
        await SystemSpeaker().play(
            tts.stream(_one(SPOKEN), voice=voices[0].id), sample_rate=tts.sample_rate
        )
        spoken_until = sum(len(block) for block in blocks)
        # A second of room after the speaker fell silent: that is the part the
        # detector must not be shown.
        await asyncio.sleep(1.0)
    finally:
        microphone.close()

    pcm = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float32)
    during = measure(pcm[:spoken_until], label="the assistant's own voice")

    # The detector is the instrument, not the average level. An answer is a few
    # seconds of sound inside a window that also holds the silence before and
    # after it, and a mean over the whole window buries exactly the part that
    # matters. What is being asked is what the detector would have done.
    if during.loudest < SPEECH_THRESHOLD:
        print(f"\n  The detector never heard the speakers: loudest frame {during.loudest:.3f}")
        print(f"  against a threshold of {SPEECH_THRESHOLD}. This run measured nothing.")
        print("  Check the speakers are on, audible, and not headphones, then run it again.")
        return during

    print(f"\n  The assistant's own voice reads {during.loudest:.3f} at its own microphone -")
    print("  as loud as a person in the room. This is why it is deafened while it speaks.")

    tail = measure(pcm[spoken_until:], label="after it stopped")
    over = [at for at, one in enumerate(tail.heard) if one >= SPEECH_THRESHOLD]
    lasted = (max(over) + 1) * FRAME_SAMPLES / SAMPLE_RATE if over else 0.0
    print(f"\n  The room read as speech for {lasted:.2f} s after the answer ended.")
    print(f"  ECHO_TAIL_SECONDS is {ECHO_TAIL_SECONDS}.")
    if lasted > ECHO_TAIL_SECONDS:
        print("  Too short: the assistant would hear the end of its own sentence.")
    else:
        print("  Long enough on this machine, at this volume.")
    return during


def summary(takes: list[Take]) -> None:
    """The takes side by side, which is the only way any of them mean anything."""
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'take':<28} {'peak':>7} {'rms':>9} {'loudest':>8} {'speech':>9} {'answered':>9}")
    for take in takes:
        share = f"{take.speaking}/{take.frames}"
        answered = "-" if not take.read else ("yes" if take.answered else "NO")
        print(
            f"{take.label:<28} {take.peak:>7.3f} {take.rms:>9.5f} "
            f"{take.loudest:>8.3f} {share:>9} {answered:>9}"
        )

    read = [take for take in takes if take.read]
    if read:
        print(f"\nWhat was read out: {SPOKEN}")
        print(f"  {'':<26} {'conf':>5} {'nsp':>5}")
        for take in read:
            confidence = "-" if take.confidence is None else f"{take.confidence:.2f}"
            no_speech = "-" if take.no_speech is None else f"{take.no_speech:.2f}"
            print(f'  {take.label:<26} {confidence:>5} {no_speech:>5}  "{take.transcript}"')

    print(f"\nThresholds: detector {SPEECH_THRESHOLD}, no-speech ceiling {NO_SPEECH_CEILING}.")
    print("A take answered NO is one app.py would not have sent to the model: silence")
    print("gets nothing back, speech that made no words gets 'I did not catch that'.")


async def guided(*, seconds: float, device: int | str | None, no_read: bool) -> None:
    """The room, two distances and the echo, in one sitting."""
    language = stt_language()
    takes: list[Take] = []

    ready("First the room: say nothing, and do not type, for two seconds.")
    floor = measure(record(2.0, device=device), label="the room, nobody talking")
    verdict(floor, quiet=True)
    takes.append(floor)

    # Loaded once, before the first distance, so that the pause between the two
    # takes is the user walking rather than a gigabyte of weights.
    speech = None
    if not no_read:
        print("\nLoading Whisper once, for both distances...")
        speech = await whisper()

    for distance in ("1 m", "2 m"):
        ready(
            f"Now from {distance}. Read this out, exactly as written, for about "
            f"{seconds:.0f} seconds:\n\n    {SPOKEN}\n"
        )
        take = measure(record(seconds, device=device), label=f"speaking, {distance}")
        verdict(take, quiet=False)
        if speech is not None:
            await read_back(take, language=language, speech=speech)
        takes.append(take)

    heard = await echo(device=device, floor=floor)
    if heard is not None:
        takes.append(heard)

    summary(takes)


async def _one(said: str) -> AsyncIterator[str]:
    yield said


def main() -> int:
    # The transcript below is in whatever language was spoken, and Windows
    # hands a redirected stream its legacy code page - which has no `ğ` in it.
    # The same fix `assistant run` makes, through the same function.
    from assistant.__main__ import use_utf8

    use_utf8(sys.stdout)
    use_utf8(sys.stderr)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="Room, 1 m, 2 m and echo, in order.")
    parser.add_argument("--seconds", type=float, default=6.0, help="How long to record.")
    parser.add_argument("--at", default="", help="What to call this take, e.g. '2 m'.")
    parser.add_argument("--quiet", action="store_true", help="Measure the room, saying nothing.")
    parser.add_argument("--echo", action="store_true", help="Measure what the speakers put back.")
    parser.add_argument("--device", default=None, help="Input device, by index or name.")
    parser.add_argument(
        "--no-read", action="store_true", help="Skip the transcript, and measure levels only."
    )
    args = parser.parse_args()

    if args.all:
        asyncio.run(guided(seconds=args.seconds, device=args.device, no_read=args.no_read))
        return 0

    if args.echo:
        asyncio.run(echo(device=args.device))
        return 0

    label = args.at or ("the room, with nobody talking" if args.quiet else "speaking")
    if args.quiet:
        print(f"Say nothing for {args.seconds:.0f} seconds.")
    else:
        print(f"Speak normally for {args.seconds:.0f} seconds, from where you would sit.")
        print(f"Read this out: {SPOKEN}")

    take = measure(record(args.seconds, device=args.device), label=label)
    verdict(take, quiet=args.quiet)

    # The levels answered the detector's question. Whisper answers the one the
    # risk table actually asks, and only a take with words in it has one.
    if not args.quiet and not args.no_read:
        asyncio.run(read_back(take, language=stt_language()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
