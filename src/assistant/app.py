"""The state machine - where the pieces of phase 1 become a product (item 1.10).

`IDLE`, then `LISTENING` while the key is held, then `TRANSCRIBING`, `THINKING`
and `SPEAKING`, and back to `IDLE`. `CONFIRMING` arrives with the first tool in
phase 2.3 and `ANNOUNCING` with the announce queue in phase 4.2; the diagram in
section 3.1 is the target, not this file (rule 6).

Everything is injected - the microphone, the recogniser, the model, the voice,
the sound card - so the whole turn can be driven in a test without any of them.
That is also the reason this file is short: each piece already knows how to do
its own job, and what is left here is the order they do it in, and what happens
when one of them fails.

**Nothing is transcribed that was not speech.** Whisper answers a recording of
silence with confident looking words; measured on this machine, an empty
recording came back as a subtitle credit at 0.49 confidence against 0.75 for a
real sentence. The guard is the number and the length of the recording, never
the words themselves: a list of known hallucinations would be a language
constant in code, which section 3.12 does not allow.

**The key going down cuts the answer off.** Not the key coming up: the user is
speaking from the moment they press, so an assistant still talking is both rude
and something the microphone is recording. The same press abandons a turn that
is still being transcribed or thought about - there is no point paying for an
answer to a question that has been withdrawn.

**The microphone is deaf for exactly as long as the answer lasts.** Push to
talk never needed that - nobody holds the key while the assistant is talking -
but hands-free (`audio/capture.py`) would otherwise hear the answer come out of
the speakers, take it for a question and answer it, once per API call, until
somebody noticed. Which microphone is being deafened is not this file's
business: it calls `mute` and `unmute`, and the capture that has nothing to
mute does nothing.

**Three failures are said out loud, and no others.** A refused key means the
user has to go and renew it (section 3.2), a provider that cannot be reached
means try again, and a minute of thinking means the same. Anything else is a
bug in this project, and swallowing it into "I could not connect" is how it
would never get fixed.

**No user-facing sentence is written here.** The pack answers first and the
English constants below are the end of the chain, exactly as in the wizard
(section 3.12).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from assistant.agent.core import Agent
from assistant.audio.player import Speaker
from assistant.llm.base import AuthenticationError, ProviderError, Usage
from assistant.locales import Locale
from assistant.stt.base import SAMPLE_RATE, Audio, STTProvider, Transcript
from assistant.tts.base import TTSProvider, VoiceInfo

__all__ = [
    "MIN_CONFIDENCE",
    "MIN_UTTERANCE_SECONDS",
    "TEXT",
    "THINKING_TIMEOUT",
    "Assistant",
    "Capture",
    "Heard",
    "State",
    "Turn",
    "choose_voice",
]


class State(StrEnum):
    """Where the assistant is. A string so the status line of item 1.11 and the
    logs can print it without a table of names."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


# Section 3.1 rule 5. A turn that has not finished in a minute is not going to.
THINKING_TIMEOUT = 60.0

# Shorter than this and the key was tapped rather than held. Below a syllable,
# so nothing anybody meant to say is thrown away.
MIN_UTTERANCE_SECONDS = 0.35

# Measured on the target machine: a recording of silence came back from Whisper
# as words at 0.494, a real sentence at 0.752. Halfway is not the point - what
# matters is that the two are far apart and the line is between them.
MIN_CONFIDENCE = 0.6

# The last link of the chain of section 3.12: what is said when no locale pack
# offers a translation. Keys are unique across the whole project - the pack has
# one table of sentences, and `test_locales.py` checks that no two modules
# claim the same key.
TEXT: dict[str, str] = {
    "unreachable": "I could not reach the provider. Will you try again?",
    "key_invalid": "Your API key is not being accepted any more. You need to renew it.",
    "took_too_long": "That took too long. Will you try again?",
    "not_understood": "I did not catch that. Will you say it again?",
}


@dataclass(frozen=True, slots=True)
class Heard:
    """What the recogniser made of a recording, and whether it is worth a reply.

    Three outcomes rather than two, because "nothing happened" and "you said
    something and I could not read it" are different things to the person in
    the chair. A tapped key deserves silence; a sentence that did not survive
    the confidence floor deserves to be told so, or the assistant looks broken
    at exactly the moment it is working as designed.
    """

    text: str = ""
    missed: bool = False
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Turn:
    """What one turn came to, for whoever is showing or logging it.

    Nothing else keeps any of this: the transcript is gone once the model has
    it and the answer once it has been spoken. Item 1.11 puts the tokens in the
    log every turn, which is what the cost report of section 6 is built from -
    so they have to leave the turn, and a turn that only reported success would
    hide the ones that cost tokens and still failed.

    A turn that was *missed* carries no `heard` on purpose: there is no
    transcript this application is willing to stand behind. What it carries
    instead is the number that decided it, which is the only thing that makes
    a run of them diagnosable afterwards.
    """

    heard: str = ""
    said: str = ""
    usage: Usage = field(default_factory=Usage)
    missed: bool = False
    confidence: float | None = None


class Capture(Protocol):
    """The microphone side of push to talk, as the state machine needs it."""

    # Called on the event loop the moment recording starts. The state machine
    # stops the speaker from here; waiting for the finished utterance instead
    # would leave the assistant talking over the user until they let go.
    on_listening: Callable[[], None] | None

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def mute(self) -> None:
        """Stops listening, because the assistant is about to speak.

        A microphone that is only live while a key is held has nothing to do
        here. One that is live on its own would otherwise hear the answer come
        out of the speakers, take it for a question, and answer it - once per
        API call, for as long as nobody stopped it.
        """
        ...

    def unmute(self) -> None:
        """Listens again, now that the answer has been said."""
        ...

    async def utterance(self) -> Audio:
        """Waits for the next completed recording."""
        ...


class Assistant:
    """One turn after another, for as long as the program runs."""

    def __init__(
        self,
        *,
        capture: Capture,
        stt: STTProvider,
        agent: Agent,
        tts: TTSProvider,
        speaker: Speaker,
        locale: Locale,
        thinking_timeout: float = THINKING_TIMEOUT,
        on_state: Callable[[State], None] | None = None,
        on_turn: Callable[[Turn], None] | None = None,
    ) -> None:
        self._capture = capture
        self._stt = stt
        self._agent = agent
        self._tts = tts
        self._speaker = speaker
        self._locale = locale
        self._thinking_timeout = thinking_timeout
        self._on_state = on_state
        self._on_turn = on_turn

        self._said = {key: locale.say(key, default) for key, default in TEXT.items()}
        self._state = State.IDLE
        self._voice = ""

    @property
    def state(self) -> State:
        return self._state

    async def begin(self) -> None:
        """Picks the voice and opens the microphone, before anything is said.

        The voice is settled here rather than at the first answer: a machine
        with no voice installed can never answer at all, and that is worth
        finding out at startup instead of at two in the morning.
        """
        self._voice = await self._pick_voice()
        self._capture.on_listening = self._key_went_down
        self._capture.start()

        # Said out loud to whoever is watching, rather than merely being true.
        # The state has been `IDLE` since the constructor, but nothing had ever
        # announced it, so the status line went on showing the last thing it
        # was told - which is "loading the speech model" - until the first turn
        # was over. A program that looks like it never finished starting is one
        # nobody presses a key at.
        self._enter(State.IDLE)

    async def run(self) -> None:
        """Answers utterances until something stops the program."""
        await self.begin()
        try:
            while True:
                finished = await self.turn(await self._capture.utterance())
                if self._on_turn is not None:
                    self._on_turn(finished)
        finally:
            self._capture.stop()

    async def turn(self, pcm: Audio) -> Turn:
        """One recording, from what was heard to what was said back."""
        self._enter(State.TRANSCRIBING)
        heard = await self._heard(pcm)

        # A question the user has already withdrawn is answered by saying
        # nothing at all, including about not having understood it.
        if self._withdrawn():
            self._rest()
            return Turn()

        if not heard.text:
            return await self._missed(heard)

        self._enter(State.THINKING)
        said, usage = await self._answer(heard.text)
        await self._speak(said)
        self._rest()
        return Turn(heard=heard.text, said=said, usage=usage)

    async def _missed(self, heard: Heard) -> Turn:
        """Nothing usable came back. Whether that is worth saying depends.

        A recording too short to be a word is a key touched by accident, and an
        assistant that announced every one of those would be unusable. A
        recording that held speech the recogniser could not read is the
        opposite: silence there is indistinguishable from a broken program,
        and the user has no way to learn that speaking up would fix it.
        """
        if not heard.missed:
            self._rest()
            return Turn()

        said = self._said["not_understood"]
        await self._speak(said)
        self._rest()
        return Turn(said=said, missed=True, confidence=heard.confidence)

    # ----------------------------------------------------------------------
    # The turn, one stage at a time
    # ----------------------------------------------------------------------

    async def _heard(self, pcm: Audio) -> Heard:
        """What the user said, and what to make of it when they said nothing."""
        if len(pcm) < MIN_UTTERANCE_SECONDS * SAMPLE_RATE:
            # A tap rather than a hold. Transcribing it costs seconds of four
            # cores and answering it costs an API call, both for nothing - and
            # there is nothing here to have misheard, so nothing to say about.
            return Heard()

        transcript = await self._stt.transcribe(pcm, hint=self._locale.stt_language)
        if is_speech(transcript):
            return Heard(text=transcript.text.strip(), confidence=transcript.confidence)

        # Long enough to have been a sentence, and it was not one this can
        # stand behind. The number is kept because a run of these is only
        # diagnosable with it (`logs.py`).
        return Heard(missed=True, confidence=transcript.confidence)

    async def _answer(self, heard: str) -> tuple[str, Usage]:
        """The model's answer, or the sentence that explains why there is none.

        A turn that failed spent no tokens anybody can account for: what the
        provider counted before it refused is not reported to us, and guessing
        would put a number in the cost report that nothing backs.
        """
        try:
            answer = await asyncio.wait_for(self._agent.reply(heard), self._thinking_timeout)
        except TimeoutError:
            return self._said["took_too_long"], Usage()
        except AuthenticationError:
            # Never retried and never failed over: the key will not start
            # working on its own, and quietly using another model would put the
            # user on a bill they did not agree to (section 3.2).
            return self._said["key_invalid"], Usage()
        except (ProviderError, OSError):
            # OSError as well as our own: a socket that was refused never
            # reaches an adapter to be translated.
            return self._said["unreachable"], Usage()

        return answer.text, answer.usage

    async def _speak(self, said: str) -> None:
        if not said or self._withdrawn():
            return

        self._enter(State.SPEAKING)
        # The microphone is deaf for exactly as long as there is something for
        # it to mishear, and in a `finally` because an answer that failed
        # halfway through must not leave the assistant unable to hear at all.
        self._capture.mute()
        try:
            await self._speaker.play(
                self._tts.stream(_one(said), voice=self._voice),
                sample_rate=self._tts.sample_rate,
            )
        finally:
            self._capture.unmute()

    # ----------------------------------------------------------------------
    # Where it is
    # ----------------------------------------------------------------------

    def _enter(self, state: State) -> None:
        self._state = state
        if self._on_state is not None:
            self._on_state(state)

    def _withdrawn(self) -> bool:
        """Whether the user has started saying something else in the meantime."""
        return self._state is State.LISTENING

    def _rest(self) -> None:
        # A turn ending must not undo a press that arrived while it was
        # finishing: that press is a recording already in progress.
        if not self._withdrawn():
            self._enter(State.IDLE)

    def _key_went_down(self) -> None:
        was_speaking = self._state is State.SPEAKING
        self._enter(State.LISTENING)
        if was_speaking:
            self._speaker.stop()

    async def _pick_voice(self) -> str:
        preferred = self._locale.voice(self._tts.id)

        for language in (self._locale.code, None):
            # The locale's own language first. Failing that, anything installed:
            # the wrong accent is a poor answer, and no answer is worse.
            voice = choose_voice(await self._tts.list_voices(language), preferred)
            if voice:
                return voice

        raise RuntimeError(f"no speech voice is installed, for {self._locale.code!r} or otherwise")


def choose_voice(voices: Sequence[VoiceInfo], preferred: str | None) -> str:
    """Which of the installed voices to speak with, given the pack's preference.

    The pack names a preference rather than an identifier (item 1.8): `tr.toml`
    says `Tolga`, and what is installed is `Microsoft Tolga` under a registry
    path nobody would put in a TOML file. A preference that matches nothing is
    not an error - it is a machine where that voice was never installed.
    """
    if not voices:
        return ""

    if preferred:
        wanted = preferred.casefold()
        for voice in voices:
            if wanted in voice.display_name.casefold():
                return voice.id

    return voices[0].id


def is_speech(heard: Transcript) -> bool:
    """Whether a transcript is worth answering.

    Words with nothing behind them is what a speech recogniser produces from
    silence, and `confidence` is what tells the two apart in any language.
    An engine that reports no confidence at all is believed: "no opinion" is
    not "unsure", and treating it as such would make the assistant deaf the day
    it moves to a cloud recogniser.
    """
    if not heard.text.strip():
        return False
    return heard.confidence is None or heard.confidence >= MIN_CONFIDENCE


async def _one(said: str) -> AsyncIterator[str]:
    """The whole answer as a stream of one.

    Phase 1 waits for the model to finish before it speaks (section 4 budgets
    for it), so there is nothing to stream yet. `tts.base.sentences` regroups
    whatever arrives, so phase 2.8 changes this line and nothing else.
    """
    yield said
