"""The state machine - where the pieces of phase 1 become a product (item 1.10).

`IDLE`, then `LISTENING` while the key is held, then `TRANSCRIBING`, `THINKING`
and `SPEAKING`, and back to `IDLE`. `CONFIRMING` is the window of phase 2.3,
opened from inside `THINKING` when a tool wants a yes; `ANNOUNCING` arrives
with the announce queue in phase 4.2. The diagram in section 3.1 is the
target, not this file (rule 6).

Everything is injected - the microphone, the recogniser, the model, the voice,
the sound card - so the whole turn can be driven in a test without any of them.
That is also the reason this file is short: each piece already knows how to do
its own job, and what is left here is the order they do it in, and what happens
when one of them fails.

**A recording is judged by whether it held speech, never by how sure the
decoder was of its words.** Whisper answers a recording of silence with
confident looking words, and answers a correct single word with unsure ones:
measured on this machine (2026-09-05), a subtitle credit over silence scored
0.54 and "Merhaba." alone 0.48, so no confidence floor can separate them - the
one at 0.6 that used to live here dropped short sentences and let the owner
say "merhaba" to a program that did nothing. The engine's own estimate of
whether anything was said does separate them (0.86 against 0.06), and `hear`
below rests on that and on nothing else. The words themselves are never
looked at: a list of known hallucinations would be a language constant in
code, which section 3.12 does not allow.

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

**A tool that wants a yes gets one out loud, or does not run.** The gate of
section 3.9 hands `confirm` the sentence with the real argument values in it;
this file reads it, tells the user how to answer, and opens the microphone
for six seconds without waiting for a key (section 3.1 rule 2). A "no"
anywhere in the answer wins over a "yes"; silence is a no; an answer with
neither word in it is asked about once more, and a second such answer is a no
as well. Every path that is not a clear yes ends in nothing being done - the
window exists so that an action the user did not agree to cannot happen, not
so that one they did agree to happens quickly. The exchange belongs to the
gate, not to the model: nothing said in it reaches the conversation. The
words that count as yes and no come from the locale pack; the English ones
below are the end of the chain.

**What a turn cost is written down, and what a day cost is said out loud.**
Every turn that reached the model goes to `usage_log` through the tracker,
priced (section 6). Past the day's or the month's limit (section 3.11)
every answer starts with a warning, and with `hard_stop` on the model is
not asked at all. An answer the token limit cut short says so at its end:
a sentence that stops halfway with nothing said about it is a bug nobody
can find.

**A short command the pack lists never reaches the model.** "saat kaç",
"dur", "iptal": the phrases in the pack's `[intents]` table are matched
whole (`agent/intents.py`) before anything is sent, and answered here. The
time is asked of `get_current_time` through the same gate the model's calls
go through - the fast path skips the model, not the gate (invariant 1) -
and said in the pack's words. Stop and cancel are answered by silence: in
phase 2 the microphone is deaf while the assistant talks and a key press
already cuts it off, so all there is to do about them is not spend a
request. A turn like this costs nothing and says so: no tokens, no row on
the bill.

**The first sentence is spoken while the model is still writing the
second** (2.8, architecture guide section 11). The loop hands the words out
as they arrive (`Agent.stream_reply`), `tts.base.sentences` cuts them at the
first boundary, and the speaker has that sentence while the rest is on its
way - phase 1 waited for the whole answer, and the first sound came 2.5-6 s
after the key (measured 3.3 s). The minute of section 3.1 rule 6 is the
time to the first word; after it the model finishes at its own pace, since
a long answer read out loud is not a turn that hung. The speaker pulls: the
model is read only as fast as the speaker asks for more, so a tool the
model calls runs while the speaker is waiting - which is also why the
question a tool asks is not spoken over an answer under way.

**A tool that takes long is said to be taking long.** The loop announces a
tool round the moment it turns to run one; when no word has followed within
`FILLER_DELAY_SECONDS`, the filler is said instead, once per turn - "bir
saniye, bakıyorum" is what makes two seconds of silence feel like none
(section 4). The words come from the pack's `[speech] filler`; `FILLERS`
below is the end of the chain. Never while a question is being asked: the
user is answering it.

**The key going down ends the stream, request and all.** Not only the
sound: the loop is left where it stood, the provider's stream is closed
under it, and nothing is remembered - the user is asking something else.

**Three failures are said out loud, and no others.** A refused key means the
user has to go and renew it (section 3.2), a provider that cannot be reached
means try again, and a minute of thinking means the same. Anything else is a
bug in this project, and swallowing it into "I could not connect" is how it
would never get fixed. A sound device that fails while the answer is being
played is the one exception, and it is not said out loud either - there is
nothing left to say it with. The words are already on the screen and in the
turn; the failure goes to the log, and the turn ends the way any other does.

**No user-facing sentence is written here.** The pack answers first and the
English constants below are the end of the chain, exactly as in the wizard
(section 3.12).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from loguru import logger

from assistant.agent.core import Agent, Answer, Dispatch
from assistant.agent.intents import GET_TIME, TIME_TOOL, match_intent
from assistant.agent.limits import Limits
from assistant.audio.player import PlaybackError, Speaker
from assistant.llm.base import AuthenticationError, ProviderError, ToolCall, Usage
from assistant.locales import Locale
from assistant.store.normalize import normalize_search
from assistant.stt.base import NO_SPEECH_CEILING, SAMPLE_RATE, Audio, STTProvider, Transcript
from assistant.tts.base import TTSProvider, VoiceInfo
from assistant.usage.tracker import UsageTracker

__all__ = [
    "CONFIRM_WINDOW_SECONDS",
    "FILLERS",
    "FILLER_DELAY_SECONDS",
    "MIN_UTTERANCE_SECONDS",
    "NO_WORDS",
    "TEXT",
    "THINKING_TIMEOUT",
    "YES_WORDS",
    "Assistant",
    "Capture",
    "Heard",
    "NoVoiceError",
    "State",
    "Turn",
    "choose_voice",
    "hear",
    "read_answer",
]


class NoVoiceError(RuntimeError):
    """No speech voice is installed, for the locale or otherwise.

    The user can install one; the program cannot. Named so that `assistant
    run` can say so in a sentence instead of a traceback.
    """


class State(StrEnum):
    """Where the assistant is. A string so the status line of item 1.11 and the
    logs can print it without a table of names."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    CONFIRMING = "confirming"
    SPEAKING = "speaking"


# Section 3.1 rule 5. A turn that has not finished in a minute is not going to.
# The number is the section 3.11 table's, written once in `limits.py`;
# `config.toml` `[limits] turn_seconds` replaces it through the constructor.
THINKING_TIMEOUT = Limits().turn_seconds

# Section 3.1 rule 6. How long the microphone stays open for a yes or a no
# after the question has been read; what comes after it is a no.
CONFIRM_WINDOW_SECONDS = 6.0

# Shorter than this and the key was tapped rather than held. Below a syllable,
# so nothing anybody meant to say is thrown away.
MIN_UTTERANCE_SECONDS = 0.35

# The last link of the chain of section 3.12 for the two words the window
# listens for, as `TEXT` is for the sentences: the pack's `[speech]` table
# answers first, and a pack that has none gets these.
YES_WORDS = ("yes", "ok", "okay", "confirm")
NO_WORDS = ("no", "cancel", "stop")

# The end of the same chain for what is said while a tool takes its time
# (2.8): the pack's `[speech] filler` answers first. Said in turn, so a
# pack that lists two is not heard saying the same one every time.
FILLERS = ("One moment, let me check...",)

# How long a tool round may stay silent before the filler is said. A native
# tool returns in 5-50 ms and the model's next first token takes 400-1500
# ms after it (section 4), so this is the model's silence that is covered,
# not the tool's; below it the answer is on its way and a "one moment"
# would only push it back. The right number is measured, not argued.
FILLER_DELAY_SECONDS = 0.3

# The last link of the chain of section 3.12: what is said when no locale pack
# offers a translation. Keys are unique across the whole project - the pack has
# one table of sentences, and `test_locales.py` checks that no two modules
# claim the same key.
TEXT: dict[str, str] = {
    "unreachable": "I could not reach the provider. Will you try again?",
    "key_invalid": "Your API key is not being accepted any more. You need to renew it.",
    "took_too_long": "That took too long. Will you try again?",
    "not_understood": "I did not catch that. Will you say it again?",
    # Read after the gate's question, so the user knows what kind of answer is
    # being listened for - and again, alone, when the answer had neither.
    "confirm_hint": "Say yes or no.",
    "confirm_again": "I did not catch that. Yes, or no?",
    # The token limit of section 3.11 ended the answer; said at its end,
    # where the cut is.
    "answer_cut_off": "The end of the answer was cut off.",
    # Said before the answer once a spending limit is passed - every turn,
    # until the day or the month turns; and instead of an answer with
    # `hard_stop` on.
    "daily_over": "You have gone over today's spending limit.",
    "monthly_over": "You have gone over this month's spending limit.",
    "spend_stopped": "The spending limit has been passed, so I am not asking the model.",
    # The fast path's answer to "what time is it" (section 4), with the
    # hour and the minute as two numbers: the number formatter is phase 3's.
    "time_is": "It is {hour} {minute}.",
}


@dataclass(frozen=True, slots=True)
class Heard:
    """What the recogniser made of a recording, and whether it is worth a reply.

    Three outcomes rather than two, because "nothing happened" and "you said
    something and I could not read it" are different things to the person in
    the chair. A tapped key or a held one over a quiet room deserves silence;
    speech that came back as no words deserves to be told so, or the assistant
    looks broken at exactly the moment it is working as designed.
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
    transcript this application is willing to stand behind. It carries the
    decoder's doubt when there was one, for whoever reads the log later.

    A turn that *failed* carries the key of the sentence that was said instead
    of an answer - `unreachable`, `key_invalid`, `took_too_long` - and nothing
    else about the failure. The log needs the kind (a line of zero tokens
    reads as a free success, and the log of 2026-09-04 had one nobody could
    explain); the provider's own words are where a key could travel and stay
    in the exception.

    `turn_id` is the name the turn goes by in `tool_audit` (section 3.9),
    so that a row there and a line in the log can be read together. A turn
    that made no call has none: a tapped key, a stop, a turn the spending
    limit stopped.

    `cost_usd` is what the turn cost at its model's price - `None` when the
    price is not known, or the turn never reached the model - and
    `tool_calls` how many calls the gate ran. Both go to the log (2.4).

    `intent` names the short command the turn was, when the fast path
    answered it without the model (2.5) - `get_time`, `stop`, `cancel` -
    and is `None` for every turn the model heard. Such a turn's usage is a
    real zero: no request was made.

    `first_sound_ms` is how long after the recording arrived the first
    sound was made (2.8) - the number the whole of 2.8 is about, and the
    one the log keeps so that it can be watched. `None` for a turn that
    made none, and for the fast path, whose answer is a tool away.
    """

    heard: str = ""
    said: str = ""
    usage: Usage = field(default_factory=Usage)
    missed: bool = False
    confidence: float | None = None
    failure: str | None = None
    turn_id: str = ""
    cost_usd: float | None = None
    tool_calls: int = 0
    intent: str | None = None
    first_sound_ms: float | None = None


@dataclass(slots=True)
class _Saying:
    """What the turn is saying, gathered as it is said (2.8): the pieces
    in order, the failure that ended it if one did, and how long the first
    sound took. `Turn` is made of it once the saying is over."""

    started: float
    pieces: list[str] = field(default_factory=list)
    failure: str | None = None
    first_sound_ms: float | None = None

    def add(self, piece: str) -> str:
        self.pieces.append(piece)
        return piece

    def own(self, sentence: str) -> str:
        """One of the assistant's own sentences after the model's words,
        set off from them - which the model's own pieces are not, since a
        piece may end in the middle of a word."""
        if self.pieces and not self.pieces[-1].endswith((" ", "\n")):
            return f" {sentence}"
        return sentence

    def first_sound(self) -> None:
        self.first_sound_ms = (time.perf_counter() - self.started) * 1000

    @property
    def said(self) -> str:
        return "".join(self.pieces).strip()


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

    async def listen_for(self, seconds: float) -> Audio | None:
        """One sentence within `seconds`, key or no key; `None` if none came.

        The window of section 3.1 rule 2. What is said in it is an answer to
        the assistant, not a question for it, so it is not announced through
        `on_listening` and does not come back through `utterance`.
        """
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
        filler_delay: float = FILLER_DELAY_SECONDS,
        on_state: Callable[[State], None] | None = None,
        on_turn: Callable[[Turn], None] | None = None,
        tracker: UsageTracker | None = None,
        dispatch: Dispatch | None = None,
    ) -> None:
        self._capture = capture
        self._stt = stt
        self._agent = agent
        self._tts = tts
        self._speaker = speaker
        self._locale = locale
        self._thinking_timeout = thinking_timeout
        self._filler_delay = filler_delay
        self._on_state = on_state
        self._on_turn = on_turn
        self._tracker = tracker
        # The gate of section 3.9 - the same one the loop runs the model's
        # calls through - for the one call the fast path makes on its own.
        # Without it the fast path cannot tell the time, and "saat kaç" goes
        # to the model as it did before 2.5.
        self._dispatch = dispatch

        self._said = {key: locale.say(key, default) for key, default in TEXT.items()}
        self._yes = locale.yes_words or YES_WORDS
        self._no = locale.no_words or NO_WORDS
        self._fillers = locale.fillers or FILLERS
        self._fillers_said = 0
        self._state = State.IDLE
        self._voice = ""
        # Set the moment the key goes down, so that whatever is waiting for
        # the model's next word wakes up and stops waiting (`_as_they_come`).
        self._pressed = asyncio.Event()
        # How many answers are being played at once: the answer, and inside
        # it the question a tool asks (2.8). The microphone is deafened by
        # the outermost and listens again when that one is done.
        self._playing = 0

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
        started = time.perf_counter()
        self._pressed.clear()
        self._enter(State.TRANSCRIBING)
        heard = await self._heard(pcm)

        # A question the user has already withdrawn is answered by saying
        # nothing at all, including about not having understood it.
        if self._withdrawn():
            self._rest()
            return Turn()

        if not heard.text:
            return await self._missed(heard)

        # Minted here, where the turn becomes something that may act: every
        # tool call it makes, on the fast path or through the model, is
        # written down under this name.
        turn_id = uuid.uuid4().hex

        intent = match_intent(heard.text, self._locale)
        if intent is not None:
            # A short command the pack lists (section 4): answered here when
            # it can be, and the model never hears of it.
            answered = await self._fast(heard.text, intent, turn_id)
            if answered is not None:
                return answered

        if self._tracker is not None and self._tracker.stopped():
            # `hard_stop` and a limit passed (section 3.11): the model is
            # not asked, and the user hears why instead of an answer.
            return await self._stopped(heard.text)

        self._enter(State.THINKING)
        saying = _Saying(started=started)
        await self._speak_all(self._answering(heard.text, turn_id, saying), saying)

        # What the turn came to is known once its stream has ended, and not
        # at all when it was cut short: by a press, or by a failure on the
        # way. A turn cut short spent no tokens anybody can account for.
        answer = self._agent.last_answer if saying.failure is None else None
        cost = self._record(turn_id, answer, saying.failure)
        self._rest()
        return Turn(
            heard=heard.text,
            said=saying.said,
            usage=answer.usage if answer is not None else Usage(),
            failure=saying.failure,
            turn_id=turn_id,
            cost_usd=cost,
            tool_calls=answer.tool_calls if answer is not None else 0,
            first_sound_ms=saying.first_sound_ms,
        )

    async def confirm(self, question: str) -> bool:
        """Asks `question` out loud and listens for a yes (section 3.1 rule 2).

        The gate's `Confirm`. `question` already holds the real argument
        values; what is added is how to answer, since the user cannot know
        that only two words are being listened for. Everything that is not a
        clear yes is a no: silence, a no beside a yes, a press of the key
        while the question is still being read, and two answers with neither
        word in them.
        """
        if self._withdrawn():
            return False

        # Where the turn was: `THINKING`, or `SPEAKING` when the model had
        # said something before it asked (2.8).
        before = self._state
        self._enter(State.CONFIRMING)
        answer = await self._ask(f"{question} {self._said['confirm_hint']}")
        if answer is None:
            answer = await self._ask(self._said["confirm_again"])

        # Back to where the turn was: the loop that asked is still running.
        if not self._withdrawn():
            self._enter(before)
        return answer is True

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

    async def _fast(self, heard: str, intent: str, turn_id: str) -> Turn | None:
        """A short command answered without the model (section 4) - or
        `None` when it could not be, and the turn goes on to the model as
        though nothing had matched.

        `stop` and `cancel` are answered by silence: the request they would
        have cost is the whole of what there was to save. `get_time` asks
        `get_current_time` through the gate - the one way a tool runs
        (invariant 1), so the call is judged and written down like the
        model's - and says the answer in the pack's words. What the gate
        hands back is addressed to a model: the tool's own line when it
        ran, a sentence when it was refused or failed. Only the first reads
        as a time, and a turn without one is not answered here.
        """
        if intent != GET_TIME:
            self._rest()
            return Turn(heard=heard, intent=intent)

        if self._dispatch is None:
            return None
        call = ToolCall(id="", name=TIME_TOOL, arguments={})
        result = await self._dispatch(call, turn_id=turn_id, confirm=self.confirm)
        moment = _time_in(result)
        if moment is None:
            logger.warning("the fast path was told {result!r} instead of the time", result=result)
            return None

        said = self._said["time_is"].format(hour=moment.hour, minute=moment.minute)
        await self._speak(said)
        self._rest()
        return Turn(heard=heard, said=said, intent=intent, turn_id=turn_id, tool_calls=1)

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

        return hear(await self._stt.transcribe(pcm, hint=self._locale.stt_language))

    async def _answering(self, heard: str, turn_id: str, saying: _Saying) -> AsyncGenerator[str]:
        """Everything the turn says, in the order it is said: the day's
        warning first, the model's words as they come with the filler among
        them, and last either the cut-off notice or the sentence that says
        why the answer stopped (sections 3.11 and 3.2).

        The warning comes first because the answer may be long and the
        warning is the one sentence the user has to act on. The notice
        comes last because that is where the cut is. A failure's sentence
        comes after whatever was already said: a connection that dropped in
        the second request of a tool turn has left half an answer behind,
        and the user has heard it. Nothing is said after a press.

        A turn that failed spent no tokens anybody can account for: what
        the provider counted before it refused is not reported to us, and
        guessing would put a number in the cost report that nothing backs.
        """
        warning = None if self._tracker is None else self._tracker.warning()
        if warning is not None:
            yield saying.add(f"{self._said[warning]} ")

        tool_round = asyncio.Event()
        stream = self._agent.stream_reply(
            heard, turn_id=turn_id, confirm=self.confirm, on_tool_round=tool_round.set
        )
        pieces = _as_they_come(
            stream,
            tool_round=tool_round,
            pressed=self._pressed,
            filler=self._filler,
            delay=self._filler_delay,
            asking=lambda: self._state is State.CONFIRMING,
        )
        try:
            async with aclosing(pieces):
                # The minute of section 3.1 rule 6 is the time to the first
                # word. After it the model finishes at its own pace: a long
                # answer read out loud is not a turn that hung, and a key
                # press ends one that did.
                first = await asyncio.wait_for(anext(pieces, None), self._thinking_timeout)
                if first is not None:
                    yield saying.add(first)
                    async for piece in pieces:
                        yield saying.add(piece)
        except TimeoutError:
            saying.failure = "took_too_long"
        except AuthenticationError:
            # Never retried and never failed over: the key will not start
            # working on its own, and quietly using another model would put
            # the user on a bill they did not agree to (section 3.2).
            saying.failure = "key_invalid"
        except (ProviderError, OSError):
            # OSError as well as our own: a socket that was refused below the
            # adapter's transport never reaches it to be translated.
            saying.failure = "unreachable"

        if self._withdrawn():
            return
        if saying.failure is not None:
            yield saying.add(saying.own(self._said[saying.failure]))
            return
        answer = self._agent.last_answer
        if answer is not None and answer.cut_off:
            yield saying.add(saying.own(self._said["answer_cut_off"]))

    def _filler(self) -> str:
        """The next of the pack's fillers, in turn."""
        chosen = self._fillers[self._fillers_said % len(self._fillers)]
        self._fillers_said += 1
        return chosen

    def _record(self, turn_id: str, answer: Answer | None, failure: str | None) -> float | None:
        """The turn's tokens to `usage_log`, priced (section 6).

        A turn that failed or was cut short reports no tokens and is not a
        row: a row of zeros would read as a free turn, which is the mistake
        `logs.py` already refuses to make. Without a tracker - most tests -
        nothing is written and nothing costs anything.
        """
        if self._tracker is None or failure is not None or answer is None:
            return None
        return self._tracker.record(turn_id, answer.usage)

    async def _stopped(self, heard: str) -> Turn:
        """`hard_stop` and a limit passed: what the user hears instead of an answer.

        No `turn_id`: the turn never reached the model and made no call to
        be filed under one. The fast path of 2.5 comes before this, not
        through it - a limit on spending has nothing to say about a turn
        that costs nothing.
        """
        said = self._said["spend_stopped"]
        await self._speak(said)
        self._rest()
        return Turn(heard=heard, said=said, failure="spend_stopped")

    async def _ask(self, prompt: str) -> bool | None:
        """Reads `prompt`, opens the window, and reads the answer.

        `None` is "neither word was heard": something was said, or the
        recogniser could not read it, and it is worth one more try. `False`
        is every way of not saying yes that is not worth one: silence, a no,
        the key going down while the question was still being read.
        """
        await self._play(_one(prompt))
        if self._withdrawn():
            return False

        # The window has to hear. An answer under way keeps the microphone
        # deaf (`_play`); it is opened for the window and closed again after
        # it - and `unmute` is also what lets the room's echo of the
        # question pass before the window listens (`audio/capture.py`).
        if self._playing:
            self._capture.unmute()
        try:
            pcm = await self._capture.listen_for(CONFIRM_WINDOW_SECONDS)
        finally:
            if self._playing:
                self._capture.mute()
        if pcm is None or self._withdrawn():
            return False

        heard = await self._heard(pcm)
        if not heard.text:
            return None if heard.missed else False
        return read_answer(heard.text, yes=self._yes, no=self._no)

    async def _speak(self, said: str) -> None:
        """Says one sentence of the assistant's own: the time, an apology,
        a limit."""
        if not said or self._withdrawn():
            return

        self._enter(State.SPEAKING)
        await self._play(_one(said))

    async def _speak_all(self, pieces: AsyncGenerator[str], saying: _Saying) -> None:
        """Says what `pieces` yields, from the first of them on (2.8).

        `SPEAKING` from the first piece - not before, since a model that
        answers with nothing is not something to open the sound card for -
        and nothing at all once the user has moved on.
        """
        async with aclosing(pieces):
            first = await anext(pieces, None)
            if first is None or self._withdrawn():
                return

            self._enter(State.SPEAKING)
            await self._play(_chain(first, pieces), on_first_sound=saying.first_sound)

    async def _play(
        self, pieces: AsyncIterator[str], *, on_first_sound: Callable[[], None] | None = None
    ) -> None:
        """Says `pieces` through the sound card, with the microphone deaf meanwhile.

        Deaf for exactly as long as there is something for it to mishear, and
        in a `finally` because an answer that failed halfway through must not
        leave the assistant unable to hear at all. The question a tool asks
        is played from inside the answer's own stream (2.8): the microphone
        is deafened once, by the outermost of the two, and listens again
        when that one is done.
        """
        if self._playing == 0:
            self._capture.mute()
        self._playing += 1
        try:
            buffers = self._tts.stream(pieces, voice=self._voice)
            if on_first_sound is not None:
                buffers = _marking_first(buffers, on_first_sound)
            await self._speaker.play(buffers, sample_rate=self._tts.sample_rate)
        except PlaybackError as failure:
            # The answer is on the screen and in the turn; only the sound of
            # it was lost. A headset switched off between two questions is
            # not a bug, so it is a line in the log rather than the program.
            logger.warning("playback failed: {problem}", problem=failure)
        finally:
            self._playing -= 1
            if self._playing == 0:
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
        # A question being read is cut off like an answer: the user is talking
        # over it. A press inside the window itself never arrives here - the
        # capture keeps it as the answer (`Capture.listen_for`).
        was_talking = self._state in (State.SPEAKING, State.CONFIRMING)
        self._enter(State.LISTENING)
        self._pressed.set()
        if was_talking:
            self._speaker.stop()

    async def _pick_voice(self) -> str:
        preferred = self._locale.voice(self._tts.id)

        for language in (self._locale.code, None):
            # The locale's own language first. Failing that, anything installed:
            # the wrong accent is a poor answer, and no answer is worse.
            voice = choose_voice(await self._tts.list_voices(language), preferred)
            if voice:
                return voice

        raise NoVoiceError(f"no speech voice is installed, for {self._locale.code!r} or otherwise")


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


def hear(transcript: Transcript) -> Heard:
    """What to make of a transcript: nothing, unreadable speech, or words.

    The decision rests on whether there was speech, never on how sure the
    decoder was of its words. Measured on the target machine (2026-09-05): a
    correct single "Merhaba." scores 0.48 confidence and silence scores up to
    0.54, so no confidence floor can tell them apart - but the engine's own
    no-speech estimate can (0.06 against 0.86). Confidence is still carried,
    for the log and the screen, because a run of low numbers is how a bad
    microphone is diagnosed afterwards.

    An engine with no opinion is believed about its words, and its silence is
    read as speech it could not make out: it cannot tell the two apart, and
    neither can this. Treating "no opinion" as "nothing was said" would make
    the assistant mute the day it moves to a cloud recogniser.
    """
    no_speech = transcript.no_speech_probability
    if no_speech is not None and no_speech >= NO_SPEECH_CEILING:
        # The engine says nothing was said. Words it produced anyway are what
        # a recogniser makes of silence, and they are not answered.
        return Heard()

    text = transcript.text.strip()
    if text:
        return Heard(text=text, confidence=transcript.confidence)

    # There was speech, or an engine with no opinion, and no words came of it.
    return Heard(missed=True, confidence=transcript.confidence)


def _time_in(result: str) -> datetime | None:
    """The moment `get_current_time` reported, or `None` when `result` is
    not its line.

    The tool writes the time first and in ISO form, so that a model reads
    it without ambiguity (`tools/system.py`). The gate's refusals and the
    tool's own failure are sentences, and a sentence does not start with a
    time.
    """
    try:
        return datetime.fromisoformat(result.split(" ", 1)[0])
    except ValueError:
        return None


# A word, for the purpose of hearing "yes" in an answer: letters and digits in
# any script. Punctuation and the spaces between are where words end.
_WORD = re.compile(r"\w+")


def read_answer(text: str, *, yes: Iterable[str], no: Iterable[str]) -> bool | None:
    """Whether `text` says yes, says no, or says neither.

    Whole words, folded the way search is (`store/normalize.py`): "Evet." and
    "EVET" are the same word, and "evetlemedim" is not it. A `no` word
    anywhere wins over a `yes` word - "yes, but no" is a no - because the
    window only ever guards something that should not happen by mistake.
    `None` means neither was heard, and is the caller's cue to ask once more.
    """
    said = _spaced(text)
    if any(phrase in said for phrase in _phrases(no)):
        return False
    if any(phrase in said for phrase in _phrases(yes)):
        return True
    return None


def _phrases(words: Iterable[str]) -> list[str]:
    """Each entry as it would appear inside `_spaced` text; blanks dropped."""
    spaced = (_spaced(word) for word in words)
    return [phrase for phrase in spaced if phrase.strip()]


def _spaced(text: str) -> str:
    """The words of `text`, folded, one space between and one either side -
    so that a phrase of one or more words can be found only at word edges."""
    return f" {' '.join(_WORD.findall(normalize_search(text)))} "


async def _one(said: str) -> AsyncIterator[str]:
    """A sentence of the assistant's own - a question, the time, an
    apology - as a stream of one. The model's answer is no longer one of
    these (2.8): it comes through `_as_they_come`, a piece at a time."""
    yield said


async def _chain(first: str, rest: AsyncIterator[str]) -> AsyncIterator[str]:
    """`first`, then `rest`: the piece already taken to see whether there
    was one, put back in front of the others."""
    yield first
    async for piece in rest:
        yield piece


async def _marking_first(
    buffers: AsyncIterator[bytes], mark: Callable[[], None]
) -> AsyncIterator[bytes]:
    """`buffers`, with `mark` called as the first of them goes by: the
    moment the first sound is made, give or take the sound card."""
    marked = False
    async for buffer in buffers:
        if buffer and not marked:
            mark()
            marked = True
        yield buffer


async def _as_they_come(
    stream: AsyncGenerator[str],
    *,
    tool_round: asyncio.Event,
    pressed: asyncio.Event,
    filler: Callable[[], str],
    delay: float,
    asking: Callable[[], bool],
) -> AsyncGenerator[str]:
    """The model's words as they come, with the filler among them, ending
    the moment the key goes down (2.8).

    Three things can happen while the next piece is waited for. It arrives,
    and is handed on. A tool round begins - then the piece has `delay` more
    to arrive, and when it has not the filler is said in its place, once per
    turn and never while a question is being asked (`asking`), because the
    user is answering it. Or the key goes down - then the stream is left
    where it stands, the request with it (`Agent.stream_reply`), and nothing
    more is said.

    The stream is read in a task of its own, so that the wait can be for
    whichever of the three comes first; the task is always waited out before
    the stream is closed, because an async generator cannot be closed while
    another task is still inside it.
    """
    began = asyncio.ensure_future(tool_round.wait())
    withdrawn = asyncio.ensure_future(pressed.wait())
    filler_due = True
    said_any = False
    try:
        async with aclosing(stream):
            while True:
                upcoming = asyncio.create_task(_pull(stream))
                try:
                    watched: set[asyncio.Future[Any]] = {upcoming, withdrawn}
                    if filler_due:
                        watched.add(began)
                    done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
                    if upcoming not in done and withdrawn not in done:
                        # The round began. `delay` more for a word to come.
                        done, _ = await asyncio.wait({upcoming, withdrawn}, timeout=delay)
                        filler_due = False
                        if upcoming not in done and withdrawn not in done and not asking():
                            yield f" {filler()} " if said_any else f"{filler()} "
                            said_any = True
                    if withdrawn.done() and not upcoming.done():
                        await _gone(upcoming)
                        return
                    piece = await upcoming
                except BaseException:
                    await _gone(upcoming)
                    raise
                if piece is None or withdrawn.done():
                    return
                said_any = True
                yield piece
    finally:
        began.cancel()
        withdrawn.cancel()


async def _pull(stream: AsyncGenerator[str]) -> str | None:
    return await anext(stream, None)


async def _gone(task: asyncio.Future[Any]) -> None:
    """Cancels `task` and waits until it is gone, whatever it ends with."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
