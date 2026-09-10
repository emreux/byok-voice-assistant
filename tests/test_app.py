"""The state machine: the file where phase 1 becomes a product (item 1.10).

`IDLE` to `LISTENING` while the key is held, `TRANSCRIBING`, `THINKING`,
`SPEAKING`, and back. Everything it needs is injected, so the whole turn can be
driven here without a microphone, a model or a sound card.

Four of these tests are the ones worth keeping if the rest were deleted.

**Silence is not a turn, and a quiet word is.** Whisper answers a silent
recording with confident looking words, and answers a correct single word with
low ones - measured on this machine (2026-09-05), a subtitle credit over
silence scored 0.54 and "Merhaba." alone scored 0.48. No confidence floor tells
them apart; the engine's own estimate of whether there was speech does (0.86
against 0.06), and that is what decides.

**A press cuts the answer off.** Not the release: the user is speaking from the
moment they press, and an assistant still talking is both rude and something
the microphone is recording.

**A refused key is not a network problem.** The user has to renew it, so the
sentence they hear has to say so (section 3.2).

**Nothing said out loud is written in this file.** The sentences come from the
locale pack, with the English constants of `app.TEXT` as the end of the chain
(section 3.12), exactly as in the setup wizard.

**The first sentence is spoken while the model writes the second** (2.8).
The tests of the last section hold the model back until the speaker has
the first sentence, say the filler only when a tool round has gone quiet,
and end the request itself - not just the sound - the moment the key goes
down.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest
from loguru import logger

from assistant import app
from assistant.agent.core import Agent, Confirm, Dispatch
from assistant.agent.intents import CANCEL, GET_TIME, STOP, TIME_TOOL
from assistant.agent.limits import Limits
from assistant.agent.policy import DECLINED, NO_SUCH_TOOL, dispatch
from assistant.app import (
    CONFIRM_WINDOW_SECONDS,
    FILLER_DELAY_SECONDS,
    THINKING_TIMEOUT,
    Assistant,
    Heard,
    NoVoiceError,
    State,
    Turn,
    choose_voice,
    hear,
    read_answer,
)
from assistant.audio.player import PlaybackError
from assistant.llm.base import (
    AuthenticationError,
    Delta,
    Message,
    ProviderError,
    ToolCall,
    Usage,
)
from assistant.locales import Locale
from assistant.store.db import open_database
from assistant.store.repos import AuditRepo, UsageRepo
from assistant.stt.base import SAMPLE_RATE, Audio, Transcript
from assistant.tools import system
from assistant.tools.registry import ToolRegistry, tool
from assistant.tools.system import get_current_time
from assistant.tts.base import VoiceInfo
from assistant.usage.tracker import Pricing, UsageTracker
from tests.test_agent_loop import ScriptedProvider

TOLGA = VoiceInfo(id=r"HKLM\...\TR-TR_TOLGA", display_name="Microsoft Tolga", language="tr")
ZIRA = VoiceInfo(id=r"HKLM\...\EN-US_ZIRA", display_name="Microsoft Zira", language="en")

TURKISH = Locale(
    code="tr",
    name="Türkçe",
    stt_language="tr",
    voices={"fake": "Tolga"},
    ui={
        "key_invalid": "API anahtarın geçersiz görünüyor, yenilemen gerekiyor.",
        "unreachable": "Sağlayıcıya bağlanamadım, tekrar dener misin?",
        "took_too_long": "Bu iş uzadı, tekrar dener misin?",
        "not_understood": "Seni anlayamadım, tekrar söyler misin?",
        "confirm_hint": "Evet ya da hayır de.",
        "confirm_again": "Anlayamadım. Evet mi, hayır mı?",
        "answer_cut_off": "Cevabın sonu kesildi.",
        "daily_over": "Bugünkü harcama sınırını aştın.",
        "monthly_over": "Bu ayki harcama sınırını aştın.",
        "spend_stopped": "Harcama sınırı aşıldı, bu yüzden modele sormuyorum.",
        "time_is": "Saat {hour} {minute}.",
    },
    yes_words=("evet", "tamam"),
    no_words=("hayır", "iptal"),
)


def speech(seconds: float = 2.0) -> Audio:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


class StopError(Exception):
    """Ends `run()` in a test the way closing the program ends it in life."""


class FakeCapture:
    """Push to talk without a keyboard: utterances arrive in the order given."""

    def __init__(self, *utterances: Audio, answers: Sequence[Audio | None] = ()) -> None:
        self.on_listening: Callable[[], None] | None = None
        self.started = False
        # Whether the state machine has told it to stop listening, and how many
        # times it has been told either thing - a microphone left deaf and one
        # that was never deafened look the same from the flag alone.
        self.deaf = False
        self.switches: list[bool] = []
        self._waiting = list(utterances)
        # What each confirmation window hears, in order - a window with
        # nothing scripted hears silence - and, for each one opened, how long
        # it was opened for and whether the microphone was deaf at the time.
        self.windows: list[float] = []
        self.deaf_windows: list[bool] = []
        self._answers = list(answers)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def mute(self) -> None:
        self.deaf = True
        self.switches.append(True)

    def unmute(self) -> None:
        self.deaf = False
        self.switches.append(False)

    async def utterance(self) -> Audio:
        if not self._waiting:
            raise StopError
        return self._waiting.pop(0)

    async def listen_for(self, seconds: float) -> Audio | None:
        self.windows.append(seconds)
        self.deaf_windows.append(self.deaf)
        return self._answers.pop(0) if self._answers else None

    def press(self) -> None:
        """What the hotkey does the moment it goes down."""
        if self.on_listening is not None:
            self.on_listening()


class FakeSTT:
    """A recogniser that hears whatever the test says it hears."""

    id = "fake_stt"
    supports_streaming = False

    def __init__(self, *heard: Transcript, before: Callable[[], None] | None = None) -> None:
        self.heard = list(heard) or [Transcript(text="saat kaç", confidence=0.9)]
        self.hints: list[str | None] = []
        self._before = before

    async def transcribe(self, pcm: Audio, *, hint: str | None = None) -> Transcript:
        self.hints.append(hint)
        if self._before is not None:
            # Somebody pressing the key while the machine is still listening to
            # the last thing they said.
            self._before()
        # The last one is kept: a recogniser does not run out of hearing, and a
        # test that scripts one utterance still gets a second turn transcribed.
        return self.heard.pop(0) if len(self.heard) > 1 else self.heard[0]

    def transcribe_stream(
        self, pcm_chunks: AsyncIterator[Audio], *, hint: str | None = None
    ) -> AsyncIterator[Transcript]:
        raise NotImplementedError


class FakeTTS:
    """An engine that returns the text it was given instead of speech.

    It speaks at Azure's rate rather than Windows', deliberately: a rate that
    happened to match the one the player would default to would let a hardcoded
    16 kHz pass the test written to catch it.
    """

    id = "fake"
    sample_rate = 24_000

    def __init__(self, *, installed: list[VoiceInfo] | None = None) -> None:
        self.installed = [TOLGA, ZIRA] if installed is None else installed
        self.said: list[str] = []
        self.voices_used: list[str] = []
        self.asked_for: list[str | None] = []

    async def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        self.asked_for.append(language)
        if language is None:
            return list(self.installed)
        return [voice for voice in self.installed if voice.language == language]

    async def stream(self, chunks: AsyncIterator[str], *, voice: str) -> AsyncIterator[bytes]:
        self.voices_used.append(voice)
        async for chunk in chunks:
            self.said.append(chunk)
            yield chunk.encode("utf-8")


class FakeSpeaker:
    """A sound card that remembers what it was asked to play."""

    def __init__(self, *, on_play: Callable[[], None] | None = None) -> None:
        self.played: list[bytes] = []
        self.rates: list[int] = []
        self.stopped = False
        self.on_play = on_play

    async def play(self, buffers: AsyncIterator[bytes], *, sample_rate: int) -> None:
        self.rates.append(sample_rate)
        if self.on_play is not None:
            self.on_play()
        async for buffer in buffers:
            self.played.append(buffer)

    def stop(self) -> None:
        self.stopped = True

    @property
    def heard(self) -> str:
        return b"".join(self.played).decode("utf-8")


def assistant_with(
    *,
    capture: FakeCapture | None = None,
    stt: FakeSTT | None = None,
    provider: ScriptedProvider | None = None,
    tts: FakeTTS | None = None,
    speaker: FakeSpeaker | None = None,
    locale: Locale = TURKISH,
    thinking_timeout: float = THINKING_TIMEOUT,
    filler_delay: float = FILLER_DELAY_SECONDS,
    on_state: Callable[[State], None] | None = None,
    on_turn: Callable[[Turn], None] | None = None,
    agent: Agent | None = None,
    tracker: UsageTracker | None = None,
    dispatch: Dispatch | None = None,
) -> Assistant:
    return Assistant(
        capture=capture if capture is not None else FakeCapture(),
        stt=stt if stt is not None else FakeSTT(),
        agent=agent
        if agent is not None
        else Agent(
            provider if provider is not None else ScriptedProvider([Delta(text="Üç.")]),
            model="fake-1",
        ),
        tts=tts if tts is not None else FakeTTS(),
        speaker=speaker if speaker is not None else FakeSpeaker(),
        locale=locale,
        thinking_timeout=thinking_timeout,
        filler_delay=filler_delay,
        on_state=on_state,
        on_turn=on_turn,
        tracker=tracker,
        dispatch=dispatch,
    )


async def one_turn(assistant: Assistant, pcm: Audio | None = None) -> Turn:
    """Starts the machine and runs a single turn through it."""
    await assistant.begin()
    return await assistant.turn(speech() if pcm is None else pcm)


# --------------------------------------------------------------------------
# The states
# --------------------------------------------------------------------------


async def test_a_turn_walks_through_the_states_the_design_names() -> None:
    seen: list[State] = []

    await one_turn(assistant_with(on_state=seen.append))

    assert seen == [
        # `begin` says so, and it has to: the state has been `IDLE` since the
        # constructor, but a status line that was never told goes on showing
        # "loading the speech model" until the first turn is over.
        State.IDLE,
        State.TRANSCRIBING,
        State.THINKING,
        State.SPEAKING,
        State.IDLE,
    ]


async def test_the_key_going_down_is_what_starts_listening() -> None:
    capture = FakeCapture()
    seen: list[State] = []
    assistant = assistant_with(capture=capture, on_state=seen.append)

    await assistant.begin()
    capture.press()

    assert seen == [State.IDLE, State.LISTENING]
    assert assistant.state is State.LISTENING


async def test_the_microphone_opens_when_the_assistant_does() -> None:
    """It stays open for the whole session: opening a stream takes long enough
    to swallow the first syllable (item 1.5)."""
    capture = FakeCapture()

    await assistant_with(capture=capture).begin()

    assert capture.started


async def test_the_microphone_is_closed_when_the_program_ends() -> None:
    capture = FakeCapture(speech())

    with pytest.raises(StopError):
        await assistant_with(capture=capture).run()

    assert not capture.started


async def test_one_utterance_after_another_is_one_turn_after_another() -> None:
    capture = FakeCapture(speech(), speech())
    provider = ScriptedProvider([Delta(text="Bir.")], [Delta(text="İki.")])
    speaker = FakeSpeaker()

    with pytest.raises(StopError):
        await assistant_with(capture=capture, provider=provider, speaker=speaker).run()

    assert speaker.heard == "Bir.İki."


# --------------------------------------------------------------------------
# What counts as having been said
# --------------------------------------------------------------------------


async def test_what_was_heard_is_what_the_model_is_asked() -> None:
    provider = ScriptedProvider([Delta(text="Üç.")])
    stt = FakeSTT(Transcript(text="saat kaç", confidence=0.9))

    await one_turn(assistant_with(stt=stt, provider=provider))

    assert provider.calls[-1].turns[-1].content == "saat kaç"


async def test_the_recogniser_is_told_which_language_to_expect() -> None:
    """Section 3.12: Whisper left to guess writes Turkish phonetic nonsense for
    a language it was not expecting, and the model never sees the question."""
    stt = FakeSTT()

    await one_turn(assistant_with(stt=stt))

    assert stt.hints == ["tr"]


async def test_a_stray_press_is_not_a_turn() -> None:
    """Tapping the hotkey by accident costs a Whisper run, an API call and an
    answer to nothing."""
    stt, speaker = FakeSTT(), FakeSpeaker()
    provider = ScriptedProvider([Delta(text="Efendim?")])

    await one_turn(assistant_with(stt=stt, provider=provider, speaker=speaker), speech(0.1))

    assert (stt.hints, provider.calls, speaker.played) == ([], [], [])


async def test_silence_the_engine_recognised_as_silence_is_not_a_turn() -> None:
    """A held key over a quiet room. The engine says so itself, and nothing
    was said that could have been misheard - so nothing is said back, and
    nothing is asked of the model."""
    speaker = FakeSpeaker()
    provider = ScriptedProvider([Delta(text="Buyurun?")])
    stt = FakeSTT(Transcript(text="", confidence=None, no_speech_probability=1.0))

    turn = await one_turn(assistant_with(stt=stt, provider=provider, speaker=speaker))

    assert provider.calls == []
    assert (turn.heard, turn.missed, speaker.heard) == ("", False, "")


async def test_words_the_engine_does_not_stand_behind_are_not_a_turn_either() -> None:
    """An engine that filters nothing and reports 0.9 'no speech' next to a
    subtitle credit: the number wins over the words. The words themselves are
    not something code may look at - a list of them would be a language
    constant (section 3.12)."""
    speaker = FakeSpeaker()
    provider = ScriptedProvider([Delta(text="Buyurun?")])
    stt = FakeSTT(Transcript(text="Altyazı M.K.", confidence=0.52, no_speech_probability=0.92))

    turn = await one_turn(assistant_with(stt=stt, provider=provider, speaker=speaker))

    assert provider.calls == [], "words nobody said were sent to the model"
    assert (turn.heard, turn.missed, speaker.heard) == ("", False, "")


async def test_a_single_quiet_word_is_answered() -> None:
    """Measured: "Merhaba." alone scores 0.48 confidence on clean audio. The
    old floor of 0.6 dropped it, and "merhaba - nothing happened" was the
    owner's first complaint. The number that matters is 0.06 no-speech."""
    provider = ScriptedProvider([Delta(text="Merhaba.")])
    stt = FakeSTT(Transcript(text="Merhaba.", confidence=0.48, no_speech_probability=0.06))

    turn = await one_turn(assistant_with(stt=stt, provider=provider))

    assert len(provider.calls) == 1
    assert turn.heard == "Merhaba."


async def test_speech_the_engine_could_not_read_is_said_out_loud() -> None:
    """There was speech - the engine says 0.0 no-speech - and no words came of
    it. That is the one case worth an apology: silence here is what a broken
    program sounds like. What must not happen is the model being asked."""
    speaker = FakeSpeaker()
    provider = ScriptedProvider([Delta(text="Buyurun?")])
    stt = FakeSTT(Transcript(text="", confidence=None, no_speech_probability=0.0))

    turn = await one_turn(assistant_with(stt=stt, provider=provider, speaker=speaker))

    assert provider.calls == []
    assert speaker.heard == TURKISH.ui["not_understood"]
    assert turn.missed is True


def test_hear_has_exactly_three_outcomes() -> None:
    assert hear(Transcript(text="x", no_speech_probability=0.9)) == Heard()
    assert hear(Transcript(text="", no_speech_probability=0.1)) == Heard(missed=True)
    assert hear(Transcript(text=" Merhaba ", confidence=0.4, no_speech_probability=0.1)) == Heard(
        text="Merhaba", confidence=0.4
    )
    # An engine with no opinion is believed about its words, and its silence
    # is unreadable speech: it cannot tell the two apart, and neither can we.
    assert hear(Transcript(text="saat kaç")) == Heard(text="saat kaç")
    assert hear(Transcript(text="")) == Heard(missed=True)


async def test_the_assistant_says_it_is_ready_before_anybody_presses_anything() -> None:
    """The state has been `IDLE` since the constructor, and for a while nobody
    was ever told. The status line went on showing "loading the speech model"
    until the first turn was over, which reads as a program that never finished
    starting - so nobody pressed the key that would have cleared it."""
    seen: list[State] = []

    await assistant_with(on_state=seen.append).begin()

    assert seen == [State.IDLE]


async def test_a_garbled_sentence_still_goes_to_the_model() -> None:
    """Measured 2026-08-31 at 1 m in a quiet voice: "bu cümlemik takılın" at
    0.55, and the old floor dropped it. Section 3.4 says the opposite: hand the
    raw transcript over, the model is told to expect misheard words and reads
    through them. The decoder's doubt is kept on the turn for the log."""
    provider = ScriptedProvider([Delta(text="Anlayamadım, tekrar eder misin?")])
    garbled = Transcript(text="bu cümlemik takılın", confidence=0.55, no_speech_probability=0.1)
    stt = FakeSTT(garbled)

    turn = await one_turn(assistant_with(stt=stt, provider=provider))

    assert len(provider.calls) == 1
    assert (turn.heard, turn.missed) == ("bu cümlemik takılın", False)


async def test_a_key_touched_by_accident_is_not_apologised_for() -> None:
    """Nothing was said, so there is nothing to have misheard. An assistant
    that announced every brushed key would be unusable."""
    speaker = FakeSpeaker()

    turn = await one_turn(assistant_with(speaker=speaker), speech(0.1))

    assert speaker.played == []
    assert turn.missed is False


async def test_a_question_the_user_withdrew_is_not_apologised_for_either() -> None:
    """They are already saying the next thing. Talking over it to say the last
    one was not understood is worse than saying nothing."""
    capture = FakeCapture()
    speaker = FakeSpeaker()
    stt = FakeSTT(Transcript(text="", no_speech_probability=0.0), before=capture.press)

    turn = await one_turn(assistant_with(capture=capture, stt=stt, speaker=speaker))

    assert speaker.played == []
    assert turn.missed is False


async def test_the_microphone_is_deaf_while_the_apology_is_spoken() -> None:
    """It is an answer like any other, and a live microphone would hear it."""
    capture = FakeCapture(speech())
    stt = FakeSTT(Transcript(text="", no_speech_probability=0.0))
    during: list[bool] = []
    speaker = FakeSpeaker(on_play=lambda: during.append(capture.deaf))

    await one_turn(assistant_with(capture=capture, stt=stt, speaker=speaker))

    assert during == [True]


async def test_a_transcript_the_recogniser_is_sure_of_is_a_turn() -> None:
    provider = ScriptedProvider([Delta(text="Üç.")])
    stt = FakeSTT(Transcript(text="saat kaç", confidence=0.752))

    await one_turn(assistant_with(stt=stt, provider=provider))

    assert len(provider.calls) == 1


async def test_an_engine_that_reports_no_confidence_at_all_is_believed() -> None:
    """Not every engine has an opinion. Treating "no opinion" as "not sure"
    would make the assistant deaf on the day it switches to a cloud one."""
    provider = ScriptedProvider([Delta(text="Üç.")])
    stt = FakeSTT(Transcript(text="saat kaç", confidence=None))

    await one_turn(assistant_with(stt=stt, provider=provider))

    assert len(provider.calls) == 1


async def test_a_transcript_of_no_words_is_no_turn() -> None:
    provider = ScriptedProvider([Delta(text="Efendim?")])
    stt = FakeSTT(Transcript(text="   ", confidence=0.99))

    await one_turn(assistant_with(stt=stt, provider=provider))

    assert provider.calls == []


# --------------------------------------------------------------------------
# Saying it out loud
# --------------------------------------------------------------------------


async def test_the_answer_is_spoken_in_the_voice_the_locale_asks_for() -> None:
    tts = FakeTTS()

    await one_turn(assistant_with(tts=tts))

    assert tts.voices_used == [TOLGA.id]


async def test_the_answer_reaches_the_speaker_at_the_rate_the_engine_declares() -> None:
    speaker = FakeSpeaker()
    provider = ScriptedProvider([Delta(text="Saat üç.")])

    await one_turn(assistant_with(provider=provider, speaker=speaker))

    assert speaker.heard == "Saat üç."
    assert speaker.rates == [FakeTTS.sample_rate]


async def test_the_microphone_is_deaf_for_exactly_as_long_as_the_answer_lasts() -> None:
    """A microphone that is live on its own hears the answer come out of the
    speakers, takes it for a question and answers it - once per API call, for
    as long as nobody stops it."""
    capture = FakeCapture(speech())
    during: list[bool] = []
    speaker = FakeSpeaker(on_play=lambda: during.append(capture.deaf))

    await one_turn(assistant_with(capture=capture, speaker=speaker))

    assert during == [True], "the microphone was still listening while it spoke"
    assert capture.deaf is False
    assert capture.switches == [True, False]


async def test_an_answer_that_failed_halfway_still_leaves_the_microphone_listening() -> None:
    """Deafness is the state that has to be undone. A sound card that threw
    would otherwise leave an assistant that can speak and never hear again."""
    capture = FakeCapture(speech())
    speaker = FakeSpeaker(on_play=_fails)

    with pytest.raises(OSError, match="sound card"):
        await one_turn(assistant_with(capture=capture, speaker=speaker))

    assert capture.deaf is False


class DeafSpeaker(FakeSpeaker):
    """A sound card that is not there any more: a headset switched off."""

    async def play(self, buffers: AsyncIterator[bytes], *, sample_rate: int) -> None:
        raise PlaybackError("the sound device could not be opened: Device unavailable")


async def test_an_answer_the_sound_card_refused_still_ends_the_turn() -> None:
    """A Bluetooth headset switched off between two questions. The words are
    on the screen and in the turn; only the sound of them was lost, and that
    is not worth the program."""
    capture = FakeCapture(speech())
    states: list[State] = []
    assistant = assistant_with(capture=capture, speaker=DeafSpeaker(), on_state=states.append)

    turn = await one_turn(assistant)

    assert turn.said == "Üç."
    assert states[-1] is State.IDLE
    assert capture.deaf is False


async def test_an_answer_the_sound_card_refused_is_written_down() -> None:
    """Not said out loud - there is nothing to say it with - but a user who
    reports that the assistant "went quiet" deserves a log that says why."""
    lines: list[str] = []
    handle = logger.add(lines.append, format="{message}")
    try:
        await one_turn(assistant_with(speaker=DeafSpeaker()))
    finally:
        logger.remove(handle)

    assert any("playback failed" in line and "unavailable" in line for line in lines)


async def test_a_turn_with_nothing_to_say_never_deafens_the_microphone() -> None:
    """Nothing is played, so there is nothing to mishear - and a mode switched
    off and on again for no reason is a gap in what the user can be heard in."""
    capture = FakeCapture(speech())
    provider = ScriptedProvider([Delta(finish_reason="SAFETY")])

    await one_turn(assistant_with(capture=capture, provider=provider))

    assert capture.switches == []


def _fails() -> None:
    raise OSError("the sound card went away")


async def test_a_locale_with_no_voice_of_its_own_still_gets_a_voice() -> None:
    """A user whose language Windows has no voice for is better served by the
    wrong accent than by an assistant that cannot answer at all."""
    tts = FakeTTS(installed=[ZIRA])

    await one_turn(assistant_with(tts=tts))

    assert tts.voices_used == [ZIRA.id]


async def test_a_machine_with_no_voice_at_all_says_so_before_it_listens() -> None:
    """Silently starting an assistant that can never answer is the worst of
    the options; this is caught at startup, not on the first question - and
    by name, so that `assistant run` can say it in a sentence."""
    with pytest.raises(NoVoiceError, match="voice"):
        await assistant_with(tts=FakeTTS(installed=[])).begin()


def test_the_named_voice_is_preferred_over_the_others_that_fit() -> None:
    """The pack names a preference, not an identifier: what is installed is
    `Microsoft Tolga`, what `tr.toml` says is `Tolga` (item 1.8)."""
    assert choose_voice([ZIRA, TOLGA], "Tolga") == TOLGA.id


def test_a_preference_nothing_matches_still_leaves_a_voice() -> None:
    assert choose_voice([ZIRA], "Tolga") == ZIRA.id


def test_a_locale_that_names_no_voice_takes_what_there_is() -> None:
    assert choose_voice([ZIRA], None) == ZIRA.id


def test_nothing_installed_is_no_voice() -> None:
    assert choose_voice([], "Tolga") == ""


async def test_an_answer_of_nothing_is_not_spoken() -> None:
    """A model that returned no text has said nothing, and the sound card
    should not be opened to prove it."""
    speaker = FakeSpeaker()
    seen: list[State] = []

    await one_turn(
        assistant_with(
            provider=ScriptedProvider([Delta(finish_reason="SAFETY")]),
            speaker=speaker,
            on_state=seen.append,
        )
    )

    assert speaker.rates == []
    assert State.SPEAKING not in seen


# --------------------------------------------------------------------------
# Being interrupted
# --------------------------------------------------------------------------


async def test_pressing_the_key_while_the_assistant_talks_stops_it() -> None:
    capture = FakeCapture()
    speaker = FakeSpeaker()
    speaker.on_play = capture.press  # the user cuts in mid answer

    await one_turn(assistant_with(capture=capture, speaker=speaker))

    assert speaker.stopped


async def test_the_machine_is_left_listening_by_the_press_that_cut_it_off() -> None:
    """The key is still down and the microphone is recording. A turn ending
    tidily into `IDLE` would say the opposite of what is happening."""
    capture = FakeCapture()
    speaker = FakeSpeaker()
    speaker.on_play = capture.press
    assistant = assistant_with(capture=capture, speaker=speaker)

    await one_turn(assistant)

    assert assistant.state is State.LISTENING


async def test_a_press_when_nothing_is_being_said_stops_nothing() -> None:
    """The speaker is stopped, not silenced: a stop left behind would cut off
    the next answer before it began."""
    capture = FakeCapture()
    speaker = FakeSpeaker()
    assistant = assistant_with(capture=capture, speaker=speaker)

    await assistant.begin()
    capture.press()

    assert not speaker.stopped


async def test_a_turn_the_user_interrupted_never_reaches_the_model() -> None:
    """Whisper takes seconds. A question abandoned while it runs is not worth
    paying an API call for."""
    capture = FakeCapture()
    provider = ScriptedProvider([Delta(text="Üç.")])
    stt = FakeSTT(Transcript(text="saat kaç", confidence=0.9), before=capture.press)

    await one_turn(assistant_with(capture=capture, stt=stt, provider=provider))

    assert provider.calls == []


async def test_an_answer_nobody_is_waiting_for_any_more_is_not_spoken() -> None:
    """The key went down while the model was still writing. By the time the
    answer arrives the user is halfway through a different question."""
    capture = FakeCapture()
    speaker = FakeSpeaker()
    provider = ScriptedProvider([capture.press, Delta(text="Saat üç.")])

    await one_turn(assistant_with(capture=capture, provider=provider, speaker=speaker))

    assert speaker.played == []


# --------------------------------------------------------------------------
# When the turn goes wrong
# --------------------------------------------------------------------------


async def test_a_refused_key_is_said_out_loud_in_words_that_help() -> None:
    """Section 3.2: the key is not retried and not failed over. The user has to
    renew it, so they have to be told that is what happened."""
    speaker = FakeSpeaker()
    provider = ScriptedProvider([AuthenticationError("gemini refused the request (403)")])

    turn = await one_turn(assistant_with(provider=provider, speaker=speaker))

    assert speaker.heard == TURKISH.ui["key_invalid"]
    assert turn.failure == "key_invalid"


async def test_a_provider_that_cannot_be_reached_is_said_out_loud() -> None:
    speaker = FakeSpeaker()
    provider = ScriptedProvider([ProviderError("gemini refused the request (503)")])

    turn = await one_turn(assistant_with(provider=provider, speaker=speaker))

    assert speaker.heard == TURKISH.ui["unreachable"]
    assert turn.failure == "unreachable"


async def test_a_network_that_is_not_there_is_said_out_loud_too() -> None:
    """The adapter never sees a socket that was refused - the SDK raises it
    straight through - so the state machine has to know that shape as well."""
    speaker = FakeSpeaker()
    provider = ScriptedProvider([ConnectionError("no route to host")])

    await one_turn(assistant_with(provider=provider, speaker=speaker))

    assert speaker.heard == TURKISH.ui["unreachable"]


async def test_a_turn_that_takes_too_long_is_given_up_on() -> None:
    provider = ScriptedProvider([3600.0, Delta(text="eventually")])
    speaker = FakeSpeaker()

    turn = await one_turn(assistant_with(provider=provider, speaker=speaker, thinking_timeout=0.01))

    assert speaker.heard == TURKISH.ui["took_too_long"]
    assert turn.failure == "took_too_long"


def test_thinking_gives_up_after_the_minute_section_3_1_allows() -> None:
    assert THINKING_TIMEOUT == 60


async def test_the_sentences_it_says_are_the_ones_in_the_locale_pack() -> None:
    """Nothing user-facing is written in `app.py`; the pack answers first and
    the English constant is the end of the chain (section 3.12)."""
    speaker = FakeSpeaker()
    provider = ScriptedProvider([ProviderError("503")])
    english = Locale(code="en", name="English", stt_language="en", voices={}, ui={})

    await one_turn(assistant_with(provider=provider, speaker=speaker, locale=english))

    assert speaker.heard == app.TEXT["unreachable"]


async def test_a_bug_in_our_own_code_is_not_reported_as_a_network_problem() -> None:
    """Three sentences cover what the user can act on. Everything else is ours
    to fix, and swallowing it into "I could not connect" is how it never gets
    fixed."""
    provider = ScriptedProvider([TypeError("this is a bug")])

    with pytest.raises(TypeError):
        await one_turn(assistant_with(provider=provider))


# --------------------------------------------------------------------------
# What the turn came to
# --------------------------------------------------------------------------


async def test_a_turn_reports_what_was_heard_and_what_was_said() -> None:
    """The status line of item 1.11 shows both, and neither is anywhere else:
    the transcript is gone once the model has it, the answer once it is spoken."""
    stt = FakeSTT(Transcript(text="saat kaç", confidence=0.9))
    provider = ScriptedProvider([Delta(text="Saat üç.")])

    turn = await one_turn(assistant_with(stt=stt, provider=provider))

    assert (turn.heard, turn.said) == ("saat kaç", "Saat üç.")
    # A turn the model answered failed in no way, and says so by saying nothing.
    assert turn.failure is None


async def test_the_tokens_a_turn_spent_leave_the_turn() -> None:
    """Item 1.11 writes them to the log every turn, and the cost report of
    section 6 is built out of that log. A turn that swallows them cannot."""
    spent = Usage(input_tokens=302, output_tokens=8)
    provider = ScriptedProvider([Delta(text="Ankara."), Delta(usage=spent)])

    turn = await one_turn(assistant_with(provider=provider))

    assert turn.usage == spent


# --------------------------------------------------------------------------
# The name a turn goes by (2.1d)
# --------------------------------------------------------------------------


@tool(risk="safe")
async def clock() -> str:
    """Tells the time."""
    return "15:04"


class FakeGate:
    """Lets every call through and keeps the turn id each came with."""

    def __init__(self) -> None:
        self.turn_ids: list[str] = []

    async def __call__(self, call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        self.turn_ids.append(turn_id)
        return "15:04"


async def test_a_turn_that_reached_the_model_has_a_name_of_its_own() -> None:
    """The name `tool_audit` files the turn's calls under (section 3.9)."""
    turn = await one_turn(assistant_with())

    assert len(turn.turn_id) == 32
    assert set(turn.turn_id) <= set("0123456789abcdef")


async def test_two_turns_have_two_names() -> None:
    assistant = assistant_with(
        provider=ScriptedProvider([Delta(text="Bir.")], [Delta(text="İki.")])
    )
    await assistant.begin()

    first = await assistant.turn(speech())
    second = await assistant.turn(speech())

    assert first.turn_id != second.turn_id


async def test_every_call_a_turn_makes_is_filed_under_the_turn_s_name() -> None:
    """The id minted here is the one the gate is handed, so a row in the
    audit table and the turn's line in the log can be read together."""
    gate = FakeGate()
    provider = ScriptedProvider(
        [Delta(tool_call=ToolCall(id="c1", name="clock", arguments={}))], [Delta(text="Üç.")]
    )
    agent = Agent(provider, model="fake-1", tools=ToolRegistry([clock]), dispatch=gate)

    turn = await one_turn(assistant_with(agent=agent))

    assert turn.turn_id
    assert gate.turn_ids == [turn.turn_id]


async def test_a_turn_that_failed_keeps_its_name() -> None:
    """The calls it made before failing are in the audit table under it."""
    turn = await one_turn(assistant_with(provider=ScriptedProvider([ProviderError("503")])))

    assert turn.failure == "unreachable"
    assert turn.turn_id


async def test_a_turn_that_never_reached_the_model_has_no_name() -> None:
    """A tapped key made no call and has nothing to be filed under."""
    turn = await one_turn(assistant_with(), speech(0.1))

    assert turn.turn_id == ""


async def test_speech_that_could_not_be_read_is_a_turn_that_came_to_nothing() -> None:
    """It cost no tokens and left no transcript, and it is still marked as
    missed so that the log can count them (`logs.py`)."""
    stt = FakeSTT(Transcript(text="", confidence=None, no_speech_probability=0.0))

    turn = await one_turn(assistant_with(stt=stt))

    assert turn == Turn(
        heard="",
        said=TURKISH.ui["not_understood"],
        usage=Usage(),
        missed=True,
        confidence=None,
    )


async def test_what_the_assistant_said_about_a_failure_is_part_of_the_turn() -> None:
    """It is what the user heard, so it is what the log has to show."""
    provider = ScriptedProvider([ProviderError("503")])

    turn = await one_turn(assistant_with(provider=provider))

    assert turn.said == TURKISH.ui["unreachable"]


async def test_every_turn_is_handed_to_whoever_is_watching() -> None:
    capture = FakeCapture(speech(), speech())
    provider = ScriptedProvider([Delta(text="Bir.")], [Delta(text="İki.")])
    seen: list[Turn] = []

    with pytest.raises(StopError):
        await assistant_with(capture=capture, provider=provider, on_turn=seen.append).run()

    assert [turn.said for turn in seen] == ["Bir.", "İki."]


# --------------------------------------------------------------------------
# The confirmation window (2.3)
# --------------------------------------------------------------------------


@tool(risk="confirm", confirm_prompt="{name} will be opened.")
async def open_app(name: str) -> str:
    """Opens an application, once the user has said yes."""
    ran.append(f"open_app:{name}")
    return f"{name} opened"


# What ran, in order. The body of the one risky tool above writes here, so
# "did not run" is something a test sees rather than assumes.
ran: list[str] = []


@pytest.fixture(autouse=True)
def _nothing_ran_before() -> Iterator[None]:
    ran.clear()
    yield
    ran.clear()


async def through_the_gate(call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
    """The real gate of section 3.9 over this file's one risky tool."""
    return await dispatch(call, turn_id=turn_id, registry=ToolRegistry([open_app]), confirm=confirm)


def wants_spotify() -> ScriptedProvider:
    """A model that asks to open Spotify, then says whatever it has to say."""
    return ScriptedProvider(
        [Delta(tool_call=ToolCall(id="c1", name="open_app", arguments={"name": "Spotify"}))],
        [Delta(text="Tamam.")],
    )


def asking(provider: ScriptedProvider | None = None, **parts: Any) -> Assistant:
    """An assistant whose gate is the real one and whose model wants Spotify."""
    agent = Agent(
        provider if provider is not None else wants_spotify(),
        model="fake-1",
        tools=ToolRegistry([open_app]),
        dispatch=through_the_gate,
    )
    return assistant_with(agent=agent, **parts)


def says(*answers: str) -> FakeSTT:
    """A recogniser that hears the request first, then each answer in turn."""
    return FakeSTT(
        Transcript(text="Spotify'ı aç", confidence=0.9),
        *(Transcript(text=answer, confidence=0.9) for answer in answers),
    )


async def test_a_tool_that_asks_runs_when_the_user_says_yes() -> None:
    capture = FakeCapture(answers=[speech()])

    await one_turn(asking(capture=capture, stt=says("Evet.")))

    assert ran == ["open_app:Spotify"]
    assert capture.windows == [CONFIRM_WINDOW_SECONDS]


async def test_the_question_says_how_to_answer() -> None:
    """The user cannot know only two words are being listened for, so the
    gate's sentence - real argument values and all - is followed by the
    pack's hint on how to answer it."""
    tts = FakeTTS()

    await one_turn(asking(capture=FakeCapture(answers=[speech()]), stt=says("evet"), tts=tts))

    assert tts.said[0] == "Spotify will be opened. Evet ya da hayır de."


async def test_a_tool_that_asks_does_not_run_when_the_user_says_no() -> None:
    await one_turn(asking(capture=FakeCapture(answers=[speech()]), stt=says("Hayır.")))

    assert ran == []


async def test_the_model_is_told_the_user_declined() -> None:
    """In the tool's own channel, so the model can say so rather than pretend."""
    provider = wants_spotify()

    await one_turn(asking(provider, capture=FakeCapture(answers=[speech()]), stt=says("hayır")))

    assert provider.calls[-1].turns[-1].content == DECLINED


async def test_silence_in_the_window_is_a_no() -> None:
    """Six seconds of nothing is the safe side (section 3.1 rule 2), and
    not worth asking again: nobody was there to hear the second question."""
    capture = FakeCapture()
    tts = FakeTTS()

    await one_turn(asking(capture=capture, stt=says(), tts=tts))

    assert ran == []
    assert capture.windows == [CONFIRM_WINDOW_SECONDS]
    assert TURKISH.ui["confirm_again"] not in tts.said


async def test_an_answer_with_neither_word_is_asked_about_once_more() -> None:
    capture = FakeCapture(answers=[speech(), speech()])
    tts = FakeTTS()

    await one_turn(asking(capture=capture, stt=says("belki", "evet"), tts=tts))

    assert ran == ["open_app:Spotify"]
    assert len(capture.windows) == 2
    assert tts.said[1] == TURKISH.ui["confirm_again"]


async def test_two_answers_with_neither_word_are_a_no() -> None:
    capture = FakeCapture(answers=[speech(), speech()])

    await one_turn(asking(capture=capture, stt=says("belki", "olabilir")))

    assert ran == []
    assert len(capture.windows) == 2


async def test_speech_that_could_not_be_read_is_asked_about_once_more() -> None:
    """The engine says there was speech and gives no words for it: worth a
    second question, unlike silence."""
    capture = FakeCapture(answers=[speech(), speech()])
    stt = FakeSTT(
        Transcript(text="Spotify'ı aç", confidence=0.9),
        Transcript(text="", no_speech_probability=0.0),
        Transcript(text="evet"),
    )

    await one_turn(asking(capture=capture, stt=stt))

    assert ran == ["open_app:Spotify"]
    assert len(capture.windows) == 2


async def test_a_recording_the_engine_calls_silence_is_silence() -> None:
    """The detector fired on a cough; the engine says nothing was said. Words
    it produced anyway are not an answer, and silence is not a reason to ask
    again."""
    capture = FakeCapture(answers=[speech(), speech()])
    stt = FakeSTT(
        Transcript(text="Spotify'ı aç", confidence=0.9),
        Transcript(text="evet", no_speech_probability=1.0),
    )

    await one_turn(asking(capture=capture, stt=stt))

    assert ran == []
    assert len(capture.windows) == 1


async def test_a_no_beside_a_yes_is_a_no() -> None:
    await one_turn(asking(capture=FakeCapture(answers=[speech()]), stt=says("evet, yok hayır")))

    assert ran == []


async def test_the_turn_passes_through_confirming_and_back_into_thinking() -> None:
    """The loop that asked is still running; the window is a detour from
    `THINKING`, not a state the turn ends in."""
    seen: list[State] = []

    await one_turn(
        asking(capture=FakeCapture(answers=[speech()]), stt=says("evet"), on_state=seen.append)
    )

    assert seen == [
        State.IDLE,
        State.TRANSCRIBING,
        State.THINKING,
        State.CONFIRMING,
        State.THINKING,
        State.SPEAKING,
        State.IDLE,
    ]


async def test_the_microphone_is_deaf_while_the_question_is_read_and_not_after() -> None:
    """A question is something the microphone could mishear like any answer,
    and the window right after it is the one moment it has to hear."""
    capture = FakeCapture(answers=[speech()])
    during: list[bool] = []
    speaker = FakeSpeaker(on_play=lambda: during.append(capture.deaf))

    await one_turn(asking(capture=capture, stt=says("evet"), speaker=speaker))

    assert during == [True, True], "the question, then the answer"
    assert capture.deaf_windows == [False]


async def test_a_press_while_the_question_is_read_is_a_no() -> None:
    """The user is talking over the question: a new question, and the window
    is never opened. Nothing runs, and what the model says about being
    refused is not said over them either."""
    capture = FakeCapture()
    speaker = FakeSpeaker()
    speaker.on_play = capture.press
    assistant = asking(capture=capture, stt=says("evet"), speaker=speaker)

    await one_turn(assistant)

    assert ran == []
    assert capture.windows == []
    assert speaker.stopped
    assert assistant.state is State.LISTENING


async def test_what_was_said_in_the_window_never_reaches_the_model() -> None:
    """The exchange belongs to the gate. The model asked a question of the
    user through the gate and gets the gate's answer, not the words."""
    provider = wants_spotify()

    turn = await one_turn(
        asking(provider, capture=FakeCapture(answers=[speech()]), stt=says("evet"))
    )

    assert turn.heard == "Spotify'ı aç"
    assert turn.said == "Tamam."
    assert not any(
        "evet" in str(message.content) for request in provider.calls for message in request.turns
    )


async def test_the_words_that_count_come_from_the_pack() -> None:
    """A pack that names no words gets the English ones beside the code."""
    english = Locale(code="en", name="English", stt_language="en", voices={}, ui={})

    await one_turn(
        asking(capture=FakeCapture(answers=[speech()]), stt=says("Yes."), locale=english)
    )

    assert ran == ["open_app:Spotify"]


async def test_the_pack_s_words_replace_the_english_ones_rather_than_adding_to_them() -> None:
    """The Turkish pack says nothing about "yes", so "yes" is not a yes."""
    await one_turn(asking(capture=FakeCapture(answers=[speech()]), stt=says("yes")))

    assert ran == []


async def test_the_hint_falls_back_to_english_together_with_the_words() -> None:
    """Whichever words are listened for, the hint names them: the two fall
    back as one, or the user would be told to say words nobody hears."""
    english = Locale(code="en", name="English", stt_language="en", voices={}, ui={})
    tts = FakeTTS()

    await one_turn(
        asking(capture=FakeCapture(answers=[speech()]), stt=says("no"), tts=tts, locale=english)
    )

    assert tts.said[0] == "Spotify will be opened. " + app.TEXT["confirm_hint"]


def test_the_window_is_the_six_seconds_section_3_1_allows() -> None:
    assert CONFIRM_WINDOW_SECONDS == 6


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("evet", True),
        ("Evet.", True),
        ("EVET", True),
        ("tamam, olur", True),
        ("hayır", False),
        ("Hayır!", False),
        ("HAYIR", False),
        ("evet ama hayır", False),
        ("belki", None),
        ("", None),
        ("evetlemedim", None),
        ("hayırlısı olsun", None),
    ],
)
def test_read_answer_hears_whole_words_in_any_case(text: str, expected: bool | None) -> None:
    """Folded the way search is: "HAYIR" is "hayır" once the dotless i is
    folded, which `casefold` alone gets wrong. Whole words, so that a word
    that merely begins with "evet" is not a yes."""
    assert read_answer(text, yes=("evet", "tamam", "olur"), no=("hayır", "iptal")) is expected


def test_read_answer_hears_a_phrase_of_more_than_one_word() -> None:
    assert read_answer("boş ver artık", yes=("evet",), no=("boş ver",)) is False
    assert read_answer("boş verme, evet", yes=("evet",), no=("boş ver",)) is True


def test_a_blank_entry_in_the_pack_matches_nothing() -> None:
    """An empty string is inside every string; it must not be a yes."""
    assert read_answer("belki", yes=("", " "), no=("",)) is None


# --------------------------------------------------------------------------
# What the turn cost (2.4)
# --------------------------------------------------------------------------

# A round price for the fake model: a token in is a millionth of a dollar and
# a token out ten of them, so 300 in and 10 out is $0.0004.
PRICED = Pricing.from_toml('[fake."fake-1"]\ninput_per_mtok = 1.0\noutput_per_mtok = 10.0\n')


@pytest.fixture
def ledger() -> Iterator[sqlite3.Connection]:
    connection = open_database(":memory:")
    yield connection
    connection.close()


def tracking(ledger: sqlite3.Connection, limits: Limits | None = None) -> UsageTracker:
    return UsageTracker(UsageRepo(ledger), PRICED, provider="fake", model="fake-1", limits=limits)


def priced() -> ScriptedProvider:
    """A model that answers and says what it counted."""
    return ScriptedProvider([Delta(text="Üç."), Delta(usage=Usage(300, 10))])


def already_spent(ledger: sqlite3.Connection, dollars: float) -> None:
    """A turn from earlier in the day, on the books."""
    UsageRepo(ledger).insert(
        turn_id="earlier", provider="fake", model="fake-1", usage=Usage(1000, 0), cost_usd=dollars
    )


async def test_a_turn_that_reached_the_model_is_priced_and_written_down(
    ledger: sqlite3.Connection,
) -> None:
    turn = await one_turn(assistant_with(provider=priced(), tracker=tracking(ledger)))

    assert turn.cost_usd == pytest.approx(0.0004)
    [row] = ledger.execute("SELECT turn_id, in_tokens, out_tokens, cost_usd FROM usage_log")
    assert (row["turn_id"], row["in_tokens"], row["out_tokens"]) == (turn.turn_id, 300, 10)
    assert row["cost_usd"] == pytest.approx(0.0004)


async def test_a_turn_that_failed_is_not_written_down(ledger: sqlite3.Connection) -> None:
    """It reports no tokens, and a row of zeros would read as a free turn."""
    provider = ScriptedProvider([ProviderError("503")])

    turn = await one_turn(assistant_with(provider=provider, tracker=tracking(ledger)))

    assert turn.cost_usd is None
    assert ledger.execute("SELECT COUNT(*) FROM usage_log").fetchone()[0] == 0


async def test_without_a_tracker_nothing_costs_anything() -> None:
    turn = await one_turn(assistant_with(provider=priced()))

    assert turn.cost_usd is None


async def test_past_the_day_s_limit_the_answer_starts_with_a_warning(
    ledger: sqlite3.Connection,
) -> None:
    """Every turn after the line was crossed. The warning is the first
    thing said and the answer starts before its own cost is known (2.8),
    so it is the spend before this turn that decides."""
    already_spent(ledger, 0.5)
    speaker = FakeSpeaker()
    spending = tracking(ledger, Limits(daily_usd=0.1))

    turn = await one_turn(assistant_with(provider=priced(), speaker=speaker, tracker=spending))

    assert speaker.heard == f"{TURKISH.ui['daily_over']} Üç."
    assert turn.said == speaker.heard


async def test_the_turn_that_crosses_the_line_is_warned_about_on_the_next_one(
    ledger: sqlite3.Connection,
) -> None:
    """Its first word is out before its cost is; the warning follows one
    turn later, which is at most one turn's worth of money late."""
    speaker = FakeSpeaker()
    provider = ScriptedProvider(
        [Delta(text="Bir."), Delta(usage=Usage(300, 10))],
        [Delta(text="İki."), Delta(usage=Usage(300, 10))],
    )
    assistant = assistant_with(
        provider=provider, speaker=speaker, tracker=tracking(ledger, Limits(daily_usd=0.0001))
    )
    await assistant.begin()

    first = await assistant.turn(speech())
    second = await assistant.turn(speech())

    assert first.said == "Bir."
    assert second.said == f"{TURKISH.ui['daily_over']} İki."


async def test_past_the_month_s_limit_the_warning_is_the_month_s(
    ledger: sqlite3.Connection,
) -> None:
    already_spent(ledger, 0.5)
    speaker = FakeSpeaker()
    spending = tracking(ledger, Limits(daily_usd=100.0, monthly_usd=0.1))

    await one_turn(assistant_with(provider=priced(), speaker=speaker, tracker=spending))

    assert speaker.heard.startswith(TURKISH.ui["monthly_over"])


async def test_under_the_limit_nothing_is_said_about_money(ledger: sqlite3.Connection) -> None:
    speaker = FakeSpeaker()

    await one_turn(assistant_with(provider=priced(), speaker=speaker, tracker=tracking(ledger)))

    assert speaker.heard == "Üç."


async def test_with_hard_stop_on_the_model_is_not_asked_past_the_limit(
    ledger: sqlite3.Connection,
) -> None:
    """Section 3.11: `hard_stop = true` and the day's limit passed. The user
    hears why, the model hears nothing, and the turn has no name because
    nothing was filed under one."""
    already_spent(ledger, 0.5)
    provider = priced()
    speaker = FakeSpeaker()
    spending = tracking(ledger, Limits(daily_usd=0.1, hard_stop=True))

    turn = await one_turn(assistant_with(provider=provider, speaker=speaker, tracker=spending))

    assert provider.calls == []
    assert speaker.heard == TURKISH.ui["spend_stopped"]
    assert (turn.heard, turn.failure, turn.turn_id) == ("saat kaç", "spend_stopped", "")


async def test_without_hard_stop_the_model_is_still_asked_past_the_limit(
    ledger: sqlite3.Connection,
) -> None:
    """The default of decision 10: warn, do not silence."""
    already_spent(ledger, 0.5)
    provider = priced()

    await one_turn(
        assistant_with(provider=provider, tracker=tracking(ledger, Limits(daily_usd=0.1)))
    )

    assert len(provider.calls) == 1


async def test_an_answer_the_token_limit_cut_short_says_so_at_its_end() -> None:
    speaker = FakeSpeaker()
    provider = ScriptedProvider(
        [Delta(text="Uzun bir hikâyenin başı"), Delta(finish_reason="MAX_TOKENS")]
    )

    turn = await one_turn(assistant_with(provider=provider, speaker=speaker))

    assert speaker.heard == f"Uzun bir hikâyenin başı {TURKISH.ui['answer_cut_off']}"
    assert turn.said == speaker.heard


async def test_an_answer_that_ended_on_its_own_says_nothing_about_being_cut() -> None:
    speaker = FakeSpeaker()
    provider = ScriptedProvider([Delta(text="Üç."), Delta(finish_reason="STOP")])

    await one_turn(assistant_with(provider=provider, speaker=speaker))

    assert speaker.heard == "Üç."


async def test_the_warning_comes_first_and_the_cut_off_notice_last(
    ledger: sqlite3.Connection,
) -> None:
    """The warning is the one sentence to act on; the notice is where the cut is."""
    already_spent(ledger, 0.5)
    speaker = FakeSpeaker()
    provider = ScriptedProvider(
        [Delta(text="Başı"), Delta(finish_reason="MAX_TOKENS", usage=Usage(300, 10))]
    )
    spending = tracking(ledger, Limits(daily_usd=0.1))

    await one_turn(assistant_with(provider=provider, speaker=speaker, tracker=spending))

    assert speaker.heard == f"{TURKISH.ui['daily_over']} Başı {TURKISH.ui['answer_cut_off']}"


async def test_the_calls_a_turn_ran_leave_the_turn() -> None:
    gate = FakeGate()
    provider = ScriptedProvider(
        [Delta(tool_call=ToolCall(id="c1", name="clock", arguments={}))], [Delta(text="Üç.")]
    )
    agent = Agent(provider, model="fake-1", tools=ToolRegistry([clock]), dispatch=gate)

    turn = await one_turn(assistant_with(agent=agent))

    assert turn.tool_calls == 1


def test_the_thinking_timeout_is_the_turn_seconds_of_section_3_11() -> None:
    """One number, written once: the constant here reads it off `Limits`."""
    assert Limits().turn_seconds == THINKING_TIMEOUT


# --------------------------------------------------------------------------
# The fast path (2.5)
# --------------------------------------------------------------------------

# The Turkish pack's short commands, on a locale of this section's own. The
# `TURKISH` above lists none on purpose: "saat kaç" is the default thing the
# fake recogniser hears, and every test above expects it to reach the model.
WITH_COMMANDS = replace(
    TURKISH,
    intents={
        GET_TIME: ("saat kaç", "saat kaçta"),
        STOP: ("dur", "sus"),
        CANCEL: ("iptal", "vazgeç"),
    },
)

# What `get_current_time` answers with, three minutes past two.
TOLD = "2026-09-09T14:03+03:00 Wednesday, Turkey Standard Time"
ANKARA = timezone(timedelta(hours=3))


class TimeGate:
    """A gate that answers what it is told to, and keeps what it was asked."""

    def __init__(self, answer: str = TOLD) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        self.calls.append((call.name, turn_id))
        return self.answer


def commanding(
    text: str = "saat kaç", *, locale: Locale = WITH_COMMANDS, **parts: Any
) -> Assistant:
    """An assistant that hears `text` and knows the short commands."""
    stt = FakeSTT(Transcript(text=text, confidence=0.9))
    return assistant_with(stt=stt, locale=locale, **parts)


async def test_a_short_command_the_pack_lists_is_answered_without_the_model() -> None:
    """Section 4: "saat kaç" costs nothing and waits for nobody."""
    provider = ScriptedProvider([Delta(text="Üç.")])
    speaker = FakeSpeaker()

    turn = await one_turn(commanding(provider=provider, speaker=speaker, dispatch=TimeGate()))

    assert provider.calls == []
    assert speaker.heard == "Saat 14 3."
    assert (turn.heard, turn.said, turn.intent) == ("saat kaç", "Saat 14 3.", GET_TIME)


async def test_the_time_is_asked_of_the_gate_and_not_of_a_clock() -> None:
    """The fast path skips the model, not the gate (invariant 1): the one
    tool it runs is run the way every tool is, under the turn's own name."""
    gate = TimeGate()

    turn = await one_turn(commanding(dispatch=gate))

    assert gate.calls == [(TIME_TOOL, turn.turn_id)]
    assert len(turn.turn_id) == 32
    assert turn.tool_calls == 1


async def test_a_fast_turn_spent_nothing_and_is_not_on_the_bill(
    ledger: sqlite3.Connection,
) -> None:
    """A real zero, not a guess: no request was made. And no row, because
    `usage_log` is the record of requests."""
    turn = await one_turn(commanding(dispatch=TimeGate(), tracker=tracking(ledger)))

    assert turn.usage == Usage()
    assert turn.cost_usd is None
    assert ledger.execute("SELECT COUNT(*) FROM usage_log").fetchone()[0] == 0


async def test_the_fast_path_never_thinks() -> None:
    """There is nothing to wait for, so nothing to show as waiting."""
    seen: list[State] = []

    await one_turn(commanding(dispatch=TimeGate(), on_state=seen.append))

    assert seen == [State.IDLE, State.TRANSCRIBING, State.SPEAKING, State.IDLE]


@pytest.mark.parametrize(("text", "intent"), [("dur", STOP), ("iptal", CANCEL)])
async def test_stop_and_cancel_are_answered_by_silence(text: str, intent: str) -> None:
    """In phase 2 the microphone is deaf while the assistant talks and a
    press already cuts it off, so there is nothing to stop. What the two
    words save is the request they would have cost."""
    provider = ScriptedProvider([Delta(text="Tamam.")])
    tts = FakeTTS()
    seen: list[State] = []

    turn = await one_turn(
        commanding(text, provider=provider, tts=tts, dispatch=TimeGate(), on_state=seen.append)
    )

    assert provider.calls == []
    assert tts.said == []
    assert turn == Turn(heard=text, intent=intent)
    assert seen == [State.IDLE, State.TRANSCRIBING, State.IDLE]


async def test_a_sentence_that_merely_contains_the_command_goes_to_the_model() -> None:
    """ "saat kaçta toplantım var" is a question about the calendar."""
    provider = ScriptedProvider([Delta(text="Üçte.")])

    turn = await one_turn(
        commanding("saat kaçta toplantım var", provider=provider, dispatch=TimeGate())
    )

    assert len(provider.calls) == 1
    assert (turn.said, turn.intent) == ("Üçte.", None)


async def test_the_command_is_heard_however_the_recogniser_spells_it() -> None:
    """ "Saat kac?" - no cedilla, a question mark - is what Whisper sometimes
    writes down."""
    provider = ScriptedProvider([Delta(text="Üç.")])

    turn = await one_turn(commanding("Saat kac?", provider=provider, dispatch=TimeGate()))

    assert provider.calls == []
    assert turn.intent == GET_TIME


async def test_without_a_gate_the_command_goes_to_the_model() -> None:
    """No gate, no tool: the fast path cannot tell the time on its own and
    does not try to."""
    provider = ScriptedProvider([Delta(text="Üç.")])

    turn = await one_turn(commanding(provider=provider))

    assert len(provider.calls) == 1
    assert turn.intent is None


async def test_a_gate_that_did_not_tell_the_time_leaves_the_turn_to_the_model() -> None:
    """What comes back from the gate is addressed to a model: a refusal is
    a sentence, not a time, and the model is who can do something with a
    sentence."""
    provider = ScriptedProvider([Delta(text="Bilmiyorum.")])
    refusing = TimeGate(NO_SUCH_TOOL.format(name=TIME_TOOL))

    turn = await one_turn(commanding(provider=provider, dispatch=refusing))

    assert len(provider.calls) == 1
    assert (turn.said, turn.intent) == ("Bilmiyorum.", None)


async def test_the_fast_path_runs_past_the_spending_limit_even_with_hard_stop(
    ledger: sqlite3.Connection,
) -> None:
    """A limit on spending has nothing to say about a turn that costs nothing."""
    already_spent(ledger, 0.5)
    provider = priced()
    speaker = FakeSpeaker()
    spending = tracking(ledger, Limits(daily_usd=0.1, hard_stop=True))

    turn = await one_turn(
        commanding(provider=provider, speaker=speaker, dispatch=TimeGate(), tracker=spending)
    )

    assert provider.calls == []
    assert speaker.heard == "Saat 14 3."
    assert turn.failure is None


async def test_the_fast_path_runs_through_the_permission_gate_and_is_written_down(
    ledger: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real gate over the real tool: one `ok` row in `tool_audit`, under
    the turn's name, exactly as if the model had asked for it."""
    monkeypatch.setattr(system, "_now", lambda: datetime(2026, 9, 9, 14, 3, tzinfo=ANKARA))
    audit = AuditRepo(ledger)

    async def gate(call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        registry = ToolRegistry([get_current_time])
        return await dispatch(
            call, turn_id=turn_id, registry=registry, confirm=confirm, audit=audit
        )

    turn = await one_turn(commanding(dispatch=gate))

    assert turn.said == "Saat 14 3."
    [row] = ledger.execute("SELECT tool, status, turn_id FROM tool_audit").fetchall()
    assert (row["tool"], row["status"], row["turn_id"]) == (TIME_TOOL, "ok", turn.turn_id)


async def test_the_time_sentence_comes_from_the_pack_and_falls_back_with_the_rest() -> None:
    speaker = FakeSpeaker()

    await one_turn(
        commanding(locale=replace(WITH_COMMANDS, ui={}), speaker=speaker, dispatch=TimeGate())
    )

    assert speaker.heard == "It is 14 3."


async def test_a_press_during_the_fast_path_is_not_spoken_over() -> None:
    """The same rule as for an answer from the model: the user is talking."""
    capture = FakeCapture()
    speaker = FakeSpeaker()

    async def pressed_meanwhile(call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        capture.press()
        return TOLD

    assistant = commanding(capture=capture, speaker=speaker, dispatch=pressed_meanwhile)
    await one_turn(assistant)

    assert speaker.played == []
    assert assistant.state is State.LISTENING


# --------------------------------------------------------------------------
# Speaking as the words come (2.8)
# --------------------------------------------------------------------------

WITH_FILLERS = replace(TURKISH, fillers=("Bir saniye, bakıyorum...", "Hemen bakıyorum..."))


class SlowGate:
    """A gate whose tool takes `seconds`: a slow tool, or the silence of the
    model's next request. The tool's body is never run."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.calls: list[str] = []

    async def __call__(self, call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        self.calls.append(call.name)
        await asyncio.sleep(self.seconds)
        return "15:04"


def wants_the_clock(*rounds: int) -> ScriptedProvider:
    """A model that asks for the clock once per round, then answers."""
    return ScriptedProvider(
        *(
            [Delta(tool_call=ToolCall(id=f"c{number}", name="clock", arguments={"n": number}))]
            for number in rounds
        ),
        [Delta(text="Üç.")],
    )


def looking(seconds: float, provider: ScriptedProvider | None = None, **parts: Any) -> Assistant:
    """An assistant whose model asks for the clock and waits `seconds` for it.

    The filler's clock is shortened to keep the test quick; what is tested
    is the rule, not the number.
    """
    agent = Agent(
        provider if provider is not None else wants_the_clock(1),
        model="fake-1",
        tools=ToolRegistry([clock]),
        dispatch=SlowGate(seconds),
    )
    parts.setdefault("locale", WITH_FILLERS)
    return assistant_with(agent=agent, filler_delay=0.05, **parts)


class Noticing(FakeSpeaker):
    """A sound card that says when it has the first buffer in hand."""

    def __init__(self) -> None:
        super().__init__()
        self.has_first = asyncio.Event()

    async def play(self, buffers: AsyncIterator[bytes], *, sample_rate: int) -> None:
        self.rates.append(sample_rate)
        async for buffer in buffers:
            self.played.append(buffer)
            self.has_first.set()


async def test_the_first_sentence_is_spoken_before_the_model_has_written_the_second() -> None:
    """The whole of 2.8 in one claim. The model is held back until the
    speaker has the first sentence; a turn that waited for the whole
    answer would wait for ever here, so the hold gives up after two
    seconds and the turn fails instead of hanging."""
    speaker = Noticing()
    provider = ScriptedProvider(
        [
            Delta(text="Birinci cümle burada. "),
            lambda: asyncio.wait_for(speaker.has_first.wait(), 2.0),
            Delta(text="İkinci cümle burada."),
        ]
    )

    turn = await one_turn(assistant_with(provider=provider, speaker=speaker))

    assert turn.failure is None
    assert speaker.heard == "Birinci cümle burada. İkinci cümle burada."
    assert turn.said == speaker.heard


async def test_speaking_begins_with_the_first_word_and_not_when_the_model_is_done() -> None:
    seen: list[State] = []
    held: list[Assistant] = []
    provider = ScriptedProvider(
        [Delta(text="Bir. "), lambda: seen.append(held[0].state), Delta(text="İki.")]
    )
    assistant = assistant_with(provider=provider)
    held.append(assistant)

    await one_turn(assistant)

    assert seen == [State.SPEAKING]


async def test_the_filler_is_said_when_a_tool_round_goes_quiet() -> None:
    """Two seconds of silence do not feel like two seconds once something
    has been said about them (section 4)."""
    speaker = FakeSpeaker()

    turn = await one_turn(looking(0.2, speaker=speaker))

    assert speaker.heard == "Bir saniye, bakıyorum... Üç."
    assert turn.said == speaker.heard
    assert turn.tool_calls == 1


async def test_a_tool_that_answers_at_once_needs_no_filler() -> None:
    """A "one moment" in front of an answer that was already on its way
    would only push it back."""
    speaker = FakeSpeaker()

    await one_turn(looking(0.0, speaker=speaker))

    assert speaker.heard == "Üç."


async def test_the_filler_is_said_once_however_many_rounds_go_quiet() -> None:
    speaker = FakeSpeaker()

    await one_turn(looking(0.2, wants_the_clock(1, 2), speaker=speaker))

    assert speaker.heard == "Bir saniye, bakıyorum... Üç."


async def test_the_fillers_are_said_in_turn() -> None:
    """A pack that lists two is not heard saying the same one every time."""
    speaker = FakeSpeaker()
    provider = ScriptedProvider(
        [Delta(tool_call=ToolCall(id="c1", name="clock", arguments={}))],
        [Delta(text="Üç.")],
        [Delta(tool_call=ToolCall(id="c2", name="clock", arguments={}))],
        [Delta(text="Dört.")],
    )
    assistant = looking(0.2, provider, speaker=speaker)
    await assistant.begin()

    first = await assistant.turn(speech())
    second = await assistant.turn(speech())

    assert first.said == "Bir saniye, bakıyorum... Üç."
    assert second.said == "Hemen bakıyorum... Dört."


async def test_the_filler_comes_from_the_code_when_the_pack_has_none() -> None:
    speaker = FakeSpeaker()

    await one_turn(looking(0.2, speaker=speaker, locale=TURKISH))

    assert speaker.heard == f"{app.FILLERS[0]} Üç."


async def test_the_filler_follows_what_the_model_said_before_it_looked() -> None:
    """Set off from the model's words, which may have ended mid-word."""
    speaker = FakeSpeaker()
    provider = ScriptedProvider(
        [Delta(text="Bakıyorum"), Delta(tool_call=ToolCall(id="c1", name="clock", arguments={}))],
        [Delta(text="Üç.")],
    )

    await one_turn(looking(0.2, provider, speaker=speaker))

    assert speaker.heard == "Bakıyorum Bir saniye, bakıyorum... Üç."


class SlowWindow(FakeCapture):
    """A confirmation window that takes a while to hear the answer."""

    async def listen_for(self, seconds: float) -> Audio | None:
        await asyncio.sleep(0.2)
        return await super().listen_for(seconds)


async def test_no_filler_is_said_while_a_question_is_being_asked() -> None:
    """The user is answering it; "one moment" over a question is noise at
    best and a second question at worst."""
    tts = FakeTTS()

    await one_turn(
        asking(
            capture=SlowWindow(answers=[speech()]),
            stt=says("evet"),
            tts=tts,
            locale=WITH_FILLERS,
            filler_delay=0.05,
        )
    )

    assert ran == ["open_app:Spotify"]
    assert tts.said == ["Spotify will be opened. Evet ya da hayır de.", "Tamam."]


async def test_the_state_goes_back_to_speaking_after_a_question_asked_mid_answer() -> None:
    """The model said something, then asked. The window is a detour from
    wherever the turn was, and `SPEAKING` is where it was."""
    seen: list[State] = []
    provider = ScriptedProvider(
        [
            Delta(text="Açıyorum. "),
            Delta(tool_call=ToolCall(id="c1", name="open_app", arguments={"name": "Spotify"})),
        ],
        [Delta(text="Tamam.")],
    )

    await one_turn(
        asking(
            provider,
            capture=FakeCapture(answers=[speech()]),
            stt=says("evet"),
            on_state=seen.append,
        )
    )

    assert seen == [
        State.IDLE,
        State.TRANSCRIBING,
        State.THINKING,
        State.SPEAKING,
        State.CONFIRMING,
        State.SPEAKING,
        State.IDLE,
    ]
    assert ran == ["open_app:Spotify"]


async def test_a_failure_in_the_second_request_is_said_after_what_was_already_heard() -> None:
    """Half an answer was spoken before the connection dropped; the sentence
    that says so comes after it, and the turn is not remembered - the next
    one starts clean, exactly as when the first request failed."""
    speaker = FakeSpeaker()
    provider = ScriptedProvider(
        [Delta(text="Bakıyorum. "), Delta(tool_call=ToolCall(id="c1", name="clock", arguments={}))],
        [ProviderError("503")],
        [Delta(text="Selam.")],
    )
    agent = Agent(provider, model="fake-1", tools=ToolRegistry([clock]), dispatch=FakeGate())
    assistant = assistant_with(agent=agent, speaker=speaker)
    await assistant.begin()

    failed = await assistant.turn(speech())
    await assistant.turn(speech())

    assert failed.said == f"Bakıyorum. {TURKISH.ui['unreachable']}"
    assert failed.failure == "unreachable"
    assert failed.usage == Usage()
    assert provider.calls[-1].turns == [Message.user("saat kaç")]


async def test_the_key_going_down_ends_the_request_and_not_only_the_sound() -> None:
    """The model would have gone on for a minute. The press ends the wait
    at once, the provider's stream is closed under it, and the turn is
    over - because the next one, the one the user is speaking now, cannot
    start until it is."""
    capture = FakeCapture()
    provider = ScriptedProvider([Delta(text="Bir. "), capture.press, 60.0, Delta(text="İki.")])
    assistant = assistant_with(capture=capture, provider=provider)

    turn = await asyncio.wait_for(one_turn(assistant), 2.0)

    assert turn.said == "Bir."
    assert provider.open_streams == 0
    assert assistant.state is State.LISTENING


async def test_the_clock_runs_only_until_the_first_word() -> None:
    """A long answer read out loud is not a turn that hung."""
    speaker = FakeSpeaker()
    provider = ScriptedProvider([Delta(text="Başladı. "), 0.05, Delta(text="Bitti.")])

    turn = await one_turn(assistant_with(provider=provider, speaker=speaker, thinking_timeout=0.02))

    assert turn.failure is None
    assert speaker.heard == "Başladı. Bitti."


async def test_how_long_the_first_sound_took_leaves_the_turn() -> None:
    """The number 2.8 is about, kept by the log so that it can be watched."""
    spoken = await one_turn(assistant_with())
    quiet = ScriptedProvider([Delta(finish_reason="SAFETY")])
    silent = await one_turn(assistant_with(provider=quiet))

    assert spoken.first_sound_ms is not None
    assert 0 <= spoken.first_sound_ms < 5000
    assert silent.first_sound_ms is None


async def test_a_press_before_the_first_word_leaves_the_model_s_answer_unspoken() -> None:
    """The stream is closed under the model, and no sentence follows: not the
    answer, and not a failure's either."""
    capture = FakeCapture()
    speaker = FakeSpeaker()
    provider = ScriptedProvider([capture.press, ProviderError("503")])

    turn = await one_turn(assistant_with(capture=capture, provider=provider, speaker=speaker))

    assert speaker.played == []
    assert turn.said == ""
