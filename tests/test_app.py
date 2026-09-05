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
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import numpy as np
import pytest
from loguru import logger

from assistant import app
from assistant.agent.core import Agent
from assistant.app import (
    THINKING_TIMEOUT,
    Assistant,
    Heard,
    NoVoiceError,
    State,
    Turn,
    choose_voice,
    hear,
)
from assistant.audio.player import PlaybackError
from assistant.llm.base import AuthenticationError, Delta, ProviderError, Usage
from assistant.locales import Locale
from assistant.stt.base import SAMPLE_RATE, Audio, Transcript
from assistant.tts.base import VoiceInfo
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
    },
)


def speech(seconds: float = 2.0) -> Audio:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


class StopError(Exception):
    """Ends `run()` in a test the way closing the program ends it in life."""


class FakeCapture:
    """Push to talk without a keyboard: utterances arrive in the order given."""

    def __init__(self, *utterances: Audio) -> None:
        self.on_listening: Callable[[], None] | None = None
        self.started = False
        # Whether the state machine has told it to stop listening, and how many
        # times it has been told either thing - a microphone left deaf and one
        # that was never deafened look the same from the flag alone.
        self.deaf = False
        self.switches: list[bool] = []
        self._waiting = list(utterances)

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
    on_state: Callable[[State], None] | None = None,
    on_turn: Callable[[Turn], None] | None = None,
) -> Assistant:
    return Assistant(
        capture=capture if capture is not None else FakeCapture(),
        stt=stt if stt is not None else FakeSTT(),
        agent=Agent(
            provider if provider is not None else ScriptedProvider([Delta(text="Üç.")]),
            model="fake-1",
        ),
        tts=tts if tts is not None else FakeTTS(),
        speaker=speaker if speaker is not None else FakeSpeaker(),
        locale=locale,
        thinking_timeout=thinking_timeout,
        on_state=on_state,
        on_turn=on_turn,
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
