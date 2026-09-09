"""The fast path of design.md section 4 (2.5): the short commands a locale
pack lists are recognised here and never sent to the model.

Three claims. A phrase is matched whole: "saat kaçta toplantım var" is not
"saat kaçta". Both sides are folded the way search is, so the case, the
accents and the punctuation the recogniser happens to write do not decide.
And the phrases come from the pack, intent by intent, with the English
phrases beside the code as the end of the chain - `test_locales.py` checks
that the shipped packs fill the table; this file checks what is done with
it.

What an intent *does* - the gate, the sentence, the silence - is in
`test_app.py`. Here a locale is built by hand, so that rewording `tr.toml`
does not break a test about matching.
"""

from __future__ import annotations

import pytest

from assistant.agent.intents import CANCEL, GET_TIME, INTENTS, STOP, TIME_TOOL, match_intent
from assistant.locales import Locale, load
from assistant.tools.registry import ToolRegistry
from assistant.tools.system import get_current_time


def pack(**intents: tuple[str, ...]) -> Locale:
    """A locale that lists exactly these phrases, and nothing else."""
    return Locale(code="xx", name="xx", stt_language="xx", voices={}, ui={}, intents=intents)


TURKISH = pack(
    get_time=("saat kaç", "saat kaçta"),
    stop=("dur", "sus", "kes"),
    cancel=("iptal", "vazgeç", "boş ver"),
)
NO_COMMANDS = pack()


# --------------------------------------------------------------------------
# Whole phrases
# --------------------------------------------------------------------------


def test_the_command_as_the_recogniser_writes_it_is_the_command() -> None:
    assert match_intent("Saat kaç?", TURKISH) == GET_TIME


@pytest.mark.parametrize("text", ["saat kaçta toplantım var", "saat kaç lütfen", "bugün saat kaç"])
def test_a_sentence_that_merely_contains_the_command_is_not_the_command(text: str) -> None:
    """A question about the calendar, answered with the time, would make the
    assistant look deaf. Anything longer than the phrase is the model's."""
    assert match_intent(text, TURKISH) is None


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("dur", STOP),
        ("Sus!", STOP),
        ("iptal", CANCEL),
        ("Vazgeç.", CANCEL),
        ("boş ver", CANCEL),
    ],
)
def test_each_of_the_pack_s_phrases_is_heard(text: str, intent: str) -> None:
    assert match_intent(text, TURKISH) == intent


def test_nothing_said_is_no_command() -> None:
    assert match_intent("", TURKISH) is None
    assert match_intent("?!", TURKISH) is None


# --------------------------------------------------------------------------
# Folded on both sides
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["SAAT KAÇ", "saat kac", "Saat, kaç?!", "  saat   kaç  ", "Saat Kac"]
)
def test_the_case_the_accents_and_the_punctuation_do_not_decide(text: str) -> None:
    """ "saat kac" is what the recogniser sometimes writes, and "SAAT KAÇ" is
    a Turkish `I` problem in the making: both fold to the same phrase as
    the pack's (`store/normalize.py`)."""
    assert match_intent(text, TURKISH) == GET_TIME


def test_the_pack_s_own_phrase_is_folded_too() -> None:
    """A translator who wrote "Saat Kaç?" has written "saat kaç"."""
    assert match_intent("saat kaç", pack(get_time=("Saat Kaç?",))) == GET_TIME


# --------------------------------------------------------------------------
# The chain: the pack, then the English beside the code
# --------------------------------------------------------------------------


def test_a_pack_with_no_commands_gets_the_english_ones() -> None:
    assert match_intent("What time is it?", NO_COMMANDS) == GET_TIME
    assert match_intent("stop", NO_COMMANDS) == STOP
    assert match_intent("Cancel.", NO_COMMANDS) == CANCEL


def test_the_pack_s_phrases_replace_the_english_ones_for_that_command() -> None:
    """Replace, not add: an English "what time is it" is not something a
    Turkish pack asked to listen for."""
    assert match_intent("what time is it", TURKISH) is None


def test_a_command_the_pack_leaves_out_is_still_heard_in_english() -> None:
    """Intent by intent, like a sentence: forgetting "dur" does not switch
    stop off."""
    only_the_time = pack(get_time=("saat kaç",))

    assert match_intent("saat kaç", only_the_time) == GET_TIME
    assert match_intent("stop", only_the_time) == STOP


def test_a_command_the_code_cannot_answer_is_not_a_command() -> None:
    """A pack may list `play_music`; nothing in the code answers it, so the
    sentence goes to the model, which can."""
    assert match_intent("müzik çal", pack(play_music=("müzik çal",))) is None


def test_the_english_phrases_cover_every_intent_the_code_answers() -> None:
    assert set(INTENTS) == {GET_TIME, STOP, CANCEL}
    assert all(INTENTS[name] for name in INTENTS)


# --------------------------------------------------------------------------
# What the shipped pack and the registry agree on
# --------------------------------------------------------------------------


def test_the_shipped_turkish_pack_answers_all_three() -> None:
    """The promise of section 4 for the language phase 1 ships complete."""
    turkish = load("tr")

    assert match_intent("saat kaç", turkish) == GET_TIME
    assert match_intent("dur", turkish) == STOP
    assert match_intent("iptal", turkish) == CANCEL


def test_the_time_is_asked_of_the_tool_of_phase_2_1c() -> None:
    """`app.py` runs `TIME_TOOL` through the gate. A name the registry does
    not know would be refused at the gate, and the fast path would quietly
    hand every "saat kaç" back to the model."""
    found = ToolRegistry([get_current_time]).get(TIME_TOOL)

    assert found is not None
    assert found.spec.name == TIME_TOOL
