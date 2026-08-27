"""Locale packs: everything that depends on the language, kept out of the code.

Section 3.12 turns "which language does the product speak" into a
configuration dimension, and two claims make that true rather than decorative.

The first is the fallback chain - the pack that was asked for, then `en`, then
the English constant in the code that says the sentence. It is what makes an
unsupported language *usable* instead of broken, and it is tested against packs
written into `tmp_path`: the mechanism is the point, and a test that read the
real `tr.toml` would fail every time a sentence was reworded.

The second is that the packs this project ships answer everything phase 1 asks
of them. A missing key is survivable by design, which is exactly why nobody
would ever notice one.
"""

from __future__ import annotations

import string
import tomllib
from pathlib import Path
from typing import Any

from assistant import locales
from assistant.locales import FALLBACK_CODE, available, iso_code, load, system_code
from assistant.setup_wizard import TEXT

PACKAGED = Path(locales.__file__).parent


def shipped() -> list[Path]:
    return sorted(PACKAGED.glob("*.toml"))


def read(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def write(directory: Path, code: str, body: str) -> None:
    (directory / f"{code}.toml").write_text(body, encoding="utf-8")


def fields(sentence: str) -> set[str]:
    """The `{name}` placeholders a sentence expects to be given."""
    return {name for _, name, _, _ in string.Formatter().parse(sentence) if name}


# --------------------------------------------------------------------------
# The packs this project ships
# --------------------------------------------------------------------------


def test_phase_one_ships_the_two_languages_it_promises() -> None:
    """A list rather than a set: a file that is not a language still loads as
    one, and would sit in the menu as a second, identical English."""
    assert [pack.code for pack in available()] == ["en", "tr"]


def test_the_template_is_a_scaffold_and_not_a_language() -> None:
    """It exists to be copied. Offered in the menu it would be a language that
    speaks nothing but placeholders."""
    assert (PACKAGED / "_template.toml").is_file()
    assert not any(pack.code.startswith("_") for pack in available())


def test_every_shipped_pack_answers_what_phase_one_asks_of_it() -> None:
    for pack in available():
        assert pack.name.strip(), f"{pack.code} has no name to show in a menu"
        assert pack.stt_language, f"{pack.code} gives the speech recogniser no hint"
        assert pack.voice("sapi"), f"{pack.code} names no Windows voice"


def test_a_pack_is_filed_under_the_name_of_its_own_file() -> None:
    """The file name is what the loader trusts. A `code` that disagrees with it
    means `config.toml` and the pack are quietly talking about two languages."""
    for path in shipped():
        if path.stem.startswith("_"):
            continue
        assert read(path).get("code") == path.stem, path.name


def test_english_is_not_written_down_twice() -> None:
    """`en.toml` carries no sentences: English is already the constant the
    chain ends at, and a second copy is a second thing to keep in step."""
    assert "ui" not in read(PACKAGED / "en.toml")


def test_the_template_offers_every_sentence_a_translator_has_to_write() -> None:
    """A key the template forgets is a sentence nobody ever translates, and the
    product answers it in English for good without anyone noticing."""
    assert set(read(PACKAGED / "_template.toml")["ui"]) == set(TEXT)


def test_a_pack_translates_only_sentences_the_product_actually_says() -> None:
    """A key left behind by a reworded question is dead weight that reads like
    finished work."""
    for path in shipped():
        offered = set(read(path).get("ui", {}))
        assert offered <= set(TEXT), f"{path.name} translates {offered - set(TEXT)}"


def test_turkish_is_complete_because_it_is_one_of_the_two_that_is_promised() -> None:
    """The fallback chain would hide a gap, and an English sentence in the
    middle of a Turkish wizard is a defect, not a graceful degradation."""
    assert set(read(PACKAGED / "tr.toml")["ui"]) == set(TEXT)


def test_a_translation_asks_for_the_same_fields_as_the_sentence_it_replaces() -> None:
    """The sentence is formatted with what the caller passes. A renamed
    placeholder is a `KeyError` in front of the user, halfway through setup."""
    for path in shipped():
        for key, sentence in read(path).get("ui", {}).items():
            assert fields(sentence) == fields(TEXT[key]), f"{path.name}: {key}"


# --------------------------------------------------------------------------
# The fallback chain: the pack, then English, then the code
# --------------------------------------------------------------------------


def test_the_pack_that_was_asked_for_outranks_english(tmp_path: Path) -> None:
    """The first link of the chain. Both packs answer this key, and the whole
    point of asking for one of them is that its answer is used."""
    write(tmp_path, "en", '[ui]\nmodel = "Which model should answer?"\n')
    write(tmp_path, "de", '[ui]\nmodel = "Welches Modell?"\n')

    assert load("de", directory=tmp_path).say("model", "constant") == "Welches Modell?"


def test_a_sentence_the_pack_forgets_is_said_in_english(tmp_path: Path) -> None:
    write(tmp_path, "en", '[ui]\nmodel = "Which model should answer?"\n')
    write(tmp_path, "de", '[ui]\nwelcome = "Willkommen."\n')

    said = load("de", directory=tmp_path).say("model", "constant in the code")

    assert said == "Which model should answer?"


def test_a_sentence_no_pack_offers_falls_back_to_the_constant_in_the_code(tmp_path: Path) -> None:
    write(tmp_path, "en", 'name = "English"\n')

    assert load("de", directory=tmp_path).say("model", "Which model?") == "Which model?"


def test_a_translation_left_blank_is_not_a_translation(tmp_path: Path) -> None:
    write(tmp_path, "de", '[ui]\nmodel = ""\n')

    assert load("de", directory=tmp_path).say("model", "Which model?") == "Which model?"


def test_a_language_with_no_pack_at_all_is_still_a_working_locale(tmp_path: Path) -> None:
    """Section 3.12's promise: an unsupported language gets an English
    interface, not a broken product."""
    pack = load("de", directory=tmp_path)

    assert (pack.code, pack.name, pack.stt_language) == ("de", "de", "de")
    assert pack.say("welcome", "Hello.") == "Hello."


def test_english_does_not_lend_its_voice_to_another_language(tmp_path: Path) -> None:
    """Sentences fall back; identity does not. An English voice reading German
    is worse than no preference at all."""
    write(tmp_path, "en", '[tts.voice]\nsapi = "Zira"\n')

    assert load("de", directory=tmp_path).voice("sapi") is None


def test_a_voice_left_blank_is_no_preference_at_all(tmp_path: Path) -> None:
    """The template ships the key with a placeholder in it. Emptying it is how
    a translator says their language has no voice worth naming."""
    write(tmp_path, "de", '[tts.voice]\nsapi = ""\n')

    assert load("de", directory=tmp_path).voice("sapi") is None


def test_english_does_not_lend_its_speech_language_either(tmp_path: Path) -> None:
    """Whisper told to expect English would return English-shaped nonsense for
    every German sentence, and the model would never see the German."""
    write(tmp_path, "en", '[stt]\nlanguage = "en"\n')

    assert load("de", directory=tmp_path).stt_language == "de"


# --------------------------------------------------------------------------
# Asking for a pack
# --------------------------------------------------------------------------


def test_the_region_and_the_case_are_not_part_of_the_question(tmp_path: Path) -> None:
    """`tr-TR` comes out of Windows, `TR` out of a settings file somebody
    typed by hand; both name the same pack."""
    write(tmp_path, "tr", 'name = "Turkce"\n')

    assert load("tr-TR", directory=tmp_path).name == "Turkce"
    assert load("TR", directory=tmp_path).name == "Turkce"
    # The code is carried around afterwards - as the speech hint, and as what
    # the pack is called. Windows finds `TR.toml` whatever the case, so the
    # name alone would not notice this.
    assert load("TR", directory=tmp_path).code == "tr"


def test_asking_for_no_language_at_all_gives_the_fallback(tmp_path: Path) -> None:
    write(tmp_path, "en", 'name = "English"\n')

    assert load(None, directory=tmp_path).code == FALLBACK_CODE
    assert load("", directory=tmp_path).code == FALLBACK_CODE


def test_a_code_that_could_name_a_file_somewhere_else_is_refused(tmp_path: Path) -> None:
    """The code comes out of a settings file. A language is letters; anything
    else is a path, and reading one would be this module's own fault."""
    write(tmp_path, "en", 'name = "English"\n')

    assert load("../secrets", directory=tmp_path).code == FALLBACK_CODE


def test_the_languages_are_offered_in_a_settled_order(tmp_path: Path) -> None:
    """A menu that reshuffles itself between runs is a menu nobody learns."""
    for code in ("tr", "en", "de"):
        write(tmp_path, code, f'name = "{code}"\n')

    assert [pack.code for pack in available(directory=tmp_path)] == ["de", "en", "tr"]


# --------------------------------------------------------------------------
# Packs written by other people
# --------------------------------------------------------------------------


def test_a_pack_that_does_not_parse_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    """A contributor's typo must not stop the assistant from starting."""
    write(tmp_path, "en", 'name = "English"\n[ui]\nwelcome = "Hello."\n')
    write(tmp_path, "de", "this is not toml at all\n")

    assert load("de", directory=tmp_path).say("welcome", "constant") == "Hello."
    assert [pack.code for pack in available(directory=tmp_path)] == ["en"]


def test_a_pack_of_the_wrong_shape_is_read_as_far_as_it_makes_sense(tmp_path: Path) -> None:
    """Every value here is the wrong kind of thing, and each one falls back on
    its own rather than taking the whole pack down with it."""
    write(tmp_path, "de", 'name = 5\nstt = "German"\n[tts]\nvoice = 3\n[ui]\nmodel = 7\n')

    pack = load("de", directory=tmp_path)

    assert (pack.code, pack.name, pack.stt_language) == ("de", "de", "de")
    assert pack.voice("sapi") is None
    assert pack.say("model", "Which model?") == "Which model?"


# --------------------------------------------------------------------------
# The language Windows itself is in
# --------------------------------------------------------------------------


def test_windows_language_identifiers_are_read_as_iso_codes() -> None:
    assert iso_code(0x041F) == "tr"
    assert iso_code(0x0409) == "en"


def test_a_language_windows_will_not_name_is_read_as_the_fallback() -> None:
    """Unlike a voice, which may honestly have no language, the interface has
    to be in *some* language before the first question is asked."""
    assert iso_code(0xFFFF) == FALLBACK_CODE


def test_the_language_windows_is_in_can_be_asked_for() -> None:
    assert system_code().isalpha()
    assert system_code() == system_code().casefold()
