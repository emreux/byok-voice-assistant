"""The three questions of `assistant setup`, without a terminal.

The wizard is the only place where a key the user typed exists in memory, so
the tests that matter are about what happens to it: it is checked before it is
kept, a key that does not work is never written anywhere, and it is never
printed back. The rest - which provider, which model, which language - is
ordinary bookkeeping.

Nothing here draws on a screen. `run_setup` talks to a `Prompter`, and this
suite hands it a scripted one, so a wizard run is a function call with a
recorded transcript rather than a session someone has to sit through.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import pytest
import questionary

from assistant import locales
from assistant.config import (
    KEYRING_SERVICE,
    LLMSettings,
    LocaleSettings,
    Settings,
    config_path,
    load_settings,
    save_settings,
    store_api_key,
)
from assistant.llm.base import Delta, Message, ModelInfo, ToolSpec
from assistant.llm.registry import ADAPTERS, ProviderEntry
from assistant.setup_wizard import TEXT, Option, TerminalPrompter, run_setup, wording
from tests.conftest import MemoryKeyring

GOOD_KEY = "good-key"


class FakeProvider:
    """A provider that accepts one key and offers two models."""

    id = "fake"
    models: ClassVar[list[ModelInfo]] = [
        ModelInfo(id="fast", display_name="Fast"),
        ModelInfo(id="smart", display_name="Smart"),
    ]

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def validate_credentials(self) -> bool:
        return self.api_key == GOOD_KEY

    async def list_models(self) -> list[ModelInfo]:
        return list(self.models)

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        raise NotImplementedError("the wizard never sends a message")


class ScriptedPrompter:
    """Answers the wizard from a script and records the whole exchange.

    Questions are addressed by a stable key rather than by their wording, so a
    test says what it answers instead of repeating a sentence. Every key is
    checked against `TEXT`: a wizard that asks something the text table has no
    words for fails here rather than in front of the user.
    """

    def __init__(self, **answers: str | list[str | None] | None) -> None:
        self._script: dict[str, list[str | None]] = {
            key: list(value) if isinstance(value, list) else [value]
            for key, value in answers.items()
        }
        self.asked: list[str] = []
        self.offered: dict[str, list[str]] = {}
        self.said: list[tuple[str, dict[str, object]]] = []

    def say(self, key: str, **fields: object) -> None:
        assert key in TEXT, f"the wizard said {key!r}, which has no text"
        self.said.append((key, fields))

    def choose(self, key: str, options: Sequence[Option]) -> str | None:
        self.offered[key] = [option.value for option in options]
        return self._answer(key)

    def secret(self, key: str) -> str | None:
        return self._answer(key)

    def _answer(self, key: str) -> str | None:
        assert key in TEXT, f"the wizard asked {key!r}, which has no text"
        self.asked.append(key)
        queue = self._script.get(key)
        assert queue, f"the wizard asked {key!r} more often than the script answers"
        return queue.pop(0)


def fake_catalog(*provider_ids: str) -> dict[str, ProviderEntry]:
    return {
        provider_id: ProviderEntry(
            id=provider_id,
            adapter="fake",
            display_name=provider_id.title(),
            key_url=f"https://{provider_id}.example/apikey",
        )
        for provider_id in (provider_ids or ("gemini",))
    }


@pytest.fixture(autouse=True)
def fake_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registers an adapter that answers without a network."""
    monkeypatch.setitem(ADAPTERS, "fake", lambda entry, api_key: FakeProvider(api_key))


def complete_run(**overrides: str | list[str | None] | None) -> ScriptedPrompter:
    """A prompter scripted to walk the wizard from end to end."""
    answers: dict[str, str | list[str | None] | None] = {
        "locale": "tr",
        "api_key": GOOD_KEY,
        "model": "fast",
    }
    answers.update(overrides)
    return ScriptedPrompter(**answers)


# --------------------------------------------------------------------------
# What the wizard leaves behind
# --------------------------------------------------------------------------


async def test_the_answers_end_up_in_the_settings_and_the_vault(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run(model="smart")

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    settings = load_settings()
    assert exit_code == 0
    assert settings.llm.primary == "gemini:smart"
    assert settings.locale.code == "tr"
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}


async def test_the_key_is_not_written_to_the_settings_file(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """The one claim section 3.3 makes about key safety, from the other end."""
    await run_setup(complete_run(), catalog=fake_catalog())

    assert GOOD_KEY not in config_path().read_text(encoding="utf-8")


async def test_the_key_is_never_printed_back(config_home: Path, vault: MemoryKeyring) -> None:
    """A key echoed to the terminal outlives the session in the scrollback."""
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    printed = [str(value) for _, fields in prompter.said for value in fields.values()]
    assert not any(GOOD_KEY in text for text in printed)


async def test_setup_run_again_changes_the_model_and_keeps_the_rest(
    config_home: Path, vault: MemoryKeyring
) -> None:
    save_settings(Settings(llm=LLMSettings(primary="gemini:fast")))

    await run_setup(complete_run(model="smart"), catalog=fake_catalog())

    assert load_settings().llm.primary == "gemini:smart"


# --------------------------------------------------------------------------
# The provider
# --------------------------------------------------------------------------


async def test_one_buildable_provider_is_not_worth_a_question(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert "provider" not in prompter.asked
    assert load_settings().llm.provider == "gemini"


async def test_only_providers_this_build_can_construct_are_offered(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """An entry naming a later phase's adapter is a dead end the user would
    only discover after typing their key in."""
    catalog: Mapping[str, ProviderEntry] = {
        **fake_catalog("gemini", "openrouter"),
        "groq": ProviderEntry(id="groq", adapter="openai_compat", display_name="Groq"),
    }
    prompter = complete_run(provider="openrouter")

    await run_setup(prompter, catalog=catalog)

    assert prompter.offered["provider"] == ["gemini", "openrouter"]
    assert load_settings().llm.provider == "openrouter"


# --------------------------------------------------------------------------
# The key
# --------------------------------------------------------------------------


async def test_a_key_that_does_not_work_is_asked_again(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run(api_key=["typo", GOOD_KEY])

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    assert exit_code == 0
    assert prompter.asked.count("api_key") == 2
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}


async def test_a_key_that_does_not_work_is_never_stored(
    config_home: Path, vault: MemoryKeyring
) -> None:
    exit_code = await run_setup(complete_run(api_key=["typo", None]), catalog=fake_catalog())

    assert exit_code != 0
    assert vault.vault == {}


async def test_an_empty_answer_is_not_a_key(config_home: Path, vault: MemoryKeyring) -> None:
    """Enter on an empty prompt is a slip, not a key to go and check.

    The provider is never asked about an empty string: `genai.Client` raises on
    one, so what should be "you left it blank" would be a traceback.
    """
    prompter = complete_run(api_key=["", GOOD_KEY])

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    said = [key for key, _ in prompter.said]
    assert exit_code == 0
    assert prompter.asked.count("api_key") == 2
    assert said.count("key_needed") == 1
    assert "bad_key" not in said


async def test_a_stored_key_is_kept_when_the_answer_is_empty(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """Changing the model must not mean fetching the key from the browser again."""
    store_api_key("gemini", GOOD_KEY)
    prompter = complete_run(api_key_keep="")

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    assert exit_code == 0
    assert "api_key" not in prompter.asked
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}


async def test_a_stored_key_that_stopped_working_is_replaced(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """A revoked key is the reason the owner reruns setup in the first place."""
    store_api_key("gemini", "revoked")
    prompter = complete_run(api_key_keep="")

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    assert exit_code == 0
    assert prompter.asked == ["locale", "api_key_keep", "api_key", "model"]
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}


# --------------------------------------------------------------------------
# The model and the language
# --------------------------------------------------------------------------


async def test_the_models_offered_are_the_ones_the_key_can_reach(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert prompter.offered["model"] == ["fast", "smart"]


async def test_every_locale_pack_in_the_package_is_offered(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """The menu is the contents of `locales/`, which is what makes a new
    language one TOML file and no code change (section 3.12)."""
    prompter = complete_run(locale="en")

    await run_setup(prompter, catalog=fake_catalog())

    assert prompter.offered["locale"] == [pack.code for pack in locales.available()]
    assert set(prompter.offered["locale"]) == {"en", "tr"}
    assert load_settings().locale.code == "en"


def test_the_wizard_speaks_the_language_it_was_told_to_last_time(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """Coming back to change the model should not mean reading English again."""
    save_settings(Settings(locale=LocaleSettings(code="tr")))

    assert wording()["model"] == locales.load("tr").say("model", TEXT["model"])
    assert wording()["model"] != TEXT["model"]


def test_the_first_run_speaks_whatever_language_windows_speaks(
    config_home: Path, vault: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing has been chosen yet, so the machine's own language is the only
    thing there is to go on."""
    monkeypatch.setattr(locales, "system_code", lambda: "tr")

    assert wording()["model"] == locales.load("tr").say("model", TEXT["model"])


def test_every_question_has_words_whatever_language_the_wizard_starts_in(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """With no answer from last time it is Windows' own language, and that may
    be one nobody has translated - which is what `TEXT` is for."""
    assert set(wording()) == set(TEXT)
    assert all(sentence.strip() for sentence in wording().values())


# --------------------------------------------------------------------------
# Giving up
# --------------------------------------------------------------------------


async def test_walking_away_at_the_last_question_writes_nothing(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """Everything is written at the end, so a wizard that was not finished
    leaves the machine exactly as it found it."""
    exit_code = await run_setup(complete_run(model=None), catalog=fake_catalog())

    assert exit_code != 0
    assert vault.vault == {}
    assert not config_path().exists()


async def test_a_settings_file_that_exists_survives_a_cancelled_run(
    config_home: Path, vault: MemoryKeyring
) -> None:
    save_settings(Settings(llm=LLMSettings(primary="gemini:fast")))

    await run_setup(complete_run(model=None), catalog=fake_catalog())

    assert load_settings().llm.primary == "gemini:fast"


async def test_a_key_that_reaches_no_model_stops_the_wizard(
    config_home: Path, vault: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key works but the account has no model; offering an empty list
    would put the user in a menu with nothing in it."""
    monkeypatch.setattr(FakeProvider, "models", [])

    exit_code = await run_setup(
        ScriptedPrompter(locale="tr", api_key=GOOD_KEY), catalog=fake_catalog()
    )

    assert exit_code != 0
    assert not config_path().exists()


# --------------------------------------------------------------------------
# The real terminal
# --------------------------------------------------------------------------


class FakeQuestion:
    """What `questionary` hands back: something with an `ask()`."""

    def __init__(self, answer: object) -> None:
        self._answer = answer

    def ask(self) -> object:
        return self._answer


def test_the_terminal_says_the_sentence_with_its_fields_filled_in(
    config_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    TerminalPrompter().say("key_url", url="https://example.test/apikey")

    assert "https://example.test/apikey" in capsys.readouterr().out


def test_the_terminal_stores_the_value_and_shows_the_label(
    config_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swap these two and the wizard writes "Fast" into config.toml."""
    seen: dict[str, object] = {}

    def select(message: str, choices: list[questionary.Choice], **kwargs: object) -> FakeQuestion:
        seen["message"] = message
        seen["titles"] = [choice.title for choice in choices]
        seen["values"] = [choice.value for choice in choices]
        return FakeQuestion(choices[0].value)

    monkeypatch.setattr(questionary, "select", select)

    answer = TerminalPrompter().choose("model", [Option("fast", "Fast"), Option("smart", "Smart")])

    assert answer == "fast"
    assert seen["values"] == ["fast", "smart"]
    assert seen["titles"] == ["Fast", "Smart"]
    assert seen["message"] == wording()["model"]


def test_the_terminal_reads_a_key_without_echoing_it(
    config_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`password` is what hides the typing; `text` would put the key on screen."""
    monkeypatch.setattr(questionary, "password", lambda message: FakeQuestion("typed-key"))
    monkeypatch.setattr(
        questionary, "text", lambda *a, **k: pytest.fail("the key must not be echoed")
    )

    assert TerminalPrompter().secret("api_key") == "typed-key"


def test_an_interrupted_question_is_not_an_answer(
    config_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl+C makes `questionary` return None, which the wizard reads as walking away."""
    monkeypatch.setattr(questionary, "password", lambda message: FakeQuestion(None))

    assert TerminalPrompter().secret("api_key") is None
