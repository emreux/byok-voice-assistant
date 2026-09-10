"""The three questions of `assistant setup`, without a terminal.

The wizard is the only place where a key the user typed exists in memory, so
the tests that matter are about what happens to it: it is checked before it is
kept, a key that does not work is never written anywhere, and it is never
printed back. The rest - which provider, which model, which language - is
ordinary bookkeeping.

Nothing here draws on a screen. `run_setup` talks to a `Prompter`, and this
suite hands it a scripted one, so a wizard run is a function call with a
recorded transcript rather than a session someone has to sit through.

Since 2.6 the wizard also sends the chosen model one request to see whether
it calls a tool, and refuses one that does not. The fake provider below
answers that request from a class attribute, so a test can say which of
its models call tools and which only talk.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import pytest
import questionary
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

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
from assistant.llm.base import Delta, Message, ModelInfo, ProviderError, ToolCall, ToolSpec
from assistant.llm.probe import NO_TOOL_CALL, QUESTION, ProbeResult, remembered
from assistant.llm.registry import ADAPTERS, ProviderEntry
from assistant.setup_wizard import TEXT, Option, TerminalPrompter, run_setup, wording
from assistant.store import db
from assistant.store.db import open_database
from assistant.store.repos import SettingsRepo
from tests.conftest import MemoryKeyring

GOOD_KEY = "good-key"
# A key checked while the network is down: the provider cannot say whether it
# works, and the adapter reports that as a refusal of the request, not the key.
OFFLINE_KEY = "offline"


class FakeProvider:
    """A provider that accepts one key, offers two models, and answers the
    probe of 2.6 for each of them as the class attributes say: a model in
    `tool_callers` calls the clock, one in `refusing` makes the provider
    refuse the request, any other only talks."""

    id = "fake"
    models: ClassVar[list[ModelInfo]] = [
        ModelInfo(id="fast", display_name="Fast"),
        ModelInfo(id="smart", display_name="Smart"),
    ]
    tool_callers: ClassVar[set[str]] = {"fast", "smart"}
    refusing: ClassVar[set[str]] = set()

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        # What the probe asked, as (model, question, tool names).
        self.probed: list[tuple[str, str, list[str]]] = []

    async def validate_credentials(self) -> bool:
        if self.api_key == OFFLINE_KEY:
            raise ProviderError("fake could not be reached (ConnectError)")
        return self.api_key == GOOD_KEY

    async def list_models(self) -> list[ModelInfo]:
        return list(self.models)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        self.probed.append((model, messages[-1].content, [tool.name for tool in tools]))
        if model in self.refusing:
            raise ProviderError("fake refused the request (429): slow down")
        if model in self.tool_callers:
            yield Delta(
                tool_call=ToolCall(id="c1", name="get_current_time", arguments={"city": "x"})
            )
        else:
            yield Delta(text="It is about three.")
        yield Delta(finish_reason="stop")


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

    async def choose(self, key: str, options: Sequence[Option]) -> str | None:
        self.offered[key] = [option.value for option in options]
        return self._answer(key)

    async def secret(self, key: str) -> str | None:
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
def fake_adapter(monkeypatch: pytest.MonkeyPatch) -> list[FakeProvider]:
    """Registers an adapter that answers without a network, and keeps every
    provider it built so a test can read what the wizard asked of it."""
    built: list[FakeProvider] = []

    def build(entry: ProviderEntry, api_key: str) -> FakeProvider:
        provider = FakeProvider(api_key)
        built.append(provider)
        return provider

    monkeypatch.setitem(ADAPTERS, "fake", build)
    return built


@pytest.fixture(autouse=True)
def own_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The wizard opens the machine's database to write the verdict on the
    model (2.6). Here that is a file under this test's directory."""
    path = tmp_path / "data" / "assistant.db"
    monkeypatch.setattr(db, "database_path", lambda: path)
    return path


@pytest.fixture
def verdicts() -> Iterator[sqlite3.Connection]:
    """A database handed to the wizard, so that what it wrote can be read."""
    connection = open_database(":memory:")
    yield connection
    connection.close()


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
        "claude": ProviderEntry(id="claude", adapter="anthropic", display_name="Claude"),
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


async def test_a_provider_that_cannot_be_reached_is_not_called_a_bad_key(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """Offline during setup. "That key did not work" would send the user to
    the provider's console to replace a key that is fine; the wizard says
    what actually happened and asks again."""
    prompter = complete_run(api_key=[OFFLINE_KEY, GOOD_KEY])

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    said = [key for key, _ in prompter.said]
    assert exit_code == 0
    assert "provider_unreachable" in said
    assert "bad_key" not in said
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}


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
# The tool-use probe (2.6)
# --------------------------------------------------------------------------


async def test_a_model_that_does_not_call_tools_cannot_be_chosen(
    config_home: Path, vault: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The most valuable thirty lines of section 3.2: the model is offered
    again until one that calls the tool is picked, and only that one is
    written down."""
    monkeypatch.setattr(FakeProvider, "tool_callers", {"smart"})
    prompter = complete_run(model=["fast", "smart"])

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    said = [key for key, _ in prompter.said]
    assert exit_code == 0
    assert prompter.asked.count("model") == 2
    assert said.count("tools_failed") == 1
    assert said.count("tools_ok") == 1
    assert load_settings().llm.primary == "gemini:smart"


async def test_a_model_that_calls_tools_is_taken_at_the_first_answer(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    said = [key for key, _ in prompter.said]
    assert prompter.asked.count("model") == 1
    assert "tools_failed" not in said
    assert said.index("probing_tools") < said.index("tools_ok") < said.index("saved")


async def test_the_first_token_time_is_shown_as_a_whole_number(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    [fields] = [fields for key, fields in prompter.said if key == "tools_ok"]
    assert str(fields["ms"]).isdigit()


async def test_the_probe_asks_in_the_language_that_was_just_chosen(
    config_home: Path, vault: MemoryKeyring, fake_adapter: list[FakeProvider]
) -> None:
    """Section 3.2 wanted the question in the user's language, so that one
    request checks tool calling and understanding together; the pack is
    where the language comes from (section 3.12)."""
    await run_setup(complete_run(locale="tr"), catalog=fake_catalog())

    [(model, question, tools)] = fake_adapter[-1].probed
    assert model == "fast"
    assert question == locales.load("tr").probe_question
    assert question != QUESTION
    assert tools == ["get_current_time"]


async def test_the_probe_falls_back_to_the_english_question(
    config_home: Path, vault: MemoryKeyring, fake_adapter: list[FakeProvider]
) -> None:
    """`en.toml` carries no question; the constant beside the code asks."""
    await run_setup(complete_run(locale="en"), catalog=fake_catalog())

    [(_, question, _)] = fake_adapter[-1].probed
    assert question == QUESTION


async def test_the_verdict_on_the_chosen_model_is_written_down(
    config_home: Path, vault: MemoryKeyring, verdicts: sqlite3.Connection
) -> None:
    """So that `assistant run` need not ask the same question for a week."""
    await run_setup(complete_run(model="smart"), catalog=fake_catalog(), database=verdicts)

    found = remembered(SettingsRepo(verdicts), "gemini", "smart")
    assert found is not None
    assert found.ok is True
    assert remembered(SettingsRepo(verdicts), "gemini", "fast") is None


async def test_the_wizard_opens_the_machine_s_database_when_none_is_handed_over(
    config_home: Path, vault: MemoryKeyring, own_database: Path
) -> None:
    """On a fresh machine the wizard is the first thing to touch the
    database, and it has to build it - with its schema - to write into it."""
    await run_setup(complete_run(), catalog=fake_catalog())

    connection = open_database(own_database)
    try:
        found = remembered(SettingsRepo(connection), "gemini", "fast")
    finally:
        connection.close()
    assert found is not None
    assert found.ok is True


async def test_a_provider_that_refuses_the_probe_has_said_nothing_about_the_model(
    config_home: Path, vault: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate limit or a bad day is not "this model cannot call tools":
    that sentence would send the user away from a model that is fine. The
    provider's own words are shown and the list is offered again."""
    monkeypatch.setattr(FakeProvider, "refusing", {"smart"})
    prompter = complete_run(model=["smart", "fast"])

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    said = [key for key, _ in prompter.said]
    refusals = [fields for key, fields in prompter.said if key == "probe_refused"]
    assert exit_code == 0
    assert "tools_failed" not in said
    assert "slow down" in str(refusals[0]["problem"])
    assert load_settings().llm.primary == "gemini:fast"


async def test_walking_away_from_the_probe_s_verdict_writes_nothing(
    config_home: Path,
    vault: MemoryKeyring,
    monkeypatch: pytest.MonkeyPatch,
    verdicts: sqlite3.Connection,
) -> None:
    """Every model the key reaches only talks; the user gives up at the
    second question. Nothing is stored - not the key, not the settings,
    not the failed verdict."""
    monkeypatch.setattr(FakeProvider, "tool_callers", set())
    prompter = complete_run(model=["fast", None])

    exit_code = await run_setup(prompter, catalog=fake_catalog(), database=verdicts)

    assert exit_code != 0
    assert vault.vault == {}
    assert not config_path().exists()
    assert verdicts.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0


def test_the_failed_verdict_has_the_reason_of_section_3_2() -> None:
    """What the probe writes down is what `assistant run` will read a week
    later; the word is the one the design names."""
    assert ProbeResult(ok=False, reason=NO_TOOL_CALL).reason == "no_tool_call_emitted"


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
    """What `questionary` hands back: something with an `ask_async()`."""

    def __init__(self, answer: object) -> None:
        self._answer = answer

    async def ask_async(self) -> object:
        return self._answer


def test_the_terminal_says_the_sentence_with_its_fields_filled_in(
    config_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    TerminalPrompter().say("key_url", url="https://example.test/apikey")

    assert "https://example.test/apikey" in capsys.readouterr().out


async def test_the_terminal_stores_the_value_and_shows_the_label(
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

    answer = await TerminalPrompter().choose(
        "model", [Option("fast", "Fast"), Option("smart", "Smart")]
    )

    assert answer == "fast"
    assert seen["values"] == ["fast", "smart"]
    assert seen["titles"] == ["Fast", "Smart"]
    assert seen["message"] == wording()["model"]


async def test_the_terminal_reads_a_key_without_echoing_it(
    config_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`password` is what hides the typing; `text` would put the key on screen."""
    monkeypatch.setattr(questionary, "password", lambda message: FakeQuestion("typed-key"))
    monkeypatch.setattr(
        questionary, "text", lambda *a, **k: pytest.fail("the key must not be echoed")
    )

    assert await TerminalPrompter().secret("api_key") == "typed-key"


async def test_an_interrupted_question_is_not_an_answer(
    config_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl+C makes `questionary` return None, which the wizard reads as walking away."""
    monkeypatch.setattr(questionary, "password", lambda message: FakeQuestion(None))

    assert await TerminalPrompter().secret("api_key") is None


# --------------------------------------------------------------------------
# The terminal the wizard really runs in
#
# Everything above hands the wizard a scripted prompter, which is the right
# way to test what it asks and what it does with the answers - and is exactly
# why the terminal itself went untested until somebody ran `assistant setup`.
# These two drive the real one, through a pipe instead of a keyboard.
# --------------------------------------------------------------------------


@contextmanager
def typed(keys: str) -> Iterator[None]:
    """A terminal that answers with `keys` and draws nowhere."""
    with create_pipe_input() as keyboard:
        keyboard.send_text(keys)
        with create_app_session(input=keyboard, output=DummyOutput()):
            yield


async def test_a_question_can_be_asked_from_inside_the_event_loop() -> None:
    """`run_setup` is a coroutine, so every prompt is drawn while an event loop
    is already running. `questionary`'s synchronous `ask` starts a second one
    and Python refuses outright - which no scripted prompter can find out."""
    prompter = TerminalPrompter(text={"locale": "Which language?"})

    with typed("\r"):
        chosen = await prompter.choose("locale", [Option("tr", "Türkçe"), Option("en", "English")])

    assert chosen == "tr"


async def test_a_key_can_be_typed_from_inside_the_event_loop() -> None:
    """The same for the one question whose answer is a secret."""
    prompter = TerminalPrompter(text={"api_key": "Paste your API key"})

    with typed("a-key\r"):
        assert await prompter.secret("api_key") == "a-key"
