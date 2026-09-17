"""What the command line does with each command.

`setup` is wired to the wizard here rather than driven through it - the wizard
has its own suite, and a test that opened a real prompt would hang. `run` is
the same idea one level up: the state machine, the speech model and the model
that answers all have their own suites, so what is tested here is that they are
handed to each other correctly and that starting up fails in words.

**Nothing may touch the hardware.** A test that loaded Whisper would take two
seconds and a gigabyte, and one that opened the microphone would record the
room. Both are replaced; everything else is the code that ships.

**A machine that is not set up is not a crash.** No settings, or a key that has
been removed from the Credential Manager, are things the user can fix - so they
are a sentence and an exit code, and neither of them costs a model load.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar

import pytest
from loguru import logger

from assistant import app, locales, logs, setup_wizard
from assistant.__main__ import TEXT, build_parser, main, use_utf8
from assistant.agent import core
from assistant.agent.limits import Limits
from assistant.agent.policy import NO_SUCH_TOOL
from assistant.app import State, Turn
from assistant.audio import capture
from assistant.config import (
    KEYRING_SERVICE,
    AudioSettings,
    LimitSettings,
    LLMSettings,
    LocaleSettings,
    MessagingSettings,
    Settings,
    STTSettings,
    config_path,
    load_settings,
    save_settings,
    store_api_key,
)
from assistant.llm import probe
from assistant.llm.base import ProviderError, ToolCall, Usage
from assistant.llm.probe import ProbeResult, remember, remembered
from assistant.messaging.contacts import CONTACTS_FILE_NAME
from assistant.store import db
from assistant.store.memory import MEMORY_FILE_NAME
from assistant.store.repos import SettingsRepo, UsageRepo
from assistant.stt import gemini_stt, local_whisper
from assistant.tools import system
from assistant.tools.system import AppCatalog, AppEntry
from assistant.ui import status
from assistant.usage.tracker import UsageTracker
from tests.conftest import MemoryKeyring

MODEL = "gemini-2.5-flash"
TURN = Turn(heard="saat kaç", said="Üç buçuk.", usage=Usage(300, 10))
# What the probe of 2.6 answers at startup unless a test says otherwise.
PASSED = ProbeResult(ok=True, first_token_ms=12.0)


@pytest.fixture(autouse=True)
def own_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The log goes into this test's directory, not the developer's machine."""
    monkeypatch.setattr(logs, "log_path", lambda: tmp_path / "logs" / "assistant.log")
    yield
    logger.remove()


@pytest.fixture
def configured(config_home: Path, vault: MemoryKeyring) -> Path:
    """A machine `assistant setup` has already been run on."""
    save_settings(
        Settings(
            llm=LLMSettings(primary=f"gemini:{MODEL}"),
            locale=LocaleSettings(code="tr"),
        )
    )
    store_api_key("gemini", "AIza-not-a-real-key")
    return config_home


@dataclass
class Wiring:
    """What the pieces did, and what they were given, while `run` ran."""

    happened: list[str] = field(default_factory=list)
    built: list[dict[str, Any]] = field(default_factory=list)
    agents: list[tuple[str, str]] = field(default_factory=list)
    # What the agent was handed to work with: the names of the tools on
    # offer, and the gate they run through.
    tools: list[list[str]] = field(default_factory=list)
    registries: list[Any] = field(default_factory=list)
    prompts: list[Any] = field(default_factory=list)
    gates: list[Any] = field(default_factory=list)
    limits: list[Any] = field(default_factory=list)
    microphones: list[Any] = field(default_factory=list)
    databases: list[sqlite3.Connection] = field(default_factory=list)
    # What the speech model was told to expect (2.2).
    vocabularies: list[list[str]] = field(default_factory=list)
    prompts_for_speech: list[str] = field(default_factory=list)
    # Google's recogniser, when the settings ask for it: what it was built
    # with (2026-09-14).
    recognisers: list[dict[str, Any]] = field(default_factory=list)
    stop: BaseException | None = None
    # The probe of 2.6 at startup: what it was asked, as (provider id,
    # model, question), and what it answers.
    probes: list[tuple[str, str, str]] = field(default_factory=list)
    verdict: ProbeResult = field(default_factory=lambda: PASSED)
    probe_refusal: Exception | None = None


@pytest.fixture
def wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Wiring:
    """Replaces the speech model, the agent and the state machine, and points
    the database at this test's directory - the real `open_database` runs,
    so that the schema is built the way it is built in life."""
    seen = Wiring()
    really_open = db.open_database

    def open_here(path: Path | str | None = None) -> sqlite3.Connection:
        seen.happened.append("database")
        connection = really_open(path)
        seen.databases.append(connection)
        return connection

    class FakeWhisper:
        def __init__(self, *, vocabulary: Iterable[str] = (), prompt: str = "") -> None:
            seen.vocabularies.append(list(vocabulary))
            seen.prompts_for_speech.append(prompt)

        async def load(self) -> None:
            seen.happened.append("speech model")

    class FakeGemini:
        def __init__(self, api_key: str, **rest: Any) -> None:
            self.fallback = rest.get("fallback")
            seen.recognisers.append(
                {
                    "api_key": api_key,
                    "model": rest.get("model"),
                    "vocabulary": list(rest.get("vocabulary", ())),
                    "fallback": self.fallback,
                }
            )

        async def load(self) -> None:
            if self.fallback is not None:
                await self.fallback.load()
            seen.happened.append("gemini")

    async def catalogue_here(**_: Any) -> AppCatalog:
        seen.happened.append("app catalogue")
        return AppCatalog(INSTALLED)

    class FakeAgent:
        def __init__(self, provider: Any, *, model: str, **rest: Any) -> None:
            seen.agents.append((provider.id, model))
            seen.tools.append([spec.name for spec in rest["tools"].specs()])
            seen.registries.append(rest["tools"])
            seen.prompts.append(rest["system_prompt"])
            seen.gates.append(rest["dispatch"])
            seen.limits.append(rest["limits"])

    class FakeMicrophone:
        def __init__(self, *, device: Any = None) -> None:
            seen.microphones.append(device)

        def open(self, on_chunk: Any) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeAssistant:
        def __init__(self, **parts: Any) -> None:
            seen.happened.append("assistant")
            seen.built.append(parts)

        async def run(self) -> None:
            if seen.stop is not None:
                raise seen.stop
            parts = seen.built[-1]
            parts["on_state"](State.THINKING)
            parts["on_turn"](TURN)

    async def probed(provider: Any, model: str, *, question: str) -> ProbeResult:
        seen.happened.append("probe")
        seen.probes.append((provider.id, model, question))
        if seen.probe_refusal is not None:
            raise seen.probe_refusal
        return seen.verdict

    monkeypatch.setattr(probe, "probe_tool_support", probed)
    monkeypatch.setattr(local_whisper, "LocalWhisper", FakeWhisper)
    monkeypatch.setattr(gemini_stt, "GeminiSTT", FakeGemini)
    monkeypatch.setattr(system.AppCatalog, "load", catalogue_here)
    monkeypatch.setattr(capture, "SystemMicrophone", FakeMicrophone)
    monkeypatch.setattr(core, "Agent", FakeAgent)
    monkeypatch.setattr(app, "Assistant", FakeAssistant)
    monkeypatch.setattr(db, "database_path", lambda: tmp_path / "data" / "assistant.db")
    monkeypatch.setattr(db, "open_database", open_here)
    return seen


# What the machine is pretended to have, so that no test scans the real one.
INSTALLED = [
    AppEntry("Spotify", r"shell:AppsFolder\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"),
    AppEntry("Google Chrome", r"C:\Programs\Google Chrome.lnk"),
]


def said(key: str, code: str = "tr") -> str:
    """A sentence as the pack for `code` has it - whatever it was reworded to."""
    return locales.load(code).say(key, {**TEXT, **status.TEXT}[key])


# --------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "assistant" in capsys.readouterr().out


def test_no_command_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["setup", "run", "cost", "mic"])
def test_the_command_names_are_declared(command: str) -> None:
    assert build_parser().parse_args([command]).command == command


def test_telegram_login_is_a_command_of_its_own() -> None:
    """`assistant telegram login` (2026-09-15); a bare `telegram` is a usage error."""
    parsed = build_parser().parse_args(["telegram", "login"])

    assert (parsed.command, parsed.telegram_command) == ("telegram", "login")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["telegram"])


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------


def test_setup_runs_the_wizard(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    async def wizard(prompter: object, **kwargs: object) -> int:
        calls.append(prompter)
        return 0

    monkeypatch.setattr(setup_wizard, "run_setup", wizard)

    assert main(["setup"]) == 0
    assert isinstance(calls[0], setup_wizard.TerminalPrompter)


def test_the_exit_code_of_the_wizard_is_the_exit_code_of_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled setup must not look like a successful one to a script."""

    async def wizard(prompter: object, **kwargs: object) -> int:
        return 1

    monkeypatch.setattr(setup_wizard, "run_setup", wizard)

    assert main(["setup"]) == 1


# --------------------------------------------------------------------------
# mic
# --------------------------------------------------------------------------


def test_mic_asks_the_microphone_question_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """The day a headset comes out: one command, the list, and the file is
    updated - no text editor."""
    calls: list[object] = []

    async def wizard(prompter: object, **kwargs: object) -> int:
        calls.append(prompter)
        return 0

    monkeypatch.setattr(setup_wizard, "run_microphone_setup", wizard)

    assert main(["mic"]) == 0
    assert isinstance(calls[0], setup_wizard.TerminalPrompter)


def test_the_exit_code_of_mic_is_the_exit_code_of_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def wizard(prompter: object, **kwargs: object) -> int:
        return 1

    monkeypatch.setattr(setup_wizard, "run_microphone_setup", wizard)

    assert main(["mic"]) == 1


# --------------------------------------------------------------------------
# run: a machine that cannot start
# --------------------------------------------------------------------------


def test_a_machine_that_was_never_set_up_is_told_to_run_setup(
    config_home: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run"]) == 1
    assert said("not_set_up", locales.system_code()) in capsys.readouterr().out


def test_nothing_is_loaded_before_it_is_known_there_is_anything_to_run(
    config_home: Path, wiring: Wiring
) -> None:
    """Whisper is two seconds and a gigabyte. Neither is spent finding out that
    the user has not run setup."""
    main(["run"])

    assert wiring.happened == []


def test_a_key_that_is_gone_is_a_sentence_rather_than_a_traceback(
    configured: Path, vault: MemoryKeyring, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """Somebody cleaned out the Credential Manager. The provider is named,
    because that is what tells the user which key to put back."""
    vault.vault.clear()

    assert main(["run"]) == 1

    printed = capsys.readouterr().out
    assert "gemini" in printed
    assert wiring.happened == []


@pytest.mark.parametrize(
    "problem",
    [
        app.NoVoiceError("no speech voice is installed, for 'tr' or otherwise"),
        local_whisper.ModelUnavailableError("the speech model 'small' could not be loaded"),
        capture.MicrophoneUnavailableError("the microphone 'nope' could not be opened"),
    ],
    ids=["voice", "model", "microphone"],
)
def test_what_the_user_can_fix_is_a_sentence_rather_than_a_traceback(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str], problem: Exception
) -> None:
    """A voice that is not installed, weights that could not be fetched, a
    microphone that would not open. Each is the user's to fix, and a traceback
    tells them nothing about how."""
    wiring.stop = problem

    assert main(["run"]) == 1

    printed = capsys.readouterr().out
    assert str(problem) in printed
    assert "Traceback" not in printed


# --------------------------------------------------------------------------
# run: a machine that starts
# --------------------------------------------------------------------------


def test_the_model_that_answers_is_the_one_that_was_configured(
    configured: Path, wiring: Wiring
) -> None:
    assert main(["run"]) == 0
    assert wiring.agents == [("gemini", MODEL)]


def test_the_address_in_the_settings_reaches_the_provider(
    config_home: Path, vault: MemoryKeyring, wiring: Wiring
) -> None:
    """`custom` has no address in the catalogue; the one setup kept in
    `config.toml` is what the adapter is built under (2.7)."""
    save_settings(
        Settings(
            llm=LLMSettings(primary="custom:llama3", base_url="http://localhost:1234/v1"),
            locale=LocaleSettings(code="tr"),
        )
    )
    store_api_key("custom", "any")

    assert main(["run"]) == 0
    assert wiring.agents == [("openai_compat", "llama3")]


def test_a_custom_server_without_an_address_is_a_sentence_rather_than_a_traceback(
    config_home: Path, vault: MemoryKeyring, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `config.toml` edited by hand to `custom:...` with no address: the
    user can fix it, so it is a sentence that names the fix, before anything
    slow is loaded."""
    save_settings(Settings(llm=LLMSettings(primary="custom:llama3")))
    store_api_key("custom", "any")

    assert main(["run"]) == 1
    assert "assistant setup" in capsys.readouterr().out
    assert "speech model" not in wiring.happened


def test_the_language_that_was_chosen_is_the_one_it_speaks(
    configured: Path, wiring: Wiring
) -> None:
    """`config.toml` says `tr`, so the pack the state machine gets says `tr` -
    otherwise the recogniser is told to expect English (section 3.12)."""
    main(["run"])

    assert wiring.built[0]["locale"].code == "tr"


def test_the_microphone_in_the_settings_is_the_one_opened(configured: Path, wiring: Wiring) -> None:
    """`[audio] input_device` names it in `sounddevice`'s words - words rather
    than an index, because the indices shift whenever a Bluetooth device
    connects."""
    save_settings(
        Settings(
            llm=LLMSettings(primary=f"gemini:{MODEL}"),
            locale=LocaleSettings(code="tr"),
            audio=AudioSettings(input_device="Microphone Array WASAPI"),
        )
    )

    main(["run"])

    assert wiring.microphones == ["Microphone Array WASAPI"]


def test_no_microphone_in_the_settings_means_the_system_default(
    configured: Path, wiring: Wiring
) -> None:
    main(["run"])

    assert wiring.microphones == [None]


def test_the_device_flag_outranks_the_settings(configured: Path, wiring: Wiring) -> None:
    """One evening with a headset should not need the settings file edited."""
    main(["run", "--device", "9"])

    assert wiring.microphones == [9]


def test_the_speech_model_is_ready_before_the_assistant_is(
    configured: Path, wiring: Wiring
) -> None:
    """Loading Whisper at the first press would swallow the first sentence.
    The database comes first of all: cheap, and a disk that refuses is
    better found out about before two seconds of four cores are spent. The
    probe of 2.6 comes next, for the same reason: one request on the
    network, and worth knowing about before the load. The app catalogue
    comes before the speech model, which is told its names."""
    main(["run"])

    assert wiring.happened == ["database", "probe", "app catalogue", "speech model", "assistant"]


# --------------------------------------------------------------------------
# run: the tools, the gate and the database (2.1c, 2.1d, 2.2)
# --------------------------------------------------------------------------


def test_every_tool_of_phase_two_is_on_offer(configured: Path, wiring: Wiring) -> None:
    main(["run"])

    assert wiring.tools == [
        [
            "get_current_time",
            "system_status",
            "get_weather",
            "open_app",
            "open_url",
            "search_web",
            "read_clipboard",
            "fetch_page",
            "open_settings",
            "media_control",
            "play_music",
            "play_video",
            "open_media",
            "add_note",
            "search_notes",
            "delete_note",
            "create_reminder",
            "list_reminders",
            "cancel_reminder",
            "remember",
            "forget",
            "install_app",
            "send_message",
        ]
    ]


def test_a_tool_file_beside_the_settings_is_on_offer(configured: Path, wiring: Wiring) -> None:
    """Section 3.9 (13 Sep 2026): the owner's own tools, read from
    `%APPDATA%\\assistant\\tools`, join the same registry as everything else."""
    folder = configured / "tools"
    folder.mkdir()
    (folder / "mine.py").write_text(
        "from assistant.tools.registry import tool\n"
        "\n"
        "\n"
        '@tool(risk="safe")\n'
        "async def start_my_project() -> str:\n"
        '    """Starts the owner\'s project."""\n'
        '    return "started"\n',
        encoding="utf-8",
    )

    main(["run"])

    assert wiring.tools[0][-1] == "start_my_project"
    assert wiring.tools[0][:-1] == [
        "get_current_time",
        "system_status",
        "get_weather",
        "open_app",
        "open_url",
        "search_web",
        "read_clipboard",
        "fetch_page",
        "open_settings",
        "media_control",
        "play_music",
        "play_video",
        "open_media",
        "add_note",
        "search_notes",
        "delete_note",
        "create_reminder",
        "list_reminders",
        "cancel_reminder",
        "remember",
        "forget",
        "install_app",
        "send_message",
    ]


def test_the_speech_model_is_told_the_locales_sentence_and_the_apps_names(
    configured: Path, wiring: Wiring
) -> None:
    """Section 3.4: the pack's `[stt] prompt` with the catalogue's names as
    people say them (2026-09-13)."""
    main(["run"])

    assert wiring.vocabularies == [["Spotify", "Google Chrome"]]
    assert wiring.prompts_for_speech == [locales.load("tr").stt_prompt]


def test_the_people_in_the_address_book_lead_the_recognisers_names(
    configured: Path, wiring: Wiring
) -> None:
    """Spec A6 (2026-09-15): people's names are the shortest, most ambiguous
    words the recogniser hears, and there are few of them."""
    (configured / CONTACTS_FILE_NAME).write_text(
        '[[contact]]\nname = "Ahmet Yılmaz"\naliases = ["abi"]\nphone = "+90 532 000 00 00"\n',
        encoding="utf-8",
    )

    main(["run"])

    assert wiring.vocabularies == [["Ahmet Yılmaz", "abi", "Spotify", "Google Chrome"]]


def test_send_message_asks_its_question_in_the_language_of_the_pack(
    configured: Path, wiring: Wiring
) -> None:
    from assistant.tools import messaging as messaging_tools

    main(["run"])

    [registry] = wiring.registries
    send_message = registry.get("send_message")
    assert send_message is not None
    assert send_message.risk == "confirm"
    assert send_message.confirm_prompt == locales.load("tr").say(
        "send_message_confirm", messaging_tools.TEXT["send_message_confirm"]
    )
    assert send_message.confirm_prompt != messaging_tools.TEXT["send_message_confirm"]


def test_a_contacts_file_that_names_one_person_twice_is_a_sentence_before_anything_slow(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """The worst failure of messaging is a message to the wrong person; the
    file that would cause one is refused at startup, not at the first send."""
    (configured / CONTACTS_FILE_NAME).write_text(
        '[[contact]]\nname = "Ahmet"\n\n[[contact]]\nname = "ahmet"\n', encoding="utf-8"
    )

    assert main(["run"]) == 1

    assert "speech model" not in wiring.happened
    out = capsys.readouterr().out
    assert said("cannot_start", "tr").split("{")[0] in out
    assert CONTACTS_FILE_NAME in out


def test_a_default_messaging_app_that_is_not_one_is_a_sentence(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    save_settings(
        Settings(
            llm=LLMSettings(primary=f"gemini:{MODEL}"),
            locale=LocaleSettings(code="tr"),
            messaging=MessagingSettings(default_app="Signal"),
        )
    )

    assert main(["run"]) == 1
    assert "Signal" in capsys.readouterr().out


def test_by_default_the_recogniser_is_whisper_alone(configured: Path, wiring: Wiring) -> None:
    """ADR-001: local, no key, the audio stays; `[stt]` left out means that."""
    main(["run"])

    assert wiring.recognisers == []
    assert "gemini" not in wiring.happened
    assert type(wiring.built[-1]["stt"]).__name__ == "FakeWhisper"


def test_with_the_setting_the_recogniser_is_gemini_with_whisper_behind_it(
    configured: Path, wiring: Wiring
) -> None:
    """On trial (2026-09-14): Google first, the local engine loaded behind it
    for the free tier's three requests a minute; the same names, the same
    key entry the LLM uses."""
    save_settings(
        Settings(
            llm=LLMSettings(primary=f"gemini:{MODEL}"),
            locale=LocaleSettings(code="tr"),
            stt=STTSettings(provider="gemini", model="gemini-3.5-transcribe-live"),
        )
    )

    main(["run"])

    (built,) = wiring.recognisers
    assert built["api_key"] == "AIza-not-a-real-key"
    assert built["model"] == "gemini-3.5-transcribe-live"
    assert built["vocabulary"] == ["Spotify", "Google Chrome"]
    assert type(built["fallback"]).__name__ == "FakeWhisper"
    assert wiring.happened[-4:] == ["app catalogue", "speech model", "gemini", "assistant"]
    assert type(wiring.built[-1]["stt"]).__name__ == "FakeGemini"


def test_gemini_as_recogniser_without_a_gemini_key_is_one_sentence(
    configured: Path, vault: MemoryKeyring, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """The LLM is elsewhere and has its key; the recogniser's is missing.
    Named, with the way out, and nothing is loaded first."""
    save_settings(
        Settings(
            llm=LLMSettings(primary="groq:llama"),
            locale=LocaleSettings(code="tr"),
            stt=STTSettings(provider="gemini"),
        )
    )
    vault.vault.clear()
    store_api_key("groq", "gsk-not-a-real-key")

    assert main(["run"]) == 1

    printed = capsys.readouterr().out
    assert "gemini" in printed and "assistant setup" in printed
    assert "speech model" not in wiring.happened
    assert wiring.recognisers == []


def test_the_gate_the_agent_is_handed_is_the_permission_gate(
    configured: Path, wiring: Wiring
) -> None:
    """Not any callable: the one that refuses what it was not offered, which
    is the one thing a fake gate would not do."""
    main(["run"])

    [gate] = wiring.gates
    made_up = ToolCall(id="c1", name="format_disk", arguments={})

    refused = asyncio.run(gate(made_up, turn_id="t1", confirm=nobody_asked))
    assert refused == NO_SUCH_TOOL.format(name="format_disk")


async def nobody_asked(question: str) -> bool:
    raise AssertionError(f"nobody should have been asked {question!r}")


def test_who_answers_a_tool_s_question_comes_with_the_turn_and_reaches_the_gate(
    configured: Path, wiring: Wiring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is built before the state machine, so who to ask cannot be
    bound into it; each turn brings its own, and the gate hands it to
    `policy.dispatch` unchanged."""
    from assistant.agent import policy

    handed: list[Any] = []

    async def recording(call: ToolCall, **rest: Any) -> str:
        handed.append(rest["confirm"])
        return "recorded"

    monkeypatch.setattr(policy, "dispatch", recording)
    main(["run"])
    [gate] = wiring.gates
    order = ToolCall(id="c1", name="get_current_time", arguments={})

    async def says_yes(question: str) -> bool:
        return True

    assert asyncio.run(gate(order, turn_id="t1", confirm=says_yes)) == "recorded"
    assert handed == [says_yes]


def test_the_fast_path_is_handed_the_same_gate_as_the_loop(
    configured: Path, wiring: Wiring
) -> None:
    """One gate, two callers (2.5): a second gate would be a second way to
    run a tool, which is the thing section 3.9 forbids."""
    main(["run"])

    [gate] = wiring.gates
    assert wiring.built[-1]["dispatch"] is gate


# --------------------------------------------------------------------------
# run: what the user asked to be kept (2.10)
# --------------------------------------------------------------------------


def remembered_by_hand(configured: Path, body: str) -> Path:
    path = configured / MEMORY_FILE_NAME
    path.write_text(body, encoding="utf-8")
    return path


def test_what_the_user_asked_to_be_kept_is_in_front_of_every_request(
    configured: Path, wiring: Wiring
) -> None:
    """Section 3.7: the file is read at startup and the prompt the loop
    is handed carries it - behind the frozen prompt, which stays as it is."""
    from assistant.agent.prompts import SYSTEM_PROMPT

    remembered_by_hand(
        configured, '[assistant]\nname = "Ada"\n\n[user]\nfacts = ["Bana Emre de."]\n'
    )

    main(["run"])

    [source] = wiring.prompts
    prompt = source()
    assert prompt.startswith(SYSTEM_PROMPT)
    assert "Bana Emre de." in prompt
    assert "Ada" in prompt


def test_with_nothing_remembered_the_prompt_is_the_frozen_one_and_the_time(
    configured: Path, wiring: Wiring
) -> None:
    """The frozen prompt first, byte for byte, so that a provider's cache
    holds; the time last (4.2), the one line that changes between requests."""
    from assistant.agent.prompts import SYSTEM_PROMPT
    from assistant.tools.reminders import NOW_LINE

    main(["run"])

    [source] = wiring.prompts
    prompt = source()
    assert prompt.startswith(SYSTEM_PROMPT + "\n\n")
    assert prompt.rsplit("\n", 1)[1].startswith(NOW_LINE.split("{")[0])


def test_a_memory_file_that_does_not_parse_is_a_sentence_rather_than_a_traceback(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hand edit gone wrong. The user can fix it; the program must not
    write an empty file over it, and must not load a speech model first."""
    path = remembered_by_hand(configured, "[user]\nfacts = [oops\n")

    assert main(["run"]) == 1

    assert "speech model" not in wiring.happened
    assert path.read_text(encoding="utf-8") == "[user]\nfacts = [oops\n"
    assert said("cannot_start", "tr").split("{")[0] in capsys.readouterr().out


def test_forget_asks_its_question_in_the_language_of_the_pack(
    configured: Path, wiring: Wiring
) -> None:
    """The first question of phase 2 a user actually hears (2.3, 2.10)."""
    from assistant.tools import memory as memory_tools

    main(["run"])

    [registry] = wiring.registries
    forget = registry.get("forget")
    assert forget is not None
    assert forget.risk == "confirm"
    assert forget.confirm_prompt == locales.load("tr").say(
        "forget_confirm", memory_tools.TEXT["forget_confirm"]
    )
    assert forget.confirm_prompt != memory_tools.TEXT["forget_confirm"]


# --------------------------------------------------------------------------
# run: the verdict on the model (2.6)
# --------------------------------------------------------------------------


def unwrapped(printed: str) -> str:
    """What was printed, with the line breaks the eighty-column terminal
    put into a long sentence taken out again."""
    return " ".join(printed.split())


def stored_verdict() -> ProbeResult | None:
    connection = sqlite3.connect(db.database_path())
    connection.row_factory = sqlite3.Row
    try:
        return remembered(SettingsRepo(connection), "gemini", MODEL)
    finally:
        connection.close()


def write_verdict(result: ProbeResult, *, age: float) -> None:
    """A verdict `age` seconds old, as setup would have left it."""
    connection = db.open_database()
    try:
        remember(SettingsRepo(connection), "gemini", MODEL, result, now=time.time() - age)
    finally:
        connection.close()


def test_a_model_nobody_has_tested_is_probed_at_startup_and_the_verdict_kept(
    configured: Path, wiring: Wiring
) -> None:
    """Setup wrote nothing - an older build, or a database that was deleted.
    The question is asked once and the answer kept, so that the next start
    does not ask again."""
    assert main(["run"]) == 0
    assert main(["run"]) == 0

    assert [(p, m) for p, m, _ in wiring.probes] == [("gemini", MODEL)]
    assert stored_verdict() == PASSED


def test_the_probe_asks_in_the_language_of_the_pack(configured: Path, wiring: Wiring) -> None:
    main(["run"])

    [(_, _, question)] = wiring.probes
    assert question == locales.load("tr").probe_question


def test_the_model_is_checked_before_the_speech_model_is_loaded(
    configured: Path, wiring: Wiring
) -> None:
    """One request on the network, before two seconds of loading are spent
    on a model that may turn out not to call tools."""
    main(["run"])

    assert wiring.happened.index("probe") < wiring.happened.index("speech model")


def test_a_fresh_verdict_is_not_asked_again(configured: Path, wiring: Wiring) -> None:
    write_verdict(ProbeResult(ok=True, first_token_ms=800.0), age=3 * 86400)

    main(["run"])

    assert wiring.probes == []


def test_a_verdict_a_week_old_is_asked_again(configured: Path, wiring: Wiring) -> None:
    """Section 3.2: the provider may have changed what is behind the name."""
    write_verdict(ProbeResult(ok=True, first_token_ms=800.0), age=8 * 86400)

    main(["run"])

    assert len(wiring.probes) == 1
    assert stored_verdict() == PASSED


def test_a_model_that_fails_the_probe_is_a_warning_and_not_a_refusal_to_start(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """The user may have chosen it knowing; it still answers questions. But
    they are told, in their language, on a line that stays."""
    wiring.verdict = ProbeResult(ok=False, reason="no_tool_call_emitted", first_token_ms=5.0)

    assert main(["run"]) == 0

    assert "assistant" in wiring.happened
    assert said("model_no_tools") in unwrapped(capsys.readouterr().out)


def test_a_kept_verdict_that_failed_is_warned_about_on_every_start(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    write_verdict(ProbeResult(ok=False, reason="no_tool_call_emitted"), age=60)

    main(["run"])

    assert wiring.probes == []
    assert said("model_no_tools") in unwrapped(capsys.readouterr().out)


def test_a_provider_that_cannot_be_asked_at_startup_is_left_to_the_first_turn(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """Offline at startup is not a bad model. Nothing is written down, no
    warning is shown, and the first turn says what is wrong in its own
    words (`app.py`)."""
    wiring.probe_refusal = ProviderError("gemini could not be reached (ConnectError)")

    assert main(["run"]) == 0

    assert "assistant" in wiring.happened
    assert stored_verdict() is None
    assert said("model_no_tools") not in unwrapped(capsys.readouterr().out)


def test_the_database_is_built_where_the_data_lives(configured: Path, wiring: Wiring) -> None:
    main(["run"])

    path = db.database_path()
    assert path.is_file()
    connection = sqlite3.connect(path)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
        assert "tool_audit" in tables
    finally:
        connection.close()


def test_the_database_is_closed_when_the_assistant_stops(configured: Path, wiring: Wiring) -> None:
    """However it stops - here, the way Ctrl+C stops it."""
    wiring.stop = KeyboardInterrupt()

    main(["run"])

    [connection] = wiring.databases
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_a_finished_turn_is_written_to_the_log(configured: Path, wiring: Wiring) -> None:
    """Item 1.11's own sentence: the token count of every turn goes to the log."""
    main(["run"])

    assert "300 in, 10 out" in logs.log_path().read_text(encoding="utf-8")


def test_a_finished_turn_is_shown_on_screen(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["run"])

    assert "saat kaç" in capsys.readouterr().out


def test_what_the_assistant_is_doing_is_shown_on_screen(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["run"])

    assert said("state_thinking") in capsys.readouterr().out


def test_ctrl_c_is_how_it_is_meant_to_end(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """The only way to stop it in phase 1. A traceback would read as a crash."""
    wiring.stop = KeyboardInterrupt()

    assert main(["run"]) == 0
    assert said("stopped") in capsys.readouterr().out


# --------------------------------------------------------------------------
# run: the limits and the bill (2.4)
# --------------------------------------------------------------------------


def with_limits(limits: LimitSettings) -> None:
    save_settings(
        Settings(
            llm=LLMSettings(primary=f"gemini:{MODEL}"),
            locale=LocaleSettings(code="tr"),
            limits=limits,
        )
    )


def test_the_limits_in_the_settings_are_the_ones_the_loop_gets(
    configured: Path, wiring: Wiring
) -> None:
    with_limits(LimitSettings(tool_calls_per_turn=3))

    main(["run"])

    assert wiring.limits == [Limits(tool_calls_per_turn=3)]


def test_the_assistant_is_handed_a_tracker_and_the_turn_s_clock(
    configured: Path, wiring: Wiring
) -> None:
    """The tracker is what writes `usage_log`; the clock is `[limits]
    turn_seconds`, the one source of the `THINKING` timeout."""
    with_limits(LimitSettings(turn_seconds=45.0))

    main(["run"])

    parts = wiring.built[0]
    assert isinstance(parts["tracker"], UsageTracker)
    assert parts["thinking_timeout"] == 45.0


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def spent(*, cost: float | None) -> None:
    """One turn of 300 in and 10 out on the configured model, on the books."""
    connection = db.open_database()
    try:
        UsageRepo(connection).insert(
            turn_id="t1", provider="gemini", model=MODEL, usage=Usage(300, 10), cost_usd=cost
        )
    finally:
        connection.close()


def test_cost_on_a_machine_that_has_spent_nothing_says_so(
    config_home: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cost"]) == 0
    assert TEXT["cost_none"] in capsys.readouterr().out


def test_cost_shows_today_s_spending_by_model(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    spent(cost=0.0004)

    assert main(["cost"]) == 0

    printed = capsys.readouterr().out
    assert said("cost_today") in printed
    assert f"gemini:{MODEL}" in printed
    assert "$0.0004" in printed


def test_a_turn_with_no_price_is_counted_and_reported_as_unpriced(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """Never a zero that reads as free (section 6)."""
    spent(cost=None)

    main(["cost"])

    printed = capsys.readouterr().out
    assert "pricing.toml" in printed
    assert "$0.0000" not in printed


# --------------------------------------------------------------------------
# The terminal itself
# --------------------------------------------------------------------------


def test_the_terminal_is_told_to_speak_utf8() -> None:
    """Windows hands a redirected stream the machine's legacy code page, and
    half the Turkish alphabet has no place in it - which ends the program with
    a `UnicodeEncodeError` in the middle of an answer."""

    class Stream:
        def __init__(self) -> None:
            self.asked: dict[str, str] = {}

        def reconfigure(self, **asked: str) -> None:
            self.asked = asked

    stream = Stream()
    use_utf8(stream)

    assert stream.asked["encoding"] == "utf-8"


def test_the_terminal_is_told_before_anything_is_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both streams, and before the first command runs - the wizard prints
    Turkish, and so does everything `run` shows."""

    class Recording(StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.asked: dict[str, str] = {}

        def reconfigure(self, **asked: str) -> None:
            self.asked = asked

    out, errors = Recording(), Recording()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", errors)

    main([])

    assert out.asked["encoding"] == "utf-8"
    assert errors.asked["encoding"] == "utf-8"


def test_a_stream_that_cannot_be_told_is_left_alone() -> None:
    """Something else is holding stdout - a test harness, a pipe of somebody
    else's making. Not being able to ask is not a reason to refuse to start."""
    use_utf8(object())


# --------------------------------------------------------------------------
# telegram login (2026-09-15)
# --------------------------------------------------------------------------


class ScriptedTerminal:
    """Stands in for `TerminalPrompter`: scripted answers, recorded sayings."""

    answers: ClassVar[dict[str, str | None]] = {}
    said: ClassVar[list[tuple[str, dict[str, object]]]] = []
    built_text: ClassVar[dict[str, str]] = {}

    def __init__(self, *, text: Mapping[str, str] | None = None) -> None:
        self.text = dict(text or {})
        ScriptedTerminal.built_text = self.text

    def say(self, key: str, **fields: object) -> None:
        ScriptedTerminal.said.append((key, fields))
        assert key in self.text, key

    async def choose(self, key: str, options: object) -> str | None:
        raise AssertionError("the login never offers a menu")

    async def secret(self, key: str) -> str | None:
        return ScriptedTerminal.answers.get(key)

    async def ask(self, key: str) -> str | None:
        return ScriptedTerminal.answers.get(key)


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> type[ScriptedTerminal]:
    ScriptedTerminal.answers = {}
    ScriptedTerminal.said = []
    ScriptedTerminal.built_text = {}
    monkeypatch.setattr(setup_wizard, "TerminalPrompter", ScriptedTerminal)
    return ScriptedTerminal


class FakeTelegramLogin:
    """What `messaging/telegram.py`'s real client would do, without Telegram."""

    def __init__(self, api_id: int, api_hash: str) -> None:
        self.api_id, self.api_hash = api_id, api_hash

    async def connect(self) -> None:
        pass

    async def send_code(self, phone: str) -> str:
        return "code-hash"

    async def sign_in(self, phone: str, code: str, *, code_hash: str) -> str:
        if code != "12345":
            from assistant.messaging.telegram import LoginError

            raise LoginError("The phone code entered was invalid")
        return "Emre"

    async def sign_in_with_password(self, password: str) -> str:
        return "Emre"

    def session_string(self) -> str:
        return "1BVts-the-session"

    async def disconnect(self) -> None:
        pass


@pytest.fixture
def telegram_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from assistant.messaging import telegram

    monkeypatch.setattr(telegram, "_login_client", FakeTelegramLogin)


def test_telegram_login_stores_the_id_in_the_file_and_the_secrets_in_the_vault(
    configured: Path, vault: MemoryKeyring, terminal: type[ScriptedTerminal], telegram_client: None
) -> None:
    """Section 10: the `api_id` is not a secret, the hash and the session
    are. And the file is rewritten from what was loaded: the microphone the
    user chose survives (the known `setup` finding, not repeated here)."""
    save_settings(
        Settings(
            llm=LLMSettings(primary=f"gemini:{MODEL}"),
            locale=LocaleSettings(code="tr"),
            audio=AudioSettings(input_device="Microphone Array 1"),
        )
    )
    terminal.answers = {
        "telegram_api_id": "123456",
        "telegram_api_hash": "abcdef",
        "telegram_phone": "+90 532 000 00 00",
        "telegram_code": "12345",
    }

    assert main(["telegram", "login"]) == 0

    loaded = load_settings()
    assert loaded.telegram.api_id == 123456
    assert loaded.audio.input_device == "Microphone Array 1"
    assert vault.vault[(KEYRING_SERVICE, "telegram")] == "abcdef"
    assert vault.vault[(KEYRING_SERVICE, "telegram-session")] == "1BVts-the-session"
    assert "abcdef" not in config_path().read_text(encoding="utf-8")
    assert terminal.said == [("telegram_logged_in", {"name": "Emre"})]


def test_telegram_login_asks_in_the_language_of_the_pack(
    configured: Path, terminal: type[ScriptedTerminal], telegram_client: None
) -> None:
    from assistant.messaging import telegram

    terminal.answers = {"telegram_api_id": None}

    assert main(["telegram", "login"]) == 1

    # Walking away is said, and the prompter was built with the pack's
    # sentences for every question the login asks.
    assert terminal.said == [("cancelled", {})]
    turkish = locales.load("tr")
    for key, english in telegram.TEXT.items():
        assert terminal.built_text[key] == turkish.say(key, english)
        assert terminal.built_text[key] != english


def test_a_wrong_code_is_telegrams_reason_and_nothing_is_stored(
    configured: Path, vault: MemoryKeyring, terminal: type[ScriptedTerminal], telegram_client: None
) -> None:
    terminal.answers = {
        "telegram_api_id": "1",
        "telegram_api_hash": "h",
        "telegram_phone": "+90...",
        "telegram_code": "0",
    }

    assert main(["telegram", "login"]) == 1

    assert (KEYRING_SERVICE, "telegram-session") not in vault.vault
    assert load_settings().telegram.api_id == 0
    [(key, fields)] = terminal.said
    assert key == "telegram_login_failed"
    assert "phone code entered was invalid" in str(fields["reason"])
