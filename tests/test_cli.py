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

import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from assistant import app, locales, logs, setup_wizard
from assistant.__main__ import TEXT, build_parser, main, use_utf8
from assistant.agent import core
from assistant.app import State, Turn
from assistant.config import LLMSettings, LocaleSettings, Settings, save_settings, store_api_key
from assistant.llm.base import Usage
from assistant.stt import local_whisper
from assistant.ui import status
from tests.conftest import MemoryKeyring

MODEL = "gemini-2.5-flash"
TURN = Turn(heard="saat kaç", said="Üç buçuk.", usage=Usage(300, 10))


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
    stop: BaseException | None = None


@pytest.fixture
def wiring(monkeypatch: pytest.MonkeyPatch) -> Wiring:
    """Replaces the speech model, the agent and the state machine."""
    seen = Wiring()

    class FakeWhisper:
        async def load(self) -> None:
            seen.happened.append("speech model")

    class FakeAgent:
        def __init__(self, provider: Any, *, model: str, **rest: Any) -> None:
            seen.agents.append((provider.id, model))

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

    monkeypatch.setattr(local_whisper, "LocalWhisper", FakeWhisper)
    monkeypatch.setattr(core, "Agent", FakeAgent)
    monkeypatch.setattr(app, "Assistant", FakeAssistant)
    return seen


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


@pytest.mark.parametrize("command", ["setup", "run"])
def test_the_command_names_are_declared(command: str) -> None:
    assert build_parser().parse_args([command]).command == command


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


# --------------------------------------------------------------------------
# run: a machine that starts
# --------------------------------------------------------------------------


def test_the_model_that_answers_is_the_one_that_was_configured(
    configured: Path, wiring: Wiring
) -> None:
    assert main(["run"]) == 0
    assert wiring.agents == [("gemini", MODEL)]


def test_the_language_that_was_chosen_is_the_one_it_speaks(
    configured: Path, wiring: Wiring
) -> None:
    """`config.toml` says `tr`, so the pack the state machine gets says `tr` -
    otherwise the recogniser is told to expect English (section 3.12)."""
    main(["run"])

    assert wiring.built[0]["locale"].code == "tr"


def test_the_speech_model_is_ready_before_the_assistant_is(
    configured: Path, wiring: Wiring
) -> None:
    """Loading Whisper at the first press would swallow the first sentence."""
    main(["run"])

    assert wiring.happened == ["speech model", "assistant"]


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
