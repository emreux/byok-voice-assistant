"""Command line entry point - and the composition root of the program.

`setup` asks the three questions of item 1.4. `run` is item 1.11: it puts the
pieces together, hands them to the state machine, and shows the one line of
terminal that is the entire interface until the tray icon of phase 4.2.
`doctor` and `cost` arrive in phase 3 and phase 2.4 (design.md section 8).

This is the only file that knows the concrete names: which tools are on
offer, which gate runs them, where the audit rows go. `agent/core.py` sees a
registry and a gate, `agent/policy.py` sees a registry and a repository, and
neither knows what the other is called - which is what lets a test hand
either of them a fake.

Three things about `run` are decisions rather than plumbing.

**Everything heavy is imported inside the function that needs it.** `assistant
--help` should not load PortAudio, a speech model and a vendor SDK to print
four lines of help - and `run` should not load the wizard's prompt library it
will never show.

**Starting up fails in sentences.** A machine nobody has run setup on, a key
that has since been deleted from the Credential Manager, a voice that is not
installed, a speech model that could not be fetched, a microphone that would
not open: these are things the user can fix, and each gets a sentence and an
exit code - the first two before anything slow is loaded. Anything else is a
bug in this project and comes out as a traceback, for the same reason `app.py`
refuses to say "I could not connect" about one.

**The terminal is told to speak UTF-8 first.** A redirected stream gets the
machine's legacy code page from Windows, which has no `ş` and no `ğ` in it -
and the assistant would end a turn with a `UnicodeEncodeError` instead of an
answer. That is fixed here rather than by asking the user to set an environment
variable before starting their own assistant.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from loguru import logger

from assistant import __version__, locales
from assistant.config import Settings, is_configured, load_settings

if TYPE_CHECKING:
    from assistant.agent.core import Dispatch
    from assistant.app import Turn
    from assistant.llm.base import ToolCall
    from assistant.locales import Locale
    from assistant.store.repos import AuditRepo
    from assistant.tools.registry import ToolRegistry
    from assistant.ui.status import StatusLine

__all__ = ["TEXT", "build_parser", "main", "use_utf8"]

_OK = 0
_GAVE_UP = 1

# The last link of the chain of section 3.12, as in every other module that
# says something: the pack answers first, and these are what is left if none
# does. Keys are unique across the project - `test_locales.py` checks.
TEXT: dict[str, str] = {
    "not_set_up": "Nothing is set up yet. Run 'assistant setup' first.",
    "cannot_start": "The assistant cannot start: {problem}",
    "stopped": "Stopped.",
}


def build_parser() -> argparse.ArgumentParser:
    """Builds the top level argument parser."""
    parser = argparse.ArgumentParser(
        prog="assistant",
        description="A voice assistant that runs on your own API key, model and language.",
    )
    parser.add_argument("--version", action="version", version=f"assistant {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.add_parser("setup", help="Choose a provider, store the API key, pick a model.")
    run = subparsers.add_parser("run", help="Start the assistant and watch for the hotkeys.")
    run.add_argument(
        "--device",
        default=None,
        help=(
            "Input device: an index, or words from its name as "
            "'scripts/bench_mic.py --list-devices' prints them. "
            "Overrides [audio] input_device in config.toml."
        ),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parses the arguments and dispatches to a command."""
    use_utf8(sys.stdout)
    use_utf8(sys.stderr)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "setup":
        from assistant import setup_wizard

        # Looked up on the module rather than imported by name so the tests can
        # stand in for it; the wizard itself opens a prompt and would hang.
        return asyncio.run(setup_wizard.run_setup(setup_wizard.TerminalPrompter()))

    return _run(device=args.device)


def use_utf8(stream: object) -> None:
    """Asks a stream to stop encoding in the machine's legacy code page.

    A console Windows owns is already fine; a redirected one - `assistant run >
    run.log`, or anything that reads our output through a pipe - is not, and
    the first Turkish sentence ends the program. Something that cannot be asked
    is left alone: not being able to is no reason to refuse to start.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return

    with contextlib.suppress(OSError, ValueError):
        reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# assistant run
# --------------------------------------------------------------------------


def _run(*, device: str | None = None) -> int:
    """Starts the assistant, or says why it cannot.

    `device` is the `--device` flag: a microphone by index or by words from its
    name, outranking the settings for this one run.
    """
    from rich.console import Console

    from assistant.app import NoVoiceError
    from assistant.audio.capture import MicrophoneUnavailableError, device_choice
    from assistant.llm.registry import RegistryError
    from assistant.logs import setup_logging
    from assistant.stt.local_whisper import ModelUnavailableError

    settings = load_settings()
    ready = is_configured()

    # Before setup there is no chosen language to read this in, so the machine's
    # own is the best guess available - the same one the wizard starts with.
    pack = locales.load(settings.locale.code if ready else locales.system_code())
    console = Console()
    said = {key: pack.say(key, default) for key, default in TEXT.items()}

    def say(key: str, **fields: object) -> None:
        # Without markup: the text of an exception is formatted into one of
        # these, and a square bracket in it is not a colour.
        console.print(said[key].format(**fields), markup=False, highlight=False)

    if not ready:
        say("not_set_up")
        return _GAVE_UP

    setup_logging()
    # What the user can fix and the program cannot: a key that is gone, a
    # voice that is not installed, weights that could not be fetched, a
    # microphone that would not open. Each is one sentence and exit code 1.
    # Anything else is a bug in this project and keeps its traceback.
    fixable = (RegistryError, NoVoiceError, ModelUnavailableError, MicrophoneUnavailableError)
    # The flag for one evening with a headset; the settings for every other
    # day; the system default when neither says anything.
    microphone = device_choice(device if device is not None else settings.audio.input_device)
    try:
        asyncio.run(_talk(settings, pack, device=microphone))
    except KeyboardInterrupt:
        # The only way to stop it in phase 1, so it is an ending rather than a
        # crash - and the microphone and the keyboard hook are already closed
        # by the time this is printed (`app.run`).
        say("stopped")
    except fixable as problem:
        say("cannot_start", problem=problem)
        return _GAVE_UP

    return _OK


async def _talk(settings: Settings, pack: Locale, *, device: int | str | None = None) -> None:
    """Builds the pieces and lets the state machine drive them."""
    from assistant.agent.core import Agent
    from assistant.app import Assistant
    from assistant.audio.capture import HandsFree, SystemMicrophone
    from assistant.audio.player import SystemSpeaker
    from assistant.audio.vad import Endpoint, SileroVAD
    from assistant.llm.registry import create_provider
    from assistant.store.db import open_database
    from assistant.store.repos import AuditRepo
    from assistant.stt.local_whisper import LocalWhisper
    from assistant.tools.media import media_control
    from assistant.tools.registry import ToolRegistry
    from assistant.tools.system import (
        PROMPT_NAMES,
        AppCatalog,
        get_current_time,
        open_app_for,
        open_settings,
        open_url,
    )
    from assistant.tts.sapi import SapiTTS
    from assistant.ui.status import StatusLine

    # First, and before anything slow: a provider that cannot be built is the
    # likeliest thing to be wrong, and the cheapest to find out about. The
    # database is next for the same reason - cheap, and a disk that refuses
    # is better found out about before Whisper has been loaded.
    provider = create_provider(settings.llm.provider)
    database = open_database()
    try:
        detector = SileroVAD()

        with StatusLine(pack) as screen:
            screen.starting()
            # The apps this machine can open, read once: a few seconds of
            # files and a PowerShell process, on a thread (2.2). Before the
            # speech model, because the model is told the names it will hear.
            catalog = await AppCatalog.load()
            speech = LocalWhisper(
                vocabulary=[pack.stt_vocabulary, *catalog.vocabulary(limit=PROMPT_NAMES)]
            )
            # The tools on offer, by name, in one place. Every one of them
            # runs through the gate below and nowhere else (section 3.9).
            tools = ToolRegistry(
                [get_current_time, open_app_for(catalog), open_url, open_settings, media_control]
            )
            # Loading Whisper takes seconds of four cores. Doing it now rather
            # than at the first press is what keeps the first sentence from
            # waiting for it (item 1.6). The detector is a tenth of a second
            # beside it, and is loaded here for the same reason rather than
            # inside the first block of audio it is asked about.
            await speech.load()
            await detector.load()

            assistant = Assistant(
                capture=HandsFree(
                    microphone=SystemMicrophone(device=device),
                    endpoint=Endpoint(detector),
                    on_mode=screen.hands_free,
                ),
                stt=speech,
                agent=Agent(
                    provider,
                    model=settings.llm.model,
                    tools=tools,
                    dispatch=_gate(settings, tools, AuditRepo(database)),
                ),
                tts=SapiTTS(),
                speaker=SystemSpeaker(),
                locale=pack,
                on_state=screen.state,
                on_turn=_finished(screen),
            )
            await assistant.run()
    finally:
        database.close()


def _gate(settings: Settings, tools: ToolRegistry, audit: AuditRepo) -> Dispatch:
    """The one permission gate, with everything it needs already in hand.

    What the loop gets is a function of the call alone; the registry, the
    audit rows and the user's `[tools]` settings are bound here, so that
    `agent/core.py` never imports `policy.py` and a test can hand it a fake.
    """
    from assistant.agent import policy

    async def dispatch(call: ToolCall, *, turn_id: str) -> str:
        return await policy.dispatch(
            call,
            turn_id=turn_id,
            registry=tools,
            confirm=_declines_until_phase_2_3,
            unblocked=settings.tools.unblocked,
            audit=audit,
        )

    return dispatch


async def _declines_until_phase_2_3(question: str) -> bool:
    """Nobody can be asked yet, so the answer is no.

    The `CONFIRMING` state of 2.3 is what opens the microphone for a yes or
    no. Until then a `confirm` tool cannot run - and cannot quietly run
    either, which is why this is a refusal and not a placeholder yes.
    """
    logger.info("a tool asked for confirmation; nobody can answer before phase 2.3")
    return False


def _finished(screen: StatusLine) -> Callable[[Turn], None]:
    """What happens to a turn once it is over: the numbers to the log, the
    words to the screen. Neither one keeps both (`logs.py`)."""
    from assistant.logs import log_turn

    def turn(finished: Turn) -> None:
        log_turn(finished)
        screen.turn(finished)

    return turn


if __name__ == "__main__":
    raise SystemExit(main())
