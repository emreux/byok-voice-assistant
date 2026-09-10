"""Command line entry point - and the composition root of the program.

`setup` asks the three questions of item 1.4. `run` is item 1.11: it puts the
pieces together, hands them to the state machine, and shows the one line of
terminal that is the entire interface until the tray icon of phase 4.2.
`cost` is 2.4: what the turns cost, read back from `usage_log`. `doctor`
arrives in phase 3 (design.md section 8).

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
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from assistant import __version__, locales
from assistant.config import Settings, is_configured, load_settings

if TYPE_CHECKING:
    from rich.table import Table

    from assistant.agent.core import Confirm, Dispatch
    from assistant.agent.limits import Limits
    from assistant.app import Turn
    from assistant.llm.base import LLMProvider, ToolCall
    from assistant.locales import Locale
    from assistant.store.repos import AuditRepo, ModelUsage, SettingsRepo
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
    # The probe of 2.6, run again at startup when its verdict is a week
    # old: a model that fails is warned about and used anyway, because the
    # user may have chosen it knowing (section 3.2).
    "model_no_tools": (
        "The model does not call tools, and most of what the assistant does depends on that. "
        "Run 'assistant setup' to choose another."
    ),
    # `assistant cost` (section 6): two small tables, today and this month,
    # one row per model. The amounts are written by the code as `$0.0004` -
    # number formatting is the locale formatter of phase 3, and until then a
    # dollar sign reads the same in every language.
    "cost_none": "No usage recorded yet.",
    "cost_today": "today",
    "cost_month": "this month",
    "cost_model": "model",
    "cost_turns": "turns",
    "cost_tokens": "in / out / cached",
    "cost_spent": "spent",
    "cost_total": "total",
    "cost_unpriced": (
        "{count} turns of {model} have no price in pricing.toml and are not in the total."
    ),
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
    subparsers.add_parser("cost", help="Show what the assistant has spent, today and this month.")

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

    if args.command == "cost":
        return _cost()

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
    from assistant.store.memory import MemoryFileError
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
    # microphone that would not open, a memory file edited into something
    # that does not parse. Each is one sentence and exit code 1. Anything
    # else is a bug in this project and keeps its traceback.
    fixable = (
        RegistryError,
        NoVoiceError,
        ModelUnavailableError,
        MicrophoneUnavailableError,
        MemoryFileError,
    )
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
    from assistant.agent.limits import Limits
    from assistant.agent.prompts import SYSTEM_PROMPT
    from assistant.app import Assistant
    from assistant.audio.capture import HandsFree, SystemMicrophone
    from assistant.audio.player import SystemSpeaker
    from assistant.audio.vad import Endpoint, SileroVAD
    from assistant.llm.registry import create_provider
    from assistant.store.db import open_database
    from assistant.store.memory import UserMemory
    from assistant.store.repos import AuditRepo, SettingsRepo, UsageRepo
    from assistant.stt.local_whisper import LocalWhisper
    from assistant.tools import memory as memory_tools
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
    from assistant.usage.tracker import Pricing, UsageTracker

    # First, and before anything slow: a provider that cannot be built is the
    # likeliest thing to be wrong, and the cheapest to find out about. The
    # database is next for the same reason - cheap, and a disk that refuses
    # is better found out about before Whisper has been loaded.
    provider = create_provider(settings.llm.provider, base_url=settings.llm.base_url or None)
    # What the user asked to be kept, and the assistant's name (section
    # 3.7, 2.10): read once here, written by the two tools below, and read
    # into the prompt at every request. Before the database for the same
    # reason as the provider - a file edited into nonsense is a sentence.
    memory = UserMemory.load()
    database = open_database()
    # The table of section 3.11, once, for everyone who reads a row of it:
    # the loop, the gate, the state machine's clock and the tracker.
    limits = Limits.from_settings(settings.limits)
    try:
        detector = SileroVAD()

        with StatusLine(pack) as screen:
            # Setup's verdict on the model, refreshed when it is a week old
            # (section 3.2, 2.6). Before the speech model: one request on
            # the network, and worth knowing about before two seconds of
            # loading are spent.
            await _model_checked(provider, settings, SettingsRepo(database), pack, screen)
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
            # `forget` is declared with the question it asks, in the pack's
            # words: the first question of phase 2 a user actually hears.
            tools = ToolRegistry(
                [
                    get_current_time,
                    open_app_for(catalog),
                    open_url,
                    open_settings,
                    media_control,
                    memory_tools.remember_for(memory),
                    memory_tools.forget_for(
                        memory,
                        confirm_prompt=pack.say(
                            "forget_confirm", memory_tools.TEXT["forget_confirm"]
                        ),
                    ),
                ]
            )
            # Loading Whisper takes seconds of four cores. Doing it now rather
            # than at the first press is what keeps the first sentence from
            # waiting for it (item 1.6). The detector is a tenth of a second
            # beside it, and is loaded here for the same reason rather than
            # inside the first block of audio it is asked about.
            await speech.load()
            await detector.load()

            # The one gate, built once and handed to both places a tool is
            # run from: the loop, and the fast path of 2.5 (section 4). A
            # second gate would be a second way to run a tool, which is the
            # thing section 3.9 forbids.
            gate = _gate(settings, tools, AuditRepo(database), limits=limits, pack=pack)
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
                    # The frozen prompt with the user's facts behind it, read
                    # at every request so that a fact just kept is in the
                    # next one; `prompts.py` stays without an import.
                    system_prompt=lambda: memory.prompt(SYSTEM_PROMPT),
                    tools=tools,
                    dispatch=gate,
                    limits=limits,
                ),
                tts=SapiTTS(),
                speaker=SystemSpeaker(),
                locale=pack,
                thinking_timeout=limits.turn_seconds,
                on_state=screen.state,
                on_turn=_finished(screen),
                # Every turn's tokens, priced, to `usage_log`: what `assistant
                # cost` reads and what the spending limits are checked against.
                tracker=UsageTracker(
                    UsageRepo(database),
                    Pricing.load(),
                    provider=settings.llm.provider,
                    model=settings.llm.model,
                    limits=limits,
                ),
                dispatch=gate,
            )
            await assistant.run()
    finally:
        database.close()


async def _model_checked(
    provider: LLMProvider,
    settings: Settings,
    verdicts: SettingsRepo,
    pack: Locale,
    screen: StatusLine,
) -> None:
    """Makes sure there is a verdict on the model that answers, and says so
    on screen when it is a bad one.

    Setup wrote one; a week later it is asked again here, because the
    provider may have changed what is behind the name (section 3.2). A model
    that fails is a warning and not a refusal to start: the user may have
    chosen it knowing, and it still answers questions. A provider that
    cannot be asked right now is left to the first turn, which has its own
    sentences for that (`app.py`) - and the verdict there was is kept.
    """
    from loguru import logger

    from assistant.llm import probe
    from assistant.llm.base import ProviderError

    provider_id, model = settings.llm.provider, settings.llm.model
    verdict = probe.remembered(verdicts, provider_id, model)
    if verdict is None:
        screen.checking_model()
        try:
            verdict = await probe.probe_tool_support(
                provider, model, question=pack.probe_question or probe.QUESTION
            )
        except ProviderError as refusal:
            logger.warning("the model could not be checked at startup: {why}", why=refusal)
            return
        probe.remember(verdicts, provider_id, model, verdict)
        logger.info(
            "probe {provider}:{model}: ok={ok}, first token {ms} ms",
            provider=provider_id,
            model=model,
            ok=verdict.ok,
            ms=None if verdict.first_token_ms is None else round(verdict.first_token_ms),
        )

    if not verdict.ok:
        screen.notice(pack.say("model_no_tools", TEXT["model_no_tools"]))


def _gate(
    settings: Settings, tools: ToolRegistry, audit: AuditRepo, *, limits: Limits, pack: Locale
) -> Dispatch:
    """The one permission gate, with everything it needs already in hand.

    What the loop gets is a function of the call alone; the registry, the
    audit rows, the user's `[tools]` settings, the look-back window of
    section 3.11 and the gate's own sentences in the user's language are
    bound here, so that `agent/core.py` never imports `policy.py` and a
    test can hand it a fake. Who to ask is not bound here: it comes with
    each turn, because it is the state machine's own microphone, and the
    state machine is built after the gate.
    """
    from assistant.agent import policy

    wording = {key: pack.say(key, default) for key, default in policy.TEXT.items()}

    async def dispatch(call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        return await policy.dispatch(
            call,
            turn_id=turn_id,
            registry=tools,
            confirm=confirm,
            unblocked=settings.tools.unblocked,
            audit=audit,
            wording=wording,
            duplicate_window=limits.duplicate_window_sec,
        )

    return dispatch


# --------------------------------------------------------------------------
# assistant cost
# --------------------------------------------------------------------------


def _cost() -> int:
    """What the assistant has spent, today and this month, by model (section 6).

    Read back from `usage_log`, where every turn was written with its price
    at the time; nothing is recomputed, so a price edited today does not
    rewrite last week. A model with no price in `pricing.toml` is counted
    and not billed, and the report says so rather than show a zero.
    """
    from rich.console import Console

    from assistant.store.db import open_database
    from assistant.store.repos import UsageRepo
    from assistant.usage.tracker import start_of_day, start_of_month

    pack = locales.load(load_settings().locale.code)
    said = {key: pack.say(key, default) for key, default in TEXT.items()}
    console = Console()

    database = open_database()
    try:
        usage = UsageRepo(database)
        now = time.time()
        ever = usage.by_model_since(0)
        today = usage.by_model_since(start_of_day(now))
        month = usage.by_model_since(start_of_month(now))
    finally:
        database.close()

    if not ever:
        console.print(said["cost_none"], markup=False, highlight=False)
        return _OK

    console.print(_cost_table(said["cost_today"], today, said))
    console.print(_cost_table(said["cost_month"], month, said))
    for row in month:
        if row.unpriced:
            line = said["cost_unpriced"].format(count=row.unpriced, model=_model(row))
            console.print(line, markup=False, highlight=False)
    return _OK


def _cost_table(title: str, rows: Sequence[ModelUsage], said: Mapping[str, str]) -> Table:
    """One period: a row per model, and a total."""
    from rich.table import Table

    table = Table(title=title, title_justify="left")
    table.add_column(said["cost_model"])
    table.add_column(said["cost_turns"], justify="right")
    table.add_column(said["cost_tokens"], justify="right")
    table.add_column(said["cost_spent"], justify="right")
    for row in rows:
        table.add_row(
            _model(row),
            str(row.turns),
            f"{row.input_tokens} / {row.output_tokens} / {row.cached_tokens}",
            _dollars(row.cost_usd),
        )

    priced = [row.cost_usd for row in rows if row.cost_usd is not None]
    table.add_row(
        said["cost_total"],
        str(sum(row.turns for row in rows)),
        " / ".join(
            str(sum(tokens))
            for tokens in (
                (row.input_tokens for row in rows),
                (row.output_tokens for row in rows),
                (row.cached_tokens for row in rows),
            )
        ),
        _dollars(sum(priced) if priced else None),
        style="bold",
    )
    return table


def _model(row: ModelUsage) -> str:
    """`gemini:gemini-3.5-flash-lite` - the way `config.toml` writes it."""
    return f"{row.provider}:{row.model}"


def _dollars(amount: float | None) -> str:
    """`$0.0004`, or a question mark for turns whose price is not known."""
    return "?" if amount is None else f"${amount:.4f}"


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
