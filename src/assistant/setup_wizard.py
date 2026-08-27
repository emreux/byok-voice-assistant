"""`assistant setup` - the three questions phase 1 asks (design.md section 3.3).

Provider, key, model, and the language the assistant speaks. The fourteen step
wizard of section 3.3 - device tests, tool probe, latency measurement, a
fallback model - is phase 4.5; what is here is the smallest thing that can
produce a working `config.toml`.

Two rules shape the code more than the questions do.

**Nothing is written until the wizard finishes.** The key is checked against
the provider before it is kept, and both the key and the settings are stored in
the last step. A run that was abandoned, or a key that turned out to be
revoked, leaves the machine exactly as it was found.

**No user-facing sentence lives in this module.** The wizard asks for a
question by key; the `Prompter` turns that key into words. Today the words come
from `TEXT` below - English, which section 3.12 names as the end of the
fallback chain - and item 1.8 puts `locales/<code>.toml` in front of it without
touching a line of the logic. That is also why the tests can script a wizard
run without repeating a single sentence.

Phase 2 adds what the catalogue already has room for: a provider that needs no
key at all (Ollama) and one that needs a `base_url`. Both are questions about
a `ProviderEntry` field, so they arrive with the adapter that makes them real.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import questionary
from rich.console import Console

from assistant.config import (
    LLMSettings,
    LocaleSettings,
    Settings,
    load_api_key,
    save_settings,
    store_api_key,
)
from assistant.llm.base import LLMProvider, ModelInfo
from assistant.llm.registry import ADAPTERS, ProviderEntry, create_provider, load_catalog

__all__ = ["TEXT", "Option", "Prompter", "TerminalPrompter", "run_setup"]

_OK = 0
_GAVE_UP = 1


@dataclass(frozen=True, slots=True)
class Option:
    """One answer to a question: what gets stored, and what the user reads.

    The label is data - a provider's display name, a model's name, a language's
    name in that language - so it is not part of the text table.
    """

    value: str
    label: str


class Prompter(Protocol):
    """Everything the wizard needs from a terminal, and nothing more.

    A question is named, not worded. Keeping the wording on this side of the
    line is what lets the wizard be tested without a screen and translated
    without being rewritten.
    """

    def say(self, key: str, **fields: object) -> None: ...

    def choose(self, key: str, options: Sequence[Option]) -> str | None:
        """Returns the chosen value, or `None` if the user walked away."""
        ...

    def secret(self, key: str) -> str | None:
        """Reads a line without echoing it, or `None` if the user walked away."""
        ...


# The last link of the fallback chain of section 3.12: what the wizard says
# when no locale file offers a translation. Item 1.8 adds `locales/tr.toml` and
# the loader that prefers it.
TEXT: dict[str, str] = {
    "welcome": "The assistant answers through an AI provider, using your own API key.",
    "provider": "Which provider do you want to use?",
    "only_provider": "Provider: {name} - the only one this version can talk to.",
    "no_provider": "This version cannot build any provider in the catalogue.",
    "locale": "Which language should the assistant speak?",
    "key_url": "You can create a key at {url}",
    "api_key": "Paste your API key (nothing is shown as you type)",
    "api_key_keep": "Paste your API key, or press Enter to keep the one already stored",
    "key_needed": "This provider needs a key before it will answer.",
    "checking_key": "Checking the key...",
    "bad_key": "That key did not work - mistyped, revoked, or out of credit.",
    "loading_models": "Asking which models the key can reach...",
    "no_models": "The key works, but it reaches no model. Check the provider's console.",
    "model": "Which model should answer?",
    "saved": "Ready. Settings: {path} - the key itself is in the Windows Credential Manager.",
    "cancelled": "Setup cancelled. Nothing was changed.",
}

# The languages offered in phase 1, each written in itself. Item 1.8 replaces
# this with what `locales/` actually contains, which is how a fourth language
# becomes one TOML file and no code change (section 3.12).
LOCALES: tuple[Option, ...] = (Option("tr", "Türkçe"), Option("en", "English"))


class _WalkedAwayError(Exception):
    """The user pressed Ctrl+C, or answered nothing to a question with no default."""


async def run_setup(
    prompter: Prompter,
    *,
    catalog: Mapping[str, ProviderEntry] | None = None,
) -> int:
    """Asks the questions, then writes the answers. Returns a process exit code."""
    try:
        return await _ask(prompter, catalog)
    except _WalkedAwayError:
        prompter.say("cancelled")
        return _GAVE_UP


async def _ask(prompter: Prompter, catalog: Mapping[str, ProviderEntry] | None) -> int:
    entries = load_catalog() if catalog is None else catalog
    buildable = {
        provider_id: entry for provider_id, entry in entries.items() if entry.adapter in ADAPTERS
    }
    if not buildable:
        # A catalogue listing only adapters from later phases. Worth its own
        # sentence: the user has done nothing wrong and retrying will not help.
        prompter.say("no_provider")
        return _GAVE_UP

    prompter.say("welcome")
    provider_id, entry = _pick_provider(prompter, buildable)
    locale = _answered(prompter.choose("locale", LOCALES))

    if entry.key_url is not None:
        prompter.say("key_url", url=entry.key_url)
    provider, api_key = await _working_key(prompter, provider_id, entries)

    prompter.say("loading_models")
    models = await provider.list_models()
    if not models:
        prompter.say("no_models")
        return _GAVE_UP
    model = _answered(prompter.choose("model", [_offer(m) for m in models]))

    # Everything above could still be abandoned; from here it is written down.
    store_api_key(provider_id, api_key)
    path = save_settings(
        Settings(
            llm=LLMSettings(primary=f"{provider_id}:{model}"),
            locale=LocaleSettings(code=locale),
        )
    )
    prompter.say("saved", path=path)
    return _OK


def _pick_provider(
    prompter: Prompter, buildable: Mapping[str, ProviderEntry]
) -> tuple[str, ProviderEntry]:
    """Asks which provider - unless there is only one, which is no question."""
    if len(buildable) == 1:
        provider_id, entry = next(iter(buildable.items()))
        prompter.say("only_provider", name=entry.display_name)
        return provider_id, entry

    options = [Option(provider_id, e.display_name) for provider_id, e in buildable.items()]
    chosen = _answered(prompter.choose("provider", options))
    return chosen, buildable[chosen]


async def _working_key(
    prompter: Prompter,
    provider_id: str,
    entries: Mapping[str, ProviderEntry],
) -> tuple[LLMProvider, str]:
    """Asks for a key until the provider accepts one (section 3.3).

    A key that fails is not stored, not retried and not silently swapped for a
    fallback: mistyped, revoked and out of credit all mean the same thing to
    the user, and all three are fixed by pasting a different key.
    """
    stored = load_api_key(provider_id)

    while True:
        typed = _answered(prompter.secret("api_key_keep" if stored else "api_key"))
        api_key = typed or stored or ""
        if not api_key:
            prompter.say("key_needed")
            continue

        prompter.say("checking_key")
        provider = create_provider(provider_id, api_key=api_key, catalog=entries)
        if await provider.validate_credentials():
            return provider, api_key

        prompter.say("bad_key")
        # Whatever was in the vault has just been proved useless, so it stops
        # being offered - otherwise Enter would retry the same dead key.
        stored = None


def _offer(model: ModelInfo) -> Option:
    # The id is what lands in `config.toml`, so it is shown next to the name
    # rather than hidden behind it.
    label = model.display_name
    if model.id != model.display_name:
        label = f"{model.display_name}  ({model.id})"
    return Option(model.id, label)


def _answered(value: str | None) -> str:
    if value is None:
        raise _WalkedAwayError
    return value


class TerminalPrompter:
    """The real terminal: `rich` for what is said, `questionary` for answers."""

    def __init__(self, *, text: Mapping[str, str] | None = None) -> None:
        self._text = TEXT if text is None else text
        self._console = Console()

    def say(self, key: str, **fields: object) -> None:
        self._console.print(self._text[key].format(**fields))

    def choose(self, key: str, options: Sequence[Option]) -> str | None:
        answer = questionary.select(
            self._text[key],
            choices=[
                questionary.Choice(title=option.label, value=option.value) for option in options
            ],
            # A key can reach forty models; typing part of a name beats forty
            # arrow presses. With the filter on, j and k are letters again.
            use_search_filter=True,
            use_jk_keys=False,
        ).ask()
        return _as_answer(answer)

    def secret(self, key: str) -> str | None:
        return _as_answer(questionary.password(self._text[key]).ask())


def _as_answer(value: object) -> str | None:
    """Narrows what `questionary` returns, which is typed `Any`.

    An interrupted question comes back as `None`, and so does anything else
    unexpected - which is what the wizard reads as walking away.
    """
    return value if isinstance(value, str) else None
