"""What the assistant can do on the machine itself (design.md section 3.6).

Phase 2.1c opened this file with the smallest tool there is: `get_current_time`
answers the one question the frozen system prompt of `agent/prompts.py`
cannot, because the prompt carries no clock so that its bytes never change
and the provider's cache keeps hitting (architecture guide section 2). Phase
2.2 adds what the owner asked for first (section 2): open an app, open a
site, open a page of Settings.

**Opening an app by name is a catalogue problem.** Windows has no "start
Spotify" call. It has the shortcut files in the two Start Menu folders and,
for everything else the Start menu knows - the store apps first of all - the
shell's own list, which `Get-StartApps` prints. Both are read once at startup
into an `AppCatalog`, in a worker thread, because hundreds of files and a
PowerShell process take seconds (measured 2026-09-09: 3.4 s for the listing
alone) and nothing on the event loop may (section 3.1 rule 4).

**The name the user said is not the name in the catalogue.** "krom" is
"Google Chrome"; "IŞIK" and "isik" are one word; the recogniser misspells.
`find` folds both sides with `normalize_search` and then tries the whole
name, one word of it, a prefix, and last `difflib`'s closest match. The
language of the catalogue never matters: on a Turkish Windows the shortcut is
already called "Hesap Makinesi", on an English one "Calculator", and the code
carries neither (section 3.12).

What a tool returns is addressed to the model, not the user, so it is written
the way a model reads best - unambiguous, in one line - and in English, like
everything else that never reaches the speaker. How it is said out loud, and
in which language, is the model's job.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import webbrowser
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated

from loguru import logger

from assistant.store.normalize import normalize_search
from assistant.tools.registry import Tool, tool

__all__ = [
    "PROMPT_NAMES",
    "SETTINGS_PAGES",
    "AppCatalog",
    "AppEntry",
    "get_current_time",
    "open_app_for",
    "open_settings",
    "open_url",
    "scan_start_apps",
    "scan_start_menu",
    "start_menu_folders",
]

# How many of the catalogue's names are told to the recogniser before each
# utterance. Whisper's prompt window is about 224 tokens, so hundreds of names
# would not fit; the shortest are offered, because those are the ones people
# say. The right number is measured in 2.8 (`bench_stt.py`), not guessed here.
PROMPT_NAMES = 40

# Below this ratio `difflib` is guessing rather than matching. Measured on
# this machine (2026-09-09) against its 173 apps: at 0.6, "spotify" - not
# installed - opened Sticky Notes and "krom" the Command Prompt, because a
# short word of a name is easy to be six tenths like. At 0.7 both are left
# for the model to ask about, and a misspelling ("notpad", "kalkulator",
# "vscode") still lands. Opening the wrong app is worse than asking.
CLOSE_ENOUGH = 0.7

# The bar for a *suggestion* is lower: when nothing matched, the model is
# better off with three names that were nearly it than with none.
NEAR_ENOUGH = 0.4

# A prefix shorter than this matches too much: "a" starts half the catalogue.
PREFIX_CHARS = 3

# How long the shell's listing may take before it is given up on. Measured at
# 3.4 s; a machine ten times slower still answers.
LISTING_SECONDS = 30.0

# What the shell is asked. Its output encoding is set first: through a pipe
# PowerShell writes in the machine's legacy code page, which has no `ğ`, and a
# name like "Ekran görüntüsü geçmişi" would arrive damaged.
LISTING_COMMAND = (
    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
    "Get-StartApps | ConvertTo-Json -Compress"
)

# The folder the Start menu itself shows. An app is an item in it, by its
# `AppID`, and opening the item starts the app - store apps included, which
# have no shortcut file anywhere.
APPS_FOLDER = "shell:AppsFolder\\"

# Pages of Windows Settings: the key the model chooses, and the URI that opens
# it. The keys are English because the model maps the user's words to them
# from whatever language was spoken; `home` is the one to fall back on.
SETTINGS_PAGES: dict[str, str] = {
    "home": "ms-settings:",
    "bluetooth": "ms-settings:bluetooth",
    "wifi": "ms-settings:network-wifi",
    "network": "ms-settings:network-status",
    "display": "ms-settings:display",
    "sound": "ms-settings:sound",
    "notifications": "ms-settings:notifications",
    "power": "ms-settings:powersleep",
    "storage": "ms-settings:storagesense",
    "apps": "ms-settings:appsfeatures",
    "update": "ms-settings:windowsupdate",
    "privacy": "ms-settings:privacy",
    "about": "ms-settings:about",
}
PAGE_KEYS = "One of: " + ", ".join(SETTINGS_PAGES)

# What runs the shell, and what scans the machine. Both are parameters so a
# test can hand in a fake and no test ever starts PowerShell.
Runner = Callable[..., subprocess.CompletedProcess[str]]
Scanner = Callable[[], list["AppEntry"]]


@dataclass(frozen=True, slots=True)
class AppEntry:
    """One thing the Start menu can open: its name, and what to hand the shell."""

    name: str
    # A shortcut's path, or `shell:AppsFolder\<AppID>`; `os.startfile` opens
    # either.
    launch: str


# --------------------------------------------------------------------------
# Reading the machine
# --------------------------------------------------------------------------


def start_menu_folders() -> list[Path]:
    """The user's Start Menu programs and the machine's, where either is set."""
    roots = (os.environ.get("APPDATA"), os.environ.get("PROGRAMDATA"))
    return [Path(root, "Microsoft", "Windows", "Start Menu", "Programs") for root in roots if root]


def scan_start_menu(folders: Iterable[Path] | None = None) -> list[AppEntry]:
    """Every shortcut under the Start Menu folders, named by its file name.

    An "Uninstall Foo" shortcut is kept: it is something the user can ask
    for, and it is the model, not this scan, that decides what "open Foo"
    meant. Sorted, so that when a name occurs twice the same one wins on
    every start.
    """
    found: list[AppEntry] = []
    for folder in start_menu_folders() if folders is None else folders:
        if not folder.is_dir():
            continue
        found.extend(
            AppEntry(name=shortcut.stem, launch=str(shortcut))
            for shortcut in sorted(folder.rglob("*.lnk"))
        )
    return found


def scan_start_apps(run: Runner = subprocess.run) -> list[AppEntry]:
    """Everything the Start menu lists, asked of the shell itself.

    `Get-StartApps` is the one documented way to the store apps - there is
    no shortcut file for Calculator - and it lists the classic ones too, so
    a name can come back twice; `AppCatalog` keeps the first. Nothing here
    is fatal: no PowerShell, a PowerShell that fails, output that is not
    JSON, each leaves the catalogue with the shortcuts alone and a line in
    the log. The workbook called this `scan_uwp_apps`; it is named for what
    the listing holds.
    """
    shell = shutil.which("powershell")
    if shell is None:
        logger.warning("PowerShell was not found; the catalogue has the Start Menu shortcuts only")
        return []

    try:
        listing = run(
            [shell, "-NoProfile", "-NonInteractive", "-Command", LISTING_COMMAND],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=LISTING_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as failure:
        logger.warning("the Start menu listing could not be read: {}", failure)
        return []

    if listing.returncode != 0:
        logger.warning("the Start menu listing failed: {}", listing.stderr.strip())
        return []

    try:
        printed = json.loads(listing.stdout or "null")
    except json.JSONDecodeError as failure:
        logger.warning("the Start menu listing was not JSON: {}", failure)
        return []
    return list(_listed(printed))


def _listed(printed: object) -> Iterator[AppEntry]:
    # One app is printed as an object, several as a list of them. Anything
    # without both a name and an id is not something that can be opened.
    items = [printed] if isinstance(printed, dict) else printed if isinstance(printed, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        name, app_id = item.get("Name"), item.get("AppID")
        if isinstance(name, str) and name.strip() and isinstance(app_id, str) and app_id:
            yield AppEntry(name=name.strip(), launch=APPS_FOLDER + app_id)


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------


class AppCatalog:
    """The apps this machine can open, found by whatever the user called them."""

    def __init__(self, entries: Iterable[AppEntry]) -> None:
        self._entries: list[AppEntry] = []
        self._by_name: dict[str, AppEntry] = {}
        self._by_word: dict[str, AppEntry] = {}
        for entry in entries:
            key = normalize_search(entry.name).strip()
            # The same name twice - a shortcut and its store twin - is one
            # app, and the first one offered is the one kept.
            if not key or key in self._by_name:
                continue
            self._entries.append(entry)
            self._by_name[key] = entry
            for word in _words(key):
                self._by_word.setdefault(word, entry)

    @classmethod
    async def load(cls, *, scanners: Iterable[Scanner] | None = None) -> AppCatalog:
        """Scans the machine - off the event loop, because it takes seconds."""
        chosen = tuple(scanners) if scanners is not None else (scan_start_menu, scan_start_apps)

        def scan() -> list[AppEntry]:
            return [entry for scanner in chosen for entry in scanner()]

        catalog = cls(await asyncio.to_thread(scan))
        logger.info("app catalogue: {} apps", len(catalog))
        return catalog

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> list[str]:
        return [entry.name for entry in self._entries]

    def vocabulary(self, limit: int = PROMPT_NAMES) -> list[str]:
        """The names most worth telling the recogniser about.

        The shortest first: a short name is one people say, and the prompt
        has room for few (`PROMPT_NAMES`).
        """
        ranked = sorted(self._entries, key=lambda entry: len(entry.name))
        return [entry.name for entry in ranked[:limit]]

    def find(self, spoken: str) -> AppEntry | None:
        """The app the user meant by `spoken`, or `None`.

        In order: the whole name, one word of a name ("chrome" is Google
        Chrome), the start of either ("spot"), and the closest by
        `difflib` ("krom"). Folded on both sides, so the case, the accents
        and the recogniser's spelling of a foreign name do not decide.
        """
        wanted = normalize_search(spoken).strip()
        if not wanted:
            return None

        found = self._by_name.get(wanted) or self._by_word.get(wanted)
        if found is None and len(wanted) >= PREFIX_CHARS:
            found = next((entry for key, entry in self._keys() if key.startswith(wanted)), None)
        if found is None:
            found = self._one_close_enough(wanted)
        return found

    def closest(self, spoken: str, *, limit: int = 3) -> list[str]:
        """Names near `spoken`, for the model to offer when nothing matched."""
        wanted = normalize_search(spoken).strip()
        return [entry.name for _, entry in self._ranked(wanted, cutoff=NEAR_ENOUGH)[:limit]]

    def _one_close_enough(self, wanted: str) -> AppEntry | None:
        """The closest app when one is clearly closest, otherwise nothing.

        Two apps equally like what was said - "chrome" and "prompt" are both
        six tenths of "krom" - is a question for the user, not a coin toss.
        """
        ranked = self._ranked(wanted, cutoff=CLOSE_ENOUGH)
        if not ranked:
            return None
        (best_score, best), *others = ranked
        if others and others[0][0] == best_score:
            return None
        return best

    def _ranked(self, wanted: str, *, cutoff: float) -> list[tuple[float, AppEntry]]:
        """Every app with a key at least `cutoff` like `wanted`, the most
        alike first, each app once - scored by its best key, so that a name
        and its words do not fill the list with one app."""
        matcher = SequenceMatcher()
        matcher.set_seq2(wanted)
        best: dict[AppEntry, float] = {}
        for key, entry in self._keys():
            matcher.set_seq1(key)
            # The two cheap upper bounds first, as `get_close_matches` does;
            # the real ratio is the expensive one, and this runs on the loop.
            if matcher.real_quick_ratio() < cutoff or matcher.quick_ratio() < cutoff:
                continue
            score = matcher.ratio()
            if score >= cutoff and score > best.get(entry, 0.0):
                best[entry] = score
        return sorted(((score, entry) for entry, score in best.items()), key=lambda pair: -pair[0])

    def _keys(self) -> Iterator[tuple[str, AppEntry]]:
        yield from self._by_name.items()
        yield from self._by_word.items()


def _words(key: str) -> list[str]:
    # The key is ASCII already; a single letter is not a word anyone asks for.
    return [word for word in re.findall(r"[a-z0-9]+", key) if len(word) > 1]


# --------------------------------------------------------------------------
# The tools
# --------------------------------------------------------------------------


def _now() -> datetime:
    """The local time with its zone attached. Kept apart so a test can pin it."""
    return datetime.now().astimezone()


def _launch(target: str) -> None:
    """Hands `target` to the shell, as a double-click would. Kept apart so a
    test can see what would have opened without opening it."""
    os.startfile(target)  # noqa: S606  # the target is ours: a catalogue entry or a settings URI


def _browse(address: str) -> bool:
    """The default browser, on `address`. Kept apart for the same reason."""
    return webbrowser.open(address)


@tool(risk="safe")
async def get_current_time() -> str:
    """Returns the current local date, time, weekday and time zone. Call it before
    answering anything that depends on today's date or the time of day."""
    now = _now()
    # ISO for the date and time, because every model reads it without
    # ambiguity; the weekday and the zone by name, because "yarın" and "bu
    # akşam" are questions about those.
    return f"{now.isoformat(timespec='minutes')} {now:%A}, {now.tzname()}"


def open_app_for(catalog: AppCatalog) -> Tool:
    """`open_app`, bound to the catalogue it looks names up in.

    A closure rather than a parameter: the schema the model sees is read from
    the function's signature, and the catalogue is not something the model
    chooses (python-guide 3.4).
    """

    @tool(risk="safe")
    async def open_app(name: Annotated[str, "The application, as the user said it."]) -> str:
        """Opens an application installed on this computer by name. Pass the
        name the user said, as they said it: case, accents and small spelling
        differences are forgiven. When nothing matches, the answer names the
        closest apps. Windows may list the app under its English name, so
        before telling the user it is not installed, call again with that
        name (Calculator, Settings, Notepad) or with one of the closest
        names; ask the user only when that fails too."""
        found = catalog.find(name)
        if found is None:
            near = catalog.closest(name)
            hint = f"; closest names: {', '.join(near)}" if near else ""
            return f"No app called {name!r}{hint}."

        # `os.startfile` returns as soon as the shell has taken the request,
        # but taking it can be a store app's activation - long enough to be
        # kept off the loop.
        await asyncio.to_thread(_launch, found.launch)
        return f"Opened {found.name}."

    return open_app


@tool(risk="safe")
async def open_url(url: Annotated[str, "A web address; 'https://' is added when missing."]) -> str:
    """Opens a web address in the user's default browser. Use it for a site the
    user named or an address that came up in the conversation."""
    address = url.strip()
    if "://" not in address:
        address = f"https://{address}"

    if not await asyncio.to_thread(_browse, address):
        raise RuntimeError(f"no browser would open {address}")
    return f"Opened {address}."


@tool(risk="safe")
async def open_settings(page: Annotated[str, PAGE_KEYS]) -> str:
    """Opens a page of Windows Settings. Choose the key nearest to what the
    user asked for, whatever language they asked in: screen brightness is
    display, wireless is wifi, the volume is sound. A key that is not on the
    list opens the Settings home page instead."""
    key = page.strip().casefold()
    uri = SETTINGS_PAGES.get(key)
    if uri is None:
        await asyncio.to_thread(_launch, SETTINGS_PAGES["home"])
        return (
            f"No settings page called {page!r}; opened the Settings home page. "
            f"The pages are: {', '.join(SETTINGS_PAGES)}."
        )

    await asyncio.to_thread(_launch, uri)
    return f"Opened the {key} page of Settings."
