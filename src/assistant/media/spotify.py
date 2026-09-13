"""What "on Spotify" can mean without asking the user for credentials.

Spotify *plays* anything its desktop application is handed as a `spotify:`
URI, and that application is the user's own - signed in, on whatever device
they last listened with, Premium or not. So starting something needs nothing
from this program. *Finding* something does, and that is where it stops:
Spotify's catalogue cannot be searched anonymously. The web player's token
endpoint answers `403`, the results page carries no identifiers until its own
script has run (measured again 2026-09-13), and the only documented way in
is a developer application - which, since February 2026, needs its owner to
hold a Premium subscription. This build does not ask that of anyone (the
owner's decision, 10 and 13 September 2026, design.md section 12 decision
25): a user with a free account gets everything a free account can do.

What a free account can do is *one exact result*. Spotify's own search box
understands an `isrc:` filter, and the ISRC of the recording the user meant
can be had without a key from Deezer (`media/deezer.py`). So the words go
to Deezer, the code goes to Spotify, and what opens is the one recording
rather than a page of candidates - in the installed application through
`spotify:search:isrc%3A<code>`, or on the website without it. When Deezer
cannot name the recording, the words themselves are searched, as before.

Either way **nothing starts playing**: a search with one result is still a
search, and the tool result says so in as many words, so that the model
tells the user to press play rather than announcing a song that is not on.
That asymmetry - a real song on YouTube Music, an open page on Spotify - is
visible to the user and is meant to be. The alternative was to have the
assistant claim it had played something it had not.
"""

from __future__ import annotations

import winreg
from collections.abc import Callable
from urllib.parse import quote

from loguru import logger

__all__ = ["APP_HOME", "APP_SEARCH", "WEB_HOME", "WEB_SEARCH", "Spotify", "app_installed"]

# What the installed application answers to, and what the website answers to.
# The scheme is how the caller knows which of the two ways to open it:
# `spotify:` goes to the shell, `https:` to the browser (`media/player.py`).
APP_SEARCH = "spotify:search:{words}"
APP_HOME = "spotify:"
WEB_SEARCH = "https://open.spotify.com/search/{words}"
WEB_HOME = "https://open.spotify.com"

# The registry key Windows fills in when something claims `spotify:`. Asking
# it is asking the only question that matters - "will this URI open anything
# on this machine" - rather than hunting for an executable that may be in the
# Start Menu, in a store package, or in neither.
PROTOCOL = r"spotify\shell\open\command"


def app_installed() -> bool:
    """Whether a `spotify:` URI would open an application on this machine."""
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, PROTOCOL):
            return True
    except OSError:
        # Not registered, or a registry that will not answer. Either way the
        # website is the answer, and neither is worth a warning.
        logger.debug("no application is registered for spotify: URIs")
        return False


class Spotify:
    """Where a Spotify request goes: the installed app, or the website."""

    def __init__(self, *, installed: Callable[[], bool] = app_installed) -> None:
        # Asked every time rather than once at startup: the user may install
        # Spotify while the assistant is running, and a registry read is
        # microseconds.
        self._installed = installed

    def search(self, words: str) -> str:
        """Where the search for `words` should be opened.

        The words are percent-encoded, which is what keeps a title with a
        space, an apostrophe or a Turkish letter in it from becoming a
        different search - or, in the URI, from ending it early.
        """
        wanted = quote(words.strip(), safe="")
        return (APP_SEARCH if self._installed() else WEB_SEARCH).format(words=wanted)

    def exact(self, isrc: str) -> str:
        """Where the one recording with `isrc` should be opened.

        A search like any other to Spotify, with the filter its own search
        box understands; the colon is encoded with the rest so that the URI's
        own colons stay the only ones the application has to parse.
        """
        return self.search(f"isrc:{isrc}")

    def home(self) -> str:
        """Where "open Spotify" should go."""
        return APP_HOME if self._installed() else WEB_HOME
