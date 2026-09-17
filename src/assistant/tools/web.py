"""What the assistant does on the web that is not a named site (design.md
section 3.6). Phase 3 brings `read_clipboard` and `fetch_page` here; the
first occupant, since 15 September 2026, is `search_web`.

**A search is an address with the words in it.** Every search engine
answers a `GET` with the query in the address, and which engine that is
belongs to the user - `[web] search_url` in `config.toml`, Google when the
line is not there - so the code carries no engine of its own (section 10:
configuration is not code). The address opens the way `open_url` opens one:
in the user's default browser, in the profile they are signed in to
(`shell.py`), because a Google that knows the user answers better than one
that does not.

**The words are the user's.** They are URL-encoded and nothing else: not
translated, not corrected, not padded. A model that "improves" a query is
searching for something the user did not say.
"""

from __future__ import annotations

from typing import Annotated
from urllib.parse import quote_plus

from loguru import logger

from assistant import shell
from assistant.tools.registry import Tool, tool

__all__ = ["SEARCH_URL", "search_web_for"]

# The default `[web] search_url`. `{query}` is where the words go, already
# URL-encoded.
SEARCH_URL = "https://www.google.com/search?q={query}"

NO_WORDS = "Nothing to search for: the words were empty. Ask the user what to look up."


def search_web_for(address: str = SEARCH_URL) -> Tool:
    """`search_web`, bound to the engine the user chose.

    An address without `{query}` cannot carry the words anywhere. It is not
    a reason for searching to stop working - the user's typo is theirs to
    find in the log - so the default engine stands in for it, the way a
    mistyped `[media] default_service` is handled (`media/player.py`).
    """
    if "{query}" not in address:
        logger.warning(
            "[web] search_url is {!r}, which has no {{query}} in it; using {}", address, SEARCH_URL
        )
        address = SEARCH_URL

    @tool(risk="safe")
    async def search_web(
        query: Annotated[str, "What to search for, in the user's own words."],
    ) -> str:
        """Opens a web search for the user's words in their browser. Use it
        when they ask to search, look something up or google something -
        "search for X", "look up Y" - and pass their words as they said
        them. Not for a site they named (open_url opens that) and not for
        music or video (play_music and play_video find those)."""
        words = " ".join(query.split())
        if not words:
            return NO_WORDS
        target = address.replace("{query}", quote_plus(words))
        if not await shell.open_address(target):
            raise RuntimeError(f"no browser would open {target}")
        return f"Opened a web search for {words!r}."

    return search_web
