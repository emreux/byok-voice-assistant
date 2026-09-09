"""The machine's own tools: the time, an app, a site, a settings page (2.1c, 2.2).

Nothing here opens anything. `_launch` and `_browse` are the two places the
shell is reached, and each test replaces them and looks at what would have
been opened - which is also how it is known that opening happens off the
event loop. The time tool has its clock pinned the same way.
"""

from __future__ import annotations

import re
import threading
from datetime import UTC, datetime, timedelta, timezone

import pytest

from assistant.tools import system
from assistant.tools.registry import Tool
from assistant.tools.system import (
    SETTINGS_PAGES,
    AppCatalog,
    AppEntry,
    get_current_time,
    open_app_for,
    open_settings,
    open_url,
)

TURKEY = timezone(timedelta(hours=3), "Turkey Standard Time")


class Opened:
    """What the shell was handed, and from which thread."""

    def __init__(self) -> None:
        self.targets: list[str] = []
        self.threads: list[threading.Thread] = []
        self.browser_works = True

    def launch(self, target: str) -> None:
        self.targets.append(target)
        self.threads.append(threading.current_thread())

    def browse(self, address: str) -> bool:
        self.targets.append(address)
        self.threads.append(threading.current_thread())
        return self.browser_works


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Opened:
    seen = Opened()
    monkeypatch.setattr(system, "_launch", seen.launch)
    monkeypatch.setattr(system, "_browse", seen.browse)
    return seen


@pytest.fixture
def open_app() -> Tool:
    return open_app_for(
        AppCatalog(
            [
                AppEntry("Spotify", r"C:\Start\Spotify.lnk"),
                AppEntry("Notepad", r"C:\Start\Notepad.lnk"),
            ]
        )
    )


# --------------------------------------------------------------------------
# get_current_time (2.1c)
# --------------------------------------------------------------------------


def test_the_time_is_a_safe_tool_with_nothing_to_fill_in() -> None:
    assert get_current_time.risk == "safe"
    assert get_current_time.spec.name == "get_current_time"
    assert get_current_time.spec.parameters["properties"] == {}
    assert get_current_time.spec.parameters["required"] == []


def test_the_description_tells_the_model_when_to_ask_the_time() -> None:
    """Architecture guide section 10: the description is what decides whether
    the model reaches for the tool, so it names the occasion."""
    said = get_current_time.spec.description.casefold()

    assert "date" in said
    assert "time" in said


async def test_the_answer_is_the_date_the_time_the_weekday_and_the_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(system, "_now", lambda: datetime(2026, 9, 9, 14, 3, tzinfo=TURKEY))

    assert await get_current_time.run() == "2026-09-09T14:03+03:00 Wednesday, Turkey Standard Time"


async def test_the_zone_is_whatever_the_clock_says_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system, "_now", lambda: datetime(2026, 1, 1, 0, 0, tzinfo=UTC))

    assert await get_current_time.run() == "2026-01-01T00:00+00:00 Thursday, UTC"


async def test_the_real_clock_answers_in_the_same_shape() -> None:
    """Local time with its offset, then a weekday; the zone's name is the
    machine's own and is not checked."""
    said = await get_current_time.run()

    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}[+-]\d{2}:\d{2} [A-Z][a-z]+day, .+$", said)


# --------------------------------------------------------------------------
# open_app (2.2)
# --------------------------------------------------------------------------


def test_open_app_is_a_safe_tool_that_asks_for_a_name(open_app: Tool) -> None:
    """The catalogue is bound in, not a parameter: the model sees one field."""
    assert open_app.risk == "safe"
    assert open_app.spec.name == "open_app"
    assert open_app.spec.parameters["required"] == ["name"]
    assert list(open_app.spec.parameters["properties"]) == ["name"]
    assert open_app.spec.parameters["properties"]["name"]["description"]


async def test_the_app_the_user_named_is_opened_and_the_model_is_told_which(
    open_app: Tool, opened: Opened
) -> None:
    said = await open_app.run(name="spotify")

    assert opened.targets == [r"C:\Start\Spotify.lnk"]
    assert said == "Opened Spotify."


async def test_opening_happens_off_the_event_loop(open_app: Tool, opened: Opened) -> None:
    """A store app's activation can take long enough to matter (rule 4)."""
    await open_app.run(name="spotify")

    assert threading.main_thread() not in opened.threads


async def test_an_app_that_is_not_there_is_reported_with_the_nearest_names(
    open_app: Tool, opened: Opened
) -> None:
    """Nothing opens; the model can ask which of the near ones was meant."""
    said = await open_app.run(name="spotter")

    assert opened.targets == []
    assert said == "No app called 'spotter'; closest names: Spotify, Notepad."


async def test_an_app_nothing_is_near_is_reported_alone(open_app: Tool, opened: Opened) -> None:
    said = await open_app.run(name="zzzz")

    assert opened.targets == []
    assert said == "No app called 'zzzz'."


# --------------------------------------------------------------------------
# open_url (2.2)
# --------------------------------------------------------------------------


def test_open_url_is_a_safe_tool_that_asks_for_an_address() -> None:
    assert open_url.risk == "safe"
    assert open_url.spec.parameters["required"] == ["url"]


async def test_a_bare_domain_is_opened_as_https(opened: Opened) -> None:
    said = await open_url.run(url=" example.com ")

    assert opened.targets == ["https://example.com"]
    assert said == "Opened https://example.com."


async def test_an_address_with_a_scheme_is_opened_as_it_is(opened: Opened) -> None:
    await open_url.run(url="http://example.com/a?b=1")

    assert opened.targets == ["http://example.com/a?b=1"]


async def test_the_browser_is_opened_off_the_event_loop(opened: Opened) -> None:
    await open_url.run(url="example.com")

    assert threading.main_thread() not in opened.threads


async def test_no_browser_is_an_error_the_gate_reports(opened: Opened) -> None:
    """Raised rather than returned: the gate writes `error` in the audit and
    tells the model the tool failed, which is what happened."""
    opened.browser_works = False

    with pytest.raises(RuntimeError, match=r"example\.com"):
        await open_url.run(url="example.com")


# --------------------------------------------------------------------------
# open_settings (2.2)
# --------------------------------------------------------------------------


def test_open_settings_lists_every_page_it_knows_in_the_schema() -> None:
    """The keys the model may choose are read from the same table the tool
    opens from, so the two cannot drift apart."""
    assert open_settings.risk == "safe"
    assert open_settings.spec.parameters["required"] == ["page"]
    offered = open_settings.spec.parameters["properties"]["page"]["description"]
    for key in SETTINGS_PAGES:
        assert key in offered, key


def test_every_page_is_a_settings_uri() -> None:
    assert all(uri.startswith("ms-settings:") for uri in SETTINGS_PAGES.values())
    assert SETTINGS_PAGES["home"] == "ms-settings:"


async def test_a_known_page_is_opened_by_its_uri(opened: Opened) -> None:
    said = await open_settings.run(page="bluetooth")

    assert opened.targets == ["ms-settings:bluetooth"]
    assert said == "Opened the bluetooth page of Settings."


async def test_the_key_is_folded_before_it_is_looked_up(opened: Opened) -> None:
    await open_settings.run(page=" Bluetooth ")

    assert opened.targets == ["ms-settings:bluetooth"]


async def test_an_unknown_page_opens_the_home_page_and_lists_the_keys(opened: Opened) -> None:
    said = await open_settings.run(page="brightness")

    assert opened.targets == ["ms-settings:"]
    assert said.startswith("No settings page called 'brightness'; opened the Settings home page.")
    assert "display" in said
    assert "bluetooth" in said


async def test_settings_are_opened_off_the_event_loop(opened: Opened) -> None:
    await open_settings.run(page="sound")

    assert threading.main_thread() not in opened.threads
