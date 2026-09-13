"""Which service a request goes to, and what the model is told happened.

Nothing here searches anything and nothing opens: the lookups are stand-ins
for the protocols of `media/player.py`, and `shell.launch` / `shell.browse`
are replaced by the `opened` fixture, which is the same seam
`test_tools_system.py` uses.

Two claims are worth more than the rest, because both are things the owner
actually hit:

*A song plays; it does not open a search.* The address that comes back for
YouTube Music is a `watch?v=` one, so the tab starts playing by itself.

*What cannot play says so.* Spotify cannot be searched without credentials
this build does not have, so its answer opens the exact recording - found by
its ISRC through Deezer - or, failing that, a search, **and either way forbids
the sentence the model would otherwise reach for**. An assistant that says "I
put it on" when it did not is worse than one that says what it did.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from assistant import shell
from assistant.config import MediaSettings
from assistant.media.player import NOT_PLAYING, Player
from assistant.media.spotify import Spotify
from assistant.media.track import Recording, SearchError, Track
from assistant.media.window import Browser, MediaWindow
from assistant.tools.media import open_media_for, play_music_for, play_video_for
from assistant.tools.registry import Tool
from assistant.tools.system import AppCatalog, AppEntry, open_app_for

MAIN = threading.current_thread()

KUMRALIM = Track(
    target="https://music.youtube.com/watch?v=UXK9s54VmxQ", title="Kumralım", artist="Yaşar"
)
FRONT_PAGE = Track(target="https://music.youtube.com/watch?v=frontfrontf", title="Whatever")
VIDEO = Track(
    target="https://www.youtube.com/watch?v=outny_anbdo",
    title='"Skill bilmeyen yakında işsiz kalır" - izle öğren',
    artist="İzle",
)
KUMRALIM_RECORDING = Recording(isrc="TR2240596102", title="Kumralım", artist="Yaşar")


class Opened:
    """What Windows was handed, and from which thread."""

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
    monkeypatch.setattr(shell, "launch", seen.launch)
    monkeypatch.setattr(shell, "browse", seen.browse)
    return seen


class Music:
    """A stand-in for YouTube Music that never leaves the process."""

    def __init__(self, found: Track | None = None, front: Track | None = None) -> None:
        self.found = found
        self.front = front
        self.asked: list[str] = []

    async def song(self, words: str) -> Track:
        self.asked.append(words)
        if self.found is None:
            raise SearchError(f"YouTube Music listed no song for {words!r}.")
        return self.found

    async def anything(self) -> Track:
        if self.front is None:
            raise SearchError("YouTube Music's front page offered nothing playable.")
        return self.front


class Videos:
    def __init__(self, found: Track | None = None) -> None:
        self.found = found
        self.asked: list[str] = []
        self.closed = False

    async def first_video(self, words: str) -> Track:
        self.asked.append(words)
        if self.found is None:
            raise SearchError(f"YouTube listed no video for {words!r}.")
        return self.found

    async def aclose(self) -> None:
        self.closed = True


class Recordings:
    """A stand-in for Deezer: names a recording by its ISRC, or cannot."""

    def __init__(self, found: Recording | None = None) -> None:
        self.found = found
        self.asked: list[str] = []
        self.closed = False

    async def recording(self, words: str) -> Recording:
        self.asked.append(words)
        if self.found is None:
            raise SearchError(f"Deezer listed no recording for {words!r}.")
        return self.found

    async def aclose(self) -> None:
        self.closed = True


class Paused:
    def __init__(self, playing: bool = True) -> None:
        self.playing = playing
        self.calls = 0

    async def __call__(self) -> bool:
        self.calls += 1
        return self.playing


def build(
    *,
    music: Music | None = None,
    videos: Videos | None = None,
    recordings: Recordings | None = None,
    installed: bool = False,
    pause: Paused | None = None,
    window: MediaWindow | None = None,
    **settings: object,
) -> Player:
    return Player(
        music=music or Music(KUMRALIM, FRONT_PAGE),
        videos=videos or Videos(VIDEO),
        # Nothing found by default, so that the tests of the search fallback
        # stay what they were; a test of the exact recording hands one in.
        recordings=recordings or Recordings(None),
        spotify=Spotify(installed=lambda: installed),
        settings=MediaSettings(**settings),  # type: ignore[arg-type]
        pause=pause or Paused(playing=False),
        # No browser: the address goes to `shell.browse`, which `opened` sees.
        window=window if window is not None else MediaWindow(None),
    )


# --------------------------------------------------------------------------
# The tools the model sees
# --------------------------------------------------------------------------


@pytest.fixture
def tools() -> list[Tool]:
    player = build()
    return [play_music_for(player), play_video_for(player), open_media_for(player)]


def test_every_media_tool_is_offered_to_the_model_as_safe(tools: list[Tool]) -> None:
    assert [entry.spec.name for entry in tools] == ["play_music", "play_video", "open_media"]
    assert all(entry.risk == "safe" for entry in tools)
    assert all(entry.confirm_prompt is None for entry in tools)


def test_play_music_may_be_called_with_nothing_at_all(tools: list[Tool]) -> None:
    """Music with no words names neither a song nor a service."""
    schema = tools[0].spec.parameters

    assert list(schema["properties"]) == ["query", "service"]
    assert schema["required"] == []


def test_play_video_has_to_be_told_what_to_open(tools: list[Tool]) -> None:
    assert tools[1].spec.parameters["required"] == ["query"]


def test_the_tools_tell_the_model_not_to_write_an_address_itself(tools: list[Tool]) -> None:
    """The hallucinated `watch?v=` is a prompt problem as much as a code one."""
    assert "open_url" in tools[0].spec.description
    assert "watch?v=" in tools[1].spec.description


# --------------------------------------------------------------------------
# One window
# --------------------------------------------------------------------------


class Desk:
    """Enough of a desktop to see which way an address went."""

    def __init__(self) -> None:
        self.started: list[list[str]] = []

    def start(self, command: Sequence[str]) -> None:
        self.started.append(list(command))

    def windows_of(self, executable: Path) -> set[int]:
        return {1} if self.started else set()

    def is_window(self, handle: int) -> bool:
        return True

    def close(self, handle: int) -> None:
        pass


CHROME = Browser(executable=Path(r"C:\chrome.exe"), new_window="--new-window")


async def test_a_song_opens_in_the_assistant_s_own_window(opened: Opened) -> None:
    desk = Desk()
    player = build(window=MediaWindow(CHROME, desk, appear_seconds=0.01, poll_seconds=0.001))

    said = await player.play_music("kumralım", "")

    assert said == "Playing Yaşar - Kumralım on YouTube Music."
    assert desk.started == [[r"C:\chrome.exe", "--new-window", KUMRALIM.target]]
    assert opened.targets == []


async def test_the_spotify_application_is_not_a_window(opened: Opened) -> None:
    desk = Desk()
    player = build(installed=True, window=MediaWindow(CHROME, desk, appear_seconds=0.01))

    await player.open_service("spotify")

    assert desk.started == []
    assert opened.targets == ["spotify:"]


# --------------------------------------------------------------------------
# Music
# --------------------------------------------------------------------------


async def test_a_song_with_no_service_named_plays_on_youtube_music(opened: Opened) -> None:
    said = await build().play_music("kumralım", "")

    assert opened.targets == ["https://music.youtube.com/watch?v=UXK9s54VmxQ"]
    assert said == "Playing Yaşar - Kumralım on YouTube Music."


async def test_the_browser_is_opened_off_the_event_loop(opened: Opened) -> None:
    await build().play_music("kumralım", "")

    assert opened.threads[0] is not MAIN


async def test_the_service_the_user_named_beats_the_default(opened: Opened) -> None:
    said = await build(default_service="youtube_music").play_music("kumralım", "spotify")

    assert opened.targets == ["https://open.spotify.com/search/Ya%C5%9Far%20-%20Kumral%C4%B1m"]
    assert NOT_PLAYING in said


async def test_a_music_service_nobody_offers_is_refused_with_the_list(opened: Opened) -> None:
    said = await build().play_music("kumralım", "deezer")

    assert opened.targets == []
    assert "deezer" in said
    assert "youtube_music" in said and "spotify" in said


async def test_a_mistyped_default_service_does_not_stop_the_music(opened: Opened) -> None:
    """The model's mistake is refused; the user's typo in a file is not."""
    said = await build(default_service="youtube musik").play_music("kumralım", "")

    assert opened.targets == ["https://music.youtube.com/watch?v=UXK9s54VmxQ"]
    assert said.endswith("on YouTube Music.")


async def test_music_with_no_words_uses_the_configured_default(opened: Opened) -> None:
    music = Music(KUMRALIM, FRONT_PAGE)

    await build(music=music, default_query="türkçe rock").play_music("", "")

    assert music.asked == ["türkçe rock"]
    assert opened.targets == ["https://music.youtube.com/watch?v=UXK9s54VmxQ"]


async def test_music_with_no_words_and_no_default_plays_what_the_service_offers(
    opened: Opened,
) -> None:
    music = Music(KUMRALIM, FRONT_PAGE)

    await build(music=music).play_music("", "")

    assert music.asked == []
    assert opened.targets == ["https://music.youtube.com/watch?v=frontfrontf"]


async def test_music_with_no_words_on_spotify_asks_rather_than_guesses(opened: Opened) -> None:
    said = await build().play_music("", "spotify")

    assert opened.targets == []
    assert "Ask the user" in said


async def test_a_search_too_long_to_have_been_spoken_is_refused(opened: Opened) -> None:
    said = await build().play_music("la " * 200, "")

    assert opened.targets == []
    assert NOT_PLAYING in said


async def test_a_failed_search_forbids_the_sentence_the_model_would_reach_for(
    opened: Opened,
) -> None:
    said = await build(music=Music(None, None)).play_music("nothing anyone recorded", "")

    assert opened.targets == []
    assert NOT_PLAYING in said


async def test_a_browser_that_will_not_open_is_reported_not_swallowed(opened: Opened) -> None:
    opened.browser_works = False

    said = await build().play_music("kumralım", "")

    assert NOT_PLAYING in said
    assert "Playing" not in said


async def test_a_cancelled_turn_is_never_reported_as_a_success(opened: Opened) -> None:
    class Stopped(Music):
        async def song(self, words: str) -> Track:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await build(music=Stopped()).play_music("kumralım", "")

    assert opened.targets == []


# --------------------------------------------------------------------------
# Not playing two things at once
# --------------------------------------------------------------------------


async def test_what_is_playing_is_paused_before_the_next_song_starts(opened: Opened) -> None:
    """Each song opens a tab, so without this two of them play together."""
    pause = Paused(playing=True)

    await build(pause=pause).play_music("kumralım", "")

    assert pause.calls == 1


async def test_pausing_first_can_be_switched_off(opened: Opened) -> None:
    pause = Paused(playing=True)

    await build(pause=pause, pause_before_playing=False).play_music("kumralım", "")

    assert pause.calls == 0


async def test_opening_a_service_pauses_nothing_and_plays_nothing(opened: Opened) -> None:
    pause = Paused(playing=True)

    said = await build(pause=pause).open_service("youtube_music")

    assert opened.targets == ["https://music.youtube.com"]
    assert pause.calls == 0
    assert NOT_PLAYING in said


async def test_two_songs_asked_for_at_once_are_served_one_after_the_other(
    opened: Opened,
) -> None:
    gate = asyncio.Event()
    order: list[str] = []

    class Slow(Music):
        async def song(self, words: str) -> Track:
            order.append(words)
            if words == "first":
                await gate.wait()
            return Track(target=f"https://music.youtube.com/watch?v={words}", title=words)

    player = build(music=Slow())
    one = asyncio.create_task(player.play_music("first", ""))
    two = asyncio.create_task(player.play_music("second", ""))
    for _ in range(4):
        await asyncio.sleep(0)

    assert order == ["first"], "the second request started before the first had finished"

    gate.set()
    await asyncio.gather(one, two)

    assert order == ["first", "second"]
    assert opened.targets == [
        "https://music.youtube.com/watch?v=first",
        "https://music.youtube.com/watch?v=second",
    ]


# --------------------------------------------------------------------------
# Spotify: the exact recording when Deezer can name it, a search otherwise -
# and honestly reported either way
# --------------------------------------------------------------------------


async def test_a_spotify_request_opens_the_exact_recording_in_the_application(
    opened: Opened,
) -> None:
    recordings = Recordings(KUMRALIM_RECORDING)

    said = await build(installed=True, recordings=recordings).play_music("kumralım", "spotify")

    assert recordings.asked == ["kumralım"]
    assert opened.targets == ["spotify:search:isrc%3ATR2240596102"]
    assert "Yaşar - Kumralım" in said
    assert NOT_PLAYING in said


async def test_without_the_application_the_exact_recording_opens_on_the_website(
    opened: Opened,
) -> None:
    await build(installed=False, recordings=Recordings(KUMRALIM_RECORDING)).play_music(
        "kumralım", "spotify"
    )

    assert opened.targets == ["https://open.spotify.com/search/isrc%3ATR2240596102"]


async def test_the_exact_recording_answer_says_the_user_has_to_press_play(
    opened: Opened,
) -> None:
    """One result on screen is still not a song playing."""
    said = await build(recordings=Recordings(KUMRALIM_RECORDING)).play_music("kumralım", "spotify")

    assert "press play" in said
    assert "YouTube Music" in said
    assert "Playing" not in said


async def test_nothing_is_paused_for_a_spotify_request(opened: Opened) -> None:
    """Nothing starts, so what the user is listening to is left alone."""
    paused = Paused(playing=True)

    await build(recordings=Recordings(KUMRALIM_RECORDING), pause=paused).play_music(
        "kumralım", "spotify"
    )

    assert paused.calls == 0


async def test_when_no_recording_is_found_the_words_are_searched_as_before(
    opened: Opened,
) -> None:
    said = await build(recordings=Recordings(None)).play_music("kumralım", "spotify")

    assert opened.targets == ["https://open.spotify.com/search/Ya%C5%9Far%20-%20Kumral%C4%B1m"]
    assert "search" in said
    assert NOT_PLAYING in said


async def test_closing_the_player_lets_go_of_the_recording_lookup_too() -> None:
    recordings = Recordings(KUMRALIM_RECORDING)
    player = build(recordings=recordings)

    await player.aclose()

    assert recordings.closed


async def test_spotify_without_the_application_opens_its_website(opened: Opened) -> None:
    said = await build(installed=False).play_music("billie jean", "spotify")

    assert opened.targets[0].startswith("https://open.spotify.com/search/")
    assert NOT_PLAYING in said


async def test_spotify_with_the_application_is_searched_inside_it(opened: Opened) -> None:
    """A `spotify:` URI goes to the shell, which is what wakes the app."""
    said = await build(installed=True).play_music("billie jean", "spotify")

    assert opened.targets[0].startswith("spotify:search:")
    assert NOT_PLAYING in said


async def test_the_spotify_search_uses_the_name_youtube_music_knows(opened: Opened) -> None:
    """The user's own word finds far less than "Yaşar - Kumralım" does."""
    await build().play_music("kumralım", "spotify")

    assert "Ya%C5%9Far%20-%20Kumral%C4%B1m" in opened.targets[0]


async def test_a_song_no_lookup_could_name_is_searched_for_as_the_user_said_it(
    opened: Opened,
) -> None:
    await build(music=Music(None, None)).play_music("bir şarkı", "spotify")

    assert opened.targets[0].endswith("bir%20%C5%9Fark%C4%B1")


async def test_spotify_says_which_command_would_make_it_play(opened: Opened) -> None:
    said = await build().play_music("billie jean", "spotify")

    assert "YouTube Music" in said


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------


async def test_a_video_is_opened_in_the_user_own_browser(opened: Opened) -> None:
    videos = Videos(VIDEO)

    said = await build(videos=videos).play_video("skill bilmeyen yakında işsiz kalır")

    assert videos.asked == ["skill bilmeyen yakında işsiz kalır"]
    assert opened.targets == ["https://www.youtube.com/watch?v=outny_anbdo"]
    assert VIDEO.title in said


async def test_a_video_nobody_named_is_asked_about_rather_than_guessed(opened: Opened) -> None:
    said = await build().play_video("   ")

    assert opened.targets == []
    assert "Ask the user" in said


async def test_a_video_search_that_fails_says_nothing_was_opened(opened: Opened) -> None:
    said = await build(videos=Videos(None)).play_video("something nobody uploaded")

    assert opened.targets == []
    assert "Nothing was opened" in said


# --------------------------------------------------------------------------
# Opening a service by name
# --------------------------------------------------------------------------


async def test_opening_spotify_prefers_the_installed_application(opened: Opened) -> None:
    await build(installed=True).open_service("spotify")

    assert opened.targets == ["spotify:"]


async def test_opening_spotify_without_the_app_uses_its_website(opened: Opened) -> None:
    await build(installed=False).open_service("spotify")

    assert opened.targets == ["https://open.spotify.com"]


async def test_a_service_nobody_offers_is_refused_with_the_list(opened: Opened) -> None:
    said = await build().open_service("deezer")

    assert opened.targets == []
    assert "youtube_music" in said


# --------------------------------------------------------------------------
# open_app, which knows nothing about services (invariant: one place decides)
# --------------------------------------------------------------------------


@pytest.fixture
def catalog() -> AppCatalog:
    return AppCatalog([AppEntry("Spotify", r"C:\Start\Spotify.lnk"), AppEntry("Notepad", "n.lnk")])


async def test_open_app_still_opens_an_ordinary_application(opened: Opened) -> None:
    open_app = open_app_for(AppCatalog([AppEntry("Notepad", "n.lnk")]), media=build().open_named)

    said = await open_app.run(name="notepad")

    assert opened.targets == ["n.lnk"]
    assert said == "Opened Notepad."


async def test_an_installed_application_wins_its_own_name(
    opened: Opened, catalog: AppCatalog
) -> None:
    """Someone who has Spotify gets Spotify, not its website."""
    open_app = open_app_for(catalog, media=build(installed=False).open_named)

    said = await open_app.run(name="spotify")

    assert opened.targets == [r"C:\Start\Spotify.lnk"]
    assert said == "Opened Spotify."


async def test_a_service_with_no_application_falls_through_to_the_player(
    opened: Opened, catalog: AppCatalog
) -> None:
    """YouTube Music has no application on Windows at all."""
    open_app = open_app_for(catalog, media=build().open_named)

    said = await open_app.run(name="YouTube Music")

    assert opened.targets == ["https://music.youtube.com"]
    assert "Opened YouTube Music" in said


async def test_a_name_that_is_neither_is_still_answered_by_the_catalogue(
    opened: Opened, catalog: AppCatalog
) -> None:
    open_app = open_app_for(catalog, media=build().open_named)

    said = await open_app.run(name="a program nobody installed")

    assert opened.targets == []
    assert said.startswith("No app called")


async def test_open_app_without_a_player_behaves_as_it_always_did(
    opened: Opened, catalog: AppCatalog
) -> None:
    said = await open_app_for(catalog).run(name="YouTube Music")

    assert opened.targets == []
    assert said.startswith("No app called")
