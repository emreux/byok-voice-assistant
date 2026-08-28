"""The application log: what a turn cost, and nothing that could identify it.

Item 1.11 asks for one thing - the token count of every turn written down, so
that the cost report of section 6 has something to be built from in phase 2.4.
What the log does *not* contain is the more interesting half.

**The terminal belongs to the status line.** `loguru` installs a handler that
writes to stderr the moment it is imported. Left in place it would draw log
lines through the middle of the one line phase 1 has to show the user.

**Nothing said out loud is written to disk.** The numbers are; the words are
not. A voice assistant that keeps a plaintext transcript of every sentence
spoken near a microphone is a liability nobody asked for, and section 3.7's
retention rules do not arrive until phase 4.

**No API key can reach the file.** Phase 1 logs no text that could carry one,
and the sink is configured so that a traceback cannot smuggle one out in the
value of a local variable.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest
from loguru import logger

from assistant.app import Turn
from assistant.config import config_dir
from assistant.llm.base import Usage
from assistant.logs import log_path, log_turn, setup_logging

KEY = "AIzaSyD-notarealkeyatall-000000000000000"


@pytest.fixture(autouse=True)
def close_the_log() -> Iterator[None]:
    """No test leaves a handler behind holding a file open."""
    yield
    logger.remove()


@pytest.fixture
def log(tmp_path: Path) -> Path:
    """A log file of this test's own, instead of the machine's."""
    return setup_logging(path=tmp_path / "logs" / "assistant.log")


def read(path: Path) -> str:
    """What ended up on disk. Removing the handler first closes the file."""
    logger.remove()
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Where it goes
# --------------------------------------------------------------------------


def test_the_log_goes_to_a_file(log: Path) -> None:
    logger.info("something happened")

    assert "something happened" in read(log)


def test_nothing_writes_on_the_terminal_any_more(tmp_path: Path) -> None:
    """`loguru` starts with a handler on stderr. The status line is the only
    thing allowed to draw there, so starting up has to take that handler away."""
    terminal = StringIO()
    logger.add(terminal)

    setup_logging(path=tmp_path / "assistant.log")
    logger.info("something happened")

    assert terminal.getvalue() == ""


def test_the_directory_is_made_on_the_first_run(tmp_path: Path) -> None:
    """Nothing creates `%LOCALAPPDATA%\\assistant\\logs` before this does."""
    path = setup_logging(path=tmp_path / "never" / "existed" / "assistant.log")
    logger.info("something happened")

    assert read(path).strip()


def test_starting_twice_does_not_write_everything_twice(tmp_path: Path) -> None:
    """Two handlers on one file is one duplicated line per entry, which reads
    as the assistant having done everything twice."""
    path = tmp_path / "assistant.log"
    setup_logging(path=path)
    setup_logging(path=path)

    logger.info("something happened")

    assert read(path).count("something happened") == 1


def test_the_log_is_written_in_the_language_it_is_read_in(log: Path) -> None:
    """Windows opens files in a legacy code page unless told otherwise, and
    half the Turkish alphabet has no place in it."""
    logger.info("Türkçe çıktı: ağır ışık")

    assert "Türkçe çıktı: ağır ışık" in read(log)


def test_the_machine_s_own_log_lives_where_the_data_does() -> None:
    """Not beside the settings: a log file has no business following the user
    to another machine through a roaming profile (section 3.3)."""
    assert log_path().suffix == ".log"
    assert log_path().parent.is_absolute()
    assert config_dir() not in log_path().parents


# --------------------------------------------------------------------------
# What a turn leaves in it
# --------------------------------------------------------------------------


def test_a_turn_writes_down_what_it_spent(log: Path) -> None:
    log_turn(Turn(heard="saat kaç", said="Üç buçuk.", usage=Usage(300, 10, 256)))

    # The whole line, not the numbers on their own: a timestamp has three
    # digits of milliseconds in it and would answer for any of them.
    assert "300 in, 10 out, 256 cached" in read(log)


def test_what_was_said_out_loud_is_not_written_to_disk(log: Path) -> None:
    log_turn(Turn(heard="kapıyı kilitledim mi", said="Bilmiyorum.", usage=Usage(300, 10)))

    written = read(log)

    assert "kapıyı kilitledim mi" not in written
    assert "Bilmiyorum." not in written


def test_a_turn_that_never_happened_is_not_a_line_in_the_log(log: Path) -> None:
    """A tapped key or a recording of silence. Logging those buries the turns
    that cost something under the ones that cost nothing."""
    log_turn(Turn())

    assert read(log).strip() == ""


def test_a_turn_that_failed_is_still_a_turn(log: Path) -> None:
    """It spent no tokens anybody can account for, and it is exactly the turn
    worth finding in the log afterwards."""
    log_turn(Turn(heard="saat kaç", said="Sağlayıcıya bağlanamadım."))

    assert read(log).strip()


# --------------------------------------------------------------------------
# What cannot reach it
# --------------------------------------------------------------------------


def test_the_value_of_a_variable_is_never_written_down(log: Path) -> None:
    """`loguru` decorates tracebacks with the values of the locals in every
    frame. One of those frames holds the API key (section 10)."""

    def build_a_provider(api_key: str) -> None:
        raise RuntimeError("the provider could not be built")

    try:
        build_a_provider(KEY)
    except RuntimeError:
        logger.exception("that did not work")

    assert KEY not in read(log)
