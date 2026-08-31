"""The application log (item 1.11).

One line per turn, holding what it cost. That is the whole of phase 1: the
`usage_log` table and `assistant cost` arrive in phase 2.4, and until they do
this file is where the answer to "what am I spending" is kept.

Three decisions are worth stating, because each of them is about what is *not*
written.

**Nothing goes to the terminal.** `loguru` installs a handler on stderr when it
is imported, and phase 1 has exactly one line of terminal to say what the
assistant is doing (`ui/status.py`). Setting up takes that handler away rather
than adding to it, which also means starting twice does not log everything
twice.

**The numbers are written down; the words are not.** A voice assistant that
keeps a plaintext transcript of everything said near a microphone is a
liability nobody asked for, and the retention rules that would cover one
(section 3.7) are phase 4. The screen shows what was said, and forgets it.

**No API key can reach the file.** Nothing here logs text that could carry one,
and `diagnose` is off so that a traceback cannot smuggle one out in the value
of a local variable - which is exactly where it would be, one frame below the
adapter (section 10). The masking filter section 5 describes belongs with the
first thing that logs a provider's own words back to us.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from assistant.app import Turn
from assistant.config import log_dir

__all__ = ["LOG_FILE", "RETAINED_FILES", "ROTATE_AT", "log_path", "log_turn", "setup_logging"]

LOG_FILE = "assistant.log"

# A turn is one short line, so this is tens of thousands of them - long enough
# to still hold last month when somebody asks where the money went.
ROTATE_AT = "2 MB"
RETAINED_FILES = 5


def log_path() -> Path:
    """`%LOCALAPPDATA%\\assistant\\logs\\assistant.log`.

    Beside the database rather than beside the settings: a log belongs to the
    machine that wrote it and has no business following the user to another
    one through a roaming profile (section 3.3).
    """
    return log_dir() / LOG_FILE


def setup_logging(*, path: Path | None = None, level: str = "INFO") -> Path:
    """Points the log at a file and takes every other handler away."""
    target = path if path is not None else log_path()

    # Removes `loguru`'s own stderr handler, and any handler a previous call
    # added. Adding rather than replacing is how a log ends up with every line
    # in it twice.
    logger.remove()
    logger.add(
        # `loguru` makes the directory on the way, which is the whole of what
        # is needed on a machine that has never run this before.
        target,
        level=level,
        rotation=ROTATE_AT,
        retention=RETAINED_FILES,
        # Windows opens a file in the machine's legacy code page unless it is
        # told otherwise, and half the Turkish alphabet has no place in one.
        encoding="utf-8",
        backtrace=False,
        diagnose=False,
    )
    return target


def log_turn(finished: Turn) -> None:
    """Writes down what one turn spent.

    A turn that heard nothing never happened - a tapped key, or a recording of
    silence - and logging those buries the turns that cost something under the
    ones that cost nothing. A turn that *failed* is kept: it reports no tokens,
    and it is the line worth finding afterwards.

    So is a turn that was *missed*. It cost nothing, which is exactly why the
    line matters: a user reporting that the assistant "does nothing" and a log
    full of missed turns at 0.5 have already answered each other, and the
    number is the whole of the answer. The words are still not written down -
    a transcript nothing stood behind is no more worth keeping than one that
    was.
    """
    if finished.missed:
        confidence = "-" if finished.confidence is None else f"{finished.confidence:.2f}"
        logger.info("missed: nothing worth answering, confidence {value}", value=confidence)
        return

    if not finished.heard:
        return

    usage = finished.usage
    logger.info(
        "turn: {input} in, {output} out, {cached} cached",
        input=usage.input_tokens,
        output=usage.output_tokens,
        cached=usage.cached_tokens,
    )
