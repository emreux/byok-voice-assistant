"""What is written to the database and how, one class per table (section 3.7).

A repository is the only code that knows a table's columns; everything else
calls a method named after what happened. `AuditRepo` is the one phase 2.1d
needs. `UsageRepo` arrives with the spend tracking of 2.4 and `SettingsRepo`
with the probe of 2.6 - neither is sketched here, because a class nobody
calls is a class nobody can tell is right.

**A call is written down before it runs.** `start` opens the row with
`status = 'started'` and no `finished_at`; `finish` closes it as `ok` or
`error`. A row that stays `started` for ever is what a crash between the two
looks like, and it is kept that way on purpose: it is how the assistant will
one day say "I tried, and I do not know whether it worked" (section 3.11)
instead of guessing. `deny` is the other shape - the gate said no, nothing
ran, and one row says so.

Every write is one transaction (`with connection:`), and every time is a Unix
epoch second in UTC (section 3.10): local time is for the screen, never for
the file. The arguments are kept as the model sent them, minus their order:
keys are sorted so that the same call always reads the same, which is what
the duplicate check of 2.4 will hash.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal

from assistant.llm.base import ToolCall

__all__ = ["SUMMARY_CHARS", "AuditRepo", "Outcome"]

Outcome = Literal["ok", "error"]

# How much of a tool's result is kept. Enough to tell what happened, not the
# whole of a web page; section 3.7 empties even this after thirty days.
SUMMARY_CHARS = 200


class AuditRepo:
    """The `tool_audit` table: one row per tool call, in two phases.

    `clock` is `time.time` unless a test hands over one it controls.
    """

    def __init__(
        self, connection: sqlite3.Connection, *, clock: Callable[[], float] = time.time
    ) -> None:
        self._connection = connection
        self._clock = clock

    def start(self, call: ToolCall, *, turn_id: str, risk: str) -> int:
        """Opens the row for a call that is about to run, and returns its id."""
        with self._connection:
            row = self._connection.execute(
                "INSERT INTO tool_audit (ts, turn_id, tool, args_json, risk, approved, status)"
                " VALUES (?, ?, ?, ?, ?, 1, 'started') RETURNING id",
                (self._now(), turn_id, call.name, _arguments(call.arguments), risk),
            ).fetchone()
        return int(row[0])

    def finish(
        self,
        row_id: int,
        *,
        status: Outcome,
        summary: str | None = None,
        error: str | None = None,
    ) -> None:
        """Closes the row `start` opened: how it went, and when it ended."""
        with self._connection:
            self._connection.execute(
                "UPDATE tool_audit SET status = ?, result_summary = ?, error = ?, finished_at = ?"
                " WHERE id = ?",
                (
                    status,
                    None if summary is None else summary[:SUMMARY_CHARS],
                    error,
                    self._now(),
                    row_id,
                ),
            )

    def deny(self, call: ToolCall, *, turn_id: str, risk: str) -> int:
        """One row for a call the gate refused. Nothing ran, so the row is
        finished as it is written."""
        now = self._now()
        with self._connection:
            row = self._connection.execute(
                "INSERT INTO tool_audit"
                " (ts, turn_id, tool, args_json, risk, approved, status, finished_at)"
                " VALUES (?, ?, ?, ?, ?, 0, 'denied', ?) RETURNING id",
                (now, turn_id, call.name, _arguments(call.arguments), risk, now),
            ).fetchone()
        return int(row[0])

    def _now(self) -> int:
        return int(self._clock())


def _arguments(arguments: Mapping[str, Any]) -> str:
    """The arguments as one line of JSON, keys sorted, nothing escaped that
    the reader would rather see as it is."""
    return json.dumps(dict(arguments), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
