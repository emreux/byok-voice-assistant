"""What is written to the database and how, one class per table (section 3.7).

A repository is the only code that knows a table's columns; everything else
calls a method named after what happened. `AuditRepo` is the `tool_audit`
table of 2.1d; `UsageRepo` is the `usage_log` of 2.4, where every turn's
tokens go with their price. `SettingsRepo` arrives with the probe of 2.6 and
is not sketched here, because a class nobody calls is a class nobody can
tell is right.

**A call is written down before it runs.** `start` opens the row with
`status = 'started'` and no `finished_at`; `finish` closes it as `ok` or
`error`. A row that stays `started` for ever is what a crash between the two
looks like, and it is kept that way on purpose: it is how the assistant says
"I tried, and I do not know whether it worked" (section 3.11) instead of
guessing. `deny` is the other shape - the gate said no, nothing ran, and one
row says so.

**A call is also written down by what it asked for.** `args_hash` is the
arguments as one line of canonical JSON, hashed: the same request always
reads the same, whatever order the model put the keys in. It is what `recent`
looks a call up by, so the gate can say "you already did this a minute ago"
(section 3.11), and what the repeat check of `agent/limits.py` counts by -
same name, same hash, same call. Never by the call's id, which Gemini leaves
empty. What it does not do is read meaning: `{"to": "a@x.com"}` and
`{"to": "A@x.com"}` are two calls, and the design accepts a missed repeat
over a reported one that never happened.

Every write is one transaction (`with connection:`), and every time is a Unix
epoch second in UTC (section 3.10): local time is for the screen, never for
the file.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from assistant.llm.base import ToolCall, Usage

__all__ = [
    "HASH_CHARS",
    "SUMMARY_CHARS",
    "AuditRepo",
    "EarlierCall",
    "ModelUsage",
    "Outcome",
    "UsageRepo",
    "args_hash",
]

Outcome = Literal["ok", "error"]

# How much of a tool's result is kept. Enough to tell what happened, not the
# whole of a web page; section 3.7 empties even this after thirty days.
SUMMARY_CHARS = 200

# How much of the SHA-256 is kept: sixty-four bits, enough that two different
# requests do not collide by accident and short enough to read in a row.
HASH_CHARS = 16


@dataclass(frozen=True, slots=True)
class EarlierCall:
    """The last time the same call ran, or may have: what became of it, and
    how many seconds ago. What the gate reads; the columns stay here."""

    status: Literal["ok", "started"]
    ago: int


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """What one model was used for over a period, as `assistant cost` shows it.

    `cost_usd` is the sum over the turns that have a price, and `None` when
    none has; `unpriced` counts the turns that have none, so that the report
    can say so instead of showing a zero.
    """

    provider: str
    model: str
    turns: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float | None
    unpriced: int


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
                "INSERT INTO tool_audit"
                " (ts, turn_id, tool, args_json, args_hash, risk, approved, status)"
                " VALUES (?, ?, ?, ?, ?, ?, 1, 'started') RETURNING id",
                (
                    self._now(),
                    turn_id,
                    call.name,
                    _arguments(call.arguments),
                    args_hash(call.arguments),
                    risk,
                ),
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
                " (ts, turn_id, tool, args_json, args_hash, risk, approved, status, finished_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 0, 'denied', ?) RETURNING id",
                (
                    now,
                    turn_id,
                    call.name,
                    _arguments(call.arguments),
                    args_hash(call.arguments),
                    risk,
                    now,
                ),
            ).fetchone()
        return int(row[0])

    def recent(self, call: ToolCall, *, within: float) -> EarlierCall | None:
        """The last time this same call ran, or may have, in the last `within` seconds.

        The query of section 3.11: the same tool with the same arguments,
        and only the rows that mean something happened - `ok`, or `started`
        and never finished. A `denied` or `error` row is a call that did
        nothing, and is not an earlier time the user did this.
        """
        now = self._now()
        row = self._connection.execute(
            "SELECT status, ts FROM tool_audit"
            " WHERE tool = ? AND args_hash = ? AND status IN ('ok', 'started') AND ts > ?"
            " ORDER BY ts DESC LIMIT 1",
            (call.name, args_hash(call.arguments), now - within),
        ).fetchone()
        if row is None:
            return None
        return EarlierCall(status=row["status"], ago=max(now - int(row["ts"]), 0))

    def _now(self) -> int:
        return int(self._clock())


class UsageRepo:
    """The `usage_log` table: one row per turn that reached the model (section 6)."""

    def __init__(
        self, connection: sqlite3.Connection, *, clock: Callable[[], float] = time.time
    ) -> None:
        self._connection = connection
        self._clock = clock

    def insert(
        self,
        *,
        turn_id: str,
        provider: str,
        model: str,
        usage: Usage,
        cost_usd: float | None,
    ) -> int:
        """One turn: what it used and what that cost, `None` when the price
        is not known. The row is the receipt; nothing recomputes it later."""
        with self._connection:
            row = self._connection.execute(
                "INSERT INTO usage_log"
                " (ts, provider, model, in_tokens, out_tokens, cached_tokens, cost_usd, turn_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                (
                    int(self._clock()),
                    provider,
                    model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cached_tokens,
                    cost_usd,
                    turn_id,
                ),
            ).fetchone()
        return int(row[0])

    def sum_since(self, since: float) -> float:
        """Dollars over the turns from `since` on that have a price.

        A turn with no price adds nothing, which understates: the report
        says how many such turns there were, and the warning of section
        3.11 can only be as right as the prices it was given.
        """
        row = self._connection.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM usage_log WHERE ts >= ?", (since,)
        ).fetchone()
        return float(row[0])

    def by_model_since(self, since: float) -> list[ModelUsage]:
        """The turns from `since` on, added up per model, in a settled order."""
        rows = self._connection.execute(
            "SELECT provider, model, COUNT(*) AS turns,"
            " SUM(in_tokens) AS input_tokens, SUM(out_tokens) AS output_tokens,"
            " SUM(cached_tokens) AS cached_tokens, SUM(cost_usd) AS cost_usd,"
            " SUM(cost_usd IS NULL) AS unpriced"
            " FROM usage_log WHERE ts >= ? GROUP BY provider, model ORDER BY provider, model",
            (since,),
        ).fetchall()
        return [
            ModelUsage(
                provider=row["provider"],
                model=row["model"],
                turns=int(row["turns"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                cached_tokens=int(row["cached_tokens"]),
                cost_usd=None if row["cost_usd"] is None else float(row["cost_usd"]),
                unpriced=int(row["unpriced"]),
            )
            for row in rows
        ]


def args_hash(arguments: Mapping[str, Any]) -> str:
    """The arguments as a short fingerprint that does not depend on their order.

    The first sixteen hex characters of the SHA-256 of the canonical JSON
    below (section 3.11). Two calls with the same name and the same hash are
    the same call, to the gate and to the repeat check alike.
    """
    return hashlib.sha256(_arguments(arguments).encode("utf-8")).hexdigest()[:HASH_CHARS]


def _arguments(arguments: Mapping[str, Any]) -> str:
    """The arguments as one line of JSON, keys sorted, nothing escaped that
    the reader would rather see as it is."""
    return json.dumps(dict(arguments), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
