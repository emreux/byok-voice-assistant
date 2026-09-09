"""The schema, as a list of steps that is only ever appended to (section 3.7).

`PRAGMA user_version` is SQLite's own integer slot for exactly this: it holds
how many of `MIGRATIONS` have been applied to a file. On every start the ones
past that number run, in order, each inside one transaction together with the
bump of the version - so a migration that fails halfway leaves neither a
half-built table nor a version claiming it was built. Nothing is ever edited
in place: a change to a table is a new entry at the end, because a database
on somebody's machine has already run the old one.

Phase 2.1d opens the list with `tool_audit` alone. The columns other steps
need come as their own entries when those steps arrive - `args_hash` for the
duplicate check of 2.4, `source` for MCP in phase 5 - and `usage_log` and
`settings` with the steps that first read them (2.4 and 2.6).
"""

from __future__ import annotations

import sqlite3

__all__ = ["MIGRATIONS", "migrate", "schema_version"]

MIGRATIONS: tuple[str, ...] = (
    # 1 - tool_audit: every tool call, written before it runs (section 3.9).
    # Times are Unix epoch seconds in UTC; local time is for the screen only.
    # status is started | ok | error | denied, and a row that stays `started`
    # is a call whose outcome nobody knows (section 3.11).
    """
    CREATE TABLE tool_audit (
        id             INTEGER PRIMARY KEY,
        ts             INTEGER NOT NULL,
        turn_id        TEXT    NOT NULL,
        tool           TEXT    NOT NULL,
        args_json      TEXT    NOT NULL,
        risk           TEXT    NOT NULL,
        approved       INTEGER NOT NULL,
        status         TEXT    NOT NULL,
        result_summary TEXT,
        error          TEXT,
        finished_at    INTEGER
    );
    CREATE INDEX tool_audit_turn ON tool_audit(turn_id);
    """,
)


def schema_version(connection: sqlite3.Connection) -> int:
    """How many migrations this database has had."""
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def migrate(connection: sqlite3.Connection) -> int:
    """Applies every migration not yet applied and returns the version reached.

    Each script runs as one transaction with the bump of the version, so the
    file is always at a whole number of migrations. A database from a newer
    build is left alone: the list is append-only, so what this build needs
    is already there.
    """
    applied = schema_version(connection)
    for number, script in enumerate(MIGRATIONS[applied:], start=applied + 1):
        try:
            connection.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;")
        except sqlite3.Error:
            connection.rollback()
            raise
    return schema_version(connection)
