"""The schema, as a list of steps that is only ever appended to (section 3.7).

`PRAGMA user_version` is SQLite's own integer slot for exactly this: it holds
how many of `MIGRATIONS` have been applied to a file. On every start the ones
past that number run, in order, each inside one transaction together with the
bump of the version - so a migration that fails halfway leaves neither a
half-built table nor a version claiming it was built. Nothing is ever edited
in place: a change to a table is a new entry at the end, because a database
on somebody's machine has already run the old one.

Phase 2.1d opened the list with `tool_audit` alone. Phase 2.4 is the second
entry, and the first real use of the mechanism: the owner's database was at
version 1 with rows in it, and came up to version 2 with those rows intact
and a new column beside them. The columns other steps need come the same way
when those steps arrive - `source` for MCP in phase 5. The third entry is
`settings` (2.6): not the user's settings, which are `config.toml`, but what
the program itself found out and wants to keep between runs - the probe's
verdict on a model, so that it is not asked again on every start.
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
    # 2 - the repeat check and the bill (section 3.11, section 6).
    # `args_hash` is the fingerprint of a call's arguments, so that "the same
    # call a minute ago" is one indexed lookup; rows from before this
    # migration have none and are never "the same call" again, which is the
    # honest reading of them. `usage_log` is one row per turn that reached
    # the model; `cost_usd` is NULL for a model with no known price, never a
    # made-up zero.
    """
    ALTER TABLE tool_audit ADD COLUMN args_hash TEXT;
    CREATE INDEX tool_audit_repeat ON tool_audit(tool, args_hash, ts);
    CREATE TABLE usage_log (
        id            INTEGER PRIMARY KEY,
        ts            INTEGER NOT NULL,
        provider      TEXT    NOT NULL,
        model         TEXT    NOT NULL,
        in_tokens     INTEGER NOT NULL,
        out_tokens    INTEGER NOT NULL,
        cached_tokens INTEGER NOT NULL,
        cost_usd      REAL,
        turn_id       TEXT    NOT NULL
    );
    CREATE INDEX usage_log_ts ON usage_log(ts);
    """,
    # 3 - settings: what the program found out for itself (section 3.2, 2.6).
    # One row per key, the value one line of JSON; the probe writes
    # `probe:<provider>:<model>` with the verdict and the time it was
    # reached, and reads it back for a week. Nothing here is written by the
    # user - their settings are `config.toml` (section 3.3), where a text
    # editor can reach them.
    """
    CREATE TABLE settings (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
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
