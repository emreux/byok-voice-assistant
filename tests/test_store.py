"""The database opens itself, builds itself, and writes what it is told (2.1d).

Three claims. Opening a database brings its schema to the current version
and does so only once - the second start of the program changes nothing. A
migration is all or nothing, because a file left between two versions is a
file no build can read. And the audit repository writes rows whose shape
section 3.7 fixes: epoch seconds in UTC, arguments as one line of JSON, a
`started` row that has no end until something ends it.

Every test opens its own database, in memory or under `tmp_path`; the one
under `%LOCALAPPDATA%` is never touched.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from assistant.config import data_dir
from assistant.llm.base import ToolCall
from assistant.store import migrations
from assistant.store.db import DB_FILE, database_path, open_database
from assistant.store.migrations import MIGRATIONS, migrate, schema_version
from assistant.store.repos import SUMMARY_CHARS, AuditRepo

CALL = ToolCall(id="c1", name="open_app", arguments={"name": "Spotify", "args": "ş"})


@pytest.fixture
def database() -> Iterator[sqlite3.Connection]:
    connection = open_database(":memory:")
    yield connection
    connection.close()


def tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def indexes(connection: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }


def rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute("SELECT * FROM tool_audit ORDER BY id")]


# --------------------------------------------------------------------------
# Opening and migrating
# --------------------------------------------------------------------------


def test_the_first_run_builds_the_schema(database: sqlite3.Connection) -> None:
    assert "tool_audit" in tables(database)
    assert "tool_audit_turn" in indexes(database)
    assert schema_version(database) == len(MIGRATIONS)


def test_the_second_run_finds_nothing_to_do(database: sqlite3.Connection) -> None:
    """Every start of the program calls `migrate`; a migration that ran twice
    would fail on the table it already made."""
    assert migrate(database) == len(MIGRATIONS)
    assert schema_version(database) == len(MIGRATIONS)


def test_the_file_is_made_where_it_is_told_with_its_directory(tmp_path: Path) -> None:
    path = tmp_path / "data" / "assistant.db"

    connection = open_database(path)
    try:
        assert path.is_file()
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        connection.close()


def test_a_commit_does_not_wait_for_the_disk(tmp_path: Path) -> None:
    """`synchronous=NORMAL`, SQLite's own recommendation under WAL: the row
    survives a crash of the program, which is the crash it is written for,
    and a write on the event loop costs a tenth of a millisecond instead of
    six (measured 2026-09-09). The setting is per connection, so it is
    checked on one that was just opened."""
    connection = open_database(tmp_path / "assistant.db")
    try:
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1
    finally:
        connection.close()


def test_the_default_path_is_beside_the_log_not_beside_the_settings() -> None:
    """`%LOCALAPPDATA%`, not `%APPDATA%`: what a machine did stays on it."""
    assert database_path() == data_dir() / DB_FILE


def test_columns_are_read_by_name(database: sqlite3.Connection) -> None:
    row = database.execute("SELECT 1 AS one").fetchone()

    assert row["one"] == 1


def test_a_migration_that_fails_halfway_leaves_nothing_of_itself(
    database: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The version is bumped in the same transaction as the schema change,
    so a file is always at a whole number of migrations."""
    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        (*MIGRATIONS, "CREATE TABLE half (a);\nTHIS IS NOT SQL;"),
    )

    with pytest.raises(sqlite3.Error):
        migrate(database)

    assert schema_version(database) == len(MIGRATIONS)
    assert "half" not in tables(database)
    assert not database.in_transaction


def test_a_database_from_a_newer_build_is_left_as_it_is(database: sqlite3.Connection) -> None:
    """The list is append-only, so what this build needs is already there."""
    database.execute("PRAGMA user_version = 40")

    assert migrate(database) == 40


# --------------------------------------------------------------------------
# The audit repository
# --------------------------------------------------------------------------


@pytest.fixture
def audit(database: sqlite3.Connection) -> AuditRepo:
    return AuditRepo(database, clock=lambda: 1_700_000_000.9)


def test_a_started_call_has_a_beginning_and_no_end(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    row_id = audit.start(CALL, turn_id="t1", risk="confirm")

    [row] = rows(database)
    assert row["id"] == row_id
    assert (row["status"], row["approved"], row["finished_at"]) == ("started", 1, None)
    assert (row["turn_id"], row["tool"], row["risk"]) == ("t1", "open_app", "confirm")
    assert row["ts"] == 1_700_000_000, "epoch seconds, whole"


def test_the_arguments_are_one_line_of_json_in_a_settled_order(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    """Keys sorted and nothing escaped: the same call always reads the same,
    and a Turkish letter reads as itself."""
    audit.start(CALL, turn_id="t1", risk="safe")

    assert rows(database)[0]["args_json"] == '{"args":"ş","name":"Spotify"}'


def test_finishing_ok_keeps_the_result_and_the_time(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    row_id = audit.start(CALL, turn_id="t1", risk="safe")

    audit.finish(row_id, status="ok", summary="Spotify opened")

    [row] = rows(database)
    assert (row["status"], row["result_summary"], row["error"]) == ("ok", "Spotify opened", None)
    assert row["finished_at"] == 1_700_000_000


def test_finishing_in_error_keeps_the_kind_of_error(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    row_id = audit.start(CALL, turn_id="t1", risk="safe")

    audit.finish(row_id, status="error", error="TimeoutError")

    [row] = rows(database)
    assert (row["status"], row["error"], row["result_summary"]) == ("error", "TimeoutError", None)


def test_a_long_result_is_kept_only_up_to_the_summary_length(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    """A web page is not an audit entry; section 3.7 empties even this later."""
    row_id = audit.start(CALL, turn_id="t1", risk="safe")

    audit.finish(row_id, status="ok", summary="x" * (SUMMARY_CHARS * 3))

    assert len(rows(database)[0]["result_summary"]) == SUMMARY_CHARS


def test_a_denied_call_is_finished_as_it_is_written(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    audit.deny(CALL, turn_id="t1", risk="blocked")

    [row] = rows(database)
    assert (row["status"], row["approved"], row["risk"]) == ("denied", 0, "blocked")
    assert row["finished_at"] == row["ts"] == 1_700_000_000


def test_each_row_gets_its_own_id(database: sqlite3.Connection, audit: AuditRepo) -> None:
    first = audit.start(CALL, turn_id="t1", risk="safe")
    second = audit.deny(CALL, turn_id="t1", risk="safe")

    assert first != second
    assert [row["id"] for row in rows(database)] == [first, second]


def test_the_clock_is_the_machine_s_unless_a_test_says_otherwise(
    database: sqlite3.Connection,
) -> None:
    """The default clock is `time.time`: epoch seconds, UTC, the same for
    every row and every reader (section 3.10)."""
    AuditRepo(database).start(CALL, turn_id="t1", risk="safe")

    assert rows(database)[0]["ts"] > 1_700_000_000
