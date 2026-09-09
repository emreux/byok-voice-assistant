"""A risky tool cannot run unconfirmed, and nothing runs that the gate did not let through.

This is the test CLAUDE.md points at for the one permission gate of section
3.9. Every claim is made with a fake `Confirm` that remembers what it was
asked, and with tools that leave a mark when their body runs - so "did not
run" is something the test can see, not something it assumes.

The second half is the audit trail of 2.1d. The claim there is one of order:
the row is on disk *before* the tool's body starts, which is proved by a tool
that reads the table from inside its own body. Everything else - `ok`,
`error`, `denied`, what the arguments look like - follows from that row.

The last two tests put the real `Confirm` behind the gate: the state
machine's window of 2.3, over a fake microphone. Silence there is what
"unconfirmed" means in life, and the tool still does not run.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest

from assistant.agent.core import Confirm
from assistant.agent.policy import (
    DECLINED,
    DISABLED,
    FAILED,
    MISSING_ARGUMENT,
    NO_SUCH_TOOL,
    dispatch,
)
from assistant.llm.base import ToolCall, ToolSpec
from assistant.store.db import open_database
from assistant.store.repos import AuditRepo
from assistant.stt.base import Transcript
from assistant.tools.registry import Tool, ToolRegistry, tool
from tests.test_app import FakeCapture, FakeSTT, assistant_with, speech

TURN = "turn-1"


class FakeConfirm:
    """Answers every question the same way, and keeps the questions."""

    def __init__(self, *, answer: bool) -> None:
        self.answer = answer
        self.asked: list[str] = []

    async def __call__(self, prompt: str) -> bool:
        self.asked.append(prompt)
        return self.answer


# What ran, in order. A tool body that executed writes its name here.
ran: list[str] = []


@tool(risk="safe")
async def get_current_time() -> str:
    """Returns the current local time."""
    ran.append("get_current_time")
    return "09:12"


@tool(risk="confirm", confirm_prompt="{name} will be opened.")
async def open_app(name: str) -> str:
    """Opens an application."""
    ran.append(f"open_app:{name}")
    return f"{name} opened"


@tool(risk="blocked", confirm_prompt="{path} will be deleted.")
async def delete_file(path: str) -> str:
    """Deletes a file."""
    ran.append(f"delete_file:{path}")
    return "deleted"


@tool(risk="safe")
async def broken() -> str:
    """Always fails."""
    raise RuntimeError("boom")


REGISTRY = ToolRegistry([get_current_time, open_app, delete_file, broken])


@pytest.fixture(autouse=True)
def _no_marks_left_over() -> Iterator[None]:
    ran.clear()
    yield
    ran.clear()


def call(tool_name: str, **arguments: str) -> ToolCall:
    """A model's order for `tool_name`; the keyword arguments are what it chose."""
    return ToolCall(id="c1", name=tool_name, arguments=arguments)


async def gate(
    order: ToolCall,
    *,
    confirm: Confirm,
    registry: ToolRegistry = REGISTRY,
    unblocked: list[str] | None = None,
    audit: AuditRepo | None = None,
) -> str:
    """`dispatch` with this file's defaults filled in."""
    return await dispatch(
        order,
        turn_id=TURN,
        registry=registry,
        confirm=confirm,
        unblocked=unblocked or [],
        audit=audit,
    )


# --------------------------------------------------------------------------
# confirm
# --------------------------------------------------------------------------


async def test_a_confirm_tool_does_not_run_when_the_user_says_no() -> None:
    confirm = FakeConfirm(answer=False)

    answer = await gate(call("open_app", name="Spotify"), confirm=confirm)

    assert answer == DECLINED
    assert ran == []


async def test_a_confirm_tool_runs_when_the_user_says_yes() -> None:
    confirm = FakeConfirm(answer=True)

    answer = await gate(call("open_app", name="Spotify"), confirm=confirm)

    assert answer == "Spotify opened"
    assert ran == ["open_app:Spotify"]


async def test_the_user_hears_the_real_arguments_before_saying_yes() -> None:
    """The injection defence itself: whatever the model put in the arguments
    is what the user is asked about, word for word."""
    confirm = FakeConfirm(answer=True)

    await gate(call("open_app", name="attacker.exe"), confirm=confirm)

    assert confirm.asked == ["attacker.exe will be opened."]


async def test_a_confirm_tool_built_without_a_prompt_cannot_be_asked_so_cannot_run() -> None:
    """`tool()` refuses to build one; this one is made by hand to prove the
    gate fails closed on its own, not only because of the decorator."""
    by_hand = Tool(
        spec=ToolSpec(name="odd", description="Made by hand.", parameters={}),
        risk="confirm",
        run=get_current_time.run,
    )
    confirm = FakeConfirm(answer=True)

    answer = await gate(call("odd"), confirm=confirm, registry=ToolRegistry([by_hand]))

    assert answer == DISABLED
    assert confirm.asked == []
    assert ran == []


async def test_a_call_missing_what_the_question_needs_is_answered_in_words() -> None:
    """The model left the name out. A `KeyError` out of the gate would end
    the turn with a traceback; a sentence back to the model lets it try
    again with the argument."""
    confirm = FakeConfirm(answer=True)

    answer = await gate(call("open_app"), confirm=confirm)

    assert answer == MISSING_ARGUMENT.format(name="name")
    assert confirm.asked == []
    assert ran == []


# --------------------------------------------------------------------------
# blocked, safe, unknown, failing
# --------------------------------------------------------------------------


async def test_a_blocked_tool_does_not_run_even_when_the_user_would_say_yes() -> None:
    confirm = FakeConfirm(answer=True)

    answer = await gate(call("delete_file", path="notes.md"), confirm=confirm)

    assert answer == DISABLED
    assert confirm.asked == []
    assert ran == []


async def test_an_unblocked_tool_still_asks_before_running() -> None:
    confirm = FakeConfirm(answer=True)

    answer = await gate(
        call("delete_file", path="notes.md"), confirm=confirm, unblocked=["delete_file"]
    )

    assert answer == "deleted"
    assert confirm.asked == ["notes.md will be deleted."]
    assert ran == ["delete_file:notes.md"]


async def test_a_safe_tool_runs_without_asking() -> None:
    confirm = FakeConfirm(answer=False)

    answer = await gate(call("get_current_time"), confirm=confirm)

    assert answer == "09:12"
    assert confirm.asked == []
    assert ran == ["get_current_time"]


async def test_a_tool_the_model_made_up_runs_nothing() -> None:
    confirm = FakeConfirm(answer=True)

    answer = await gate(call("format_disk"), confirm=confirm)

    assert answer == NO_SUCH_TOOL.format(name="format_disk")
    assert confirm.asked == []
    assert ran == []


async def test_a_tool_that_fails_answers_in_words_and_the_exception_stays_inside() -> None:
    answer = await gate(call("broken"), confirm=FakeConfirm(answer=True))

    assert answer == FAILED.format(kind="RuntimeError")


# --------------------------------------------------------------------------
# The audit trail: written before it runs (section 3.9)
# --------------------------------------------------------------------------


@pytest.fixture
def database() -> Iterator[sqlite3.Connection]:
    connection = open_database(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def audit(database: sqlite3.Connection) -> AuditRepo:
    return AuditRepo(database, clock=lambda: 1_700_000_000)


def rows(database: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(row) for row in database.execute("SELECT * FROM tool_audit ORDER BY id")]


async def test_the_row_is_on_disk_before_the_tool_body_runs(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    """The order is the whole design: a crash inside the tool leaves a row
    that says `started`, which is how "I do not know whether it worked" is
    ever going to be said (section 3.11). A tool that looks at the table
    from inside its own body is the only witness there is."""
    seen: list[dict[str, Any]] = []

    @tool(risk="safe")
    async def peek() -> str:
        """Looks at the audit table while running."""
        seen.extend(rows(database))
        return "peeked"

    await gate(
        call("peek"), confirm=FakeConfirm(answer=True), registry=ToolRegistry([peek]), audit=audit
    )

    assert len(seen) == 1
    assert (seen[0]["tool"], seen[0]["status"], seen[0]["finished_at"]) == ("peek", "started", None)


async def test_a_tool_that_ran_is_closed_as_ok_with_what_it_said(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    await gate(call("get_current_time"), confirm=FakeConfirm(answer=False), audit=audit)

    [row] = rows(database)
    assert (row["status"], row["approved"], row["result_summary"]) == ("ok", 1, "09:12")
    assert (row["turn_id"], row["tool"], row["risk"]) == (TURN, "get_current_time", "safe")
    assert row["finished_at"] == 1_700_000_000
    assert row["error"] is None


async def test_a_tool_that_failed_is_closed_as_error_with_the_kind_and_no_more(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    """The class name, never the message: a message can carry anything."""
    await gate(call("broken"), confirm=FakeConfirm(answer=True), audit=audit)

    [row] = rows(database)
    assert (row["status"], row["error"], row["result_summary"]) == ("error", "RuntimeError", None)
    assert row["finished_at"] is not None
    assert "boom" not in str(row)


async def test_a_refused_confirm_is_one_row_that_says_denied(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    await gate(call("open_app", name="Spotify"), confirm=FakeConfirm(answer=False), audit=audit)

    [row] = rows(database)
    assert (row["status"], row["approved"], row["risk"]) == ("denied", 0, "confirm")
    assert row["args_json"] == '{"name":"Spotify"}'
    assert row["finished_at"] == row["ts"]


async def test_a_blocked_tool_is_one_row_that_says_denied(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    await gate(call("delete_file", path="notes.md"), confirm=FakeConfirm(answer=True), audit=audit)

    [row] = rows(database)
    assert (row["status"], row["approved"], row["risk"]) == ("denied", 0, "blocked")


async def test_a_confirmed_call_is_approved_and_then_run(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    await gate(call("open_app", name="Spotify"), confirm=FakeConfirm(answer=True), audit=audit)

    [row] = rows(database)
    assert (row["status"], row["approved"], row["result_summary"]) == ("ok", 1, "Spotify opened")


async def test_a_tool_the_model_made_up_leaves_no_row(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    """No tool was judged, so there is nothing to write about."""
    await gate(call("format_disk"), confirm=FakeConfirm(answer=True), audit=audit)

    assert rows(database) == []


async def test_a_call_missing_an_argument_leaves_no_row(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    await gate(call("open_app"), confirm=FakeConfirm(answer=True), audit=audit)

    assert rows(database) == []


async def test_every_call_of_a_turn_is_filed_under_the_turn(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    confirm = FakeConfirm(answer=False)

    await gate(call("get_current_time"), confirm=confirm, audit=audit)
    await gate(call("open_app", name="Spotify"), confirm=confirm, audit=audit)

    assert [(row["tool"], row["status"], row["turn_id"]) for row in rows(database)] == [
        ("get_current_time", "ok", TURN),
        ("open_app", "denied", TURN),
    ]


async def test_without_a_repository_the_gate_writes_nothing_and_still_works() -> None:
    """Most tests, and nothing else: a gate without somewhere to write is a
    gate that judges exactly the same."""
    answer = await gate(call("get_current_time"), confirm=FakeConfirm(answer=False), audit=None)

    assert answer == "09:12"


# --------------------------------------------------------------------------
# The gate and the microphone together (2.3)
# --------------------------------------------------------------------------


async def test_a_confirm_tool_does_not_run_when_the_microphone_hears_nothing() -> None:
    """The claim CLAUDE.md names, with the real `Confirm` this time - the
    state machine's own window. Nobody answers, and the tool does not run."""
    assistant = assistant_with(capture=FakeCapture(), stt=FakeSTT())
    await assistant.begin()

    answer = await gate(call("open_app", name="Spotify"), confirm=assistant.confirm)

    assert answer == DECLINED
    assert ran == []


async def test_a_confirm_tool_runs_when_the_microphone_hears_yes() -> None:
    assistant = assistant_with(
        capture=FakeCapture(answers=[speech()]), stt=FakeSTT(Transcript(text="evet"))
    )
    await assistant.begin()

    answer = await gate(call("open_app", name="Spotify"), confirm=assistant.confirm)

    assert answer == "Spotify opened"
    assert ran == ["open_app:Spotify"]
