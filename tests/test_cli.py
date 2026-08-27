"""Phase 0 smoke test: the entry point is importable and the parser works."""

from __future__ import annotations

import pytest

from assistant.__main__ import build_parser, main


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "assistant" in capsys.readouterr().out


def test_no_command_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["setup", "run"])
def test_known_commands_are_declared_but_not_implemented(command: str) -> None:
    """The command names are final; their behaviour arrives in phase 1."""
    assert build_parser().parse_args([command]).command == command
    assert main([command]) == 2
