"""Command line entry point.

The parser and the command names are final. `setup` runs the wizard of item
1.4; `run` is item 1.11 and still says so rather than pretending. `doctor` and
`cost` arrive in phase 3 and phase 2 respectively (design.md section 8).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from assistant import __version__, setup_wizard

_NOT_IMPLEMENTED_EXIT_CODE = 2


def build_parser() -> argparse.ArgumentParser:
    """Builds the top level argument parser."""
    parser = argparse.ArgumentParser(
        prog="assistant",
        description="A voice assistant that runs on your own API key, model and language.",
    )
    parser.add_argument("--version", action="version", version=f"assistant {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.add_parser("setup", help="Choose a provider, store the API key, pick a model.")
    subparsers.add_parser("run", help="Start the assistant and listen for the hotkey.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parses the arguments and dispatches to a command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "setup":
        # Looked up on the module rather than imported by name so the tests can
        # stand in for it; the wizard itself opens a prompt and would hang.
        return asyncio.run(setup_wizard.run_setup(setup_wizard.TerminalPrompter()))

    print(
        f"'{args.command}' is not implemented yet - this is phase 0 scaffolding. "
        f"See design.md section 8 for the roadmap.",
        file=sys.stderr,
    )
    return _NOT_IMPLEMENTED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
