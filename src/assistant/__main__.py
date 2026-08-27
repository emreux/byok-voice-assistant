"""Command line entry point.

Phase 0 scaffolding: the parser and the command names are final, the commands
themselves are not implemented yet. `setup` and `run` are filled in during
phase 1 (design.md section 8, items 1.4 and 1.11); `doctor` and `cost` arrive
in phase 3 and phase 2 respectively.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from assistant import __version__

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

    print(
        f"'{args.command}' is not implemented yet - this is phase 0 scaffolding. "
        f"See design.md section 8 for the roadmap.",
        file=sys.stderr,
    )
    return _NOT_IMPLEMENTED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
