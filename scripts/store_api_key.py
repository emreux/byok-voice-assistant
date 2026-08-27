"""Stores a provider API key in the Windows Credential Manager (design.md section 3.3).

Run this once so the key never appears in a file, a shell history entry or a
chat log:

    uv run python scripts/store_api_key.py            # defaults to gemini
    uv run python scripts/store_api_key.py --provider openrouter

The key is typed blind, written straight to the credential store under the
service name `assistant` and the provider id, and read back only to confirm it
arrived. Phase 1.4 replaces this script with `assistant setup`, which also
validates the key against the provider and lists the models; until then this
is the safe way to get a key onto the machine.

    uv run python scripts/store_api_key.py --check    # is a key already stored?
    uv run python scripts/store_api_key.py --delete   # remove it
"""

from __future__ import annotations

import argparse
import getpass
import sys

SERVICE = "assistant"
DEFAULT_PROVIDER = "gemini"


def main() -> int:
    """Stores, checks or deletes the key for one provider."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--check", action="store_true", help="Report whether a key is stored.")
    parser.add_argument("--delete", action="store_true", help="Delete the stored key.")
    arguments = parser.parse_args()

    import keyring

    stored = keyring.get_password(SERVICE, arguments.provider)

    if arguments.check:
        if stored:
            print(f"A key for '{arguments.provider}' is stored ({len(stored)} characters).")
            return 0
        print(f"No key is stored for '{arguments.provider}'.")
        return 1

    if arguments.delete:
        if not stored:
            print(f"Nothing to delete for '{arguments.provider}'.")
            return 0
        keyring.delete_password(SERVICE, arguments.provider)
        print(f"Deleted the key for '{arguments.provider}'.")
        return 0

    if stored:
        answer = input(f"A key for '{arguments.provider}' already exists. Replace it? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Left the existing key alone.")
            return 0

    key = getpass.getpass(f"Paste the {arguments.provider} API key (input stays hidden): ").strip()
    if not key:
        print("Nothing entered - no key was stored.", file=sys.stderr)
        return 1

    keyring.set_password(SERVICE, arguments.provider, key)

    if keyring.get_password(SERVICE, arguments.provider) != key:
        print("The credential store did not return what was written.", file=sys.stderr)
        return 1

    print(f"Stored {len(key)} characters for '{arguments.provider}' in the Credential Manager.")
    print("Nothing was written to disk. Verify with: uv run python scripts/check_provider.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
