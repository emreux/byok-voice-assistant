"""Checks that the stored key really talks to the provider (design.md section 8, phase 0.9).

Reads the key from the Windows Credential Manager - it is never passed on the
command line - asks the provider for its model list, and prints the models that
can hold a conversation.

    uv run python scripts/store_api_key.py     # once, to store the key
    uv run python scripts/check_provider.py    # any time, to prove it works

Phase 1.3 moves this logic into `llm/registry.py` and `llm/gemini_adapter.py`,
where `validate_credentials()` and `list_models()` belong. Until those exist,
this script is the only thing that proves the key works.
"""

from __future__ import annotations

import argparse

SERVICE = "assistant"
DEFAULT_PROVIDER = "gemini"
GENERATE_METHOD = "generateContent"
SHOW_LIMIT = 15


def check_gemini(key: str) -> int:
    """Lists the Gemini models that support text generation."""
    from google import genai

    client = genai.Client(api_key=key)
    models = list(client.models.list())

    usable = [
        model
        for model in models
        if GENERATE_METHOD in (model.supported_actions or [])
        and model.name is not None
        and "embedding" not in model.name
    ]

    print(f"The key works: {len(models)} models visible, {len(usable)} can generate text.\n")
    for model in usable[:SHOW_LIMIT]:
        name = (model.name or "").removeprefix("models/")
        window = model.input_token_limit
        window_text = f"{window:,} token input".replace(",", " ") if window else "unknown window"
        print(f"  {name:<42} {window_text}")

    if len(usable) > SHOW_LIMIT:
        print(f"  ... and {len(usable) - SHOW_LIMIT} more")

    return 0 if usable else 1


def main() -> int:
    """Reads the stored key and runs the provider specific check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    arguments = parser.parse_args()

    import keyring

    key = keyring.get_password(SERVICE, arguments.provider)
    if not key:
        print(
            f"No key is stored for '{arguments.provider}'.\n"
            f"Run: uv run python scripts/store_api_key.py --provider {arguments.provider}"
        )
        return 1

    if arguments.provider != "gemini":
        print(f"Only 'gemini' is wired up in phase 1; '{arguments.provider}' arrives in phase 2.7.")
        return 1

    try:
        return check_gemini(key)
    except Exception as error:
        print(f"The provider rejected the key or could not be reached:\n  {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
