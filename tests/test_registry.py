"""Turning a name in a file into a working adapter.

The registry is the seam that keeps the provider list out of the code
(design.md section 3.2): a catalogue entry, a key from the Credential Manager,
and an adapter comes back. What it must never do is fail vaguely - "KeyError:
'gemini'" tells the owner nothing, while "no API key stored, run assistant
setup" tells them exactly what to do next.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import keyring
import pytest

from assistant.config import store_api_key
from assistant.llm.base import LLMProvider
from assistant.llm.gemini_adapter import GeminiAdapter
from assistant.llm.registry import (
    ADAPTERS,
    MissingAPIKeyError,
    ProviderEntry,
    UnknownProviderError,
    UnsupportedAdapterError,
    create_provider,
    load_catalog,
)
from tests.test_config import MemoryKeyring


@pytest.fixture
def vault() -> Iterator[MemoryKeyring]:
    previous = keyring.get_keyring()
    fake = MemoryKeyring()
    keyring.set_keyring(fake)
    try:
        yield fake
    finally:
        keyring.set_keyring(previous)


@pytest.fixture
def keys_handed_over(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Registers an adapter that records the key it was built with.

    The Gemini client refuses an empty key at construction, so the paths that
    are about key handling rather than about Gemini are exercised through this
    one instead. It is a real `GeminiAdapter`; only its vendor client is inert.
    """
    seen: list[str] = []

    def build(entry: ProviderEntry, api_key: str) -> LLMProvider:
        seen.append(api_key)
        return GeminiAdapter(api_key=api_key, client=object())

    monkeypatch.setitem(ADAPTERS, "recording", build)
    return seen


def recording_catalog(*, requires_key: bool = True) -> dict[str, ProviderEntry]:
    return {
        "x": ProviderEntry(id="x", adapter="recording", display_name="X", requires_key=requires_key)
    }


# --------------------------------------------------------------------------
# The catalogue that ships with the package
# --------------------------------------------------------------------------


def test_the_shipped_catalogue_offers_gemini() -> None:
    entry = load_catalog()["gemini"]

    assert entry.adapter == "gemini"
    assert entry.requires_key is True
    assert entry.key_url is not None


def test_every_provider_offered_can_actually_be_built() -> None:
    """A catalogue entry naming an adapter from a later phase is a dead end
    the user only discovers after typing their key in."""
    for provider_id, entry in load_catalog().items():
        assert entry.adapter in ADAPTERS, f"{provider_id} names a missing adapter"


def test_the_catalogue_can_be_read_from_a_given_file(tmp_path: Path) -> None:
    catalogue = tmp_path / "providers.toml"
    catalogue.write_text('[groq]\nadapter = "openai_compat"\ndisplay_name = "Groq"\n', "utf-8")

    entries = load_catalog(catalogue)

    assert entries["groq"].display_name == "Groq"
    assert entries["groq"].id == "groq"


def test_an_entry_may_leave_the_optional_fields_out(tmp_path: Path) -> None:
    """Only a local provider needs `base_url`; only a paid one needs `key_url`."""
    catalogue = tmp_path / "providers.toml"
    catalogue.write_text('[x]\nadapter = "gemini"\ndisplay_name = "X"\n', "utf-8")

    entry = load_catalog(catalogue)["x"]

    assert (entry.base_url, entry.key_url, entry.key_prefix) == (None, None, None)


def test_a_field_from_a_later_version_does_not_break_the_catalogue(tmp_path: Path) -> None:
    """Section 3.2 already foresees an `auth` field; an old build must still start."""
    catalogue = tmp_path / "providers.toml"
    catalogue.write_text('[x]\nadapter = "gemini"\ndisplay_name = "X"\nauth = "aws"\n', "utf-8")

    assert load_catalog(catalogue)["x"].adapter == "gemini"


# --------------------------------------------------------------------------
# Building an adapter
# --------------------------------------------------------------------------


def test_a_provider_id_becomes_an_adapter() -> None:
    provider = create_provider("gemini", api_key="test-key")

    assert isinstance(provider, LLMProvider)
    assert provider.id == "gemini"


def test_the_key_comes_from_the_credential_manager(
    vault: MemoryKeyring, keys_handed_over: list[str]
) -> None:
    store_api_key("x", "stored-key")

    create_provider("x", catalog=recording_catalog())

    assert keys_handed_over == ["stored-key"]


def test_an_explicit_key_is_used_as_given(
    vault: MemoryKeyring, keys_handed_over: list[str]
) -> None:
    """Setup validates a key before storing it, so it must be able to pass one in."""
    store_api_key("x", "stored-key")

    create_provider("x", api_key="typed-key", catalog=recording_catalog())

    assert keys_handed_over == ["typed-key"]


def test_without_a_stored_key_the_error_says_what_to_run(vault: MemoryKeyring) -> None:
    with pytest.raises(MissingAPIKeyError, match="assistant setup"):
        create_provider("gemini")


def test_a_provider_that_needs_no_key_is_built_without_one(
    vault: MemoryKeyring, keys_handed_over: list[str]
) -> None:
    """A local endpoint - Ollama, LM Studio - has nothing to authenticate with."""
    create_provider("x", catalog=recording_catalog(requires_key=False))

    assert keys_handed_over == [""]


def test_an_unknown_provider_names_the_ones_that_exist() -> None:
    with pytest.raises(UnknownProviderError, match="gemini"):
        create_provider("claude")


def test_a_provider_whose_adapter_is_not_written_yet_says_so() -> None:
    """Phase 2 adds `openai_compat`; until then the entry must fail clearly."""
    catalog = {"groq": ProviderEntry(id="groq", adapter="openai_compat", display_name="Groq")}

    with pytest.raises(UnsupportedAdapterError, match="openai_compat"):
        create_provider("groq", api_key="test-key", catalog=catalog)
