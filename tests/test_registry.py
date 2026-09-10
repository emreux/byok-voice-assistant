"""Turning a name in a file into a working adapter.

The registry is the seam that keeps the provider list out of the code
(design.md section 3.2): a catalogue entry, a key from the Credential Manager,
and an adapter comes back. What it must never do is fail vaguely - "KeyError:
'gemini'" tells the owner nothing, while "no API key stored, run assistant
setup" tells them exactly what to do next.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.config import store_api_key
from assistant.llm.base import LLMProvider
from assistant.llm.gemini_adapter import GeminiAdapter
from assistant.llm.openai_compat_adapter import OpenAICompatAdapter
from assistant.llm.registry import (
    ADAPTERS,
    MissingAPIKeyError,
    MissingBaseURLError,
    ProviderEntry,
    UnknownProviderError,
    UnsupportedAdapterError,
    create_provider,
    load_catalog,
)
from tests.conftest import MemoryKeyring


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


def test_the_shipped_catalogue_offers_the_openai_compatible_six() -> None:
    """Section 3.2's list, each under the address the adapter speaks to."""
    entries = load_catalog()

    assert {p for p, e in entries.items() if e.adapter == "openai_compat"} == {
        "openai",
        "openrouter",
        "groq",
        "deepseek",
        "ollama",
        "custom",
    }
    assert entries["groq"].base_url == "https://api.groq.com/openai/v1"
    assert entries["ollama"].requires_key is False
    assert entries["custom"].base_url is None


def test_every_paid_provider_says_where_its_key_comes_from() -> None:
    """The wizard shows the address; an entry without one leaves the user
    to search for it."""
    for provider_id, entry in load_catalog().items():
        if entry.requires_key and provider_id != "custom":
            assert entry.key_url, f"{provider_id} has no key_url"


def test_every_provider_offered_can_actually_be_built() -> None:
    """A catalogue entry naming an adapter from a later phase is a dead end
    the user only discovers after typing their key in."""
    for provider_id, entry in load_catalog().items():
        assert entry.adapter in ADAPTERS, f"{provider_id} names a missing adapter"


def test_every_shipped_entry_builds_with_a_key_and_an_address(vault: MemoryKeyring) -> None:
    """Built for real, adapter and all - only the network is never touched."""
    for provider_id in load_catalog():
        provider = create_provider(provider_id, api_key="k", base_url="http://x.test/v1")

        assert isinstance(provider, LLMProvider), provider_id


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
    """Phase 4.5 adds `anthropic`; until then the entry must fail clearly."""
    catalog = {"claude": ProviderEntry(id="claude", adapter="anthropic", display_name="Claude")}

    with pytest.raises(UnsupportedAdapterError, match="anthropic"):
        create_provider("claude", api_key="test-key", catalog=catalog)


# --------------------------------------------------------------------------
# The OpenAI-compatible entries (2.7)
# --------------------------------------------------------------------------


def test_an_openai_compatible_entry_becomes_the_one_adapter_under_its_address() -> None:
    provider = create_provider("groq", api_key="gsk-test")

    assert isinstance(provider, OpenAICompatAdapter)
    assert provider.id == "openai_compat"


def test_a_local_server_is_built_without_a_key(vault: MemoryKeyring) -> None:
    """Ollama: `requires_key = false` in the catalogue, nothing in the
    Credential Manager, and it still builds."""
    provider = create_provider("ollama")

    assert isinstance(provider, OpenAICompatAdapter)


def test_the_custom_entry_needs_an_address_and_says_so_without_one(vault: MemoryKeyring) -> None:
    """`custom` has no address of its own: the wizard asks for one and
    `config.toml` keeps it. Before that it is refused in a sentence that
    names the fix, like a missing key is."""
    with pytest.raises(MissingBaseURLError, match="assistant setup"):
        create_provider("custom", api_key="k")


def test_an_address_given_by_the_caller_outranks_the_catalogue_s() -> None:
    """How `custom` gets its address - and, for any entry, how a test or
    a proxy points it somewhere else."""
    provider = create_provider("custom", api_key="k", base_url="http://localhost:1234/v1")

    assert isinstance(provider, OpenAICompatAdapter)


def test_an_empty_address_from_the_caller_is_no_address() -> None:
    with pytest.raises(MissingBaseURLError):
        create_provider("custom", api_key="k", base_url="")
