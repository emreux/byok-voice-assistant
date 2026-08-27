"""Where settings live, what shape they have, and where the key does not go.

Two claims are worth a test each. The first is that a directory this code
writes to is the one design.md section 3.3 names - `platformdirs` is called
with arguments, and the wrong ones silently produce a different, plausible
looking path. The second is that the API key never reaches the settings file;
that is the whole reason `keyring` is a dependency.

The keyring tests run against a backend held in memory. The library is real,
so the calls are the calls production makes; only the vault is fake, because a
test suite has no business writing to the user's Credential Manager.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Iterator
from pathlib import Path

import keyring
import pytest
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError

from assistant.config import (
    KEYRING_SERVICE,
    LLMSettings,
    LocaleSettings,
    Settings,
    config_dir,
    config_path,
    data_dir,
    delete_api_key,
    is_configured,
    load_api_key,
    load_settings,
    log_dir,
    save_settings,
    store_api_key,
)

WINDOWS_ONLY = pytest.mark.skipif(os.name != "nt", reason="the product ships on Windows")


class MemoryKeyring(KeyringBackend):
    """A real keyring backend that forgets everything when the test ends."""

    priority = 1.0

    def __init__(self) -> None:
        # `keyring` is typed but this constructor is not; calling it anyway
        # keeps the fake on the same code path as a real backend.
        super().__init__()  # type: ignore[no-untyped-call]
        self.vault: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self.vault[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self.vault.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self.vault:
            raise PasswordDeleteError(f"nothing stored for {username}")
        del self.vault[(service, username)]


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
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ASSISTANT_CONFIG_DIR", str(tmp_path))
    return tmp_path


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def test_the_environment_variable_moves_the_whole_config_directory(config_home: Path) -> None:
    """Tests need somewhere to write that is not the developer's own settings."""
    assert config_dir() == config_home
    assert config_path() == config_home / "config.toml"


@WINDOWS_ONLY
def test_settings_land_in_appdata_under_a_single_assistant_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ASSISTANT_CONFIG_DIR", raising=False)

    assert config_dir() == Path(os.environ["APPDATA"]) / "assistant"


@WINDOWS_ONLY
def test_data_and_logs_are_local_not_roaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """A database and a log file must not follow the user to another machine."""
    monkeypatch.delenv("ASSISTANT_CONFIG_DIR", raising=False)
    local = Path(os.environ["LOCALAPPDATA"]) / "assistant"

    assert data_dir() == local
    assert log_dir() == local / "Logs"
    assert data_dir() != config_dir()


def test_reading_settings_creates_nothing(config_home: Path) -> None:
    """Merely asking what the settings are must not litter the disk."""
    load_settings()

    assert list(config_home.iterdir()) == []


# --------------------------------------------------------------------------
# The settings themselves
# --------------------------------------------------------------------------


def test_before_setup_the_settings_load_with_defaults(config_home: Path) -> None:
    """The first run has no file; it must not be an error."""
    settings = load_settings()

    assert settings.llm.primary == ""
    assert is_configured() is False


def test_after_setup_the_settings_come_back(config_home: Path) -> None:
    save_settings(Settings(llm=LLMSettings(primary="gemini:gemini-3.5-flash-lite")))

    settings = load_settings()

    assert settings.llm.primary == "gemini:gemini-3.5-flash-lite"
    assert is_configured() is True


def test_the_saved_file_is_toml(config_home: Path) -> None:
    save_settings(Settings(llm=LLMSettings(primary="gemini:x"), locale=LocaleSettings(code="tr")))

    written = tomllib.loads(config_path().read_text(encoding="utf-8"))

    assert written["llm"]["primary"] == "gemini:x"
    assert written["locale"]["code"] == "tr"


def test_a_value_with_quotes_survives_the_round_trip(config_home: Path) -> None:
    """There is no TOML writer in the standard library, so ours must escape."""
    awkward = 'custom:he said "hi" \\ back'
    save_settings(Settings(llm=LLMSettings(primary=awkward)))

    assert load_settings().llm.primary == awkward


def test_the_provider_and_the_model_are_read_off_one_line() -> None:
    settings = LLMSettings(primary="gemini:gemini-3.5-flash-lite")

    assert settings.provider == "gemini"
    assert settings.model == "gemini-3.5-flash-lite"


def test_a_model_name_may_contain_colons_and_slashes() -> None:
    """`ollama:llama3:8b` and `openrouter:google/gemini-2.5-flash` are real names."""
    assert LLMSettings(primary="ollama:llama3:8b").model == "llama3:8b"
    assert LLMSettings(primary="openrouter:google/gemini-2.5-flash").provider == "openrouter"


def test_a_primary_without_a_provider_is_refused() -> None:
    """Left unchecked this reads as a provider named after the model, with no model."""
    with pytest.raises(ValueError, match="provider:model"):
        LLMSettings(primary="gemini-3.5-flash-lite")


def test_the_locale_defaults_to_english(config_home: Path) -> None:
    """The fallback chain of section 3.12 ends at `en`, so it is the safe default."""
    assert load_settings().locale.code == "en"


def test_an_unknown_table_in_the_file_does_not_stop_the_assistant(config_home: Path) -> None:
    """A newer version's key, or a typo, is not worth refusing to start over."""
    config_path().write_text('[llm]\nprimary = "gemini:x"\n[watchers]\nkap = true\n', "utf-8")

    assert load_settings().llm.primary == "gemini:x"


def test_the_environment_can_override_a_setting(
    config_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """So a second model can be tried without editing the file."""
    save_settings(Settings(llm=LLMSettings(primary="gemini:one")))
    monkeypatch.setenv("ASSISTANT_LLM__PRIMARY", "gemini:two")

    assert load_settings().llm.primary == "gemini:two"


# --------------------------------------------------------------------------
# The key
# --------------------------------------------------------------------------


def test_the_key_is_stored_and_read_back(vault: MemoryKeyring) -> None:
    store_api_key("gemini", "AQ.secret")

    assert load_api_key("gemini") == "AQ.secret"


def test_the_key_is_filed_under_the_provider_id(vault: MemoryKeyring) -> None:
    """Setup writes it and the registry reads it; they must agree on the name."""
    store_api_key("gemini", "AQ.secret")

    assert vault.vault == {(KEYRING_SERVICE, "gemini"): "AQ.secret"}


def test_no_key_stored_reads_as_none(vault: MemoryKeyring) -> None:
    assert load_api_key("gemini") is None


def test_deleting_a_key_that_was_never_there_is_not_an_error(vault: MemoryKeyring) -> None:
    """`assistant purge` must not fail on a provider the user never configured."""
    delete_api_key("gemini")

    assert load_api_key("gemini") is None


def test_deleting_a_key_removes_it(vault: MemoryKeyring) -> None:
    store_api_key("gemini", "AQ.secret")

    delete_api_key("gemini")

    assert load_api_key("gemini") is None


def test_the_key_never_reaches_the_settings_file(config_home: Path, vault: MemoryKeyring) -> None:
    """The one claim section 3.3 makes about key safety."""
    store_api_key("gemini", "AQ.secret")
    save_settings(Settings(llm=LLMSettings(primary="gemini:gemini-3.5-flash-lite")))

    assert "AQ.secret" not in config_path().read_text(encoding="utf-8")


def test_the_settings_model_has_nowhere_to_put_a_key() -> None:
    """Not an oversight to be fixed later: the field is absent on purpose."""
    assert "api_key" not in LLMSettings.model_fields
    assert not any("key" in name for name in Settings.model_fields)
