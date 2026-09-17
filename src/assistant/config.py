"""Where the user's settings live, and where their API key lives instead.

Two rules from design.md section 3.3 are enforced here and nowhere else.

**Settings are not in the repository.** They belong to the machine, under
`%APPDATA%\\assistant\\`; the database and the logs belong under
`%LOCALAPPDATA%\\assistant\\`, because a log file has no business following the
user to another computer through roaming profiles. `ASSISTANT_CONFIG_DIR`
redirects the first of those, which is how the tests avoid writing into the
developer's own settings.

**The API key is never written to a file.** It goes to the Windows Credential
Manager through `keyring`, so it is encrypted at rest and bound to the user
account. That is why `Settings` has no field to hold one: an absent field
cannot be filled in by accident, serialised into a log, or committed. The
store/load/delete functions at the bottom of this module are the only way in.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import keyring
import platformdirs
from keyring.errors import PasswordDeleteError
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from assistant.agent.limits import Limits
from assistant.media.youtube import SEARCH_SECONDS
from assistant.tools.web import SEARCH_URL

__all__ = [
    "CONFIG_DIR_ENV",
    "KEYRING_SERVICE",
    "RECOGNISERS",
    "AudioSettings",
    "LLMSettings",
    "LimitSettings",
    "LocaleSettings",
    "MediaSettings",
    "STTSettings",
    "Settings",
    "ToolSettings",
    "config_dir",
    "config_path",
    "data_dir",
    "delete_api_key",
    "is_configured",
    "load_api_key",
    "load_settings",
    "log_dir",
    "save_settings",
    "store_api_key",
]

APP_NAME = "assistant"
CONFIG_DIR_ENV = "ASSISTANT_CONFIG_DIR"
CONFIG_FILE_NAME = "config.toml"

# One entry per provider in the Credential Manager: service "assistant",
# user name = the provider id from providers.toml. Setup writes it, the
# registry reads it; the two agree only because both come through here.
KEYRING_SERVICE = APP_NAME


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def config_dir() -> Path:
    """The directory holding `config.toml` - `%APPDATA%\\assistant` by default.

    `appauthor=False` matters: without it `platformdirs` inserts a vendor level
    and the settings end up in `...\\assistant\\assistant`. `roaming=True`
    matters too - settings are small and worth carrying between machines,
    which is exactly what the roaming profile is for.
    """
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override)
    return Path(platformdirs.user_config_dir(APP_NAME, appauthor=False, roaming=True))


def config_path() -> Path:
    return config_dir() / CONFIG_FILE_NAME


def data_dir() -> Path:
    """The database, audio debug captures and anything else large and local."""
    return Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))


def log_dir() -> Path:
    return Path(platformdirs.user_log_dir(APP_NAME, appauthor=False))


# --------------------------------------------------------------------------
# The settings
# --------------------------------------------------------------------------


class LLMSettings(BaseModel):
    """Which model answers, written the way section 3.2 writes it.

    One line, `provider:model`, rather than two fields. The fallback chain of
    phase 4.5 lists alternatives in exactly this form, so a config file written
    today keeps its meaning when `fallback` is added next to it.
    """

    model_config = ConfigDict(extra="ignore")

    primary: str = ""

    # The address of the server, for the one catalogue entry that has none
    # of its own (`custom`, 2.7): the wizard asked for it and keeps it here.
    # Empty for every other provider, whose address is in `providers.toml`.
    base_url: str = ""

    @field_validator("primary")
    @classmethod
    def _must_name_a_provider(cls, value: str) -> str:
        if value and ":" not in value:
            raise ValueError(f"expected 'provider:model', got {value!r}")
        return value

    @property
    def provider(self) -> str:
        return self.primary.partition(":")[0]

    @property
    def model(self) -> str:
        """Everything after the first colon.

        Only the first: `ollama:llama3:8b` and `openrouter:google/gemini-2.5-flash`
        are real model names, and splitting on every colon would truncate them.
        """
        return self.primary.partition(":")[2]


class LocaleSettings(BaseModel):
    """Which locale pack the product speaks (section 3.12).

    Not the language the model replies in - that mirrors whatever the user
    just said and is a rule in the system prompt, not a setting.
    """

    model_config = ConfigDict(extra="ignore")

    code: str = "en"


class AudioSettings(BaseModel):
    """Which microphone, in `sounddevice`'s own words.

    Empty means the system default. Otherwise an index, or words matched in
    order against "<device name>, <host API>" - `Microphone Array WASAPI`.
    Words rather than an index by default: indices shift every time a
    Bluetooth device connects (measured: the WASAPI array was 12 one day and
    9 the next), and a setting that points at a different microphone after a
    headset pairs is worse than none.
    """

    model_config = ConfigDict(extra="ignore")

    input_device: str = ""


# The recognisers `[stt] provider` may name. `local` is Whisper on this
# machine and never leaves the list (ADR-001); `gemini` is on trial since
# 2026-09-14 and sends the microphone audio to Google.
RECOGNISERS = ("local", "gemini")


class STTSettings(BaseModel):
    """Which engine turns speech into text (section 3.4).

    `local` by default: no key, no cost, and the audio never leaves the
    machine - the sentence the README makes, and the one that stays true for
    everyone who did not change this. `gemini` is a choice made in this
    file, calmly, and it is the one setting here that changes where the
    voice goes; `model` names Google's recogniser and is only read then.
    """

    model_config = ConfigDict(extra="ignore")

    provider: str = "local"
    model: str = "gemini-3.5-transcribe-live"

    @field_validator("provider")
    @classmethod
    def _must_be_a_recogniser(cls, value: str) -> str:
        if value not in RECOGNISERS:
            raise ValueError(f"expected one of {', '.join(RECOGNISERS)}, got {value!r}")
        return value


class ToolSettings(BaseModel):
    """Which `blocked` tools the user switched on, by name (section 3.9).

    A tool declared `blocked` never runs unless it is named here, and even
    then it asks first. A list in a file is deliberately the only way to open
    one: editing the file is a decision made calmly, a spoken "yes" in the
    middle of a turn is not.
    """

    model_config = ConfigDict(extra="ignore")

    unblocked: list[str] = []


class MediaSettings(BaseModel):
    """The `[media]` table: where music comes from, and what happens first.

    `default_service` is what "play something" means when the user did not
    name a service. It is YouTube Music because that is the one that actually
    *plays* from here: a song resolves to a `watch?v=` address and starts by
    itself in the browser the user is signed in to. Spotify is reached the
    other way round - it can be handed words to search, not a track to play
    (`media/spotify.py`) - so naming it here means "open a Spotify search for
    everything", which is a choice and not a default.

    `default_query` fills in for "put some music on" when it is set; empty,
    the front page of the service decides, which is the service's guess about
    this listener rather than the assistant's.
    """

    model_config = ConfigDict(extra="ignore")

    default_service: str = "youtube_music"
    default_query: str = ""
    # A song plays in the assistant's own browser window, which the next song
    # closes; what plays elsewhere would keep going underneath it without
    # this (`media/now_playing.py`). Off for anyone who would rather the
    # assistant never touched what they were listening to.
    pause_before_playing: bool = True
    search_timeout_seconds: float = SEARCH_SECONDS


class WebSettings(BaseModel):
    """The `[web]` table: which search engine `search_web` opens.

    `search_url` is the engine's own search address with `{query}` where
    the words go. Google when the line is not there; a user who would rather
    not be known to Google writes DuckDuckGo's address here and no code
    changes (section 10).
    """

    model_config = ConfigDict(extra="ignore")

    search_url: str = SEARCH_URL


class MessagingSettings(BaseModel):
    """The `[messaging]` table: which app `send_message` uses when the user
    named none (spec of 2026-09-15, section 7.2).

    `default_app` is "WhatsApp" or "Telegram"; empty means the model asks.
    A name that is neither is refused at startup, in a sentence, rather
    than in the middle of every turn.
    """

    model_config = ConfigDict(extra="ignore")

    default_app: str = ""


class TelegramSettings(BaseModel):
    """The `[telegram]` table: the one thing about the user's Telegram
    application that is not a secret.

    `api_id` comes from my.telegram.org with the user's own account; the
    `api_hash` beside it and the session `assistant telegram login` produces
    are secrets and live in the Credential Manager, never here (section 10).
    Zero means "not set up": `send_message` then says so for Telegram.
    """

    model_config = ConfigDict(extra="ignore")

    api_id: int = 0


# The numbers of section 3.11 are written once, in `agent/limits.py`; the
# file's defaults are read off them so that the two cannot drift apart.
_LIMITS = Limits()


class LimitSettings(BaseModel):
    """The `[limits]` table: what a turn may do and what a day may cost (section 3.11).

    A table left out, or a key left out, means the default - so a
    `config.toml` written before 2.4 keeps its meaning. `hard_stop` is the
    one that changes what the assistant does rather than what it says:
    with it on, a limit passed means the model is not asked at all.
    """

    model_config = ConfigDict(extra="ignore")

    tool_calls_per_turn: int = _LIMITS.tool_calls_per_turn
    output_tokens: int = _LIMITS.output_tokens
    duplicate_calls: int = _LIMITS.duplicate_calls
    turn_seconds: float = _LIMITS.turn_seconds
    daily_usd: float = _LIMITS.daily_usd
    monthly_usd: float = _LIMITS.monthly_usd
    hard_stop: bool = _LIMITS.hard_stop
    duplicate_window_sec: int = _LIMITS.duplicate_window_sec


class Settings(BaseSettings):
    """Everything phase 1 stores. There is deliberately no field for a key."""

    model_config = SettingsConfigDict(
        env_prefix="ASSISTANT_",
        env_nested_delimiter="__",
        # A key from a newer version, or a typo, is not a reason to refuse to
        # start. The wizard rewrites the file anyway.
        extra="ignore",
    )

    llm: LLMSettings = LLMSettings()
    locale: LocaleSettings = LocaleSettings()
    audio: AudioSettings = AudioSettings()
    stt: STTSettings = STTSettings()
    tools: ToolSettings = ToolSettings()
    limits: LimitSettings = LimitSettings()
    media: MediaSettings = MediaSettings()
    web: WebSettings = WebSettings()
    messaging: MessagingSettings = MessagingSettings()
    telegram: TelegramSettings = TelegramSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Highest priority first: what the caller passed, then the environment,
        then the file. The file path is resolved here rather than in
        `model_config` so `ASSISTANT_CONFIG_DIR` still works when a test sets it
        after this class was imported.
        """
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=config_path()),
        )


def load_settings() -> Settings:
    """Reads the settings, or returns the defaults if there is no file yet."""
    return Settings()


def save_settings(settings: Settings) -> Path:
    """Writes `config.toml`, creating the directory on first run."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + _dump_toml(settings.model_dump(exclude_none=True)), encoding="utf-8")
    return path


def is_configured() -> bool:
    """Whether setup has run. `assistant run` refuses to start before it has."""
    return config_path().is_file() and bool(load_settings().llm.primary)


_HEADER = (
    f"# {APP_NAME} settings. Written by `assistant setup`; safe to edit by hand.\n"
    "# API keys are NOT in this file - they are in the Windows Credential Manager.\n\n"
)


def _dump_toml(tables: Mapping[str, Mapping[str, Any]]) -> str:
    """Serialises the two-level structure `Settings` produces.

    The standard library reads TOML and does not write it, and a whole
    dependency to emit six lines is not worth it. This handles exactly the
    shapes `Settings` can hold and raises on anything else, so a future field
    of an unsupported type fails here rather than producing a broken file.
    """
    lines: list[str] = []
    for table, values in tables.items():
        lines.append(f"[{table}]")
        lines.extend(f"{key} = {_format(value)}" for key, value in values.items())
        lines.append("")
    return "\n".join(lines)


def _format(value: Any) -> str:
    if isinstance(value, bool):  # before int - a bool is an int in Python
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return _quote(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_format(item) for item in value) + "]"
    raise TypeError(f"no TOML representation for {type(value).__name__}")


_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _quote(value: str) -> str:
    out: list[str] = []
    for character in value:
        if character in _ESCAPES:
            out.append(_ESCAPES[character])
        elif character < " " or character == "\x7f":
            out.append(f"\\u{ord(character):04X}")
        else:
            out.append(character)
    return '"' + "".join(out) + '"'


# --------------------------------------------------------------------------
# The key
# --------------------------------------------------------------------------


def store_api_key(provider: str, api_key: str) -> None:
    """Writes the key to the Windows Credential Manager, never to disk."""
    keyring.set_password(KEYRING_SERVICE, provider, api_key)


def load_api_key(provider: str) -> str | None:
    """The stored key, or `None` if this provider was never set up."""
    return keyring.get_password(KEYRING_SERVICE, provider)


def delete_api_key(provider: str) -> None:
    """Removes the key. Deleting one that was never stored is not an error -
    `assistant purge` should not fail on a provider the user never used."""
    with contextlib.suppress(PasswordDeleteError):
        keyring.delete_password(KEYRING_SERVICE, provider)
