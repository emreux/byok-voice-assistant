"""The media keys (2.2): a key press, because there is no API for "pause".

`_press` is the one place the keyboard is reached; every test replaces it
and looks at which key would have been pressed.
"""

from __future__ import annotations

import pytest

from assistant.tools import media
from assistant.tools.media import KEYS, media_control


@pytest.fixture
def pressed(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []
    monkeypatch.setattr(media, "_press", seen.append)
    return seen


def test_it_is_a_safe_tool_whose_actions_are_an_enum() -> None:
    """The model is told the six actions rather than left to guess them."""
    assert media_control.risk == "safe"
    assert media_control.spec.parameters["required"] == ["action"]
    assert media_control.spec.parameters["properties"]["action"] == {
        "type": "string",
        "enum": ["play_pause", "next", "previous", "volume_up", "volume_down", "mute"],
    }
    assert list(KEYS) == media_control.spec.parameters["properties"]["action"]["enum"]


@pytest.mark.parametrize(
    ("action", "code"),
    [
        ("play_pause", 0xB3),
        ("next", 0xB0),
        ("previous", 0xB1),
        ("volume_up", 0xAF),
        ("volume_down", 0xAE),
        ("mute", 0xAD),
    ],
)
async def test_each_action_presses_its_own_key(pressed: list[int], action: str, code: int) -> None:
    said = await media_control.run(action=action)

    assert pressed == [code]
    assert said == f"Pressed the {action} key."


async def test_an_action_that_is_not_a_key_presses_nothing(pressed: list[int]) -> None:
    """A model that ignores the enum gets the list back, not a `KeyError`."""
    said = await media_control.run(action="stop")

    assert pressed == []
    assert said.startswith("No action called 'stop'")
    assert "play_pause" in said


def test_the_description_says_what_stops_the_music() -> None:
    """The user says "stop the music"; the model has to know that is play_pause."""
    said = media_control.spec.description

    assert "play_pause" in said
    assert "stop" in said
