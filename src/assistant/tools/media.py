"""The media keys (design.md section 3.6): pause, next track, volume.

Windows has no call that means "pause whatever is playing". It has the media
keys on the keyboard, which every player - Spotify, a browser tab, the Music
app - already listens to, so the tool presses one. `keybd_event` is the
oldest way of doing that and still the one every player answers to; the
call returns at once, so it runs on the loop.
"""

from __future__ import annotations

import ctypes
from typing import Literal

from assistant.tools.registry import tool

__all__ = ["KEYS", "Action", "media_control"]

Action = Literal["play_pause", "next", "previous", "volume_up", "volume_down", "mute"]

# The virtual-key code of each, from winuser.h.
KEYS: dict[str, int] = {
    "play_pause": 0xB3,  # VK_MEDIA_PLAY_PAUSE
    "next": 0xB0,  # VK_MEDIA_NEXT_TRACK
    "previous": 0xB1,  # VK_MEDIA_PREV_TRACK
    "volume_up": 0xAF,  # VK_VOLUME_UP
    "volume_down": 0xAE,  # VK_VOLUME_DOWN
    "mute": 0xAD,  # VK_VOLUME_MUTE
}

KEY_UP = 0x0002  # KEYEVENTF_KEYUP


def _press(code: int) -> None:
    """One press and release, as the keyboard would send it. Kept apart so a
    test can see which key without pressing it."""
    user32 = ctypes.windll.user32
    user32.keybd_event(code, 0, 0, 0)
    user32.keybd_event(code, 0, KEY_UP, 0)


@tool(risk="safe")
async def media_control(action: Action) -> str:
    """Controls whatever is playing, the way the media keys on a keyboard do.
    play_pause toggles between playing and paused - use it both to stop the
    music and to resume it; next and previous change the track; volume_up and
    volume_down step the system volume; mute silences it and unsilences it."""
    code = KEYS.get(action)
    if code is None:
        return f"No action called {action!r}; the actions are: {', '.join(KEYS)}."

    _press(code)
    return f"Pressed the {action} key."
