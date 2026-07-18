"""Browser-based voice input using Web Speech API."""

from __future__ import annotations

import os

import streamlit.components.v1 as components

_COMPONENT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
_voice_component = components.declare_component("voice_input_widget", path=_COMPONENT_PATH)


def render_voice_input(key: str = "voice", enabled: bool = True) -> str | None:
    """Return transcribed text when user speaks, else None."""
    if not enabled:
        return None
    try:
        result = _voice_component(key=key, default=None)
        if isinstance(result, str) and result.strip():
            return result.strip()
    except Exception:
        pass
    return None
