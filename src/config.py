"""Environment and configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def get_openrouter_api_key() -> str | None:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    return key or None


def get_openrouter_model() -> str:
    return os.getenv("OPENROUTER_MODEL", "openrouter/free").strip() or "openrouter/free"


def get_openrouter_model_fallbacks() -> list[str]:
    primary = get_openrouter_model()
    defaults = [
        "openrouter/free",
        "meta-llama/llama-3.2-3b-instruct:free",
    ]
    models = [primary, *defaults]
    seen: set[str] = set()
    ordered: list[str] = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered
