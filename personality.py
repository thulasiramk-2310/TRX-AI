"""Conversation personality modes for TRX with persistent custom profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PERSONALITIES_PATH = Path("personalities.json")

DEFAULT_PERSONALITIES: dict[str, dict[str, str]] = {
    "friendly": {
        "style": "warm, casual, helpful",
        "prefix": "😊",
        "instruction": "Be friendly, conversational, and encouraging.",
    },
    "mentor": {
        "style": "teaching, structured",
        "prefix": "🎓",
        "instruction": "Explain step-by-step, like teaching a student.",
    },
    "dev": {
        "style": "technical, concise",
        "prefix": "💻",
        "instruction": "Be direct, technical, and solution-focused.",
    },
}


def _sanitize_personality_payload(payload: Any) -> dict[str, dict[str, str]]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for raw_name, raw_meta in payload.items():
        name = str(raw_name).strip().lower()
        if not name or not isinstance(raw_meta, dict):
            continue
        prefix = str(raw_meta.get("prefix", "✨")).strip() or "✨"
        instruction = str(raw_meta.get("instruction", "")).strip()
        style = str(raw_meta.get("style", "")).strip()
        if not instruction:
            continue
        out[name] = {
            "prefix": prefix,
            "instruction": instruction,
            "style": style or "custom",
        }
    return out


def _persist_personalities(data: dict[str, dict[str, str]]) -> None:
    PERSONALITIES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_personalities() -> dict[str, dict[str, str]]:
    if not PERSONALITIES_PATH.exists():
        data = {key: dict(value) for key, value in DEFAULT_PERSONALITIES.items()}
        try:
            _persist_personalities(data)
        except OSError:
            pass
        return data
    try:
        payload = json.loads(PERSONALITIES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    data = _sanitize_personality_payload(payload)
    # Ensure built-ins always exist.
    for key, value in DEFAULT_PERSONALITIES.items():
        data.setdefault(key, dict(value))
    # Self-heal persisted file if shape drifted.
    try:
        _persist_personalities(data)
    except OSError:
        pass
    return data


def list_personality_names() -> list[str]:
    return sorted(load_personalities().keys())


def normalize_personality(value: str) -> str:
    key = str(value or "").strip().lower()
    data = load_personalities()
    return key if key in data else "friendly"


def personality_meta(value: str) -> dict[str, Any]:
    data = load_personalities()
    key = normalize_personality(value)
    return {"name": key, **data[key]}


def add_personality(name: str, instruction: str, prefix: str = "✨") -> str:
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("Personality name cannot be empty.")
    if key in {"add", "set", "list", "delete"}:
        raise ValueError("Personality name is reserved.")
    guide = str(instruction or "").strip()
    if not guide:
        raise ValueError("Instruction cannot be empty.")
    data = load_personalities()
    data[key] = {
        "prefix": str(prefix or "✨").strip() or "✨",
        "instruction": guide,
        "style": "custom",
    }
    _persist_personalities(data)
    return key


def delete_personality(name: str) -> None:
    key = str(name or "").strip().lower()
    if key in DEFAULT_PERSONALITIES:
        raise ValueError("Built-in personalities cannot be deleted.")
    data = load_personalities()
    if key not in data:
        raise ValueError("Personality not found.")
    data.pop(key, None)
    _persist_personalities(data)

