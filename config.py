"""Application configuration for Reality Debugger."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _load_local_env_file(env_file: str = ".env") -> None:
    """Loads key-value pairs from .env into process environment if missing."""
    if not os.path.exists(env_file):
        return

    try:
        with open(env_file, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        # Keep app boot resilient even if local env file cannot be read.
        return


# Load local .env once at startup so CLI runs pick up local LLM and runtime settings.
_load_local_env_file()


@dataclass
class AppConfig:
    """Holds runtime configuration loaded from environment variables."""

    use_local_llm: bool = True
    local_llm_url: str = "http://localhost:11434/api/generate"
    local_llm_model: str = "qwen3:8b"
    local_llm_timeout_seconds: int = 120
    local_llm_max_new_tokens: int = 600
    local_llm_temperature: float = 0.3
    local_llm_retries: int = 2
    cache_size: int = 120
    typing_effect_enabled: bool = False
    typing_delay_seconds: float = 0.01
    ui_transitions_enabled: bool = True
    ui_transition_delay_seconds: float = 0.35
    context_window_size: int = 3
    dev_mode: bool = False
    review_logging: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        timeout_value = os.getenv("HF_REQUEST_TIMEOUT", os.getenv("LOCAL_LLM_TIMEOUT", "120"))
        max_tokens_value = os.getenv("HF_MAX_NEW_TOKENS", os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "600"))
        temperature_value = os.getenv("HF_TEMPERATURE", os.getenv("LOCAL_LLM_TEMPERATURE", "0.3"))
        retries_value = os.getenv("LOCAL_LLM_RETRIES", "2")

        return cls(
            use_local_llm=os.getenv("RD_USE_LOCAL_LLM", "true").lower() == "true",
            local_llm_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate"),
            local_llm_model=os.getenv("LOCAL_LLM_MODEL", "qwen3:8b"),
            local_llm_timeout_seconds=max(120, int(timeout_value)),
            local_llm_max_new_tokens=max(32, int(max_tokens_value)),
            local_llm_temperature=max(0.0, min(1.0, float(temperature_value))),
            local_llm_retries=max(1, int(retries_value)),
            cache_size=int(os.getenv("RD_CACHE_SIZE", "120")),
            typing_effect_enabled=os.getenv("RD_TYPING_EFFECT", "false").lower() == "true",
            typing_delay_seconds=float(os.getenv("RD_TYPING_DELAY", "0.01")),
            ui_transitions_enabled=os.getenv("RD_UI_TRANSITIONS", "true").lower() == "true",
            ui_transition_delay_seconds=float(os.getenv("RD_UI_TRANSITION_DELAY", "0.35")),
            context_window_size=int(os.getenv("RD_CONTEXT_WINDOW", "3")),
            dev_mode=os.getenv("RD_DEV_MODE", "false").lower() == "true",
            review_logging=os.getenv("RD_REVIEW_LOGS", "false").lower() == "true",
        )
