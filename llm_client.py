"""HTTP client for the local Ollama-compatible LLM endpoint."""

from __future__ import annotations

import time
from typing import Any

import requests


def call_local_llm(
    prompt: str,
    *,
    url: str = "http://localhost:11434/api/generate",
    model: str = "qwen3:8b",
    timeout: int = 120,
    max_new_tokens: int = 600,
    temperature: float = 0.3,
    retries: int = 2,
    response_format: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Calls local Ollama and returns normalized response metadata.

    `response_format` is passed as Ollama's `format`: "json", or a JSON schema the reply must follow
    (structured outputs), which removes the need to parse free-form text.
    """
    last_status: int | None = None
    attempts = max(1, retries)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": max(32, int(max_new_tokens)),
            "temperature": max(0.0, min(1.0, float(temperature))),
        },
    }
    if response_format is not None:
        payload["format"] = response_format
    for attempt in range(attempts):
        if attempt:
            # Back off before every retry, including after non-200 or unparsable responses.
            time.sleep(0.5 * attempt)
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
            )

            last_status = response.status_code
            if response.status_code != 200:
                continue

            try:
                data = response.json()
            except ValueError:
                continue

            text = str(data.get("response", "")).strip()
            if not text:
                text = str(data.get("generated_text", "")).strip()
            if not text:
                text = str(data.get("output_text", "")).strip()
            if not text:
                message = data.get("message")
                if isinstance(message, dict):
                    text = str(message.get("content", "")).strip()
            if not text:
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        text = str(first.get("text", "")).strip()
                        if not text and isinstance(first.get("message"), dict):
                            text = str(first["message"].get("content", "")).strip()
            if text:
                return {
                    "ok": True,
                    "text": text,
                    "status_code": response.status_code,
                    "done_reason": data.get("done_reason"),
                    "attempts": attempt + 1,
                }
        except requests.RequestException:
            pass

    return {"ok": False, "text": "", "status_code": last_status, "attempts": attempts}
