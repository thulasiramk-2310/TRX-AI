"""One shared, lazily loaded sentence-embedding model.

Loading a sentence-transformers model takes seconds (more on first download). Before this module,
SemanticRouter and VectorMemory each loaded their own copy while TRX started, so the terminal
stayed blank for ~20s. Now the model loads once, in a background thread, while the prompt is
already usable; callers use it when `ready()` returns it and fall back to lexical matching until then.
"""

from __future__ import annotations

import os
import threading
from typing import Any

DEFAULT_MODEL = "all-MiniLM-L6-v2"

_lock = threading.Lock()
_models: dict[str, Any] = {}
_failed: set[str] = set()
_loading: dict[str, threading.Thread] = {}


def _quiet_hf() -> None:
    # Hugging Face progress bars ("Loading weights 0/103 ...") would print into the transcript.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def disabled() -> bool:
    return os.getenv("RD_EMBEDDINGS", "true").lower() == "false"


def load(name: str = DEFAULT_MODEL) -> Any | None:
    """Returns the model, loading it if needed (blocking). None if unavailable."""
    if disabled():
        return None
    with _lock:
        if name in _models:
            return _models[name]
        if name in _failed:
            return None
    _quiet_hf()
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        try:
            from transformers.utils import logging as hf_logging  # type: ignore

            hf_logging.disable_progress_bar()
        except Exception:
            pass
        model = SentenceTransformer(name)
    except Exception:
        with _lock:
            _failed.add(name)
        return None
    with _lock:
        _models[name] = model
    return model


def ready(name: str = DEFAULT_MODEL) -> Any | None:
    """Returns the model only if it has already finished loading (never blocks)."""
    with _lock:
        return _models.get(name)


def warm_up(name: str = DEFAULT_MODEL) -> None:
    """Starts loading the model in a daemon thread, at most once per name."""
    if disabled():
        return
    with _lock:
        if name in _models or name in _failed or name in _loading:
            return
        thread = threading.Thread(target=load, args=(name,), name=f"embeddings-{name}", daemon=True)
        _loading[name] = thread
    thread.start()


def wait(name: str = DEFAULT_MODEL, timeout: float | None = None) -> Any | None:
    """Blocks until a warm-up started earlier finishes (used by tests and batch tools)."""
    with _lock:
        thread = _loading.get(name)
    if thread is not None:
        thread.join(timeout)
    return ready(name)
