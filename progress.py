"""Request-scoped stage notifications. Never publish prompts or model output."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

ProgressCallback = Callable[[str], None]
_callback: ContextVar[ProgressCallback | None] = ContextVar("trx_progress", default=None)


@contextmanager
def report_progress(callback: ProgressCallback | None) -> Iterator[None]:
    token = _callback.set(callback)
    try:
        yield
    finally:
        _callback.reset(token)


def stage(label: str) -> None:
    callback = _callback.get()
    if callback is not None:
        callback(label)
