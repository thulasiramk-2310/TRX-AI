"""File watcher for real-time TRX-AI code review."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


class CodeChangeHandler(FileSystemEventHandler):
    """Watches Python file changes and triggers review."""

    def __init__(
        self,
        analyzer: Any,
        formatter: Any,
        *,
        auto_fix: bool = False,
        fix_writer: Callable[[str, str], Path] | None = None,
        debounce_seconds: float = 1.0,
    ) -> None:
        self.analyzer = analyzer
        self.formatter = formatter
        self.auto_fix = auto_fix
        self.fix_writer = fix_writer
        self.debounce_seconds = debounce_seconds
        self._last_seen: dict[str, float] = {}

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        src_path = str(event.src_path)
        if not self._is_supported_python_file(src_path):
            return
        if self._is_temp_or_ignored(src_path):
            return
        if self._is_debounced(src_path):
            return

        file_path = Path(src_path)
        print(f"\n[Detected change] {file_path}")

        try:
            code = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"[Watcher error] Unable to read file: {exc}")
            return

        if not code.strip():
            print("[Watcher] Skipping empty file.")
            return

        try:
            result = self.analyzer.analyze_code_multi_agent(code)
            self.formatter.render(result, "debug")

            if self.auto_fix and self.fix_writer is not None:
                fixed_code = str(result.get("fixed_code", "")).strip()
                if fixed_code:
                    fixed_path = self.fix_writer(str(file_path), fixed_code)
                    print(f"[Auto Fix] Saved: {fixed_path}")
                else:
                    print("[Auto Fix] No fixed code generated.")
        except Exception as exc:
            print(f"[Watcher error] {exc}")

    def _is_supported_python_file(self, src_path: str) -> bool:
        return src_path.lower().endswith(".py")

    def _is_temp_or_ignored(self, src_path: str) -> bool:
        path = Path(src_path)
        name = path.name.lower()
        ignored_suffixes = (
            ".tmp",
            ".swp",
            ".swo",
            "~",
            ".bak",
            ".pyc",
        )
        if name.endswith(ignored_suffixes):
            return True
        if name.startswith("."):
            return True
        if name.endswith("_fixed.py"):
            return True
        if "__pycache__" in {part.lower() for part in path.parts}:
            return True
        return False

    def _is_debounced(self, src_path: str) -> bool:
        now = time.monotonic()
        last = self._last_seen.get(src_path, 0.0)
        if now - last < self.debounce_seconds:
            return True
        self._last_seen[src_path] = now
        return False


def start_watcher(
    path: str,
    analyzer: Any,
    formatter: Any,
    *,
    auto_fix: bool = False,
    fix_writer: Callable[[str, str], Path] | None = None,
    debounce_seconds: float = 1.0,
) -> Observer:
    """Starts folder watcher and returns active observer."""
    watch_path = Path(path)
    if not watch_path.exists() or not watch_path.is_dir():
        raise ValueError(f"Invalid watch path: {path}")

    event_handler = CodeChangeHandler(
        analyzer,
        formatter,
        auto_fix=auto_fix,
        fix_writer=fix_writer,
        debounce_seconds=debounce_seconds,
    )
    observer = Observer()
    observer.schedule(event_handler, str(watch_path), recursive=True)
    observer.start()
    return observer

