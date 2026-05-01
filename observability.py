"""Minimal structured logging and in-memory metrics for TRX."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = defaultdict(int)
        self.last_values: dict[str, float] = {}
        self.states: dict[str, str] = {}
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self.counters[name] += int(value)

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self.last_values[name] = float(value)
            self.sums[name] += float(value)
            self.counts[name] += 1

    def set_state(self, name: str, value: str) -> None:
        with self._lock:
            self.states[name] = str(value)

    def average(self, name: str) -> float:
        with self._lock:
            count = self.counts.get(name, 0)
            if count <= 0:
                return 0.0
            return self.sums.get(name, 0.0) / count

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "last_values": dict(self.last_values),
                "states": dict(self.states),
                "averages": {
                    key: (self.sums[key] / self.counts[key]) if self.counts[key] else 0.0
                    for key in self.sums
                },
            }

    def export(self, path: str = "sessions/metrics.json") -> str:
        payload = {
            "ts": int(time.time() * 1000),
            **self.snapshot(),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(destination)


METRICS = MetricsCollector()


def log_event(event: str, **fields: Any) -> None:
    if os.getenv("RD_STRUCTURED_LOGS", "true").lower() != "true":
        return
    payload = {
        "ts": int(time.time() * 1000),
        "event": event,
        **fields,
    }
    try:
        print(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass
