"""Lightweight helpers for local MCP code-review graph availability and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time


@dataclass(frozen=True)
class McpGraphStatus:
    enabled: bool
    path: str
    exists: bool
    size_bytes: int
    age_seconds: float
    state: str


def detect_mcp_graph_status(graph_path: str = ".code-review-graph/graph.db") -> McpGraphStatus:
    path = Path(graph_path)
    if not path.exists():
        return McpGraphStatus(
            enabled=False,
            path=str(path),
            exists=False,
            size_bytes=0,
            age_seconds=-1.0,
            state="missing",
        )

    try:
        stat = path.stat()
    except OSError:
        return McpGraphStatus(
            enabled=False,
            path=str(path),
            exists=False,
            size_bytes=0,
            age_seconds=-1.0,
            state="unreadable",
        )

    try:
        age_seconds = max(0.0, time.time() - stat.st_mtime)
    except Exception:
        age_seconds = 0.0

    state = "ready" if stat.st_size > 0 else "empty"
    return McpGraphStatus(
        enabled=True,
        path=str(path),
        exists=True,
        size_bytes=int(stat.st_size),
        age_seconds=float(age_seconds),
        state=state,
    )
