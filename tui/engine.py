"""UI-independent TRX operations for the full-screen app.

Reuses the classic CLI's building blocks (analyzer, review loading, guardrails, history), so the
full-screen app and `trx --classic` behave the same; this module only adds timing and stats.
Everything here is synchronous and safe to call from a worker thread.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from config import AppConfig
from guardrails import check_fix
from progress import ProgressCallback, report_progress, stage


@dataclass
class SessionStats:
    runs: int = 0
    reviews: int = 0
    latencies: list[float] = field(default_factory=list)
    cache_hits: int = 0

    def record(self, seconds: float, result: dict[str, Any], *, review: bool) -> None:
        self.runs += 1
        self.reviews += int(review)
        self.latencies = (self.latencies + [round(seconds, 2)])[-40:]
        if "cache_hit" in [str(s).lower() for s in result.get("system_status", [])]:
            self.cache_hits += 1

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def cache_rate(self) -> float:
        return self.cache_hits / self.runs if self.runs else 0.0


@dataclass
class SystemStatus:
    model: str
    llm_enabled: bool
    llm_reachable: bool
    compute: str        # "GPU", "CPU", or "idle" (model not loaded yet)
    graph: str          # "ready", "degraded", "missing"
    mode: str


class Engine:
    def __init__(self, analyzer: Any = None, config: AppConfig | None = None) -> None:
        import main as trx_main  # reuse the CLI's helpers; imported lazily to keep `import tui` light

        self._main = trx_main
        self.config = config or AppConfig.from_env()
        if analyzer is None:
            from analyzer import RealityAnalyzer

            analyzer = RealityAnalyzer(self.config)
        self.analyzer = analyzer
        self.stats = SessionStats()
        self.context: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ status

    def status(self) -> SystemStatus:
        base = str(self.config.local_llm_url).split("/api/")[0]
        reachable, compute = False, "idle"
        if self.config.use_local_llm:
            try:
                reachable = requests.get(f"{base}/api/tags", timeout=1.5).ok
                loaded = requests.get(f"{base}/api/ps", timeout=1.5).json().get("models", [])
                if loaded:
                    compute = "GPU" if any(int(m.get("size_vram", 0)) > 0 for m in loaded) else "CPU"
            except (requests.RequestException, ValueError):
                reachable = False
        try:
            graph_status = self._main.detect_mcp_graph_status()
            graph = "ready" if graph_status.exists and graph_status.state == "ready" else (
                "missing" if not graph_status.exists else "degraded")
        except Exception:  # noqa: BLE001 - status must never crash the UI
            graph = "degraded"
        return SystemStatus(
            model=str(self.config.local_llm_model), llm_enabled=bool(self.config.use_local_llm),
            llm_reachable=reachable, compute=compute, graph=graph,
            mode=str(self.config.assistant_mode or "auto"),
        )

    # ------------------------------------------------------------------ operations

    def chat(self, text: str, *, progress: ProgressCallback | None = None) -> dict[str, Any]:
        with report_progress(progress):
            stage("Routing · selecting response path")
            return self._chat(text)

    def _chat(self, text: str) -> dict[str, Any]:
        started = time.perf_counter()
        owner = getattr(self.analyzer, "current_user", "guest")
        context = [entry for entry in self.context if entry.get("owner", "guest") == owner]
        result = self.analyzer.analyze(text, mode="debug", past_context=context[-3:])
        elapsed = time.perf_counter() - started
        result.setdefault("elapsed_seconds", elapsed)
        result.setdefault("model_name", self.config.local_llm_model)
        result["current_input"] = text
        self.context.append({"input": text, "owner": owner})
        self.stats.record(elapsed, result, review=result.get("response_mode") == "analysis")
        self._main._append_run_history(text, "chat", result)
        return result

    def review(self, target: str, *, for_fix: bool = False,
               progress: ProgressCallback | None = None) -> dict[str, Any]:
        with report_progress(progress):
            stage("Reading · loading code")
            return self._review(target, for_fix=for_fix)

    def _review(self, target: str, *, for_fix: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        path = Path(target)
        if for_fix:
            if not path.is_file():
                raise FileNotFoundError(target)
            if path.suffix.lower() not in self._main.SUPPORTED_CODE_SUFFIXES:
                raise ValueError("Unsupported code file format.")
            code = path.read_text(encoding="utf-8")
        else:
            code = self._main._load_review_target(target, config=self.config)
        result = self.analyzer.analyze_code_multi_agent(code)
        elapsed = time.perf_counter() - started
        result["original_code"] = code if for_fix else code[:20000]
        result["current_input"] = f"{'fix' if for_fix else 'review'} {target}"
        result["review_target"] = target
        result.setdefault("elapsed_seconds", elapsed)
        result.setdefault("model_name", self.config.local_llm_model)
        stage("Safety · checking suggested changes")
        guardrails = list(self._main._injection_findings(code))
        fixed = str(result.get("fixed_code", "")).strip()
        if for_fix and fixed:
            guardrails += check_fix(code, fixed, self._main._language_for_path(path))
        result["guardrails"] = guardrails
        self.stats.record(elapsed, result, review=True)
        self._main._append_run_history(result["current_input"], "fix" if for_fix else "review", result)
        return result

    def apply_fix(self, target: str, result: dict[str, Any]) -> Path:
        reason = result.get("final_insight") or [""]
        reason = reason[0] if isinstance(reason, list) and reason else str(reason)
        return self._main.apply_code_fix(
            target, str(result.get("fixed_code", "")).strip(),
            reason=str(reason), original_code=result.get("original_code"),
        )
