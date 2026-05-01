"""Transport-agnostic graph query adapter with local/remote MCP clients and fallback."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests
from observability import METRICS, log_event


@dataclass(frozen=True)
class GraphQueryResult:
    ok: bool
    source: str
    degraded: bool
    data: dict[str, Any]
    errors: list[str]
    elapsed_ms: int


class GraphClient(Protocol):
    def query_dependencies(self, file: str) -> GraphQueryResult:
        ...

    def query_related_files(self, symbol: str) -> GraphQueryResult:
        ...

    def query_call_graph(self, function: str) -> GraphQueryResult:
        ...


@dataclass
class ClientResilience:
    timeout_seconds: float = 1.2
    retries: int = 1
    breaker_fail_threshold: int = 3
    breaker_open_seconds: float = 15.0


class BaseGraphClient:
    """Base client providing retry and circuit-breaker behavior."""

    def __init__(self, resilience: ClientResilience) -> None:
        self.resilience = resilience
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    def _allow_request(self) -> bool:
        return time.time() >= self._breaker_open_until

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.resilience.breaker_fail_threshold:
            self._breaker_open_until = time.time() + self.resilience.breaker_open_seconds

    def _run_with_retry(
        self,
        op_name: str,
        fn: Any,
        fallback_data: dict[str, Any],
        source: str,
    ) -> GraphQueryResult:
        started = time.perf_counter()
        errors: list[str] = []
        if not self._allow_request():
            return GraphQueryResult(
                ok=False,
                source=source,
                degraded=True,
                data=fallback_data,
                errors=[f"{op_name}: circuit_breaker_open"],
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        for attempt in range(self.resilience.retries + 1):
            try:
                data = fn()
                self._record_success()
                return GraphQueryResult(
                    ok=True,
                    source=source,
                    degraded=False,
                    data=data,
                    errors=[],
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
            except Exception as exc:
                errors.append(str(exc))
                self._record_failure()
                if attempt < self.resilience.retries:
                    time.sleep(0.05 * (attempt + 1))

        return GraphQueryResult(
            ok=False,
            source=source,
            degraded=True,
            data=fallback_data,
            errors=errors[:4],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )


class LocalGraphClient(BaseGraphClient):
    def __init__(self, *, db_path: str, resilience: ClientResilience) -> None:
        super().__init__(resilience)
        self.db_path = Path(db_path)

    def query_dependencies(self, file: str) -> GraphQueryResult:
        fallback_data = {"file": file, "dependencies": [], "dependents": []}
        return self._run_with_retry(
            "query_dependencies",
            lambda: self._query_dependencies_sqlite(file),
            fallback_data,
            "code-review-graph/sqlite",
        )

    def query_related_files(self, symbol: str) -> GraphQueryResult:
        fallback_data = {"symbol": symbol, "files": []}
        return self._run_with_retry(
            "query_related_files",
            lambda: self._query_related_files_sqlite(symbol),
            fallback_data,
            "code-review-graph/sqlite",
        )

    def query_call_graph(self, function: str) -> GraphQueryResult:
        fallback_data = {"function": function, "calls": [], "called_by": []}
        return self._run_with_retry(
            "query_call_graph",
            lambda: self._query_call_graph_sqlite(function),
            fallback_data,
            "code-review-graph/sqlite",
        )

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Graph database not found: {self.db_path}")
        con = sqlite3.connect(self.db_path, timeout=self.resilience.timeout_seconds)
        con.row_factory = sqlite3.Row
        return con

    def _query_dependencies_sqlite(self, file: str) -> dict[str, Any]:
        con = self._connect()
        try:
            cur = con.cursor()
            deps_rows = cur.execute(
                """
                SELECT DISTINCT target_qualified
                FROM edges
                WHERE file_path = ? AND target_qualified IS NOT NULL AND target_qualified != ''
                LIMIT 80
                """,
                (file,),
            ).fetchall()
            dep_rows = cur.execute(
                """
                SELECT DISTINCT file_path
                FROM edges
                WHERE target_qualified IN (
                    SELECT qualified_name FROM nodes WHERE file_path = ?
                ) AND file_path != ? AND file_path IS NOT NULL AND file_path != ''
                LIMIT 80
                """,
                (file, file),
            ).fetchall()
            return {
                "file": file,
                "dependencies": [str(row[0]) for row in deps_rows if row[0]],
                "dependents": [str(row[0]) for row in dep_rows if row[0]],
            }
        finally:
            con.close()

    def _query_related_files_sqlite(self, symbol: str) -> dict[str, Any]:
        con = self._connect()
        try:
            cur = con.cursor()
            rows = cur.execute(
                """
                SELECT DISTINCT file_path
                FROM nodes
                WHERE (name = ? OR qualified_name LIKE ? OR signature LIKE ?)
                  AND file_path IS NOT NULL AND file_path != ''
                LIMIT 120
                """,
                (symbol, f"%{symbol}%", f"%{symbol}%"),
            ).fetchall()
            return {"symbol": symbol, "files": [str(row[0]) for row in rows if row[0]]}
        finally:
            con.close()

    def _query_call_graph_sqlite(self, function: str) -> dict[str, Any]:
        con = self._connect()
        try:
            cur = con.cursor()
            qualified_rows = cur.execute(
                """
                SELECT qualified_name
                FROM nodes
                WHERE name = ? OR qualified_name LIKE ?
                LIMIT 20
                """,
                (function, f"%{function}%"),
            ).fetchall()
            qualified = [str(row[0]) for row in qualified_rows if row[0]]
            if not qualified:
                return {"function": function, "calls": [], "called_by": []}

            placeholders = ",".join("?" for _ in qualified)
            calls_rows = cur.execute(
                f"""
                SELECT DISTINCT target_qualified
                FROM edges
                WHERE source_qualified IN ({placeholders})
                  AND target_qualified IS NOT NULL AND target_qualified != ''
                LIMIT 120
                """,
                tuple(qualified),
            ).fetchall()
            callers_rows = cur.execute(
                f"""
                SELECT DISTINCT source_qualified
                FROM edges
                WHERE target_qualified IN ({placeholders})
                  AND source_qualified IS NOT NULL AND source_qualified != ''
                LIMIT 120
                """,
                tuple(qualified),
            ).fetchall()
            return {
                "function": function,
                "calls": [str(row[0]) for row in calls_rows if row[0]],
                "called_by": [str(row[0]) for row in callers_rows if row[0]],
            }
        finally:
            con.close()


class RemoteGraphClient(BaseGraphClient):
    """HTTP transport for remote MCP graph server."""

    def __init__(self, *, base_url: str, api_key: str, resilience: ClientResilience) -> None:
        super().__init__(resilience)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()

    def query_dependencies(self, file: str) -> GraphQueryResult:
        fallback_data = {"file": file, "dependencies": [], "dependents": []}
        return self._run_with_retry(
            "query_dependencies",
            lambda: self._post_json("/graph/query/dependencies", {"file": file}),
            fallback_data,
            "code-review-graph/remote-http",
        )

    def query_related_files(self, symbol: str) -> GraphQueryResult:
        fallback_data = {"symbol": symbol, "files": []}
        return self._run_with_retry(
            "query_related_files",
            lambda: self._post_json("/graph/query/related-files", {"symbol": symbol}),
            fallback_data,
            "code-review-graph/remote-http",
        )

    def query_call_graph(self, function: str) -> GraphQueryResult:
        fallback_data = {"function": function, "calls": [], "called_by": []}
        return self._run_with_retry(
            "query_call_graph",
            lambda: self._post_json("/graph/query/call-graph", {"function": function}),
            fallback_data,
            "code-review-graph/remote-http",
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self.session.post(
            f"{self.base_url}{path}",
            headers=headers,
            json=payload,
            timeout=self.resilience.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid remote MCP response shape")
        return data


class FallbackGraphClient:
    """Uses primary client first, falls back to secondary on failure."""

    def __init__(self, primary: GraphClient, secondary: GraphClient) -> None:
        self.primary = primary
        self.secondary = secondary

    def query_dependencies(self, file: str) -> GraphQueryResult:
        first = self.primary.query_dependencies(file)
        if first.ok:
            return first
        second = self.secondary.query_dependencies(file)
        if second.ok:
            return GraphQueryResult(
                ok=True,
                source=f"{first.source}->fallback->{second.source}",
                degraded=False,
                data=second.data,
                errors=(first.errors + second.errors)[:4],
                elapsed_ms=first.elapsed_ms + second.elapsed_ms,
            )
        return GraphQueryResult(
            ok=False,
            source=f"{first.source}->fallback->{second.source}",
            degraded=True,
            data=second.data,
            errors=(first.errors + second.errors)[:4],
            elapsed_ms=first.elapsed_ms + second.elapsed_ms,
        )

    def query_related_files(self, symbol: str) -> GraphQueryResult:
        first = self.primary.query_related_files(symbol)
        if first.ok:
            return first
        second = self.secondary.query_related_files(symbol)
        if second.ok:
            return GraphQueryResult(
                ok=True,
                source=f"{first.source}->fallback->{second.source}",
                degraded=False,
                data=second.data,
                errors=(first.errors + second.errors)[:4],
                elapsed_ms=first.elapsed_ms + second.elapsed_ms,
            )
        return GraphQueryResult(
            ok=False,
            source=f"{first.source}->fallback->{second.source}",
            degraded=True,
            data=second.data,
            errors=(first.errors + second.errors)[:4],
            elapsed_ms=first.elapsed_ms + second.elapsed_ms,
        )

    def query_call_graph(self, function: str) -> GraphQueryResult:
        first = self.primary.query_call_graph(function)
        if first.ok:
            return first
        second = self.secondary.query_call_graph(function)
        if second.ok:
            return GraphQueryResult(
                ok=True,
                source=f"{first.source}->fallback->{second.source}",
                degraded=False,
                data=second.data,
                errors=(first.errors + second.errors)[:4],
                elapsed_ms=first.elapsed_ms + second.elapsed_ms,
            )
        return GraphQueryResult(
            ok=False,
            source=f"{first.source}->fallback->{second.source}",
            degraded=True,
            data=second.data,
            errors=(first.errors + second.errors)[:4],
            elapsed_ms=first.elapsed_ms + second.elapsed_ms,
        )


class CodeReviewGraphAdapter:
    """Graph retrieval adapter preserving existing analyzer-facing API."""

    def __init__(
        self,
        *,
        db_path: str = ".code-review-graph/graph.db",
        timeout_seconds: float = 1.2,
        retries: int = 1,
        memory_cache_size: int = 256,
        disk_cache_path: str = "sessions/graph_query_cache.json",
        disk_cache_enabled: bool = True,
    ) -> None:
        self.cache_version = 2
        self.cache_ttl_seconds = max(30, int(os.getenv("RD_GRAPH_CACHE_TTL_SECONDS", "300")))
        self.memory_cache_size = max(32, int(memory_cache_size))
        self.disk_cache_path = Path(disk_cache_path)
        self.disk_cache_enabled = disk_cache_enabled
        self.db_path = Path(db_path)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._load_disk_cache()

        transport = os.getenv("RD_GRAPH_TRANSPORT", "local").strip().lower()
        self.transport_mode = transport
        remote_url = os.getenv("RD_GRAPH_REMOTE_URL", "").strip()
        remote_key = os.getenv("RD_GRAPH_REMOTE_API_KEY", "").strip()
        breaker_fail_threshold = int(os.getenv("RD_GRAPH_CB_FAIL_THRESHOLD", "3"))
        breaker_open_seconds = float(os.getenv("RD_GRAPH_CB_OPEN_SECONDS", "15"))

        resilience = ClientResilience(
            timeout_seconds=max(0.2, float(timeout_seconds)),
            retries=max(0, int(retries)),
            breaker_fail_threshold=max(1, breaker_fail_threshold),
            breaker_open_seconds=max(1.0, breaker_open_seconds),
        )

        local_client = LocalGraphClient(db_path=db_path, resilience=resilience)
        if transport == "remote":
            if not remote_url:
                self.client: GraphClient = local_client
            else:
                remote_client = RemoteGraphClient(base_url=remote_url, api_key=remote_key, resilience=resilience)
                self.client = FallbackGraphClient(remote_client, local_client)
        elif transport == "hybrid":
            if remote_url:
                remote_client = RemoteGraphClient(base_url=remote_url, api_key=remote_key, resilience=resilience)
                self.client = FallbackGraphClient(remote_client, local_client)
            else:
                self.client = local_client
        else:
            self.client = local_client

    def transport_label(self) -> str:
        mode = (self.transport_mode or "local").lower()
        if mode in {"hybrid", "remote", "local"}:
            return mode.upper()
        return "LOCAL"

    def clear_cache(self) -> None:
        self._cache.clear()
        if self.disk_cache_enabled and self.disk_cache_path.exists():
            try:
                self.disk_cache_path.unlink()
            except OSError:
                pass

    def query_dependencies(self, file: str) -> GraphQueryResult:
        key = self._cache_key("deps", {"file": file})
        cached = self._cache_get(key)
        if cached is not None:
            METRICS.inc("cache_hit")
            return self._result_from_cached(cached)
        METRICS.inc("cache_miss")
        result = self.client.query_dependencies(file)
        self._record_query_metrics(result)
        self._cache_set(key, result)
        return result

    def query_related_files(self, symbol: str) -> GraphQueryResult:
        key = self._cache_key("related", {"symbol": symbol})
        cached = self._cache_get(key)
        if cached is not None:
            METRICS.inc("cache_hit")
            return self._result_from_cached(cached)
        METRICS.inc("cache_miss")
        result = self.client.query_related_files(symbol)
        self._record_query_metrics(result)
        self._cache_set(key, result)
        return result

    def query_call_graph(self, function: str) -> GraphQueryResult:
        key = self._cache_key("calls", {"function": function})
        cached = self._cache_get(key)
        if cached is not None:
            METRICS.inc("cache_hit")
            return self._result_from_cached(cached)
        METRICS.inc("cache_miss")
        result = self.client.query_call_graph(function)
        self._record_query_metrics(result)
        self._cache_set(key, result)
        return result

    async def aquery_dependencies(self, file: str) -> GraphQueryResult:
        return await asyncio.to_thread(self.query_dependencies, file)

    async def aquery_related_files(self, symbol: str) -> GraphQueryResult:
        return await asyncio.to_thread(self.query_related_files, symbol)

    async def aquery_call_graph(self, function: str) -> GraphQueryResult:
        return await asyncio.to_thread(self.query_call_graph, function)

    def summarize_context(
        self,
        *,
        files: list[str],
        symbols: list[str],
        max_items: int = 5,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        dependencies: list[str] = []
        impacted_modules: list[str] = []
        call_relationships: list[str] = []
        degraded = False
        errors: list[str] = []

        tasks: list[tuple[str, str]] = []
        for file in files[:max_items]:
            tasks.append(("deps", file))
        for symbol in symbols[:max_items]:
            tasks.append(("rel", symbol))
            tasks.append(("calls", symbol))

        results: list[tuple[str, str, GraphQueryResult]] = []
        with ThreadPoolExecutor(max_workers=min(12, max(2, len(tasks)))) as pool:
            future_map = {}
            for task_type, value in tasks:
                if task_type == "deps":
                    future = pool.submit(self.query_dependencies, value)
                elif task_type == "rel":
                    future = pool.submit(self.query_related_files, value)
                else:
                    future = pool.submit(self.query_call_graph, value)
                future_map[future] = (task_type, value)
            for future in as_completed(future_map):
                task_type, value = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = GraphQueryResult(
                        ok=False,
                        source="adapter",
                        degraded=True,
                        data={},
                        errors=[str(exc)],
                        elapsed_ms=0,
                    )
                results.append((task_type, value, result))

        for task_type, value, result in results:
            degraded = degraded or result.degraded
            errors.extend(result.errors[:1])
            if task_type == "deps":
                for item in result.data.get("dependencies", [])[:max_items]:
                    dependencies.append(f"{value} -> {item}")
                for item in result.data.get("dependents", [])[:max_items]:
                    impacted_modules.append(f"{item} depends on {value}")
            elif task_type == "rel":
                for file in result.data.get("files", [])[:max_items]:
                    impacted_modules.append(f"{value} touches {file}")
            elif task_type == "calls":
                for target in result.data.get("calls", [])[:max_items]:
                    call_relationships.append(f"{value} -> {target}")
                for caller in result.data.get("called_by", [])[:max_items]:
                    call_relationships.append(f"{caller} -> {value}")

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": not degraded,
            "degraded": degraded,
            "dependencies": self._dedupe(dependencies)[: max_items * 3],
            "impacted_modules": self._dedupe(impacted_modules)[: max_items * 3],
            "call_relationships": self._dedupe(call_relationships)[: max_items * 3],
            "errors": self._dedupe([e for e in errors if e])[:4],
            "elapsed_ms": elapsed_ms,
        }

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = item.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _cache_key(self, name: str, payload: dict[str, Any]) -> str:
        fingerprint = self._graph_fingerprint()
        return f"v{self.cache_version}:{fingerprint}:{name}:{json.dumps(payload, sort_keys=True, ensure_ascii=False)}"

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        ts = float(value.get("timestamp", 0.0))
        if ts and (time.time() - ts) > self.cache_ttl_seconds:
            return None
        self._cache[key] = value
        return value

    def _cache_set(self, key: str, result: GraphQueryResult) -> None:
        if key in self._cache:
            self._cache.pop(key)
        payload = {
            "ok": result.ok,
            "source": result.source,
            "degraded": result.degraded,
            "data": result.data,
            "errors": result.errors,
            "elapsed_ms": result.elapsed_ms,
            "timestamp": time.time(),
            "cache_version": self.cache_version,
        }
        self._cache[key] = payload
        while len(self._cache) > self.memory_cache_size:
            self._cache.popitem(last=False)
        self._save_disk_cache()

    def _result_from_cached(self, cached: dict[str, Any]) -> GraphQueryResult:
        return GraphQueryResult(
            ok=bool(cached.get("ok")),
            source=str(cached.get("source", "cache")),
            degraded=bool(cached.get("degraded")),
            data=dict(cached.get("data", {})),
            errors=[str(x) for x in cached.get("errors", []) if str(x)],
            elapsed_ms=int(cached.get("elapsed_ms", 0)),
        )

    def _load_disk_cache(self) -> None:
        if not self.disk_cache_enabled:
            return
        try:
            if not self.disk_cache_path.exists():
                return
            payload = json.loads(self.disk_cache_path.read_text(encoding="utf-8"))
            if int(payload.get("version", 0)) != self.cache_version:
                return
            items = payload.get("items", [])
            if not isinstance(items, list):
                return
            for item in items[-self.memory_cache_size :]:
                key = str(item.get("key", ""))
                value = item.get("value")
                if key and isinstance(value, dict):
                    ts = float(value.get("timestamp", 0.0))
                    if ts and (time.time() - ts) > self.cache_ttl_seconds:
                        continue
                    self._cache[key] = value
        except (OSError, ValueError, TypeError):
            return

    def _save_disk_cache(self) -> None:
        if not self.disk_cache_enabled:
            return
        try:
            self.disk_cache_path.parent.mkdir(parents=True, exist_ok=True)
            items = [{"key": key, "value": value} for key, value in self._cache.items()]
            payload = {"version": self.cache_version, "items": items[-self.memory_cache_size :]}
            self.disk_cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            return

    def _graph_fingerprint(self) -> str:
        try:
            if self.db_path.exists():
                stat = self.db_path.stat()
                return f"mtime={int(stat.st_mtime)}:size={int(stat.st_size)}"
        except OSError:
            pass
        return "graph:none"

    def _record_query_metrics(self, result: GraphQueryResult) -> None:
        METRICS.observe("latency_ms", float(result.elapsed_ms))
        METRICS.inc("mcp_query_status_ok" if result.ok else "mcp_query_status_degraded")
        breaker_open = any("circuit_breaker_open" in err for err in result.errors)
        METRICS.inc("circuit_breaker_open" if breaker_open else "circuit_breaker_closed")
        METRICS.set_state("circuit_breaker_state", "OPEN" if breaker_open else "CLOSED")
        METRICS.set_state("mcp_query_status", "ACTIVE" if result.ok else "DEGRADED")
        log_event(
            "mcp_query",
            source=result.source,
            ok=result.ok,
            degraded=result.degraded,
            latency_ms=result.elapsed_ms,
            circuit_breaker_state="open" if breaker_open else "closed",
        )
