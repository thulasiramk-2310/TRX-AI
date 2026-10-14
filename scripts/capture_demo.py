"""Capture a real Ollama review and confirmed fix in a disposable workspace."""

from __future__ import annotations

import asyncio
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["RD_EMBEDDINGS"] = "false"

from config import AppConfig  # noqa: E402
from guardrails import redact  # noqa: E402
from tui.app import ConfirmFix, TRXApp  # noqa: E402
from tui.engine import Engine  # noqa: E402

BUILD = ROOT / "build" / "demo"


def verify_sample_fix(fixed: str) -> dict[str, bool]:
    """Test only this small, restricted pure-function fixture in a child process."""
    tree = ast.parse(fixed)
    allowed = {ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.If, ast.UnaryOp, ast.Not,
               ast.Call, ast.Name, ast.Load, ast.Raise, ast.Constant, ast.Compare, ast.Eq, ast.Return}
    names = {"values", "isinstance", "list", "TypeError", "len", "ValueError", "max"}
    if not all(type(node) in allowed for node in ast.walk(tree)) or not all(
            node.id in names for node in ast.walk(tree) if isinstance(node, ast.Name)):
        raise ValueError("Fix needs manual inspection; it is outside the demo's restricted fixture.")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "find_max":
        raise ValueError("The demo requires exactly one find_max function.")
    test = Path("verify_demo.py")
    test.write_text(
        "import runpy\n"
        "f = runpy.run_path('examples/find_max_fixed.py')['find_max']\n"
        "assert f([3, 1, 9]) == 9\nassert f([-5, -1, -8]) == -1\nassert f([4]) == 4\n"
        "try:\n    f([])\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Empty input accepted')\n",
        encoding="utf-8")
    subprocess.run([sys.executable, "-I", str(test.resolve())], check=True, timeout=10)
    return dict(nonempty_max=True, negative_max=True, single_item=True, empty_rejected=True)


class CaptureApp(TRXApp):
    CSS_PATH = ROOT / "tui" / "app.tcss"
    captured_result = None

    def show_result(self, result, for_fix=False):
        self.captured_result = result
        super().show_result(result, for_fix)


async def capture():
    config = AppConfig.from_env()
    config.use_local_llm = True
    config.local_llm_timeout_seconds = 120
    config.local_llm_retries = 1
    config.vector_memory_enabled = False
    config.cloud_memory_enabled = False
    config.review_cache_disk_enabled = False
    engine = await asyncio.to_thread(Engine, config=config)
    app = CaptureApp(engine=engine)
    app.workspace_name = "trx-ai"
    screenshots = {}
    stage_labels = []

    async with app.run_test(size=(124, 32)) as pilot:
        await app.workers.wait_for_complete()

        async def shot(name):
            await pilot.pause()
            app.save_screenshot(filename=name + ".svg", path=str(BUILD))
            screenshots[name] = name + ".svg"

        await shot("ready")
        command = app.query_one("#command")
        text = "fix examples/find_max.py"
        for index in range(1, 9):
            command.value = text[:round(len(text) * index / 8)]
            await shot(f"typing-{index}")
        await pilot.press("f3")
        started = time.monotonic()
        await command.action_submit()
        await pilot.pause(0.1)
        while app.busy:
            if app._stages and (not stage_labels or app._stages[-1] != stage_labels[-1]):
                label = app._stages[-1]
                stage_labels.append(label)
                print(label, flush=True)
                await shot(f"live-stage-{len(stage_labels) - 1}")
            await pilot.pause(0.2)
            if time.monotonic() - started > 900:
                raise TimeoutError("Live review did not finish within 15 minutes.")
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = app.captured_result
        if not result or result.get("analysis_source") != "llm":
            raise RuntimeError("A real model review is required; no fallback demo will be published.")
        if any(f.blocking for f in result.get("guardrails", [])):
            raise RuntimeError("The generated fix was blocked; it will not be saved for the demo.")
        if not isinstance(app.screen, ConfirmFix):
            raise RuntimeError("No saveable change was generated.")
        await shot("confirm")
        await pilot.press("escape")
        await pilot.press("f3")
        app.query_one("#stream").scroll_home(animate=False)
        await shot("review")
        app.query_one("#stream").scroll_end(animate=False)
        await shot("patch")
        app._confirm_fix(result)
        await pilot.pause()
        await pilot.click("#save")
        await pilot.pause()
        await shot("saved")
        source = Path("examples/find_max.py")
        saved = source.with_name("find_max_fixed.py")
        if not saved.is_file() or source.read_text(encoding="utf-8") != ORIGINAL:
            raise RuntimeError("The fix was not saved correctly or the original changed.")
        verification = verify_sample_fix(result["fixed_code"])
        record = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "model": config.local_llm_model,
            "elapsed_seconds": result.get("elapsed_seconds"),
            "original": ORIGINAL, "fixed": result["fixed_code"],
            "saved": saved.read_text(encoding="utf-8"),
            "original_preserved": True, "output": str(saved),
            "stages": stage_labels, "screenshots": screenshots,
            "status": asdict(app.sys_status),
            "result": redact(result),
            "verification": verification,
        }
        (BUILD / "live-capture.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Real review captured in {record['elapsed_seconds']:.1f}s; fix saved; original preserved.", flush=True)


ORIGINAL = (ROOT / "examples" / "find_max.py").read_text(encoding="utf-8")


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="trx-demo-live-") as directory:
        try:
            os.chdir(directory)
            Path("examples").mkdir()
            Path("examples/find_max.py").write_text(ORIGINAL, encoding="utf-8")
            asyncio.run(capture())
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    main()
