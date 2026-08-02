"""Drives the full-screen app headlessly with a fake engine (no LLM, no cluster)."""

from __future__ import annotations

import unittest
from pathlib import Path

from textual.widgets import Button, Digits, Input, Markdown, Static

from guardrails import Finding
from tui.app import ConfirmFix, Patch, ReviewCard, TRXApp
from tui.engine import SessionStats, SystemStatus

REVIEW = {
    "response_mode": "analysis",
    "current_input": "review app.py",
    "model_name": "fake-model",
    "elapsed_seconds": 1.5,
    "analysis_source": "llm",
    "confidence_score": 80,
    "debug_analysis": ["Line 2: returns arr[0] on an empty list"],
    "improvements": ["Guard empty input"],
    "original_code": "def f(a):\n    return a[0]\n",
    "fixed_code": "def f(a):\n    if not a:\n        return None\n    return a[0]\n",
    "system_status": ["source=llm", "critic_score=0.9", "steps=3"],
    "guardrails": [],
}


class FakeEngine:
    def __init__(self, findings=None) -> None:
        self.stats = SessionStats()
        self.findings = findings or []
        self.applied: list[str] = []

    def status(self) -> SystemStatus:
        return SystemStatus("fake-model", True, True, "GPU", "ready", "auto")

    def chat(self, text: str) -> dict:
        self.stats.record(0.4, {}, review=False)
        return {"response_mode": "chat", "chat_response": f"**answer** to {text}", "model_name": "fake-model",
                "elapsed_seconds": 0.4, "analysis_source": "llm", "intent_confidence": 0.9}

    def review(self, target: str, *, for_fix: bool = False) -> dict:
        self.stats.record(1.5, REVIEW, review=True)
        verb = "fix" if for_fix else "review"
        return dict(REVIEW, current_input=f"{verb} {target}", review_target=target, guardrails=list(self.findings))

    def apply_fix(self, target: str, result: dict) -> Path:
        self.applied.append(target)
        return Path(target + "_fixed")


async def submit(pilot, app: TRXApp, text: str) -> None:
    command = app.query_one("#command", Input)
    command.value = text
    await command.action_submit()
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


class FullScreenAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_help_and_chat(self) -> None:
        engine = FakeEngine()
        app = TRXApp(engine=engine)
        async with app.run_test(size=(140, 40)) as pilot:
            await submit(pilot, app, "/help")
            self.assertTrue(any("review <file" in m._markdown for m in app.query(Markdown)))
            await submit(pilot, app, "what is a data warehouse?")
            answers = [m for m in app.query(Markdown) if "answer" in m._markdown]
            self.assertEqual(len(answers), 1)
            self.assertEqual(str(app.query_one("#confidence-digits", Digits).value), "90%")
            self.assertIn("1", str(app.query_one("#session", Static).render()))

    async def test_review_renders_card_patch_and_telemetry(self) -> None:
        app = TRXApp(engine=FakeEngine())
        async with app.run_test(size=(140, 40)) as pilot:
            await submit(pilot, app, "review app.py")
            self.assertEqual(len(app.query(ReviewCard)), 1)
            patch = app.query_one(Patch)
            self.assertIn("+2 -0", str(patch.border_title))
            self.assertEqual(str(app.query_one("#confidence-digits", Digits).value), "80%")
            self.assertIn("critic score", str(app.query_one("#pipeline-text", Static).render()))
            self.assertEqual(len(app.screen_stack), 1)  # review never asks to save

    async def test_fix_confirm_and_cancel(self) -> None:
        engine = FakeEngine()
        app = TRXApp(engine=engine)
        async with app.run_test(size=(140, 40)) as pilot:
            await submit(pilot, app, "fix app.py")
            self.assertIsInstance(app.screen, ConfirmFix)
            self.assertFalse(app.screen.query_one("#save", Button).disabled)
            await pilot.click("#cancel")
            await pilot.pause()
            self.assertEqual(engine.applied, [])
            await submit(pilot, app, "fix app.py")
            await pilot.click("#save")
            await pilot.pause()
            self.assertEqual(engine.applied, ["app.py"])

    async def test_blocked_fix_needs_force(self) -> None:
        engine = FakeEngine(findings=[Finding("block", "Fix introduces eval()")])
        app = TRXApp(engine=engine)
        async with app.run_test(size=(140, 40)) as pilot:
            await submit(pilot, app, "fix app.py")
            screen = app.screen
            self.assertIsInstance(screen, ConfirmFix)
            save = screen.query_one("#save", Button)
            self.assertTrue(save.disabled)
            screen.query_one("#force-input", Input).value = "yes"
            await pilot.pause()
            self.assertTrue(save.disabled)
            screen.query_one("#force-input", Input).value = "force"
            await pilot.pause()
            self.assertFalse(save.disabled)
            await pilot.click("#save")
            await pilot.pause()
            self.assertEqual(engine.applied, ["app.py"])
            self.assertIn("eval()", str(app.query_one("#guard-text", Static).render()))

    async def test_status_lights_and_demo_mode(self) -> None:
        app = TRXApp(demo=True)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            self.assertEqual(app.theme, "trx-ops")
            self.assertEqual(len(app.query(ReviewCard)), 1)
            await pilot.press("f2")
            self.assertEqual(app.theme, "trx-neon")


if __name__ == "__main__":
    unittest.main()
