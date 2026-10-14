"""Stage notifications follow real execution and stay scoped to each request."""

from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from analyzer_helpers import analyzer_stub
from progress import report_progress, stage
from tui.engine import Engine


class ProgressTests(unittest.TestCase):
    def test_scope_resets_after_error_and_isolates_threads(self):
        events = []
        with self.assertRaises(ValueError), report_progress(events.append):
            stage("outer")
            raise ValueError("stop")
        stage("outside")
        self.assertEqual(events, ["outer"])

        def request(label):
            own = []
            with report_progress(own.append):
                stage(label)
            return own

        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(request, ["chat", "review"])), [["chat"], ["review"]])

    def test_engine_scopes_callback_for_actual_analyzer_calls(self):
        engine = Engine.__new__(Engine)
        events = []
        with patch.object(engine, "_chat", side_effect=lambda text: stage("model") or {"text": text}):
            self.assertEqual(engine.chat("hello", progress=events.append), {"text": "hello"})
        stage("outside")
        self.assertEqual(events, ["Routing · selecting response path", "model"])

    def test_critic_pass_skips_fixer_and_low_score_runs_it(self):
        analyzer = analyzer_stub()
        for score, expected in ((0.9, ["Analyzer", "Generator", "Critic"]),
                                (0.1, ["Analyzer", "Generator", "Critic", "Fixer", "Critic", "Fixer"])):
            events = []
            responses = [
                {"ok": True, "text": "notes"},
                {"ok": True, "text": "review"},
                {"ok": True, "text": '{"score": ' + str(score) + ', "issues": []}'},
                {"ok": True, "text": "revised review"},
                {"ok": True, "text": '{"score": 0.1, "issues": []}'},
                {"ok": True, "text": "final review"},
            ]
            with report_progress(events.append), \
                    patch.object(analyzer, "_failure_context_snippet", return_value=""), \
                    patch.object(analyzer, "_call_local", side_effect=responses):
                result = analyzer._run_code_review_multi_agent_pipeline("x = 1", "python")
            self.assertTrue(result["ok"])
            self.assertEqual([e.partition(" · ")[0] for e in events], expected)


if __name__ == "__main__":
    unittest.main()
