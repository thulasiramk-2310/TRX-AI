"""Failures discovered during the complete runtime and persistence audit."""
from concurrent.futures import ThreadPoolExecutor
import json
import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cli_input
from cloud_memory_sync import CloudMemorySync
from config import AppConfig
from graph_query_adapter import CodeReviewGraphAdapter, GraphQueryResult
from guardrails import Finding, redact
from history import SessionHistory
from llm_client import call_local_llm
from personality import load_personalities
from router_memory import RouterMemory
from tui.engine import Engine
from user_profiles import UserProfiles
from vector_memory import VectorMemory
from watcher import CodeChangeHandler
from main import apply_code_fix

SECRET = "ghp_" + "z" * 36


class AuditRegressionTests(unittest.TestCase):
    def test_fix_comment_reason_cannot_inject_executable_statements(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.py"
            source.write_text("x = 1\n", encoding="utf-8")
            output = apply_code_fix(str(source), "x = 2\n", reason="safe\nprint('injected')")
            parsed = ast.parse(output.read_text(encoding="utf-8"))
            self.assertEqual(len(parsed.body), 1)
            self.assertIsInstance(parsed.body[0], ast.Assign)

    def test_fixed_output_link_cannot_overwrite_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.py"
            source.write_text("x = 1\n", encoding="utf-8")
            destination = source.with_name("sample_fixed.py")
            os.link(source, destination)
            with self.assertRaises(ValueError):
                apply_code_fix(str(source), "x = 2\n")
            self.assertEqual(source.read_text(encoding="utf-8"), "x = 1\n")
    def test_invalid_model_payload_shapes_retry_instead_of_crashing(self):
        for payload in ([], None, 42, "text", {"response": None}, {"response": ["text"]}):
            with self.subTest(payload=payload):
                bad, good = MagicMock(), MagicMock()
                bad.status_code = good.status_code = 200
                bad.json.return_value = payload
                good.json.return_value = {"response": "valid"}
                with patch("llm_client.requests.post", side_effect=[bad, good]), patch("llm_client.time.sleep"):
                    result = call_local_llm("input", retries=2)
                self.assertEqual(result["text"], "valid")
                self.assertEqual(result["attempts"], 2)

    def test_guardrail_findings_can_be_saved_in_session_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = SessionHistory()
            history.add_entry("review", "review", {"guardrails": [Finding("warn", "token " + SECRET)]})
            path = history.save(str(Path(tmp) / "history.json"))
            payload = path.read_text(encoding="utf-8")
            self.assertNotIn(SECRET, payload)
            self.assertEqual(json.loads(payload)[0]["analysis"]["guardrails"][0]["level"], "warn")

    def test_redaction_covers_dictionary_keys(self):
        self.assertNotIn(SECRET, json.dumps(redact({SECRET: {"input": SECRET}})))

    def test_router_and_profile_persistence_dont_leak_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "router.json"
            memory = RouterMemory(str(path))
            memory.record("help with " + SECRET, "code")
            self.assertNotIn(SECRET, path.read_text(encoding="utf-8"))
            self.assertEqual(RouterMemory(str(path)).get_bias("help with " + SECRET), "code")
            path.write_text(json.dumps({"old prompt": {"code": 2, "general": 0}}), encoding="utf-8")
            self.assertEqual(RouterMemory(str(path)).get_bias("old prompt"), "code")
            profiles = UserProfiles(str(Path(tmp) / "users.json"))
            profiles.set_user("alice", {"preferences": {"note": SECRET}})
            self.assertNotIn(SECRET, profiles.path.read_text(encoding="utf-8"))

    def test_prompt_history_redacts_disk_and_legacy_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt-history.txt"
            with patch.object(cli_input.sys.stdin, "isatty", return_value=True), \
                    patch.object(cli_input.sys.stdout, "isatty", return_value=True), \
                    patch("prompt_toolkit.PromptSession") as session:
                cli_input.build_prompt_fn(cli_input.StatusState(), history_path=path)
            history = session.call_args.kwargs["history"]
            history.store_string("secret " + SECRET)
            self.assertNotIn(SECRET, path.read_text(encoding="utf-8"))
            path.write_text("\n# legacy\n+secret " + SECRET + "\n", encoding="utf-8")
            self.assertNotIn(SECRET, " ".join(history.load_history_strings()))

    def test_nonobject_memory_and_graph_cache_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            for payload in ([], None, 42, {"version": 2, "items": [None, "bad"]}):
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertEqual(VectorMemory(enabled=False, path=str(path)).entries, [])
                graph = CodeReviewGraphAdapter(disk_cache_path=str(path))
                self.assertEqual(dict(graph._cache), {})

    def test_parallel_graph_cache_updates_leave_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            graph = CodeReviewGraphAdapter(disk_cache_path=str(path))
            result = GraphQueryResult(True, "test", False, {}, [], 0)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda i: graph._cache_set(str(i), result), range(80)))
                list(pool.map(lambda i: graph._cache_get(str(i)), range(80)))
            self.assertEqual(len(json.loads(path.read_text(encoding="utf-8"))["items"]), 80)

    def test_exports_reject_escaping_reports_and_create_subfolders(self):
        with tempfile.TemporaryDirectory() as tmp, patch("history.Path", wraps=Path) as paths:
            paths.return_value = Path(tmp) / "reports"
            for name in ("../outside", str(Path(tmp) / "outside")):
                with self.assertRaises(ValueError):
                    SessionHistory._report_path(name, default_name="report.txt", suffix=".txt")
            out = SessionHistory._report_path("sub/report", default_name="report.txt", suffix=".txt")
            self.assertTrue(out.parent.is_dir())

    def test_long_pdf_paginates_and_masks_secrets(self):
        history = SessionHistory()
        doc = MagicMock()
        content = "[DEBUG ANALYSIS]\n" + "\n".join("- " + "W" * 150 for _ in range(80))
        content += "\n[LLM FIXED CODE]\n" + "\n".join(f"value_{i} = {i}" for i in range(120))
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(history, "_prepare_report_path", return_value=Path(tmp) / "report.pdf"), \
                patch("reportlab.pdfgen.canvas.Canvas", return_value=doc):
            history.export_pdf_report("report.pdf", "token " + SECRET, "review", content)
        self.assertGreater(doc.showPage.call_count, 3)
        self.assertNotIn(SECRET, str(doc.drawString.call_args_list))
        self.assertTrue(all(call.args[1] >= 0 for call in doc.drawString.call_args_list))
        self.assertIn("value_119", str(doc.drawString.call_args_list))

    def test_comparison_pdf_reports_confidence_drop(self):
        history = SessionHistory()
        doc = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(history, "_prepare_report_path", return_value=Path(tmp) / "report.pdf"), \
                patch("reportlab.pdfgen.canvas.Canvas", return_value=doc):
            history.export_comparison_pdf_report("report.pdf", first_input="before", second_input=SECRET,
                mode="review", first_structured_output="[CONFIDENCE SCORE]\n80%",
                second_structured_output="[CONFIDENCE SCORE]\n50%")
        self.assertIn("confidence dropped", str(doc.drawString.call_args_list))
        self.assertNotIn(SECRET, str(doc.drawString.call_args_list))

    def test_watcher_ignores_fixed_copies_and_dependency_folders(self):
        watcher = CodeChangeHandler(object(), object())
        for path in ("src/test_fixed.js", "src/test_fixed.ts", "src/test_fixed.py", "node_modules/lib/code.js"):
            self.assertTrue(watcher._is_temp_or_ignored(path))
        self.assertFalse(watcher._is_temp_or_ignored("src/code.js"))

    def test_cloud_retrieval_filters_personality(self):
        cloud = CloudMemorySync(enabled=True)
        cloud.client = MagicMock()
        cloud.fetch_memory("alice", personality="dev")
        query = cloud.client.table.return_value.select.return_value
        query.eq.assert_called_once_with("user", "alice")
        query.eq.return_value.eq.assert_called_once_with("personality", "dev")

    def test_readonly_personality_store_does_not_prevent_startup(self):
        with patch("personality.PERSONALITIES_PATH") as path, \
                patch("personality._persist_personalities", side_effect=PermissionError("readonly")):
            path.exists.return_value = False
            self.assertIn("friendly", load_personalities())

    def test_environment_cache_and_context_limits_are_positive(self):
        with patch.dict(os.environ, {"RD_CACHE_SIZE": "-1", "RD_CONTEXT_WINDOW": "0"}):
            config = AppConfig.from_env()
        self.assertGreaterEqual(config.cache_size, 1)
        self.assertGreaterEqual(config.context_window_size, 1)

    def test_tui_fix_rejects_noncode_files_before_analysis(self):
        engine = Engine.__new__(Engine)
        engine._main = __import__("main")
        engine.analyzer = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "secret.env"
            source.write_text("not code", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                engine.review(str(source), for_fix=True)
        engine.analyzer.analyze_code_multi_agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
