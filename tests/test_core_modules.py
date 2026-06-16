from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

import llm_client
import main
from intent_rules import (
    build_routing_debug,
    classify_intent,
    detect_intent,
    detect_intent_confidence,
    has_file_signal,
    has_strong_code_signal,
    is_general_question,
)
from llm_client import call_local_llm
from rule_engine import RuleEngine


def _response(status: int = 200, payload: dict | None = None, bad_json: bool = False) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    if bad_json:
        resp.json.side_effect = ValueError("not json")
    else:
        resp.json.return_value = payload or {}
    return resp


class LocalLlmClientTests(unittest.TestCase):
    def setUp(self) -> None:
        sleeper = patch.object(llm_client.time, "sleep")
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)

    def test_ollama_response_field(self) -> None:
        with patch.object(llm_client.requests, "post", return_value=_response(payload={"response": " hi ", "done_reason": "stop"})) as post:
            out = call_local_llm("p", model="m", max_new_tokens=5, temperature=3.0)
        self.assertEqual(out, {"ok": True, "text": "hi", "status_code": 200, "done_reason": "stop", "attempts": 1})
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["options"]["num_predict"], 32)  # clamped to minimum
        self.assertEqual(body["options"]["temperature"], 1.0)  # clamped to maximum
        self.assertFalse(body["stream"])

    def test_alternate_payload_shapes(self) -> None:
        shapes = [
            {"generated_text": "a"},
            {"output_text": "a"},
            {"message": {"content": "a"}},
            {"choices": [{"text": "a"}]},
            {"choices": [{"message": {"content": "a"}}]},
        ]
        for payload in shapes:
            with self.subTest(payload=payload), patch.object(llm_client.requests, "post", return_value=_response(payload=payload)):
                self.assertEqual(call_local_llm("p")["text"], "a")

    def test_retries_on_bad_status_then_succeeds(self) -> None:
        responses = [_response(status=503), _response(bad_json=True), _response(payload={"response": "ok"})]
        with patch.object(llm_client.requests, "post", side_effect=responses):
            out = call_local_llm("p", retries=3)
        self.assertTrue(out["ok"])
        self.assertEqual(out["attempts"], 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_all_attempts_fail(self) -> None:
        with patch.object(llm_client.requests, "post", side_effect=requests.ConnectionError("down")):
            out = call_local_llm("p", retries=2)
        self.assertEqual(out, {"ok": False, "text": "", "status_code": None, "attempts": 2})

    def test_empty_text_is_failure(self) -> None:
        with patch.object(llm_client.requests, "post", return_value=_response(payload={"response": "  "})):
            out = call_local_llm("p", retries=1)
        self.assertFalse(out["ok"])
        self.assertEqual(out["status_code"], 200)


class RuleEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RuleEngine()

    def test_detects_known_triggers(self) -> None:
        out = self.engine.analyze("I failed my exam because I procrastinate and feel tired", "debug", [])
        joined = " ".join(out["debug_analysis"]).lower()
        self.assertIn("preparation", joined)
        self.assertIn("energy", joined)
        self.assertIn("task-start", joined)
        self.assertTrue(45 <= out["confidence_score"] <= 90)

    def test_no_trigger_asks_for_specifics(self) -> None:
        out = self.engine.analyze("ok", "debug", [])
        self.assertIn("No strong trigger found", out["debug_analysis"][-1])
        self.assertEqual(len(out["predictions"]), 2)

    def test_modes_change_output(self) -> None:
        optimize = self.engine.analyze("deadline missed", "optimize", [])
        predict = self.engine.analyze("deadline missed", "predict", [])
        self.assertTrue(any("impact-to-effort" in item for item in optimize["improvements"]))
        self.assertTrue(any("2-8 weeks" in item for item in predict["predictions"]))

    def test_repeated_topics_from_context(self) -> None:
        out = self.engine.analyze("team deadline again", "debug", [{"input": "the team missed a deadline"}])
        self.assertTrue(any("Recurring topics" in item for item in out["debug_analysis"]))

    def test_dedupe_is_case_insensitive(self) -> None:
        self.assertEqual(RuleEngine._dedupe(["A", "a ", " ", "b"]), ["A", "b"])


class IntentRuleTests(unittest.TestCase):
    def test_classify_intent(self) -> None:
        self.assertEqual(classify_intent("Explain recursion"), "theory")
        self.assertEqual(classify_intent("please review this"), "code")
        self.assertEqual(classify_intent("I failed again"), "problem")
        self.assertEqual(classify_intent("nice weather"), "chat")

    def test_detect_intent_and_confidence(self) -> None:
        self.assertEqual(detect_intent("fix the login bug"), "code")
        self.assertEqual(detect_intent("database is slow"), "code")
        self.assertEqual(detect_intent("what is love"), "general")
        self.assertEqual(detect_intent_confidence("hello"), ("general", 1.0))
        intent, conf = detect_intent_confidence("def foo():\n    raise an error")
        self.assertEqual(intent, "code")
        self.assertGreater(conf, 0.52)
        self.assertLessEqual(conf, 0.98)

    def test_signals(self) -> None:
        self.assertTrue(has_strong_code_signal("stacktrace attached"))
        self.assertFalse(has_strong_code_signal("prefix only"))
        self.assertTrue(has_file_signal("look at main.py please"))
        self.assertFalse(has_file_signal("look at main.txt"))
        self.assertTrue(is_general_question("Why is the sky blue"))

    def test_routing_debug_payload(self) -> None:
        payload = build_routing_debug("hi", "general", 0.12345, 0.6)
        self.assertEqual(payload["confidence"], 0.123)
        self.assertTrue(payload["signals"]["greeting"])

    def test_analyzer_reexports_moved_symbols(self) -> None:
        import analyzer

        self.assertIs(analyzer.call_local_llm, call_local_llm)
        self.assertIs(analyzer.RuleEngine, RuleEngine)
        self.assertIs(analyzer.classify_intent, classify_intent)


class CliEntryTests(unittest.TestCase):
    def test_dispatches_subcommands(self) -> None:
        with patch.object(main, "run_history_mode") as history:
            self.assertEqual(main.main(["history", "--limit", "3"]), 0)
        history.assert_called_once_with(limit=3)

        with patch.object(main, "run_analyze_mode") as analyze:
            main.main(["analyze", "def f(): pass", "--mode", "optimize"])
        analyze.assert_called_once_with(text="def f(): pass", mode="optimize", debug=False, debug_cache=False)

        with patch.object(main, "run_analyze_mode_explain_routing") as explain:
            main.main(["analyze", "x", "--explain-routing"])
        explain.assert_called_once()

        with patch.object(main, "run_cli") as cli:
            main.main([])
        cli.assert_called_once_with()

    def test_eval_forwards_raw_arguments(self) -> None:
        with patch("evaluation.main", return_value=0) as eval_main:
            self.assertEqual(main.main(["eval", "--limit", "2", "--no-baseline"]), 0)
        eval_main.assert_called_once_with(["--limit", "2", "--no-baseline"])


class ReviewJsonParsingTests(unittest.TestCase):
    """qwen3:8b often answers the review prompt as JSON keyed by the section headers."""

    FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_review_json_uppercase.txt"

    def test_uppercase_header_keys_with_nested_lists_are_parsed(self) -> None:
        from analyzer import RealityAnalyzer

        sections = RealityAnalyzer._parse_code_review_sections(self.FIXTURE.read_text(encoding="utf-8"))

        debug = " ".join(sections["code_debug"])
        self.assertIn("incorrectly returning `arr[0]`", debug)
        self.assertNotIn("No critical bugs", debug)
        self.assertNotIn("['", debug)  # nested lists are flattened, not stringified
        self.assertTrue(any("empty arrays" in item for item in sections["code_improvements"]))
        self.assertTrue(any("ValueError" in item for item in sections["fix_suggestions"]))
        self.assertNotIn("parsing incomplete", " ".join(sections["performance"]))

    def test_fixed_code_extracted_from_nested_json(self) -> None:
        from analyzer import RealityAnalyzer

        fixed = RealityAnalyzer._extract_fixed_code(self.FIXTURE.read_text(encoding="utf-8"))
        self.assertTrue(fixed.startswith("def find_max(arr):"))
        self.assertIn("raise ValueError", fixed)

    def test_snake_case_keys_still_work(self) -> None:
        from analyzer import RealityAnalyzer

        text = '{"code_debug": ["off by one"], "fixed_code": "x = 1", "confidence": "85%"}'
        sections = RealityAnalyzer._parse_code_review_sections(text)
        self.assertEqual(sections["code_debug"], ["off by one"])
        self.assertEqual(sections["confidence_score"], 85)
        self.assertEqual(RealityAnalyzer._extract_fixed_code(text), "x = 1")


class ReviewFormatDriftTests(unittest.TestCase):
    """Real qwen3:8b answers that the parser used to drop entirely (eval run 2026-10-01)."""

    FIXTURES = Path(__file__).parent / "fixtures"

    def _sections(self, name: str) -> dict:
        from analyzer import RealityAnalyzer

        return RealityAnalyzer._parse_code_review_sections((self.FIXTURES / name).read_text(encoding="utf-8"))

    def test_json_with_raw_newlines_in_strings(self) -> None:
        sections = self._sections("qwen3_review_json_control_chars.txt")
        self.assertIn("sql injection", " ".join(sections["code_debug"]).lower())

    def test_review_wrapped_in_plaintext_fence(self) -> None:
        sections = self._sections("qwen3_review_plaintext_fence.txt")
        self.assertIn("memory leak", " ".join(sections["code_debug"]).lower())
        self.assertTrue(any("free" in item.lower() for item in sections["code_improvements"]))

    def test_review_wrapped_in_yaml_fence(self) -> None:
        sections = self._sections("qwen3_review_yaml_fence.txt")
        self.assertIn("out of bounds", " ".join(sections["code_debug"]).lower())

    def test_code_fences_are_still_ignored(self) -> None:
        from analyzer import RealityAnalyzer

        text = "CODE DEBUG:\n- real issue\n```python\nCODE DEBUG:\nx = 1\n```\n"
        sections = RealityAnalyzer._parse_code_review_sections(text)
        self.assertEqual(sections["code_debug"], ["real issue"])

class SemanticScorerRegressionTests(unittest.TestCase):
    def test_synonyms_do_not_penalize_label(self) -> None:
        from semantic_scoring import SemanticMatcher

        text = "No error handling is present for empty array input, which would cause an IndexError at line 2."
        self.assertGreaterEqual(SemanticMatcher().score("empty list index error", text), 0.45)

    def test_camel_case_identifiers_are_split(self) -> None:
        from semantic_scoring import _normalize_text

        self.assertEqual(_normalize_text("raises IndexError"), "raises index error")


if __name__ == "__main__":
    unittest.main()
