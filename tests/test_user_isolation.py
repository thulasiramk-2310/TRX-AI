"""User isolation across cached responses, retrieval, CLI, and TUI."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, PropertyMock, patch
from analyzer_helpers import analyzer_stub
from config import AppConfig
from conversation_memory import ConversationMemory
from history import SessionHistory
from tui.engine import Engine
from vector_memory import VectorMemory


class OwnerIsolationTests(unittest.TestCase):
    def test_cached_response_cannot_cross_users_or_personalities(self):
        analyzer = analyzer_stub()
        analyzer.set_current_user("alice")
        key = analyzer._cache_key("Tell me about my project", "debug", [])
        analyzer._cache_set(key, {"chat_response": "Alice's project", "system_status": []})
        analyzer.set_current_user("bob")
        self.assertIsNone(analyzer._cache_get(analyzer._cache_key("Tell me about my project", "debug", [])))
        analyzer.set_current_user("alice")
        self.assertEqual(analyzer._cache_get(key)["chat_response"], "Alice's project")
        analyzer.set_personality("mentor")
        self.assertIsNone(analyzer._cache_get(analyzer._cache_key("Tell me about my project", "debug", [])))

    def test_cache_preserves_case_and_runtime_configuration(self):
        analyzer = analyzer_stub()
        key = analyzer._cache_key("Explain Foo", "debug", [])
        self.assertNotEqual(key, analyzer._cache_key("Explain foo", "debug", []))
        analyzer.config.local_llm_model = "other-model"
        self.assertNotEqual(key, analyzer._cache_key("Explain Foo", "debug", []))

    def test_conversation_round_trip_is_scoped_by_owner(self):
        memory = ConversationMemory(max_turns=1)
        memory.add("friendly", "Alice private project", "one", owner="alice")
        memory.add("friendly", "Bob project", "two", owner="bob")
        self.assertNotIn("Alice", memory.get_context("friendly", owner="bob"))
        self.assertIn("Alice", memory.get_context("friendly", owner="alice"))
        self.assertEqual(memory.get_context("friendly"), "")

    def test_guest_qa_searches_only_guest_vectors(self):
        analyzer = analyzer_stub()
        with tempfile.TemporaryDirectory() as tmp:
            vectors = VectorMemory(enabled=False, path=str(Path(tmp) / "memory.json"))
            vectors.enabled = True
            vectors.entries = [
                {"text": "private project", "user": "alice", "personality": "friendly"},
                {"text": "public project", "user": "guest", "personality": "friendly"},
            ]
            analyzer._vector_memory = vectors
            analyzer._cloud_memory = Mock()
            analyzer.call_llm = Mock(return_value="answer")
            with patch.object(VectorMemory, "_model", new_callable=PropertyMock, return_value=None), \
                    patch.object(vectors, "search", wraps=vectors.search) as search:
                analyzer._handle_general_qa("tell me about project")
            self.assertEqual(search.call_args.kwargs["user"], "guest")
            prompt = analyzer.call_llm.call_args.args[0]
            self.assertIn("public project", prompt)
            self.assertNotIn("private project", prompt)

    def test_cli_history_filters_before_taking_recent_entries(self):
        history = SessionHistory()
        history.add_entry("Alice project", "debug", {"current_user": "alice"})
        history.add_entry("Bob project", "debug", {"current_user": "bob"})
        self.assertEqual(history.recent_context(1, owner="alice")[0]["input"], "Alice project")
        self.assertEqual(history.recent_context(1, owner="guest"), [])

    def test_tui_chat_does_not_pass_another_owners_context(self):
        analyzer = Mock(current_user="alice")
        analyzer.analyze.side_effect = lambda *args, **kwargs: {"response_mode": "chat"}
        engine = Engine(analyzer=analyzer, config=AppConfig(use_local_llm=False))
        with patch.object(engine._main, "_append_run_history"):
            engine.chat("Alice project")
            analyzer.current_user = "bob"
            engine.chat("Bob project")
            self.assertEqual(analyzer.analyze.call_args.kwargs["past_context"], [])
            analyzer.current_user = "alice"
            engine.chat("continue")
            self.assertEqual(analyzer.analyze.call_args.kwargs["past_context"][0]["input"], "Alice project")
