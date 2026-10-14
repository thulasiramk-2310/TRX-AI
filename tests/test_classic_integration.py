"""Exercise real classic command sequences against disposable runtime storage."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main


class ClassicIntegrationTests(unittest.TestCase):
    def test_status_profiles_review_exports_and_cache_work_in_one_session(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "RD_USE_LOCAL_LLM": "false", "RD_EMBEDDINGS": "false", "RD_VECTOR_MEMORY": "false",
            "RD_CLOUD_MEMORY": "false", "RD_STRUCTURED_LOGS": "false",
        }):
            os.chdir(tmp)
            try:
                Path("sample.py").write_text("def first(items):\n    return items[0]\n", encoding="utf-8")
                commands = iter(["help", "status", "user create audit_a", "user login audit_a",
                    "personality set dev", "hello", "user create audit_b", "user login audit_b",
                    "review sample.py", "history", "export audit.txt", "save sessions/audit.json",
                    "metrics export", "cache clear", "code", "auto", "status", "exit"])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    main.run_cli(input_fn=lambda _: next(commands))
                self.assertIn("Goodbye", output.getvalue())
                self.assertTrue(Path("reports/audit.txt").is_file())
                self.assertIsInstance(json.loads(Path("sessions/audit.json").read_text(encoding="utf-8")), list)
                self.assertIn("audit_b", json.loads(Path("sessions/users.json").read_text(encoding="utf-8"))["users"])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
