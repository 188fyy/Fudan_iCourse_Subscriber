"""Offline budget, retry, privacy and orchestration regressions."""
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openai import APITimeoutError
from src.ai.summarizer import Summarizer, target_note_chars
from src.runtime.reporter import Reporter


def response(reason="stop", body=None, *, empty_choices=False, usage=True):
    if body is None:
        body = json.dumps({"notes": "### 内容\n完整但简短的笔记。",
                           "annotations": [], "mindmap": {"topic": "内容"}})
    return NS(choices=[] if empty_choices else [
        NS(finish_reason=reason, message=NS(content=body))],
        usage=NS(prompt_tokens=123, completion_tokens=45, total_tokens=168)
        if usage else None)


class SummarizerTests(unittest.TestCase):
    def setUp(self):
        self.engine = Summarizer.__new__(Summarizer)
        self.engine.providers = [{"name": "test", "models": ["primary", "backup"]}]
        self.client = Mock()
        self.engine._clients = {"test": self.client}
        self.create = self.client.chat.completions.create
        self.output = StringIO()
        capture = redirect_stdout(self.output)
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def generate(self, content="secret transcript and PPT", chars=24672):
        return self.engine.summarize("course", content, transcript_chars=chars)

    def messages(self, index=0):
        return self.create.call_args_list[index].kwargs["messages"][1]["content"]

    def test_budget_and_full_materials(self):
        self.assertEqual(target_note_chars(24672), 6168)
        self.assertEqual(target_note_chars(2002), 501)  # half-up
        self.assertEqual(target_note_chars(0), 500)
        self.assertEqual(target_note_chars(100000), 8000)
        with self.assertRaises(ValueError):
            target_note_chars(-1)
        self.create.side_effect = [response(), response()]
        for material in ("short PPT", "long PPT" * 16000):
            notes, model = self.generate(material)
            self.assertIn("完整但简短", notes)  # No minimum-length rejection.
        for index in (0, 1):
            self.assertIn("6168", self.messages(index))
            self.assertTrue(self.messages(index).endswith(
                ("short PPT", "long PPT" * 16000)[index]))
            args = self.create.call_args_list[index].kwargs
            self.assertEqual(args["max_tokens"], 24000)
            self.assertEqual(args["timeout"], 300)
        with self.assertRaises(TypeError):
            self.engine.summarize("course", "materials")

    def test_length_retry_reduces_budget(self):
        self.create.side_effect = [response("length", "TRUNCATED SECRET"), response()]
        notes, model = self.generate()
        self.assertNotIn("TRUNCATED", notes)
        self.assertEqual(model, "test/primary")
        self.assertEqual(self.create.call_count, 2)
        self.assertIn("4626", self.messages(1))
        self.assertIn("finish_reason=length", self.output.getvalue())
        self.assertIn("completion_tokens=45", self.output.getvalue())
        self.assertNotIn("TRUNCATED SECRET", self.output.getvalue())
        self.assertNotIn("secret transcript", self.output.getvalue())

    def test_empty_responses_and_timeouts_retry_same_target(self):
        cases = [response(empty_choices=True), response(body=""), response(body="  "),
                 APITimeoutError(request=Mock()),
                 TimeoutError("PRIVATE API TEXT")]
        for case in cases:
            with self.subTest(case=type(case).__name__):
                self.create.reset_mock()
                self.create.side_effect = [case, response()]
                self.generate()
                self.assertEqual(self.create.call_count, 2)
                self.assertEqual(self.messages(0), self.messages(1))
        self.assertNotIn("PRIVATE API TEXT", self.output.getvalue())

    def test_nonretryable_results_use_backup_immediately(self):
        for first in [response("content_filter", ""), response("tool_calls", ""),
                      response("unexpected PRIVATE TEXT"), response(None),
                      response(body="INVALID PRIVATE JSON"),
                      response(body='{"notes":"x"}'),
                      RuntimeError("PRIVATE API BODY")]:
            with self.subTest(first=type(first).__name__):
                self.create.reset_mock()
                self.create.side_effect = [first, response()]
                _, model = self.generate()
                self.assertEqual(model, "test/backup")
                self.assertEqual(self.create.call_count, 2)
        self.assertNotIn("PRIVATE", self.output.getvalue())

    def test_attempt_cap_and_target_reset_for_backup(self):
        self.create.side_effect = [response("length")] * 4
        with self.assertRaisesRegex(RuntimeError, "All LLM models failed"):
            self.generate()
        self.assertEqual(self.create.call_count, 4)
        self.assertIn("6168", self.messages(2))
        self.assertIn("4626", self.messages(3))

    def test_missing_usage_still_diagnoses_before_format_validation(self):
        self.create.side_effect = [response(body="invalid", usage=False), response()]
        self.generate()
        logs = self.output.getvalue()
        self.assertIn("prompt_tokens=unknown", logs)
        self.assertIn("finish_reason=stop", logs)
        self.assertLess(logs.index("body_chars=7"), logs.index("failed:"))

    def test_range_cap_and_sdk_retry_defaults(self):
        self.create.return_value = response()
        self.generate(chars=100000)
        self.assertIn("6800～8000", self.messages())
        with patch("src.ai.summarizer.config.resolve_model_providers", return_value=[{
                "name": "test", "models": ["model"], "api_key": "test-key",
                "base_url": "https://example.com"}]), patch(
                "src.ai.summarizer.OpenAI") as constructor:
            Summarizer()
        self.assertNotIn("max_retries", constructor.call_args.kwargs)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        # Load only orchestration functions, avoiding ASR/OCR native dependencies.
        path = Path(__file__).resolve().parents[1] / "main.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in {"_drive_lectures", "_send_email"}]
        module = ast.Module(body=[ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0)] + selected,
                            type_ignores=[])
        self.runner = Mock()
        self.scope = {"LectureRunner": Mock(return_value=self.runner),
                      "_check_session": Mock(), "traceback": Mock()}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), self.scope)
        self.reporter = Reporter()
        self.db = Mock()
        self.db.get_unsent_lectures.return_value = []
        self.emailer = Mock()
        self.emailer.send.return_value = True

    def test_mixed_result_summary_and_email_only_success(self):
        self.runner.run.side_effect = ["complete notes", RuntimeError("failed"), None]
        self.db.get_lecture.return_value = {"processed_at": "already processed"}
        lectures = [("course", "title", {"sub_id": str(i)}) for i in range(3)]
        items = []
        output = StringIO()
        with redirect_stdout(output):
            self.scope["_drive_lectures"](Mock(), self.db, Mock(), Mock(), Mock(),
                                           self.reporter, lectures, items)
            self.scope["_send_email"](self.emailer, self.db, self.reporter, items)
        self.assertIn("attempted=3 succeeded=1 failed=1 skipped=1", output.getvalue())
        self.assertIn("::warning::1 lecture(s) failed", output.getvalue())
        self.assertEqual(len(items), 1)
        self.emailer.send.assert_called_once_with(items)
        self.db.mark_emailed_batch.assert_called_once_with(["0"])

    def test_failed_without_exception_and_no_email(self):
        self.runner.run.return_value = None
        self.db.get_lecture.return_value = {"processed_at": None}
        output = StringIO()
        with redirect_stdout(output):
            self.scope["_drive_lectures"](Mock(), self.db, Mock(), Mock(), Mock(),
                self.reporter, [("course", "title", {"sub_id": "id"})], [])
            self.scope["_send_email"](self.emailer, self.db, self.reporter, [])
        self.assertIn("attempted=1 succeeded=0 failed=1 skipped=0", output.getvalue())
        self.assertIn("no email will be sent", output.getvalue())
        self.emailer.send.assert_not_called()

    def test_truncated_response_never_saved(self):
        path = Path(__file__).resolve().parents[1] / "src/pipeline/lecture_runner.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "LectureRunner")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_summarize")
        module = ast.Module(body=[ast.ImportFrom(module="__future__",
                  names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        import time
        scope = {"time": time, "bucketer": Mock()}
        scope["bucketer"].assemble.return_value = ("transcript and PPT", "flat")
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        engine = Summarizer.__new__(Summarizer)
        engine.providers = [{"name": "test", "models": ["model"]}]
        client = Mock()
        client.chat.completions.create.return_value = response("length", "PRIVATE PARTIAL")
        engine._clients = {"test": client}
        runner = NS(_db=self.db, _reporter=self.reporter, _summarizer=engine)
        output = StringIO()
        with redirect_stdout(output), self.assertRaises(RuntimeError):
            scope["_summarize"](runner, "sub", "course", "转录" * 1000, None)
        self.db.update_summary.assert_not_called()
        self.db.update_error.assert_called_once()
        self.assertNotIn("PRIVATE PARTIAL", output.getvalue())


if __name__ == "__main__":
    unittest.main()
