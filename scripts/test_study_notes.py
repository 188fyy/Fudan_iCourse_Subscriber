"""Offline regression checks: python scripts/test_study_notes.py."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ai.study_notes import format_notes, validate_tree
from src.ai.mindmap_render import render_map_pngs
from src.api.emailer import _md_to_html

CONTENT = '''=== 时间段 00:00 – 10:00 ===
【音频转录】
梯度下降是重点，这个知识点考试要考应用。推导不考试，但应用要考。历史背景了解即可。
【PPT 文字识别】
这是假设的重点
'''


def payload(annotations=None):
    return json.dumps({"notes": "### 梯度下降\n详细笔记。\n### 推导\n推导过程。\n### 历史背景\n背景介绍。",
                       "annotations": annotations or [],
                       "mindmap": {"topic": "优化方法", "children": [
                           {"topic": "梯度下降", "children": [{"topic": "推导", "children": []}]}]}}, ensure_ascii=False)


class StudyNotesTests(unittest.TestCase):
    def test_evidence_and_scope(self):
        note = format_notes(payload([
            {"topic": "梯度下降", "label": "考点", "quote": "梯度下降是重点，这个知识点考试要考应用。"},
            {"topic": "推导", "label": "不考试", "quote": "推导不考试，但应用要考。"},
            {"topic": "历史背景", "label": "补充理解", "quote": "历史背景了解即可。"},
        ]), CONTENT)
        self.assertIn("### 梯度下降 【考点】", note)
        self.assertIn("### 推导 【不考试】", note)
        self.assertIn("00:00 – 10:00", note)
        self.assertIn("推导不考试，但应用要考。", note)
        self.assertNotIn("历史背景 【不考试】", note)

    def test_reject_fabricated_and_ppt_quotes(self):
        note = format_notes(payload([
            {"topic": "梯度下降", "label": "考点", "quote": "这是假设的重点"},
            {"topic": "推导", "label": "不考试", "quote": "老师保证所有推导都不考"},
        ]), CONTENT)
        self.assertIn("未找到可核对", note)
        self.assertNotIn("【不考试】", note)

    def test_flat_has_no_invented_time(self):
        note = format_notes(payload([{"topic": "历史背景", "label": "补充理解", "quote": "历史背景了解即可。"}]),
                            "【音频转录（无时间轴）】\n历史背景了解即可。")
        self.assertIn("无时间轴", note)

    def test_invalid_response_fails(self):
        for raw in ('{}', '{', '{"notes":"笔记","annotations":[],"mindmap":{}}'):
            with self.assertRaises((ValueError, TypeError)):
                format_notes(raw, CONTENT)

    def test_depth_limit(self):
        tree = {"topic": "root", "children": []}
        child = tree
        for _ in range(4):
            new = {"topic": "child", "children": []}
            child["children"].append(new)
            child = new
        with self.assertRaises(ValueError):
            validate_tree(tree)

    def test_email_and_pdf_images(self):
        note = format_notes(payload(), CONTENT)
        cids = {}
        html = _md_to_html(note, cids)
        self.assertIn("cid:mindmap-", html)
        self.assertTrue(all(png.startswith(b'\x89PNG') for png in cids.values()))
        pdf_html = _md_to_html(note)
        self.assertIn("data:image/png;base64,", pdf_html)
        self.assertNotIn("icourse-mindmap", pdf_html)

    def test_old_notes_unchanged(self):
        self.assertIn("旧笔记", _md_to_html("### 旧笔记\n正常内容"))

    def test_image_pagination(self):
        tree = {"topic": "root", "children": [{"topic": str(i), "children": []} for i in range(30)]}
        self.assertEqual(len(render_map_pngs(tree)), 3)


if __name__ == "__main__":
    unittest.main()
