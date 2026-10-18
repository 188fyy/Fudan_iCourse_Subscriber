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


def payload(annotations=None, **overrides):
    data = {"notes": "### 梯度下降\n详细笔记。\n### 推导\n推导过程。\n### 历史背景\n背景介绍。",
            "annotations": annotations or [],
            "mainline": "围绕如何减小误差，先认识梯度下降，再理解推导依据，最后了解其发展背景。",
            "transitions": [
                {"from_topic": "梯度下降", "to_topic": "推导",
                 "text": "梯度下降给出更新方法，接着通过推导解释这一方法为何能减小误差。"},
                {"from_topic": "推导", "to_topic": "历史背景",
                 "text": "推导解释了方法的依据，随后转向历史背景，补充对方法发展过程的认识。"}]}
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def old_map_note():
    return '### 旧笔记\n正文。\n\n```icourse-mindmap\n' + json.dumps({
        "topic": "优化方法", "children": [{"topic": "梯度下降", "children": []}]},
        ensure_ascii=False) + '\n```\n'


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

    def test_legacy_email_and_pdf_images(self):
        note = old_map_note()
        cids = {}
        html = _md_to_html(note, cids)
        self.assertIn("cid:mindmap-", html)
        self.assertTrue(all(png.startswith(b'\x89PNG') for png in cids.values()))
        pdf_html = _md_to_html(note)
        self.assertIn("data:image/png;base64,", pdf_html)
        self.assertNotIn("icourse-mindmap", pdf_html)

    def test_learning_logic_order_and_no_new_maps(self):
        note = format_notes(payload(), CONTENT)
        self.assertNotIn("思维导图", note)
        self.assertNotIn("icourse-mindmap", note)
        self.assertEqual(note.count("> **本课主线**"), 1)
        self.assertEqual(note.count("> **学习逻辑｜"), 2)
        self.assertLess(note.index("本课主线"), note.index("### 梯度下降"))
        self.assertLess(note.index("详细笔记。"), note.index("学习逻辑｜梯度下降 → 推导"))
        self.assertLess(note.index("学习逻辑｜梯度下降 → 推导"), note.index("### 推导"))
        self.assertLess(note.index("背景介绍。"), note.index("老师强调与考试范围"))

    def test_reminders_and_subsections_are_not_transition_targets(self):
        note = format_notes(payload(notes="### 【课程事项提醒】\n下周交作业。\n\n"
            "### 梯度下降\n正文。\n#### 更新方向\n细节。\n##### 例子\n例子。\n"
            "### 推导\n推导过程。\n### 历史背景\n背景。"), CONTENT)
        self.assertLess(note.index("下周交作业。"), note.index("本课主线"))
        self.assertLess(note.index("本课主线"), note.index("### 梯度下降"))
        self.assertEqual(note.count("> **学习逻辑｜"), 2)
        self.assertNotIn("学习逻辑｜更新方向", note)

    def test_single_section_and_fenced_headings(self):
        note = format_notes(payload(notes="### 唯一板块\n正文。\n\n```markdown\n"
            "### 示例标题\n```\n", transitions=[]), CONTENT)
        self.assertIn("本课主线", note)
        self.assertNotIn("学习逻辑｜", note)

    def test_sections_after_fenced_code_remain_anchorable(self):
        for fence in ("```", "~~~~"):
            note = format_notes(payload(notes="### 梯度下降\n正文。\n" + fence +
                "markdown\n### 示例标题\n" + fence + "\n### 推导\n推导过程。\n"
                "### 历史背景\n背景。"), CONTENT)
            self.assertEqual(note.count("> **学习逻辑｜"), 2)
            self.assertLess(note.index("学习逻辑｜梯度下降 → 推导"), note.index("### 推导"))

    def test_invalid_learning_logic_is_rejected(self):
        original = json.loads(payload())
        cases = [dict(mainline=""), dict(mainline="两行\n文本"), dict(mainline="字" * 241),
                 dict(transitions=[]), dict(transitions=None),
                 dict(transitions=original["transitions"][::-1]),
                 dict(transitions=[original["transitions"][0]] * 2),
                 dict(transitions=[None, None]),
                 dict(notes="### 重复\n正文\n### 重复\n正文"), dict(notes="#### 子板块\n正文")]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                format_notes(payload(**change), CONTENT)
        for field in ("mainline", "transitions"):
            data = original.copy()
            del data[field]
            with self.assertRaises(ValueError):
                format_notes(json.dumps(data), CONTENT)

    def test_learning_cards_email_and_pdf_html(self):
        note = format_notes(payload(), CONTENT)
        for cids in ({}, None):
            html = _md_to_html(note, cids)
            self.assertEqual(html.count('class="icourse-learning"'), 3)
            self.assertIn("background:#eff6ff", html)
            self.assertIn("break-inside:avoid", html)
            self.assertNotIn("ICOURSE", html)
            if cids is not None:
                self.assertEqual(cids, {})
        self.assertNotIn('class="icourse-learning"', _md_to_html("> 普通引文"))

    def test_card_text_is_literal_and_cannot_inject_html_or_markdown(self):
        note = format_notes(payload(mainline='<img src=x onerror=alert(1)> **文本** [链接](x)'), CONTENT)
        html = _md_to_html(note)
        self.assertNotIn("<img", html)
        self.assertNotIn("<a ", html)
        self.assertNotIn("<strong>文本</strong>", html)
        self.assertIn("&lt;img", html)

    def test_old_notes_unchanged(self):
        self.assertIn("旧笔记", _md_to_html("### 旧笔记\n正常内容"))

    def test_image_pagination(self):
        tree = {"topic": "root", "children": [{"topic": str(i), "children": []} for i in range(30)]}
        self.assertEqual(len(render_map_pngs(tree)), 3)


if __name__ == "__main__":
    unittest.main()
