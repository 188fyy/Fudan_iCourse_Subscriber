"""Render private mind maps locally; never send course text to a diagram service."""
import base64
import json
import uuid
from html import escape
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from src.ai.study_notes import MAP_RE, validate_tree, LABELS


def render_map_pngs(tree):
    validate_tree(tree)
    font_path = Path(__file__).resolve().parents[2] / "assets" / "MSYH.TTC"
    font = ImageFont.truetype(str(font_path), 24)
    rows = []

    def walk(node, depth):
        tags = [t for t in node.get("labels", []) if t in LABELS]
        label = node["topic"] + " ".join(f"【{t}】" for t in tags)
        # Fixed character wrapping bounds even for CJK text and long labels.
        width = 40 - depth * 3
        lines = [label[i:i + width] for i in range(0, len(label), width)]
        rows.append((depth, lines))
        for child in node.get("children", []):
            walk(child, depth + 1)
    walk(tree, 0)
    pngs = []
    for offset in range(0, len(rows), 12):
        chunk = rows[offset:offset + 12]
        height = 62 + sum(len(lines) * 34 + 20 for _, lines in chunk)
        image = Image.new("RGB", (1200, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 12), f"本课次思维导图 {offset // 12 + 1}", font=font, fill="#334155")
        y = 60
        for depth, lines in chunk:
            x = 30 + depth * 64
            h = len(lines) * 34 + 10
            color = "#dbeafe" if depth == 0 else "#f1f5f9"
            draw.rounded_rectangle((x, y, 1170, y + h), radius=8, fill=color)
            if depth:
                draw.line((x - 22, y - 10, x - 22, y + 20, x - 4, y + 20), fill="#94a3b8", width=2)
            for index, line in enumerate(lines):
                draw.text((x + 12, y + 3 + index * 34), line, font=font, fill="#0f172a")
            y += h + 10
        buf = BytesIO()
        image.save(buf, format="PNG")
        pngs.append(buf.getvalue())
    return pngs


def stash_maps(text, cid_images=None):
    replacements = {}

    def replace(match):
        key = "ICOURSEMAP" + uuid.uuid4().hex
        try:
            tree = json.loads(match[1])
            images = render_map_pngs(tree)
            html = []
            for png in images:
                if cid_images is not None:
                    cid = "mindmap-" + uuid.uuid4().hex
                    cid_images[cid] = png
                    src = "cid:" + cid
                else:
                    src = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
                html.append(f'<div style="break-inside:avoid;margin:12px 0"><img src="{src}" '
                            'alt="本课次思维导图" style="width:100%;max-width:800px;height:auto"></div>')
            replacements[key] = "".join(html)
        except (ValueError, TypeError, OSError) as exc:
            print(f"[Mindmap] Render failed: {type(exc).__name__}")
            replacements[key] = "<pre>" + escape(match[1]) + "</pre>"
        return "\n\n" + key + "\n\n"
    return MAP_RE.sub(replace, text), replacements
