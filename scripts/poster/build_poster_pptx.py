#!/usr/bin/env python3
"""Assemble the SfN poster as an editable one-slide PowerPoint.

PowerPoint caps slides at 56 in, so the deck is built at exactly half scale
(36 x 24 in) with every font halved. **Print at 200 %** to get the 72 x 48 in
poster, identical in layout to ``poster.pdf``. All text is live, editable text;
panels are embedded as 300 dpi PNGs (600 dpi at the half-scale slide size).

    python scripts/poster/build_poster_pptx.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

import poster_spec as spec  # noqa: E402
from poster_style import FONT, INK, LINE_SPACING, NAVY, POSTER_H, POSTER_W  # noqa: E402
from richtext import BULLET, MONO, PARA_GAP, Run, parse, width  # noqa: E402

S = 0.5                     # slide scale; print at 1/S = 200 %
BOX_PAD = 0.35


def _in(v):
    return Inches(v * S)


def _rgb(hex_):
    return RGBColor.from_string(hex_.lstrip("#"))


def _fill_text(tf, text, size, color=INK, align="left"):
    tf.word_wrap = True
    tf.auto_size = None
    paras = parse(text)
    lh = size * LINE_SPACING
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER if align == "center" else PP_ALIGN.LEFT
        p.line_spacing = Pt(lh * S)
        p.space_before = Pt(0)
        p.space_after = Pt(PARA_GAP * lh * S) if i < len(paras) - 1 else Pt(0)
        runs = list(para.runs)
        if para.bullet:
            ind = Emu(int(width(Run(""), BULLET + "  ", size) * S * 12700))
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(ind)))
            pPr.set("indent", str(-int(ind)))
            runs = [Run(BULLET + "\t")] + runs
        for r in runs:
            run = p.add_run()
            run.text = r.text
            f = run.font
            f.name = MONO if r.code else FONT
            f.size = Pt(size * S)
            f.bold = r.bold
            f.italic = r.italic
            f.color.rgb = _rgb(color)


def _shape(slide, kind, x, y, w, h, fill=None):
    shp = slide.shapes.add_shape(kind, _in(x), _in(y), _in(w), _in(h))
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb(fill)
    else:
        shp.fill.background()
    shp.line.fill.background()
    shp.shadow.inherit = False
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = min(0.5, 0.18 / min(w, h))
    return shp


def _margins(tf, pad):
    tf.margin_left = tf.margin_right = _in(pad)
    tf.margin_top = tf.margin_bottom = _in(pad)


def main(name: str = "poster_v6") -> int:
    facts = json.loads((spec.DATA / "facts.json").read_text())
    prs = Presentation()
    prs.slide_width, prs.slide_height = _in(POSTER_W), _in(POSTER_H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])          # blank
    n_text = n_pic = 0
    for e in spec.layout():
        if e.kind == "rect":
            _shape(slide, MSO_SHAPE.RECTANGLE, e.x, e.y, e.w, e.h, e.fill)
        elif e.kind == "header":
            shp = _shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, e.x, e.y, e.w, e.h, NAVY)
            tf = shp.text_frame
            _margins(tf, 0)
            tf.margin_left = _in(0.35)
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            _fill_text(tf, f"**{e.text}**", e.size, color="#ffffff")
            n_text += 1
        elif e.kind == "box":
            shp = _shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, e.x, e.y, e.w, e.h, e.fill)
            tf = shp.text_frame
            _margins(tf, BOX_PAD)
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE if e.align == "center" else MSO_ANCHOR.TOP
            _fill_text(tf, e.text.format_map(facts), e.size, e.color or INK, e.align)
            n_text += 1
        elif e.kind == "text":
            tb = slide.shapes.add_textbox(_in(e.x), _in(e.y), _in(e.w), _in(e.h))
            tf = tb.text_frame
            _margins(tf, 0)
            tf.vertical_anchor = MSO_ANCHOR.TOP
            _fill_text(tf, e.text.format_map(facts), e.size, e.color or INK, e.align)
            n_text += 1
        elif e.kind == "figure":
            pic = slide.shapes.add_picture(str(spec.PANELS / f"{e.name}.png"),
                                           _in(e.x), _in(e.y), _in(e.w), _in(e.h))
            pic.name = f"panel: {e.name}"
            n_pic += 1
    out = spec.OUT / f"{name}_36x24in_print_at_200pct.pptx"
    prs.save(out)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB; {n_text} text boxes, {n_pic} panels)")
    print("slide is 36 x 24 in: print at 200 % for the 72 x 48 in poster")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="poster_v6", help="output stem")
    sys.exit(main(ap.parse_args().name))
