#!/usr/bin/env python3
"""Assemble the 72 x 48 in SfN poster as a print PDF (plus a small PNG preview).

Reads the layout and text from ``poster_spec.py``, numbers from
``results/poster/data/facts.json`` and panels from ``results/poster/panels/``
(run ``make_panels.py`` first). Text is laid out by ``richtext.py`` with real
font metrics, and the build fails if any text box overflows its slot.

    python scripts/poster/build_poster_pdf.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402

import poster_spec as spec  # noqa: E402
from poster_style import (INK, LINE_SPACING, NAVY, PAPER, POSTER_H, POSTER_W,  # noqa: E402
                          apply_mpl_style)
from richtext import BULLET, PARA_GAP, font_props, needed_height, parse, width, wrap  # noqa: E402

BOX_PAD = 0.35
Image.MAX_IMAGE_PIXELS = None


def fill(text: str, facts: dict) -> str:
    return text.format_map(facts)


def draw_text(ax, x, y, w, text, size, color=INK, align="left", bold_all=False):
    lh = size * LINE_SPACING / 72.0
    base = y + size * 0.93 / 72.0                    # first baseline below the box top
    for para, lines, indent in wrap(parse(text, bold_all), w, size):
        for i, line in enumerate(lines):
            if not line:
                continue
            lw = line[-1][0] + width(line[-1][1], line[-1][2].rstrip(), size)
            dx = (w * 72 - lw) / 2 if align == "center" else 0.0
            if para.bullet and i == 0:
                ax.text(x + dx / 72, base, BULLET, fontproperties=font_props(line[0][1], size),
                        color=color, va="baseline", ha="left")
            for xp, run, frag in line:
                ax.text(x + (xp + dx) / 72, base, frag.rstrip() if frag.endswith(" ") else frag,
                        fontproperties=font_props(run, size), color=color, va="baseline",
                        ha="left")
            base += lh
        base += PARA_GAP * lh


def main(name: str = "poster_v6") -> int:
    apply_mpl_style()
    facts = json.loads((spec.DATA / "facts.json").read_text())
    els = spec.layout()
    problems = [f"column {c + 1} overflows by {o:.2f} in" for c, o in spec.OVERFLOW.items() if o > 0]

    fig = plt.figure(figsize=(POSTER_W, POSTER_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, POSTER_W)
    ax.set_ylim(POSTER_H, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), POSTER_W, POSTER_H, fc=PAPER, ec="none", zorder=0))

    for e in els:
        if e.kind == "rect":
            ax.add_patch(Rectangle((e.x, e.y), e.w, e.h, fc=e.fill, ec="none", zorder=1))
        elif e.kind == "header":
            ax.add_patch(FancyBboxPatch((e.x, e.y), e.w, e.h,
                                        boxstyle="round,pad=0,rounding_size=0.18",
                                        fc=NAVY, ec="none", zorder=1))
            ax.text(e.x + 0.35, e.y + e.h / 2 + e.size * 0.36 / 72, e.text, color="white",
                    fontsize=e.size, fontweight="bold", va="baseline", ha="left", zorder=2)
        elif e.kind in ("text", "box"):
            txt = fill(e.text, facts)
            pad = BOX_PAD if e.kind == "box" else 0.0
            if e.kind == "box":
                ax.add_patch(FancyBboxPatch((e.x, e.y), e.w, e.h,
                                            boxstyle="round,pad=0,rounding_size=0.18",
                                            fc=e.fill, ec="none", zorder=1))
            need = needed_height(txt, e.w - 2 * pad, e.size) + 2 * pad
            if need > e.h + 0.02:
                problems.append(f"text overflows ({need:.2f} > {e.h:.2f} in): {e.text[:60]!r}")
            ty = e.y + pad
            if e.kind == "box" and e.align == "center":
                ty = e.y + (e.h - need) / 2 + pad
            draw_text(ax, e.x + pad, ty, e.w - 2 * pad, txt, e.size, color=e.color or INK,
                      align=e.align)
        elif e.kind == "figure":
            img = np.asarray(Image.open(spec.PANELS / f"{e.name}.png").convert("RGB"))
            ax.imshow(img, extent=(e.x, e.x + e.w, e.y + e.h, e.y), interpolation="none",
                      zorder=2)
    ax.set_xlim(0, POSTER_W)
    ax.set_ylim(POSTER_H, 0)

    if problems:
        print("LAYOUT PROBLEMS:\n  " + "\n  ".join(problems))
    for fname in (f"{name}.pdf", f"{name}b.pdf"):          # never touch older versions
        out = spec.OUT / fname
        try:
            fig.savefig(out, dpi=300)
            break
        except PermissionError:             # open in a PDF viewer (Windows locks it)
            print(f"{fname} is locked (open in a viewer?); trying the next name")
    preview = spec.OUT / f"{name}_preview.png"
    fig.savefig(preview, dpi=40)
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {preview}")
    return 1 if problems else 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="poster_v6", help="output stem")
    sys.exit(main(ap.parse_args().name))
