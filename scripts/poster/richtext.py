"""Minimal rich-text model shared by the PDF and PPTX builders.

Markup: ``**bold**``, ``*italic*``, ```code```; a paragraph (line) starting with
``- `` is a bullet with a hanging indent. That is all the poster needs, and it
keeps both renderers honest: the PDF engine below measures every word with the
same Arial metrics PowerPoint uses, so line breaks come out (nearly) the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties

from poster_style import FONT, LINE_SPACING

MONO = "Consolas"
BULLET = "•"
PARA_GAP = 0.45            # extra space between paragraphs, in line heights


@dataclass(frozen=True)
class Run:
    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False


@dataclass
class Para:
    runs: List[Run]
    bullet: bool = False


_TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*)")


def parse(text: str, bold_all: bool = False) -> List[Para]:
    paras = []
    for line in text.split("\n"):
        bullet = line.startswith("- ")
        if bullet:
            line = line[2:]
        runs = []
        for tok in _TOKEN.split(line):
            if not tok:
                continue
            if tok.startswith("**") and tok.endswith("**"):
                runs.append(Run(tok[2:-2], bold=True))
            elif tok.startswith("`") and tok.endswith("`"):
                runs.append(Run(tok[1:-1], code=True, bold=bold_all))
            elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
                runs.append(Run(tok[1:-1], italic=True, bold=bold_all))
            else:
                runs.append(Run(tok, bold=bold_all))
        paras.append(Para(runs, bullet))
    return paras


# --------------------------------------------------------------------------
# Measurement (points)
# --------------------------------------------------------------------------
_R = RendererAgg(10, 10, 72)


def font_props(run: Run, size: float) -> FontProperties:
    return FontProperties(family=MONO if run.code else FONT,
                          weight="bold" if run.bold else "normal",
                          style="italic" if run.italic else "normal", size=size)


@lru_cache(maxsize=200_000)
def _width(s: str, family: str, weight: str, style: str, size: float) -> float:
    if not s:
        return 0.0
    fp = FontProperties(family=family, weight=weight, style=style, size=size)
    # Trailing spaces are not measured by Agg; measure them via a sentinel.
    if s.endswith(" "):
        w1, _, _ = _R.get_text_width_height_descent(s.rstrip(" ") + "|", fp, ismath=False)
        w2, _, _ = _R.get_text_width_height_descent("|", fp, ismath=False)
        n = len(s) - len(s.rstrip(" "))
        ws, _, _ = _R.get_text_width_height_descent("a a", fp, ismath=False)
        wa, _, _ = _R.get_text_width_height_descent("aa", fp, ismath=False)
        return (w1 - w2) + n * (ws - wa)
    w, _, _ = _R.get_text_width_height_descent(s, fp, ismath=False)
    return w


def width(run: Run, s: str, size: float) -> float:
    return _width(s, MONO if run.code else FONT, "bold" if run.bold else "normal",
                  "italic" if run.italic else "normal", float(size))


# A laid-out line is a list of (x_pt, run, text) fragments.
Line = List[Tuple[float, Run, str]]


def wrap(paras: List[Para], width_in: float, size: float) -> List[Tuple[Para, List[Line], float]]:
    """Greedy word wrap. Returns, per paragraph, its lines and its left indent (pt)."""
    max_w = width_in * 72.0
    out = []
    for para in paras:
        indent = width(Run(""), BULLET + "  ", size) if para.bullet else 0.0
        words: List[Tuple[Run, str]] = []          # word pieces, spaces kept on the left piece
        for run in para.runs:
            for piece in re.findall(r"\S+\s*|\s+", run.text):
                words.append((run, piece))
        # merge pieces that are not separated by whitespace into one unbreakable word
        units: List[List[Tuple[Run, str]]] = []
        for run, piece in words:
            if units and not units[-1][-1][1].endswith((" ", "\t")) and not piece.startswith(" "):
                units[-1].append((run, piece))
            else:
                units.append([(run, piece)])
        lines: List[Line] = []
        cur: Line = []
        x = indent
        for unit in units:
            uw = sum(width(r, p.rstrip(" "), size) for r, p in unit[:-1]) + \
                width(unit[-1][0], unit[-1][1].rstrip(" "), size)
            if cur and x + uw > max_w:
                lines.append(cur)
                cur, x = [], indent
                if unit[0][1].strip() == "":
                    continue
            for r, p in unit:
                if not cur and p.strip() == "":
                    continue
                cur.append((x, r, p))
                x += width(r, p, size)
        if cur:
            lines.append(cur)
        out.append((para, lines, indent))
    return out


def needed_height(text: str, width_in: float, size: float, bold_all: bool = False) -> float:
    laid = wrap(parse(text, bold_all), width_in, size)
    lh = size * LINE_SPACING / 72.0
    n_lines = sum(len(lines) for _, lines, _ in laid)
    return n_lines * lh + max(0, len(laid) - 1) * PARA_GAP * lh
