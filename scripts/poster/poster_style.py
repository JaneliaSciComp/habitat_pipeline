"""Shared visual style for the SfN poster (both the PDF and the PPTX builds).

Everything is in physical units: inches for geometry, points for type. Panel
figures are rendered at the exact size of the box they occupy on the poster,
so a 24 pt tick label in a panel prints at 24 pt.
"""

from __future__ import annotations

import matplotlib as mpl

# --- page ---------------------------------------------------------------
POSTER_W, POSTER_H = 72.0, 48.0          # inches, landscape (SfN board is 8 x 4 ft)
DPI = 300                                 # panel rasters

# --- type (pt) ----------------------------------------------------------
FONT = "Arial"
TITLE_PT = 104
AUTHORS_PT = 46
AFFIL_PT = 36
HEADER_PT = 54
SUBHEAD_PT = 38
BODY_PT = 30
CAPTION_PT = 26
SMALL_PT = 22
FIG_LABEL_PT = 28                         # axis labels inside panels
FIG_TICK_PT = 24
FIG_TITLE_PT = 30
LINE_SPACING = 1.22                       # multiple of font size

# --- colour (Okabe-Ito based; colour-blind safe) ------------------------
INK = "#1d2433"
MUTED = "#5b6475"
RULE = "#c9ced8"
PAPER = "#ffffff"
PANEL_BG = "#f3f5f9"
NAVY = "#123a63"                          # title band / section bars
ACCENT = "#0b6e99"
SIG = "#d1352b"                           # significant / observed
NULL = "#aeb4bf"                          # null / not significant
COHORT = {"cohort5": "#0072B2", "cohort7": "#D55E00", "cohort9": "#009E73"}
OPP = ["#E69F00", "#56B4E9"]              # the two opponents in a binary decode
STREAM = {"ephys": "#0072B2", "video": "#009E73", "events": "#D55E00", "sync": "#CC79A7"}


def apply_mpl_style() -> None:
    mpl.rcParams.update({
        "font.family": FONT,
        "font.size": FIG_TICK_PT,
        "axes.labelsize": FIG_LABEL_PT,
        "axes.titlesize": FIG_TITLE_PT,
        "axes.titleweight": "bold",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": FIG_TICK_PT,
        "ytick.labelsize": FIG_TICK_PT,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "legend.fontsize": FIG_TICK_PT,
        "legend.frameon": False,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": PAPER,
        "figure.facecolor": PAPER,
    })
