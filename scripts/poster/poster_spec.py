"""Single source of truth for the SfN poster: layout boxes, text, figure slots.

Both builders (``build_poster_pdf.py``, ``build_poster_pptx.py``) and the panel
generator (``make_panels.py``) read this module, so a box moved here moves in
both outputs and the panel re-renders at the new size.

Text markup (kept tiny on purpose so both renderers can honour it):

* ``**bold**`` runs inside a paragraph;
* a paragraph starting with ``- `` is a bullet;
* ``{name}`` / ``{name:fmt}`` fields are filled from ``results/poster/data/facts.json``,
  which ``make_panels.py`` writes after asserting every number against its source.

Placeholders the authors must fill in are wrapped in square brackets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from poster_style import (AFFIL_PT, AUTHORS_PT, BODY_PT, CAPTION_PT, HEADER_PT,
                          POSTER_H, POSTER_W, SMALL_PT, SUBHEAD_PT, TITLE_PT)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "poster"
PANELS = OUT / "panels"
DATA = OUT / "data"
GUI = OUT / "gui"


@dataclass
class Item:
    kind: str                      # 'header' | 'text' | 'figure' | 'box' | 'rect'
    h: float = 0.0                 # height in inches (stacked items)
    text: str = ""
    name: str = ""                 # figure slot name -> PANELS/<name>.png
    size: float = BODY_PT
    color: Optional[str] = None
    fill: Optional[str] = None
    align: str = "left"
    bold: bool = False
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------
MARGIN = 0.8
TITLE_H = 7.0
GUTTER = 0.7
GAP = 0.32                         # vertical gap between stacked items
TOP = TITLE_H + 0.6
NAVY_BAR = "#123a63"
BOTTOM = POSTER_H - MARGIN
COL_W = [15.5, 17.3, 18.25, 17.25]


def _col_x() -> List[float]:
    xs, x = [], MARGIN
    for w in COL_W:
        xs.append(x)
        x += w + GUTTER
    return xs


COL_X = _col_x()
OVERFLOW: Dict[int, float] = {}          # column -> inches past the bottom margin (>0 = overflow)


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------
TITLE = ("Polis toolkit: an analysis framework for multi-animal extracellular "
         "electrophysiology in freely interacting rats")
AUTHORS = "Mikhail Proskurin¹, Yuxin Pan¹, Konrad Rokicki¹\\*, Gowan Tervo¹\\*"
AFFIL = "¹ Janelia Research Campus, HHMI, Ashburn, VA   ·   \\* Co-senior authors"

ABSTRACT = (
    "Brains evolved for social life, yet systems neuroscience has mostly recorded "
    "one animal at a time, leaving **brains in dialog** largely unexplored. Simultaneous "
    "extracellular recordings from several freely interacting animals are now possible, and "
    "**the bottleneck has shifted to analysis**: each session yields spikes, multi-animal "
    "tracking and scored behavior from separate systems on independent clocks. **Polis "
    "toolkit** is an open-source Python framework that **ingests and aligns all three streams** "
    "and exposes every animal's data through **one shared-clock interface**. Its modules decode "
    "opponent identity, interaction outcome and position, map population geometry "
    "(PCA/UMAP/Rastermap), test **social place fields** for specific conspecifics, and extract "
    "**shared and unique inter-brain subspaces**. A Python API and an interactive GUI give "
    "scripted and point-and-click access. We applied it to a **12-rat colony (4 with wireless chronic "
    "Neuropixels)** in a large arena, for analyses ranging from quality control to "
    "partner-specific coding and inter-brain coupling."
)

INTRO = (
    "**Multimodal data from many animals, across weeks.** A colony of rats lived together in "
    "a shared, enriched arena. Up to {c7_max_sim} implanted rats were recorded "
    "**simultaneously** for {c7_days} consecutive days, while overhead "
    "cameras tracked every animal and human scorers annotated social encounters. The result is "
    "one record of spikes, positions and behavior from the same interacting individuals: "
    "{n_animal_recordings} animal-recordings and {hours_c57:.0f} h of loadable "
    "electrophysiology so far."
)

AI_INTRO = (
    "An AI coding agent (Claude Code) drives the same CLIs and loaders as the GUI, through "
    "four repo skills and two subagents. It works *within* a stage; a scientist decides "
    "*between* stages (red ♦ gates), where human input is most necessary."
)

AI_GUARDS = (
    "**Guard-rails:**\n"
    "- **Held-out session:** at least one recording stays hidden from the agent until a "
    "hypothesis is pre-registered; confirmation runs only on held-out data.\n"
    "- **Multiple-comparison control, tracked:** the notebook counts every test in a "
    "campaign, records the family and applies BH-FDR or Bonferroni automatically.\n"
    "- **Power / effect-size sanity:** required N is estimated before an analysis is "
    "proposed, and under-powered designs are flagged.\n"
    "- **Assumptions stated:** every agent-written module writes its statistical assumptions "
    "into its docstring and the notebook.\n"
    "- **Domain guard-rails as tests:** agent-written modules must pass domain-invariant "
    "checks, not just run."
)

ARCH_CAPTION = (
    "**Architecture.** Per-cohort JSON configs give the network-share roots for spikes, "
    "video, tracking and scored events, from which each recording's files are discovered; three data classes and one sync object "
    "form a single aligned session model, shared by every analysis module, the GUI, notebooks "
    "and the AI agent."
)

MANIFEST = (
    "**Capability manifest:** a probe of every cohort, session and animal records what "
    "exists, what loads, sync quality and coverage, so `check_testable()` answers \"can "
    "this analysis run here?\" before any compute is spent. **{n_tests:,} automated "
    "tests** run on mock data."
)

GUI_INTRO = (
    "A **Panel/Bokeh** app, built on the pipeline's own loaders and caches, allows a researcher to "
    "browse every recording block and links neural, behavioral and positional views on one time cursor."
)

BROWSER_CAPTION = (
    "**Session browser.** **1** Every recording block, with counts of cells, video, tracking "
    "and scored events, and the analyses each block is ready for. **2** Whole-day timeline: "
    "ephys per animal, video, scored events and per-animal tracking coverage, with the "
    "selected block shaded. **3** Quality cells per animal. **4** Events in the block window."
)

EXPLORE_CAPTION = (
    "**Explore view** (rat631, 2025-12-16 morning block). **1** Encounters by opponent on the "
    "ephys clock. **2** Rastermap of all {n_q_631} quality cells. **3** Neural trajectory "
    "(here CEBRA on position; PCA, UMAP and CEBRA tabs) with events marked. **4** The rat's "
    "tracked path with events (manual or APT tracking). A shared time cursor links all panels, and zooming the timeline filters the "
    "embedding."
)

DECODE_INTRO = (
    "Each analysis was run, checked and corrected through the agent loop and logged in the "
    "lab notebook. **Opponent decoding:** does a single neuron's activity tell *which* rat "
    "the focal animal is meeting? Partner identity (two most frequent partners per rat) "
    "was decoded from each cell separately."
)

DECODE_RESULT = (
    "**Result.** Two example rat631 cells (left) fire alike before onset and split by partner "
    "only after it. Pooled over {n_cells_all:,} cells, the significant ones (red) sit in the "
    "upper tail. {decode_sig_sentence} Slow state changes could mimic identity coding, so the "
    "name-tag test adds a time-block null."
)

BARCODE_TEXT = (
    "**Opponent name tags.** A cell *passes* an opponent when its evoked response to that "
    "opponent exceeds the others' by d > 0.5; a name-tag cell passes exactly one (example "
    "below, left). If identity is coded by partner-specific cell sets, name-tag cells should be "
    "over-represented relative to shuffled labels. In rat631: {bc631_obs:.0f}% observed vs "
    "{bc631_null:.0f}% expected (p = {bc631_p_free:.3f} free, {bc631_p_block:.3f} time-block "
    "null), and the preferred opponent replicated on held-out encounters ({bc631_rep:.0f}% vs "
    "{bc631_rep_chance:.1f}% chance, p = {bc631_rep_p:.3f}). Only rat630 also passed, under the "
    "free null only."
)

CONNECT_TEXT = (
    "**Negative result: are co-tuned cells coupled?** If name-tag cells for one opponent "
    "excite one another, their cross-correlogram should peak 1–4 ms after the reference spike "
    "(schematic below, left). We tested all {conn_n_pairs:,} ordered pairs of rat631 quality cells "
    "against a hollowed-Gaussian predictor (Stark & Abeles 2012). Pairs tuned to the same "
    "opponent showed slightly more excess coupling than pairs tuned to different ones "
    "(Δz = +{conn_diff:.2f}), but inside the label-shuffle null (95% [{conn_lo:.2f}, "
    "+{conn_hi:.2f}], p = {conn_p:.2f}; p = {conn_p_resid:.2f} after regressing out distance "
    "and rate). Connection rates: {conn_exc_co:.1f}% vs {conn_exc_comp:.1f}% (Fisher p = "
    "{conn_fisher:.2f}). **Not significant**, and reported as such."
)

CONCLUSIONS = (
    "- The **Polis toolkit** permits analysis of **multimodal data** recorded from **multiple "
    "animals simultaneously** across **many days**.\n"
    "- An AI agent runs analyses in the loop; a scientist gates each step, guard-rails live "
    "in the code, and every run is logged.\n"
    "- Some neurons carried **opponent identity** in some rats, and in one rat they "
    "formed **opponent-specific sets**; a coupling link between co-tuned cells was **not** found."
)

REFS = (
    "Kilosort 4: Pachitariu et al., *Nat. Methods* 2024 · Rastermap: Stringer et al., "
    "*Nat. Neurosci.* 2024 · CEBRA: Schneider, Lee & Mathis, *Nature* 2023 · Inter-brain "
    "subspace: Zhang, Phi, Li et al., *Nature* 2025 · BH-FDR: Benjamini & Hochberg 1995 · "
    "CCG coupling: Stark & Abeles, *J. Neurosci. Methods* 2012 · AI agents in a neuroscience "
    "pipeline: Horstmann et al., arXiv 2606.07718.\n"
    "**Code:** github.com/JaneliaSciComp/habitat_pipeline · Funding: Howard Hughes Medical Institute"
)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def _stack(col: int, items: List[Item]) -> List[Item]:
    x, w = COL_X[col], COL_W[col]
    y = TOP
    for it in items:
        it.x, it.y, it.w = x, y, w
        y += it.h + GAP
    bottom = y - GAP
    OVERFLOW[col] = bottom - BOTTOM
    # spread leftover space into the gaps (at most 0.45 in extra each) so a column
    # ends near the bottom margin instead of leaving one empty band
    if OVERFLOW[col] < 0 and len(items) > 1:
        extra = min(0.45, -OVERFLOW[col] / (len(items) - 1))
        for i, it in enumerate(items):
            it.y += i * extra
    return items


def _header(n: int, text: str) -> Item:
    return Item("header", h=1.15, text=f"{n}   {text}", size=HEADER_PT)


def _sub(text: str) -> Item:
    return Item("text", h=0.75, text=f"**{text}**", size=SUBHEAD_PT, color="#123a63")


def layout() -> List[Item]:
    els: List[Item] = []
    # title band (absolute)
    logo_w = 8.6                                   # janelia_logo.png is 3.2 : 1
    tx, tw = MARGIN + logo_w + 0.4, POSTER_W - 2 * (MARGIN + logo_w + 0.4)
    els += [
        Item("rect", x=0, y=0, w=POSTER_W, h=TITLE_H, fill="#ffffff"),
        Item("rect", x=MARGIN, y=TITLE_H - 0.14, w=POSTER_W - 2 * MARGIN, h=0.14, fill=NAVY_BAR),
        Item("text", x=tx, y=0.45, w=tw, h=3.65, text=f"**{TITLE}**", size=TITLE_PT,
             color="#000000", align="center"),
        Item("text", x=tx, y=4.25, w=tw, h=0.9, text=AUTHORS, size=AUTHORS_PT,
             color="#000000", align="center"),
        Item("text", x=tx, y=5.35, w=tw, h=0.75, text=AFFIL, size=AFFIL_PT,
             color="#333333", align="center"),
        Item("figure", name="logo", x=MARGIN, y=(TITLE_H - logo_w / 3.2) / 2 - 0.1,
             w=logo_w, h=logo_w / 3.2),
        Item("figure", name="qr", x=POSTER_W - MARGIN - 5.4, y=0.9, w=5.4, h=5.2),
    ]
    els += _stack(0, [
        Item("header", h=1.15, text="Abstract", size=HEADER_PT),
        Item("box", h=8.45, text=ABSTRACT, size=BODY_PT, fill="#e8eef6"),
        _header(1, "Rats recorded together for weeks"),
        Item("figure", h=8.7, name="arena"),
        Item("text", h=0.55, text="*Schematic illustration.*", size=SMALL_PT,
             color="#5b6475", align="center"),
        Item("text", h=3.4, text=INTRO, size=BODY_PT),
        _header(2, "Multimodal data streams"),
        Item("figure", h=12.55, name="data_formats"),
    ])
    els += _stack(1, [
        _header(3, "Pipeline architecture"),
        Item("figure", h=14.6, name="architecture"),
        Item("text", h=1.4, text=ARCH_CAPTION, size=CAPTION_PT),
        _header(4, "AI agent in the loop"),
        Item("text", h=1.35, text=AI_INTRO, size=CAPTION_PT),
        Item("figure", h=5.8, name="ai_loop"),
        Item("box", h=6.6, text=AI_GUARDS, size=CAPTION_PT, fill="#e8eef6"),
        Item("box", h=2.9, text=MANIFEST, size=CAPTION_PT, fill="#e8eef6"),
    ])
    els += _stack(2, [
        _header(5, "Results from the loop"),
        Item("text", h=1.4, text=DECODE_INTRO, size=CAPTION_PT),
        Item("figure", h=2.3, name="decode_method"),
        Item("figure", h=8.6, name="decode_examples"),
        Item("text", h=2.25, text=BARCODE_TEXT, size=CAPTION_PT),
        Item("figure", h=9.4, name="barcode"),
        Item("text", h=2.7, text=CONNECT_TEXT, size=CAPTION_PT),
        Item("figure", h=9.2, name="connectivity"),
    ])
    els += _stack(3, [
        _header(6, "Interactive exploration"),
        Item("text", h=1.1, text=GUI_INTRO, size=CAPTION_PT),
        Item("figure", h=9.0, name="gui_browser"),
        Item("text", h=1.8, text=BROWSER_CAPTION, size=CAPTION_PT),
        Item("figure", h=11.0, name="gui_explore"),
        Item("text", h=1.8, text=EXPLORE_CAPTION, size=CAPTION_PT),
        _header(7, "Conclusions"),
        Item("box", h=4.3, text=CONCLUSIONS, size=CAPTION_PT, fill="#fdf1e6"),
        Item("text", h=2.1, text=REFS, size=SMALL_PT, color="#5b6475"),
    ])
    return els


def figure_sizes() -> Dict[str, tuple]:
    return {e.name: (e.w, e.h) for e in layout() if e.kind == "figure"}
