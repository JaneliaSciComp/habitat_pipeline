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
TITLE = ("An open-source analysis framework for multi-animal extracellular "
         "electrophysiology in freely interacting rodents")
AUTHORS = "Mikhail Proskurin¹, Yuxin Pan¹, Gowan Tervo¹, Konrad Rokicki¹"
AFFIL = "¹ Janelia Research Campus, HHMI, Ashburn, VA"

ABSTRACT = (
    "**Abstract.** Brains evolved for social life. Recognizing individuals, anticipating their "
    "actions, cooperating, and competing all happen in the company of conspecifics. Yet systems "
    "neuroscience has historically measured neural activity one animal at a time in isolated "
    "chambers, leaving the brains-in-dialog that animals actually evolved to be largely "
    "unexplored. Recent advances now allow simultaneous extracellular recordings from many "
    "neurons across several freely-behaving animals interacting in a shared arena. The "
    "bottleneck has shifted to analysis. Each session yields electrophysiology data, "
    "multi-animal location tracking, and scored behavioral events from distinct acquisition "
    "systems on independent clocks. We introduce Polis toolkit, an open-source Python framework "
    "that ingests spike-sorted data, animal-resolved location tracking, and scored events; "
    "aligns electrophysiology and behavioral clocks; and exposes every animal's data through a "
    "unified shared-clock interface. Modules cover decoding of opponent identity, interaction "
    "outcome, and spatial position; PCA/UMAP/Rastermap population geometry; allocentric "
    "\"social place fields\" testing whether individual neurons encode the position of "
    "specific conspecifics; and a shared/unique subspace decomposition that captures "
    "coordinated dynamics neither brain shows alone. A Python API and interactive GUI provide "
    "scripted and point-and-click access. By unifying the multi-animal data plane and bundling "
    "cross-brain analyses that previously required custom implementation, Polis toolkit "
    "shortens analysis time. Applied to a 12-rat cohort (4 with wireless chronic Neuropixels) "
    "in a large arena, it supports analyses spanning quality control, partner-specific spatial "
    "coding, and inter-brain coupling. We are extending it to include analysis of social "
    "vocalizations, unsupervised social-behavior segmentation linked to neural activity, and "
    "detection of social-experience replay during rest as well as support of NWB data format. "
    "By making cohort-scale analysis routine, Polis toolkit aims to shift the field from "
    "studying isolated brains to studying brains as they evolved — in interaction with each "
    "other."
)

INTRO = (
    "Rats live together continuously in a shared, enriched arena under overhead cameras. "
    "Several colony members carry chronic implants and are recorded **simultaneously** for "
    "hours, while every animal is tracked and human scorers annotate chases, fights and "
    "other social encounters. Brain, video and behavior each arrive on their own clock, in "
    "their own formats, across cohorts, days and recording blocks."
)

AI_INTRO = (
    "An AI coding agent (Claude Code) drives the same CLIs and loaders as the GUI, through "
    "four repo skills and two subagents. It works *within* a stage; a scientist decides "
    "*between* stages (red ♦ gates), where agents fail most often (Horstmann et al. 2026)."
)

AI_GUARDS = (
    "**Guard-rails, built into the code the agent calls:**\n"
    "- **Add-one permutation p-values**, (1+k)/(n+1), and **majority-class baselines**, "
    "not 1/n.\n"
    "- **Resolution check.** 200 shuffles over 149 cells cap the best BH q at "
    "{q_floor_200_149:.2f}, so \"0 significant cells\" was set by the budget. "
    "`fdr_resolution` now flags it.\n"
    "- **Declared-test ledger.** The FDR denominator is fixed before running. *Caught:* a "
    "{fam4_n_declared}-target location sweep reported at m = {fam4_m_logged} gave q = "
    "{fam4_q_logged:.3f}; at the honest m = {fam4_m_declared}, q = {fam4_q_declared:.3f}, "
    "not significant.\n"
    "- **Holdout registry + derived evidence tier.** All {n_hypotheses} hypotheses are "
    "*exploratory* until a reserved session confirms them."
)

ARCH_CAPTION = (
    "**Architecture.** Per-cohort JSON configs, not code, locate the data. Path discovery "
    "feeds three data classes and one sync object (sync-pulse matching; Greyware clock fit "
    "for cohort 9; NWB arrives pre-synchronized). Every module consumes the same aligned "
    "objects, caches results keyed on parameters, and backs the GUI, notebooks and AI agent."
)

MANIFEST = (
    "**Capability manifest:** a probe of every cohort, session and animal records what "
    "exists, what loads, sync quality and coverage, so `check_testable()` answers \"can "
    "this analysis run here?\" before any compute is spent. **{n_tests:,} automated "
    "tests** run on mock data."
)

GUI_INTRO = (
    "A **Panel/Bokeh** app, built on the pipeline's own loaders and caches, browses every "
    "recording block and links neural, behavioral and positional views on one time cursor."
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
    "the focal animal is meeting? Partner identity (two most frequent partners per rat) is "
    "decoded from each cell separately."
)

DECODE_RESULT = (
    "**Result.** {decode_sig_sentence} Identity is carried by individual cells in some rats "
    "but not all. Slow state changes could mimic it, so the barcode test adds a time-block null."
)

BARCODE_TEXT = (
    "**Opponent barcodes.** Using all partners, a cell *passes* an opponent when its evoked "
    "response to that opponent exceeds the others' by d > 0.5. If identity is coded by "
    "partner-specific cell sets, cells that pass exactly one opponent should be "
    "over-represented relative to shuffled labels. In rat631: {bc631_obs:.0f}% observed vs "
    "{bc631_null:.0f}% expected (p = {bc631_p_free:.3f} free, {bc631_p_block:.3f} time-block "
    "null). The preferred opponent replicates on held-out encounters in {bc631_rep:.0f}% of "
    "cells (chance {bc631_rep_chance:.1f}%, p = {bc631_rep_p:.3f}). Only rat630 also passes, "
    "and only under the free null."
)

CONNECT_TEXT = (
    "**Negative result: are co-tuned cells coupled?** Short-lag cross-correlograms for all "
    "{conn_n_pairs:,} ordered pairs of rat631 quality cells, with a hollowed-Gaussian "
    "predictor (Stark & Abeles 2012). Pairs tuned to the same opponent show slightly more "
    "excess coupling than pairs tuned to different ones (Δz = +{conn_diff:.2f}), but inside "
    "the label-shuffle null (95% [{conn_lo:.2f}, +{conn_hi:.2f}], p = {conn_p:.2f}; "
    "p = {conn_p_resid:.2f} after regressing out distance and rate). Connection rates: "
    "{conn_exc_co:.1f}% vs {conn_exc_comp:.1f}% (Fisher p = {conn_fisher:.2f}). **Not "
    "significant**, and reported as such."
)

CONCLUSIONS = (
    "- One config-driven pipeline turns three unsynchronized streams into one aligned "
    "session model, browsable in interactive dashboards.\n"
    "- An AI agent runs analyses in the loop; a scientist gates each step, guard-rails live "
    "in the code, and every run is logged.\n"
    "- Some neurons carry opponent identity in some rats, and in rat631 they form "
    "opponent-specific sets; a coupling link between co-tuned cells was **not** found. All "
    "**exploratory**: each cohort has one scored session, so held-out confirmation is next."
)

REFS = (
    "Kilosort 4: Pachitariu et al., *Nat. Methods* 2024 · Rastermap: Stringer et al., "
    "*Nat. Neurosci.* 2024 · CEBRA: Schneider, Lee & Mathis, *Nature* 2023 · Inter-brain "
    "subspace: Zhang, Phi, Li et al., *Nature* 2025 · BH-FDR: Benjamini & Hochberg 1995 · "
    "CCG coupling: Stark & Abeles, *J. Neurosci. Methods* 2012 · AI agents in a neuroscience "
    "pipeline: Horstmann et al., arXiv 2606.07718.\n"
    "**Code:** github.com/JaneliaSciComp/habitat_pipeline · [Acknowledgments and funding]"
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
        Item("box", h=11.75, text=ABSTRACT, size=CAPTION_PT, fill="#e8eef6"),
        _header(1, "A rat colony, recorded"),
        Item("figure", h=9.5, name="arena"),
        Item("text", h=2.25, text=INTRO, size=CAPTION_PT),
        _header(2, "Data: three streams, three formats"),
        Item("figure", h=12.05, name="data_formats"),
    ])
    els += _stack(1, [
        _header(3, "Pipeline architecture"),
        Item("figure", h=14.6, name="architecture"),
        Item("text", h=2.0, text=ARCH_CAPTION, size=CAPTION_PT),
        _header(4, "AI agent in the loop"),
        Item("text", h=1.35, text=AI_INTRO, size=CAPTION_PT),
        Item("figure", h=5.8, name="ai_loop"),
        Item("box", h=5.4, text=AI_GUARDS, size=CAPTION_PT, fill="#e8eef6"),
        Item("box", h=2.9, text=MANIFEST, size=CAPTION_PT, fill="#e8eef6"),
    ])
    els += _stack(2, [
        _header(5, "Results from the loop"),
        Item("text", h=1.4, text=DECODE_INTRO, size=CAPTION_PT),
        Item("figure", h=2.3, name="decode_method"),
        Item("figure", h=9.6, name="decode_by_rat"),
        Item("text", h=1.4, text=DECODE_RESULT, size=CAPTION_PT),
        Item("figure", h=8.4, name="barcode"),
        Item("text", h=2.25, text=BARCODE_TEXT, size=CAPTION_PT),
        Item("figure", h=7.4, name="connectivity"),
        Item("text", h=2.25, text=CONNECT_TEXT, size=CAPTION_PT),
    ])
    els += _stack(3, [
        _header(6, "Interactive exploration"),
        Item("text", h=1.1, text=GUI_INTRO, size=CAPTION_PT),
        Item("figure", h=10.3, name="gui_browser"),
        Item("text", h=1.8, text=BROWSER_CAPTION, size=CAPTION_PT),
        Item("figure", h=11.5, name="gui_explore"),
        Item("text", h=1.8, text=EXPLORE_CAPTION, size=CAPTION_PT),
        _header(7, "Conclusions"),
        Item("box", h=4.3, text=CONCLUSIONS, size=CAPTION_PT, fill="#fdf1e6"),
        Item("text", h=2.1, text=REFS, size=SMALL_PT, color="#5b6475"),
    ])
    return els


def figure_sizes() -> Dict[str, tuple]:
    return {e.name: (e.w, e.h) for e in layout() if e.kind == "figure"}
