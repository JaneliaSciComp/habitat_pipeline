"""Session browser: table of recording blocks (left) + detail panel (right).

Landing view of ``gui/interactive_app.py``. All data comes from the cached
index in :mod:`gui.session_index`, so selecting a row never touches Kilosort or
DIO. The one slow thing — per-animal tracking coverage, which reads every
tracking CSV of the day — is shown from cache when available and otherwise
computed on request in a worker thread.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import (
    BoxAnnotation, ColumnDataSource, CustomJSTickFormatter, FixedTicker,
    HoverTool, Range1d, Span,
)
from bokeh.palettes import Blues256
from bokeh.plotting import figure

from gui.session_index import (
    MIN_GOOD_CELLS,
    cached_day_tracking_grid,
    day_sync,
    get_day_tracking_grid,
    load_session_index,
    local_midnight_epoch,
)

logger = logging.getLogger(__name__)

# Light → dark, with the near-white end trimmed so 0 % still reads as a bar.
_BLUES = list(Blues256[::-1][40:])
C_EPHYS = "#2b6cb0"
C_EPHYS_OTHER = "#a3bfdc"
C_TRACKED = "#2b6cb0"
C_UNTRACKED = "#cbd5e0"
C_BROKEN = "#e53e3e"
C_EVENT = "#dd6b20"
C_BLOCK = "#f6ad55"

_HHMM = CustomJSTickFormatter(code="""
    const h = Math.floor(tick) % 24;
    const m = Math.round((tick - Math.floor(tick)) * 60);
    return String(h).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0');
""")

#: Columns shown in the table, with their header titles.
TABLE_TITLES = {
    "recording": "Recording",      # hidden: row identity for selection only
    "date": "Date",
    "start": "Start",
    "block": "Block",
    "ephys_animals": "Ephys<br>rats",
    "good_neurons": "Good<br>cells",
    "video": "Video<br>chunks",
    "tracked": "Tracked<br>chunks",
    "annotated": "Annot.<br>chunks",
    "events": "Events",
    "top_events": "Top<br>events",
    "loadable": "Load-<br>able",
    "broken": "Broken<br>chunks",
    "ephys_h": "Ephys<br>hours",
    "ready": "Ready<br>for",
    "animals": "Animals",
}

#: Narrow fixed widths (px); titles wrap onto two lines via TABLE_CSS.
TABLE_WIDTHS = {
    "date": 92, "start": 58, "block": 58, "ephys_animals": 58, "good_neurons": 58,
    "video": 62, "tracked": 66, "annotated": 62, "events": 62, "top_events": 150,
    "loadable": 58, "broken": 62, "ephys_h": 58, "ready": 150, "animals": 170,
}

TABLE_CSS = """
.tabulator .tabulator-header .tabulator-col .tabulator-col-content {
    padding: 4px 3px; position: relative;
}
.tabulator .tabulator-header .tabulator-col .tabulator-col-title {
    white-space: normal !important; text-overflow: clip !important; overflow: visible;
    font-size: 11px; line-height: 1.15; padding-right: 0 !important;
}
/* Sort arrow pinned to the corner so it doesn't eat the narrow column's width. */
.tabulator .tabulator-header .tabulator-col .tabulator-col-content .tabulator-col-sorter {
    position: absolute; right: 1px; top: 3px;
}
.tabulator .tabulator-header .tabulator-col .tabulator-col-sorter .tabulator-arrow {
    transform: scale(0.7);
}
.tabulator-row .tabulator-cell { padding: 4px 3px; }
"""


def _hours(epoch, midnight: float):
    return (np.asarray(epoch, dtype=np.float64) - midnight) / 3600.0


def _fmt_clock(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S")


def _flag_low(column: pd.Series, limit: float):
    return ["color: #e53e3e; font-weight: bold" if pd.notna(v) and v < limit else ""
            for v in column]


def _status_reason(animal: Mapping[str, Any]) -> str:
    err = animal.get("load_error") or ""
    if "timestamps.dat" in err:
        return "no .timestamps.dat"
    if err:
        return err.split(":")[0]
    return animal.get("status", "")


class SessionBrowser:
    """Two-panel session browser.

    ``on_explore(cohort_label, recording_id, animal_id)`` is called by the
    Explore button; the host app owns everything after that.
    """

    def __init__(self, cohort_labels: Mapping[str, str],
                 on_explore: Callable[[str, str, str], None]):
        self._cohort_labels = dict(cohort_labels)      # label -> 'cohort7'
        self._on_explore = on_explore
        self._index: Optional[Dict[str, Any]] = None
        self._selected: Optional[str] = None

        self.cohort_sel = pn.widgets.Select(name="Cohort", options=list(self._cohort_labels))
        self.rescan_btn = pn.widgets.Button(name="Rescan share", button_type="default",
                                            width=220)
        self.status = pn.pane.Markdown("", styles={"font-size": "12px", "color": "#888"})

        self.table = pn.widgets.Tabulator(
            pd.DataFrame(columns=list(TABLE_TITLES)),
            titles=TABLE_TITLES,
            show_index=False,
            disabled=True,
            selectable=1,
            pagination=None,
            layout="fit_data_table",
            hidden_columns=["recording"],
            frozen_columns=["date"],
            widths=TABLE_WIDTHS,
            stylesheets=[TABLE_CSS],
            header_filters={
                "date": {"type": "input", "func": "like", "placeholder": "filter"},
                "ready": {"type": "input", "func": "like", "placeholder": "filter"},
                "animals": {"type": "input", "func": "like", "placeholder": "rat…"},
            },
            sizing_mode="stretch_both",
            min_height=500,
        )
        self.detail = pn.Column(
            pn.pane.Alert("Select a session on the left.", alert_type="info"),
            sizing_mode="stretch_width",
        )
        self.animal_sel = pn.widgets.Select(name="Animal", options=[], width=200)
        self.explore_btn = pn.widgets.Button(name="Explore session ▶", button_type="primary",
                                             width=200, disabled=True)
        self.track_btn = pn.widgets.Button(name="Compute per-animal tracking coverage",
                                           button_type="light", width=300)

        self.cohort_sel.param.watch(lambda *_: self.reload(), "value")
        self.rescan_btn.on_click(self._on_rescan)
        self.table.param.watch(self._on_select, "selection")
        self.explore_btn.on_click(self._on_explore_click)
        # The explore view's controls live in the template sidebar, which the
        # browser keeps collapsed; open it client-side (FastListTemplate's openNav).
        self.explore_btn.js_on_click(code="if (window.openNav) { openNav(); }")
        self.track_btn.on_click(self._on_compute_tracking)

        self.view = pn.Row(
            pn.Column(self.table, sizing_mode="stretch_both", min_width=480),
            pn.Column(self.detail, width=700),
            sizing_mode="stretch_both",
        )
        self.sidebar = pn.Column(
            "## Session browser",
            self.cohort_sel,
            self.rescan_btn,
            self.status,
            pn.pane.Markdown(
                "**Video** = 30-min raw chunks in the block · **Tracked** = chunk has "
                "APT or manual tracking · **Annotated** = ≥1 scored event inside the "
                "chunk · **Good cells** = manifest quality cells (rate ≥ 0.5 Hz, "
                "presence ≥ 0.8, CV-ISI ≤ 5). ★ = primary block of the .rec day.",
                styles={"font-size": "12px"},
            ),
            width=280,
        )

    # ── Index loading ──────────────────────────────────────────────────────────

    @property
    def cohort(self) -> str:
        return self._cohort_labels[self.cohort_sel.value]

    def reload(self, refresh: bool = False, select: Optional[str] = None):
        cohort = self.cohort
        cache_key = f"session_index__{cohort}"
        index = None if refresh else pn.state.cache.get(cache_key)
        self.table.loading = True
        try:
            if index is None:
                index = load_session_index(cohort, refresh=refresh)
                pn.state.cache[cache_key] = index
        except Exception as exc:
            logger.exception("session index failed")
            self.status.object = f"⚠ index failed: {exc}"
            self.table.loading = False
            return
        self._index = index
        df = index["table"]
        self.table.value = df[[c for c in TABLE_TITLES if c in df.columns]] \
            if not df.empty else pd.DataFrame(columns=list(TABLE_TITLES))
        self.table.style.apply(_flag_low, subset=["loadable"], limit=1)
        self.table.loading = False

        built = index.get("meta", {}).get("built_at")
        built_s = datetime.fromtimestamp(built).strftime("%Y-%m-%d %H:%M") if built else "?"
        manifest = index.get("manifest_generated_at") or "missing"
        self.status.object = (f"{len(df)} recordings · index built {built_s}<br>"
                              f"capability manifest: {manifest[:10]}")

        target = select or self._selected
        if not df.empty:
            rows = df.index[df["recording"] == target].tolist() if target else []
            self.table.selection = rows[:1] or [0]

    async def _on_rescan(self, event):
        self.rescan_btn.disabled = True
        self.status.object = "rescanning the share…"
        try:
            cohort = self.cohort
            index = await asyncio.to_thread(load_session_index, cohort, None, True)
            pn.state.cache[f"session_index__{cohort}"] = index
        except Exception as exc:
            # The previous index stays on disk and on screen; say why it wasn't replaced.
            logger.exception("rescan failed")
            self.status.object = (f"⚠ rescan failed, showing the previous index:<br>"
                                  f"{type(exc).__name__}: {exc}")
            return
        finally:
            self.rescan_btn.disabled = False
        self.reload()

    # ── Selection → detail panel ──────────────────────────────────────────────

    def _on_select(self, event):
        if not event.new or self._index is None:
            return
        # selected_dataframe resolves the index through any header filter/sort.
        picked = self.table.selected_dataframe
        if picked.empty:
            return
        self._selected = picked.iloc[0]["recording"]
        pn.state.cache["session_browser_state"] = {
            "cohort": self.cohort_sel.value, "recording": self._selected}
        self._render_detail()

    def _render_detail(self):
        rid = self._selected
        det = self._index["details"].get(rid)
        if det is None:
            self.detail[:] = [pn.pane.Alert(f"No details for {rid}.", alert_type="warning")]
            return
        day = self._index["days"][det["date"]]
        row = self._index["table"].set_index("recording").loc[rid]

        animals = det["animals"]
        loadable = [a for a in animals if a["loadable"]]
        loadable.sort(key=lambda a: -(a["n_quality_cells"] or -1))
        self.animal_sel.options = [a["animal"] for a in loadable]
        if loadable:
            self.animal_sel.value = loadable[0]["animal"]
        self.explore_btn.disabled = not loadable
        if not loadable:
            reasons = sorted({_status_reason(a) for a in animals}) or ["no animals"]
            self.explore_btn.name = f"Not loadable ({', '.join(reasons)})"
        else:
            self.explore_btn.name = "Explore session ▶"

        a, b = det["block_window"]
        span = "" if np.isnan(a) else (
            f"{_fmt_clock(a)} – {_fmt_clock(b)} ({(b - a) / 3600:.2f} h"
            + (", from directory stamp — no sync" if det["block_window_source"] != "sync"
               else "") + ")")
        header = pn.pane.Markdown(
            f"### {rid}\n"
            f"{row['date']} · block {det['block_index'] + 1} of {det['n_blocks']}"
            f"{' (primary)' if det['is_primary'] else ''} · {span}<br>"
            f"**{row['video']}** video chunks · **{row['tracked']}** tracked · "
            f"**{row['annotated']}** annotated · **{row['events']}** events in window"
            + (f" · ⚠ {row['broken']} broken chunk(s)" if row["broken"] else ""),
            sizing_mode="stretch_width",
        )

        grid = cached_day_tracking_grid(self.cohort, day)
        timeline = pn.pane.Bokeh(self._timeline_fig(det, day, grid),
                                 sizing_mode="stretch_width")
        track_note = [] if grid is not None or not day.get("tracking_files") else [
            pn.Column(self.track_btn, pn.pane.Markdown(
                f"_{len(day['tracking_files'])} tracking file(s) for this date — per-animal "
                "coverage not cached yet (reads every CSV; ~1 min)._",
                styles={"font-size": "12px"}))]

        neurons = pn.pane.Bokeh(self._neurons_fig(animals), sizing_mode="stretch_width")
        ev_types = det.get("event_types") or {}
        events_md = pn.pane.Markdown(
            "**Events in block window**\n\n" + (
                "\n".join(f"- `{k}` {v}" for k, v in
                          sorted(ev_types.items(), key=lambda kv: -kv[1])[:10])
                if ev_types else "_none scored_"),
            width=200,
        )
        chunks = pd.DataFrame([c for c in day["chunks"] if c["block"] == rid])
        chunk_card = pn.Card(
            pn.widgets.Tabulator(
                chunks.assign(
                    start=chunks["t0"].map(_fmt_clock),
                    minutes=((chunks["t1"] - chunks["t0"]) / 60).round(1),
                )[["key", "start", "minutes", "tracked_by", "n_events", "reason"]]
                if not chunks.empty else pd.DataFrame(),
                show_index=False, disabled=True, layout="fit_data_table",
                sizing_mode="stretch_width", max_height=260,
            ),
            title=f"Video chunks in this block ({len(chunks)})",
            collapsed=True, sizing_mode="stretch_width",
        )

        self.detail[:] = [
            header,
            timeline,
            *track_note,
            pn.Row(neurons, events_md, sizing_mode="stretch_width"),
            pn.Row(self.animal_sel, self.explore_btn, align="end"),
            chunk_card,
        ]

    # ── Figures ────────────────────────────────────────────────────────────────

    def _timeline_fig(self, det, day, grid):
        """Whole-day timeline on local clock hours; the selected block is shaded."""
        midnight = local_midnight_epoch(det["date"])
        index = self._index
        quads = {k: [] for k in ("left", "right", "bottom", "top", "color", "alpha",
                                 "label", "info")}

        def add(left, right, y, color, label, info, alpha=1.0, h=0.38):
            quads["left"].append(left)
            quads["right"].append(right)
            quads["bottom"].append(y - h)
            quads["top"].append(y + h)
            quads["color"].append(color)
            quads["alpha"].append(alpha)
            quads["label"].append(label)
            quads["info"].append(info)

        # Ephys: one row per animal recorded on this day, one bar per block.
        ephys_rows: Dict[str, int] = {}
        for rid in day["recordings"]:
            for an in (index["details"].get(rid) or {}).get("animals", []):
                if an["animal"] not in ephys_rows:
                    ephys_rows[an["animal"]] = len(ephys_rows)
        n_e = len(ephys_rows)
        for rid in day["recordings"]:
            d = index["details"].get(rid) or {}
            s = d.get("sync")
            for an in d.get("animals", []):
                w = an["ephys_window"]
                if not w or not s:
                    continue
                t = np.array(w) * s["slope"] + s["intercept"]
                y = n_e - 1 - ephys_rows[an["animal"]]
                mine = rid == det["recording"]
                add(*_hours(t, midnight), y, C_EPHYS if mine else C_EPHYS_OTHER,
                    f"{an['animal']} ephys",
                    f"{rid}: {an['n_quality_cells']} good / {an['n_clusters']} clusters, "
                    f"{_fmt_clock(t[0])}–{_fmt_clock(t[1])}")

        # Rows top to bottom: ephys (n_e-1 … 0), video (-1), events (-2),
        # then per-animal tracking from the cached grid (-3 …).
        track_rows = []
        if grid is not None:
            edges_h = _hours(grid["edges_epoch"], midnight)
            track_rows = list(zip(grid["animals"], grid["grid"]))
        y_video, y_events = -1, -2
        for c in day["chunks"]:
            if c["broken"]:
                color, state = C_BROKEN, f"broken: {c['reason']}"
            elif c["tracked"]:
                color, state = C_TRACKED, f"tracked ({c['tracked_by']})"
            else:
                color, state = C_UNTRACKED, "not tracked"
            mine = c["block"] == det["recording"]
            add(*_hours([c["t0"], c["t1"]], midnight), y_video, color,
                "video chunk", f"{c['key']}: {state}, {c['n_events']} events"
                + ("" if c["block"] else " (no ephys block)"),
                alpha=1.0 if mine else 0.45, h=0.3)
        for i, (an, values) in enumerate(track_rows):
            y = -3 - i
            for j, v in enumerate(values):
                if np.isnan(v):
                    continue
                add(edges_h[j], edges_h[j + 1], y,
                    _BLUES[int(round(v * (len(_BLUES) - 1)))],
                    f"{an} tracked", f"{_fmt_clock(grid['edges_epoch'][j])}: {v:.0%} of frames",
                    h=0.45)

        y_min = -3 - len(track_rows) + 0.5 if track_rows else -2.5
        labels = {n_e - 1 - v: f"{k} · ephys" for k, v in ephys_rows.items()}
        labels[y_video] = "video chunks"
        labels[y_events] = "scored events"
        for i, (an, values) in enumerate(track_rows):
            pct = np.nanmean(values) if np.isfinite(values).any() else np.nan
            labels[-3 - i] = f"{an} · tracked {'' if np.isnan(pct) else f'{pct:.0%}'}"

        # x extent: everything drawn, padded.
        xs = quads["left"] + quads["right"]
        ev_h = _hours(day["events"]["t_epoch"].to_numpy(), midnight) \
            if len(day["events"]) else np.array([])
        if len(ev_h):
            xs += [float(ev_h.min()), float(ev_h.max())]
        lo, hi = (min(xs), max(xs)) if xs else (0, 24)
        pad = max(0.25, 0.03 * (hi - lo))

        n_rows = len(labels)
        p = figure(
            height=max(170, 26 * n_rows + 70), sizing_mode="stretch_width",
            x_range=Range1d(lo - pad, hi + pad, bounds=(lo - 2 * pad, hi + 2 * pad)),
            y_range=Range1d(y_min - 0.2, n_e - 0.3),
            tools="xpan,xwheel_zoom,reset", active_scroll="xwheel_zoom",
            toolbar_location="right",
            title=f"{det['date']} — whole day, selected block shaded",
        )
        p.xaxis.formatter = _HHMM
        p.xaxis.axis_label = "local time"
        p.yaxis.ticker = FixedTicker(ticks=sorted(labels))
        p.yaxis.major_label_overrides = {k: v for k, v in labels.items()}
        p.ygrid.grid_line_color = None

        a, b = det["block_window"]
        if not np.isnan(a):
            p.add_layout(BoxAnnotation(left=_hours(a, midnight).item(),
                                       right=_hours(b, midnight).item(),
                                       fill_color=C_BLOCK, fill_alpha=0.12, level="underlay"))
        src = ColumnDataSource(quads)
        r = p.quad(left="left", right="right", bottom="bottom", top="top", source=src,
                   fill_color="color", fill_alpha="alpha", line_color=None)
        p.add_tools(HoverTool(renderers=[r], tooltips=[("", "@label"), ("", "@info")]))

        if len(ev_h):
            ev_src = ColumnDataSource({
                "x": ev_h, "type": day["events"]["type"].astype(str).to_numpy(),
                "clock": [_fmt_clock(t) for t in day["events"]["t_epoch"].to_numpy()],
            })
            er = p.segment(x0="x", x1="x", y0=y_events - 0.35, y1=y_events + 0.35,
                           source=ev_src, line_color=C_EVENT, line_alpha=0.35, line_width=1)
            p.add_tools(HoverTool(renderers=[er], tooltips=[("event", "@type @clock")]))
        return p

    def _neurons_fig(self, animals):
        names = [a["animal"] for a in animals][::-1]
        clusters = [a["n_clusters"] or 0 for a in animals][::-1]
        good = [a["n_quality_cells"] or 0 for a in animals][::-1]
        notes = []
        for a in animals[::-1]:
            if a["status"] == "ok":
                note = f"{a['n_quality_cells']} / {a['n_clusters']}"
            else:
                note = _status_reason(a)
            notes.append(note)
        color = ["#2b6cb0" if g >= MIN_GOOD_CELLS else C_BROKEN for g in good]
        src = ColumnDataSource(dict(animal=names, clusters=clusters, good=good,
                                    note=notes, color=color,
                                    x_note=[max(c, g) for c, g in zip(clusters, good)]))
        top = max(clusters + good + [MIN_GOOD_CELLS]) * 1.35
        p = figure(y_range=names, height=60 + 30 * len(names), sizing_mode="stretch_width",
                   x_range=(0, top), tools="", toolbar_location=None,
                   title="Good cells per animal (quality cells / Kilosort-good clusters)")
        p.hbar(y="animal", right="clusters", height=0.7, source=src,
               fill_color="#cbd5e0", line_color=None)
        p.hbar(y="animal", right="good", height=0.7, source=src,
               fill_color="color", line_color=None)
        p.text(x="x_note", y="animal", text="note", source=src, x_offset=6,
               text_font_size="11px", text_baseline="middle")
        p.add_layout(Span(location=MIN_GOOD_CELLS, dimension="height",
                          line_color=C_BROKEN, line_dash="dashed", line_alpha=0.6))
        p.ygrid.grid_line_color = None
        return p

    # ── Actions ────────────────────────────────────────────────────────────────

    async def _on_compute_tracking(self, event):
        det = self._index["details"].get(self._selected)
        if det is None:
            return
        day = self._index["days"][det["date"]]
        sync_obj = day_sync(self._index, det["date"])
        if sync_obj is None:
            self.track_btn.name = "No sync for this day — cannot place tracking"
            return
        required = sorted({a["animal"] for rid in day["recordings"]
                           for a in (self._index["details"].get(rid) or {}).get("animals", [])})
        self.track_btn.disabled = True
        self.track_btn.name = "Reading tracking files…"
        try:
            await asyncio.to_thread(get_day_tracking_grid, self.cohort, day, sync_obj, required)
        except Exception as exc:
            logger.exception("tracking grid failed")
            self.track_btn.name = f"Failed: {exc}"[:80]
            return
        finally:
            self.track_btn.disabled = False
        self.track_btn.name = "Compute per-animal tracking coverage"
        self._render_detail()

    async def _on_explore_click(self, event):
        if self._selected and self.animal_sel.value:
            result = self._on_explore(self.cohort_sel.value, self._selected,
                                      self.animal_sel.value)
            if inspect.isawaitable(result):
                await result
