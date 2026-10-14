# Run from project root: panel serve gui/interactive_app.py --show
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import panel as pn
import param
from bokeh import events as bk_events
from bokeh.layouts import column as bk_col
from bokeh.models import (
    ColumnDataSource, CustomJS, FixedTicker, HoverTool, LinearColorMapper,
    Range1d, Span, WheelZoomTool,
)
from bokeh.palettes import Category10, Inferno256
from bokeh.plotting import figure

from ephys.decode_opponent_identity import align_spikes_to_events, extract_firing_rate_features
from ingestion.data_paths import (
    DataStorageManager, get_animals_and_sessions, get_tracking_files_by_date,
)
from ingestion.ephys_sync import DataSyncManager
from gui import explore_cache as xc
from gui.session_browser import SessionBrowser
from ingestion.kilosort_data_import import load_kilosort_data
from video.behavioral_events import BehavioralEventsData, load_behavioral_events

pn.extension("plotly", "tabulator")

PALETTE = Category10[10]
logger = logging.getLogger(__name__)

#: The shared time cursor: one moment, shown on every panel. Plotly traces with
#: this name are moved in place (a small restyle message, not a redraw).
CURSOR_NAME = "cursor"
CURSOR_COLOR = "#e53e3e"
#: How often the server looks at the pending cursor time (ms).
CURSOR_PERIOD_MS = 100
#: Plotly markers jump to the cursor once it has been still this long (s) ...
CURSOR_SETTLE_S = 0.12
#: ... and, while it is still moving, follow at most this often (s). A 3D
#: restyle redraws the whole scene, so following every mouse move would queue
#: redraws in the browser and the marker would crawl behind the mouse.
CURSOR_FOLLOW_S = 0.25
#: Browser-side throttle on the vertical lines (ms); each move is one sync message.
CURSOR_JS_THROTTLE_MS = 33


def _cursor_trace_3d(point):
    import plotly.graph_objects as go
    x, y, z = ([point[0]], [point[1]], [point[2]]) if point is not None else ([], [], [])
    return go.Scatter3d(x=x, y=y, z=z, mode="markers", name=CURSOR_NAME, showlegend=False,
                        hoverinfo="skip",
                        marker=dict(size=14, color=CURSOR_COLOR, symbol="diamond",
                                    line=dict(width=3, color="white")))


def _cursor_trace_2d(point):
    import plotly.graph_objects as go
    x, y = ([point[0]], [point[1]]) if point is not None else ([], [])
    # WebGL, like the trajectory: an SVG marker would sit under the gl canvas.
    return go.Scattergl(x=x, y=y, mode="markers", name=CURSOR_NAME, showlegend=False,
                        hoverinfo="skip",
                        # An open symbol is stroked with marker.color, not marker.line.color.
                        marker=dict(size=22, color=CURSOR_COLOR, symbol="circle-open",
                                    line=dict(width=4)))


#: Most rastermap columns ever sent to the browser. The full image (one column
#: per raster bin, ~19k for a 5 h block) is ~10x what a screen can show and was
#: ~21 MB per draw; the visible window is block-averaged down to this and
#: re-sliced at full resolution as you zoom in.
MAX_RASTER_COLS = 3000

# ── Per-process data cache (survives theme-toggle page reloads) ────────────────

def _cache_key(cohort: str, session_id: str, animal_id: str) -> str:
    return f"habitat_data__{cohort}__{session_id}__{animal_id}"
CONFIG_OPTIONS = {
    "Cohort 7 (default)": None,
    "Cohort 5": "cohort5_paths.json",
}
#: Same labels -> cohort names as the capability manifest / session index spell them.
COHORT_NAMES = {
    "Cohort 7 (default)": "cohort7",
    "Cohort 5": "cohort5",
}


@pn.cache(max_items=4)
def _recordings(cfg):
    """``get_animals_and_sessions`` walks the share; once per process is enough."""
    return get_animals_and_sessions(config_path=cfg)
# Two parallel lists: labels displayed in the widget, abbreviations used in code
BTYPE_LABELS = [f"{k} — {v}" for k, v in BehavioralEventsData.BEHAVIOR_TYPES.items()]
BTYPE_ABBREVS = list(BehavioralEventsData.BEHAVIOR_TYPES.keys())


def _label_to_abbrev(label: str) -> str:
    """Convert a behavior display label to its abbreviation."""
    try:
        return BTYPE_ABBREVS[BTYPE_LABELS.index(label)]
    except ValueError:
        return BTYPE_ABBREVS[0]


# ── Module-level helpers ───────────────────────────────────────────────────────

def _focal_events(events_df, behavior_type, animal_id):
    """Events of one type involving ``animal_id``, with an ``opponent`` column.

    Not filtered by ``min_events``: that threshold decides which opponents the
    PCA marks, not which events exist, so the timeline shows all of them.
    """
    df = events_df[(events_df["type"] == behavior_type) &
                   ((events_df["initiator"] == animal_id) |
                    (events_df["victim"] == animal_id))].copy()
    df["opponent"] = np.where(df["initiator"] == animal_id, df["victim"], df["initiator"])
    return df


def _rat_colors(df):
    """One colour per rat in ``df`` (sorted), shared by the timeline and the PCA."""
    all_rats = sorted(set(df["initiator"].dropna().tolist() + df["victim"].dropna().tolist()))
    return {r: PALETTE[i % 10] for i, r in enumerate(all_rats)}


def _build_pop_data(events, pca_base, behavior_type, animal_id, min_events,
                    label="PCA", subtitle=""):
    """Lay one behaviour type's event markers over a precomputed 3-D embedding.

    ``pca_base`` is :func:`gui.explore_cache.fit_pca_trajectory`'s output (or
    :func:`~gui.explore_cache.fit_cebra_embedding`'s, with ``label="CEBRA"``);
    the fit itself is behaviour-independent, so changing the behaviour type or
    ``min_events`` never refits it.

    Returns dict with keys:
      scores        : ndarray (n_bins, 3)  — full trajectory in PC space
      bin_centers   : ndarray (n_bins,)   — time of each bin center (s)
      var_explained : ndarray (3,)
      ev_starts     : ndarray of event ts_start_ephys
      ev_opponents  : ndarray of opponent labels (str)
      opp_colors    : dict {opponent: hex_color}  — same mapping as timeline
      btype_map     : dict {abbrev: full_name}
    or None when there is no trajectory (< 3 quality cells).
    """
    if pca_base is None:
        return None

    # Gather events of the selected behavior type
    try:
        ev_starts, _, ev_labels = events.extract_opponent_labels(
            animal_of_interest=animal_id,
            behavior_type=behavior_type,
            min_events_per_class=min_events,
        )
    except Exception:
        ev_starts = ev_labels = np.array([])

    # Opponent colors: the same mapping the timeline uses
    df = _focal_events(events.events_data, behavior_type, animal_id)
    opp_colors = _rat_colors(df.dropna(subset=["ts_start_ephys", "initiator", "victim"]))

    return {
        "scores": pca_base["scores"],
        "bin_centers": pca_base["bin_centers"],
        # None for embeddings without a variance notion (CEBRA).
        "var_explained": pca_base.get("var_explained"),
        "label": label,
        "subtitle": subtitle,
        "ev_starts": ev_starts,
        "ev_opponents": ev_labels,
        "btype_map": BehavioralEventsData.BEHAVIOR_TYPES,
        "opp_colors": opp_colors,
        "behavior_type": behavior_type,
    }


def _load_full(cfg, session_id, animal_id, raster_bin, pca_bin, cached_signature):
    """Everything the explore view needs, from the share. Blocking; runs in a thread.

    Rastermap and PCA are recomputed only when the sources differ from the
    cached view's (``cached_signature``); otherwise ``products`` is ``None``
    and the cached image/trajectory stay valid.
    """
    dsm = DataStorageManager(animal_id, session_id, config_path=cfg, auto_load=True)
    ks_path = dsm.get_kilosort_path()
    event_files = dsm.get_behavioral_event_files()
    signature = xc.source_signature(ks_path, event_files)

    ks_data = load_kilosort_data(ks_path)
    events = load_behavioral_events(event_files, session_id=dsm.session_id)
    sync = DataSyncManager(dsm, dio_channel=1)
    events.synchronize_with_ephys(sync, create_new_columns=True)

    if signature != cached_signature:
        products = xc.compute_view_products(ks_data, raster_bin, pca_bin)
    else:
        products = None
        ks_data.filter_cells_by_firing_patterns()   # what later re-binning expects
    # Resolved fresh: the DSM's on-disk path cache has no staleness check and
    # can list fewer APT chunks than the share has (CLAUDE.md).
    try:
        tracking_files = get_tracking_files_by_date(session_id, config_path=cfg)
    except Exception as exc:
        logger.info("no tracking files for %s: %s", session_id, exc)
        tracking_files = []
    return {"ks_data": ks_data, "events": events, "signature": signature,
            "products": products, "sync": sync, "tracking_files": tracking_files}


def _make_pca_plotly(pop_data_full, t_view_start, t_view_end, cursor=None):
    """Full continuous PCA trajectory with event markers filtered to the current view.

    Background: thin line colored by time (viridis) clipped to [t_view_start, t_view_end].
    Foreground: one marker-trace per opponent, showing only events in the view window.
    """
    import plotly.graph_objects as go

    scores = pop_data_full["scores"]
    bin_centers = pop_data_full["bin_centers"]
    var = pop_data_full["var_explained"]
    label = pop_data_full.get("label", "PCA")
    subtitle = pop_data_full.get("subtitle", "")
    axis = "PC" if label == "PCA" else f"{label} "
    if var is not None:
        axis_titles = [f"{axis}{i + 1} ({v * 100:.1f}%)" for i, v in enumerate(var)]
    else:
        axis_titles = [f"{axis}{i + 1}" for i in range(3)]
    ev_starts = pop_data_full["ev_starts"]
    ev_opponents = pop_data_full["ev_opponents"]
    opp_colors = pop_data_full["opp_colors"]
    btype_map = pop_data_full["btype_map"]
    behavior_type = pop_data_full["behavior_type"]

    fig = go.Figure()

    # ── Background trajectory clipped to view ──────────────────────────────────
    view_mask = (bin_centers >= t_view_start) & (bin_centers <= t_view_end)
    view_scores = scores[view_mask]
    view_times = bin_centers[view_mask]

    if len(view_scores) >= 2:
        fig.add_trace(go.Scatter3d(
            x=view_scores[:, 0], y=view_scores[:, 1], z=view_scores[:, 2],
            mode="lines",
            line=dict(
                color=view_times, colorscale="Viridis", width=3,
                # Same window-pinned scale as the trajectory panel.
                cmin=t_view_start, cmax=t_view_end,
                showscale=True,
                colorbar=dict(title="Time (s)", x=1.05, len=0.6),
            ),
            opacity=0.15,
            name="Trajectory",
            hovertemplate="t=%{customdata:.1f}s<extra></extra>",
            customdata=view_times,
        ))

    # ── Event markers for events in the view window ────────────────────────────
    n_events_shown = 0
    if len(ev_starts) > 0:
        ev_mask = (ev_starts >= t_view_start) & (ev_starts <= t_view_end)
        ev_bin_idx = np.searchsorted(bin_centers, ev_starts[ev_mask]).clip(0, len(bin_centers) - 1)
        ev_opps_view = ev_opponents[ev_mask]
        ev_ts_view = ev_starts[ev_mask]

        for opp in np.unique(ev_opps_view):
            opp_mask = ev_opps_view == opp
            idx = ev_bin_idx[opp_mask]
            color = opp_colors.get(opp, "grey")
            hover = [
                f"t={ev_ts_view[opp_mask][j]:.1f}s<br>"
                f"{btype_map.get(behavior_type, behavior_type)} vs {opp}"
                for j in range(opp_mask.sum())
            ]
            fig.add_trace(go.Scatter3d(
                x=scores[idx, 0], y=scores[idx, 1], z=scores[idx, 2],
                mode="markers",
                marker=dict(size=6, color=color, line=dict(width=0.5, color="black")),
                name=opp,
                hovertext=hover,
                hoverinfo="text",
                customdata=ev_ts_view[opp_mask],   # hover → shared cursor time
            ))
            n_events_shown += opp_mask.sum()

    fig.add_trace(_cursor_trace_3d(cursor))

    if len(view_scores) < 2 and n_events_shown == 0:
        fig.add_annotation(
            text="No data in current view.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13),
        )

    plotly_template = "plotly_dark" if pn.config.theme == "dark" else "plotly"
    fig.update_layout(
        template=plotly_template,
        scene=dict(
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1],
            zaxis_title=axis_titles[2],
        ),
        title=(
            f"{label} trajectory — {n_events_shown} events in view"
            f"  [{t_view_start:.0f}–{t_view_end:.0f} s]"
            + (f"<br><sup>{subtitle}</sup>" if subtitle else "")
        ),
        height=460,
        legend=dict(title="Opponent", x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


#: Most trajectory samples drawn at once (tracking runs at ~10 Hz: a 5 h block
#: is ~180k points); the visible window is strided down to this.
MAX_TRAJ_POINTS = 20000
#: An event is placed on the trajectory only if a tracked frame lies this close.
EVENT_SNAP_S = 1.0


def _make_trajectory_plotly(traj, ev_df, opp_colors, min_events, t_view_start, t_view_end,
                            animal_id, behavior_type, btype_map, status=None,
                            other_sources=False, cursor=None):
    """The focal animal's (x, y) path in the view window, coloured by time.

    Mirrors :func:`video.behavioral_visualization.plot_events_on_trajectory`:
    events sit at the animal's position at event start, coloured by opponent —
    the same colours as the timeline and PCA. As on the timeline, opponents
    below ``min_events`` are faded rather than hidden.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    plotly_template = "plotly_dark" if pn.config.theme == "dark" else "plotly"
    layout = dict(template=plotly_template, height=460,
                  margin=dict(l=0, r=0, t=60, b=0),
                  legend=dict(title="Opponent", x=0.01, y=0.99))

    if traj is None:
        fig.add_annotation(text=status or f"No tracking for {animal_id} in this block.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=13))
        fig.update_layout(title="Trajectory", xaxis_visible=False, yaxis_visible=False,
                          **layout)
        return fig

    t, x, y = traj["t"], traj["x"], traj["y"]
    in_view = (t >= t_view_start) & (t <= t_view_end)
    tv, xv, yv = t[in_view], x[in_view], y[in_view]
    fmt = {"apt_tqt": "APT", "mask_metrics": "manual"}.get(traj["format"], traj["format"])
    if len(tv) == 0:
        fig.add_annotation(
            text=(f"{animal_id} is not tracked by {fmt} in "
                  f"[{t_view_start:.0f}–{t_view_end:.0f} s].<br>"
                  f"This source covers {t[0]:.0f}–{t[-1]:.0f} s"
                  + (" — try the other tracking source above." if other_sources else ".")),
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=13))
        fig.update_layout(title=f"{animal_id} trajectory ({fmt})",
                          xaxis_visible=False, yaxis_visible=False, **layout)
        return fig
    stride = max(1, int(np.ceil(len(tv) / MAX_TRAJ_POINTS)))
    tv, xv, yv = tv[::stride], xv[::stride], yv[::stride]

    if len(tv) >= 2:
        # Faint path, broken where tracking has a gap so jumps aren't drawn.
        gap = np.r_[False, np.diff(tv) > 2.0 * stride]
        lx = np.where(gap, np.nan, xv)
        ly = np.where(gap, np.nan, yv)
        fig.add_trace(go.Scattergl(
            x=lx, y=ly, mode="lines", line=dict(color="rgba(128,128,128,0.25)", width=1),
            hoverinfo="skip", showlegend=False, connectgaps=False,
        ))
        fig.add_trace(go.Scattergl(
            x=xv, y=yv, mode="markers",
            marker=dict(size=3, color=tv, colorscale="Viridis", opacity=0.5,
                        cmin=t_view_start, cmax=t_view_end, showscale=True,
                        colorbar=dict(title="Time (s)", x=1.02, len=0.6)),
            name="Trajectory", customdata=tv,
            hovertemplate="t=%{customdata:.1f}s<extra></extra>",
        ))

    # Events at the animal's tracked position at event start.
    n_shown = n_untracked = 0
    if ev_df is not None and not ev_df.empty and len(t):
        ev = ev_df[(ev_df["ts_start_ephys"] >= t_view_start)
                   & (ev_df["ts_start_ephys"] <= t_view_end)]
        et = ev["ts_start_ephys"].to_numpy(dtype=float)
        idx = np.clip(np.searchsorted(t, et), 1, len(t) - 1)
        idx = np.where(np.abs(t[idx - 1] - et) < np.abs(t[idx] - et), idx - 1, idx)
        snapped = np.abs(t[idx] - et) <= EVENT_SNAP_S
        n_untracked = int((~snapped).sum())
        counts = ev_df["opponent"].value_counts()
        ev = ev.assign(ex=x[idx], ey=y[idx])[snapped]
        for opp in sorted(ev["opponent"].unique()):
            sub = ev[ev["opponent"] == opp]
            faded = counts.get(opp, 0) < min_events
            # WebGL like the path: an SVG trace would sit *under* the gl canvas.
            fig.add_trace(go.Scattergl(
                x=sub["ex"], y=sub["ey"], mode="markers",
                marker=dict(size=10, color=opp_colors.get(opp, "grey"),
                            opacity=0.35 if faded else 0.9,
                            line=dict(width=0.5, color="black")),
                name=opp + (" (faded)" if faded else ""),
                hovertext=[f"t={v:.1f}s<br>{btype_map.get(behavior_type, behavior_type)} "
                           f"{r.initiator} → {r.victim}"
                           for v, r in zip(sub["ts_start_ephys"], sub.itertuples())],
                hoverinfo="text",
                customdata=sub["ts_start_ephys"].to_numpy(),   # hover → shared cursor
            ))
            n_shown += len(sub)

    fig.add_trace(_cursor_trace_2d(cursor))

    untracked = f" · {n_untracked} not tracked at event time" if n_untracked else ""
    fig.update_layout(
        title=dict(text=(f"{animal_id} trajectory — {n_shown} {behavior_type} events in view"
                         f"<br><sup>[{t_view_start:.0f}–{t_view_end:.0f} s]{untracked}</sup>"),
                   font=dict(size=15)),
        xaxis=dict(title=f"x ({fmt} pixels)", constrain="domain"),
        yaxis=dict(title=f"y ({fmt} pixels)", autorange="reversed",
                   scaleanchor="x", scaleratio=1),
        **layout,
    )
    return fig


# ── App class ──────────────────────────────────────────────────────────────────

def _raster_window(img, t0, bin_s, start, end, max_cols=MAX_RASTER_COLS):
    """Display slice of the rastermap image for ``[start, end]`` (s, ephys clock).

    Returns ``dict(image, x, dw)`` for an image glyph: the columns covering the
    window, block-averaged so at most ``max_cols`` remain, as float32.
    """
    n_cols = img.shape[1]
    i0 = int(np.clip(np.floor((start - t0) / bin_s), 0, n_cols - 1))
    i1 = int(np.clip(np.ceil((end - t0) / bin_s), i0 + 1, n_cols))
    factor = max(1, int(np.ceil((i1 - i0) / max_cols)))
    i1 = i0 + max(1, (i1 - i0) // factor) * factor
    sub = img[:, i0:i1]
    if factor > 1:
        sub = sub.reshape(sub.shape[0], -1, factor).mean(axis=2)
    return {"image": np.ascontiguousarray(sub, dtype=np.float32),
            "x": t0 + i0 * bin_s, "dw": (i1 - i0) * bin_s}


class HabitatApp:
    def __init__(self):
        self._ks_data = None
        self._events = None
        self._raster_img = None
        self._raster_bin = xc.DEFAULT_RASTER_BIN
        self._raster_src = None   # image glyph source, re-sliced on zoom
        self._t0 = None
        self._t1 = None
        self._full_pop = None
        self._pca = None          # behaviour-independent PCA trajectory (see explore_cache)
        self._traj = None         # focal animal's (t, x, y) in pixels, or None
        self._traj_status = None  # message while the trajectory is unavailable
        self._traj_pane = None
        self._cebra = None        # CEBRA-Time fit (see explore_cache), fitted on demand
        self._cebra_pop = None    # ... with event markers laid on top
        self._cebra_pane = None
        self._cebra_pos = None    # CEBRA-Behavior fit supervised by the focal rat's (x, y)
        self._cebra_pos_pop = None
        self._cebra_pos_pane = None
        self._cebra_ev = None     # CEBRA-Behavior fit on opponent labels after event onsets
        self._cebra_ev_pop = None
        self._cebra_ev_pane = None
        self._cebra_evp = None    # ... and with the focal rat's (x, y) as well
        self._cebra_evp_pop = None
        self._cebra_evp_pane = None
        self._track_sig = None    # signature of the tracking files the trajectory came from
        self._umap = None         # UMAP fit (see explore_cache), fitted on demand
        self._umap_pop = None
        self._umap_pane = None
        self._embed_tab = 0       # 0 PCA, 1 CEBRA, 2 CEBRA (position), 3 UMAP, 4 CEBRA (events), 5 CEBRA (events + position); kept across refreshes
        self._cebra_fitting = False
        self._signature = None    # source signature of the loaded session
        # Shared cursor: the latest requested time, and the one on screen.
        self._cursor_pending = None
        self._cursor_source = None     # 'bokeh' | 'plotly': who asked for the move
        self._cursor_requested_at = 0.0
        self._cursor_t = None          # on the Bokeh lines / readout
        self._cursor_plotly_t = None   # on the Plotly markers
        self._cursor_plotly_at = 0.0
        self._cursor_spans = []
        self._cursor_md = pn.pane.Markdown(
            "", styles={"font-size": "12px", "color": "#888"}, margin=(0, 10))
        self._load_token = 0      # bumps per load; stale background results are dropped
        self._x_range = None      # the live Range1d shared by both Bokeh figures
        self._last_x_range = [None, None]
        self._bokeh_pane = None
        self._plotly_pane = None
        self._loading = False
        self._view = "browser"

        self._status = pn.pane.Alert("", alert_type="info", visible=False,
                                     sizing_mode="stretch_width", margin=(0, 10))
        self._content = pn.Column(
            pn.pane.Alert(
                "Select a session and animal, then press **Load Session**.",
                alert_type="info",
            ),
            sizing_mode="stretch_width",
        )

        # ── Sidebar widgets ────────────────────────────────────────────────────
        self.cohort_sel = pn.widgets.Select(
            name="Cohort", options=list(CONFIG_OPTIONS.keys())
        )
        self.session_sel = pn.widgets.Select(name="Session", options=[])
        self.animal_sel = pn.widgets.Select(name="Animal", options=[])
        self.load_btn = pn.widgets.Button(
            name="Load Session", button_type="primary", width=220
        )
        self.btype_sel = pn.widgets.Select(
            name="Behavior type",
            options=BTYPE_LABELS,
            value=BTYPE_LABELS[13], # Encounter
        )
        self.min_events_sl = pn.widgets.IntSlider(
            name="Min events / opponent", value=10, start=1, end=50
        )
        self.pca_bin_sl = pn.widgets.FloatSlider(
            name="PCA bin size (s)", value=0.5, start=0.1, end=2.0, step=0.1
        )
        self.raster_bin_sl = pn.widgets.FloatSlider(
            name="Rastermap bin size (s)", value=1.0, start=0.1, end=2.0, step=0.1
        )
        # CEBRA-Time is fitted on demand (minutes on a CPU), never on load.
        self.cebra_bin_sl = pn.widgets.FloatSlider(
            name="CEBRA bin size (s)", value=xc.DEFAULT_CEBRA_BIN, start=0.1, end=2.0, step=0.1
        )
        self.cebra_iter_sl = pn.widgets.IntSlider(
            name="CEBRA iterations", value=xc.DEFAULT_CEBRA_ITERATIONS,
            start=500, end=10000, step=500
        )
        self.cebra_btn = pn.widgets.Button(
            name="Fit CEBRA", button_type="primary", width=220,
            disabled=True,
        )
        self.cebra_btn.on_click(self._on_fit_cebra)
        self.cebra_pos_btn = pn.widgets.Button(
            name="Fit CEBRA (position)", button_type="primary", width=220, disabled=True)
        self.cebra_pos_btn.on_click(self._on_fit_cebra_position)
        self.cebra_ev_btn = pn.widgets.Button(
            name="Fit CEBRA (events)", button_type="primary", width=220, disabled=True)
        self.cebra_ev_btn.on_click(self._on_fit_cebra_events)
        self.cebra_evp_btn = pn.widgets.Button(
            name="Fit CEBRA (events + position)", button_type="primary", width=220, disabled=True)
        self.cebra_evp_btn.on_click(self._on_fit_cebra_events_position)
        # UMAP is fitted on demand too (a minute or so, single-threaded for a fixed seed).
        self.umap_bin_sl = pn.widgets.FloatSlider(
            name="UMAP bin size (s)", value=xc.DEFAULT_UMAP_BIN, start=0.1, end=2.0, step=0.1)
        self.umap_neighbors_sl = pn.widgets.IntSlider(
            name="UMAP n_neighbors", value=xc.DEFAULT_UMAP_NEIGHBORS, start=5, end=200, step=5)
        self.umap_min_dist_sl = pn.widgets.FloatSlider(
            name="UMAP min_dist", value=xc.DEFAULT_UMAP_MIN_DIST, start=0.0, end=0.99, step=0.01)
        self.umap_btn = pn.widgets.Button(
            name="Fit UMAP", button_type="primary", width=220, disabled=True)
        self.umap_btn.on_click(self._on_fit_umap)
        self._umap_note = pn.pane.Markdown(
            "", styles={"font-size": "12px", "color": "#888"})
        self._cebra_note = pn.pane.Markdown(
            "", styles={"font-size": "12px", "color": "#888"})

        # Tracking source for the trajectory panel; shown only when a block has
        # more than one (their pixel spaces differ, so they are never mixed).
        self.traj_src_sel = pn.widgets.RadioButtonGroup(
            options={}, button_type="light", visible=False, margin=(0, 10))
        self.traj_src_sel.param.watch(lambda *_: self._update_traj_pane(), "value")
        self.back_btn = pn.widgets.Button(
            name="← Session browser", button_type="light", width=220
        )
        self.back_btn.on_click(lambda *_: self._show_view("browser"))
        self.back_btn.js_on_click(code="if (window.closeNav) { closeNav(); }")
        self.browser = SessionBrowser(COHORT_NAMES, on_explore=self._explore_from_browser)
        self._main = pn.Column(self.browser.view, sizing_mode="stretch_both")
        self._sidebar = pn.Column(self.browser.sidebar, width=280)

        self.cohort_sel.param.watch(self._update_sessions, "value")
        self.session_sel.param.watch(self._update_animals, "value")
        self.load_btn.on_click(self._on_load)
        self.btype_sel.param.watch(self._on_behavior_change, "value")
        self.min_events_sl.param.watch(self._on_behavior_change, "value")
        self.pca_bin_sl.param.watch(self._on_behavior_change, "value")
        self._update_sessions()
        pn.state.add_periodic_callback(self._check_range_update, period=600)
        pn.state.add_periodic_callback(self._flush_cursor, period=CURSOR_PERIOD_MS)
        self._try_restore_from_cache()
        # Return to the cohort/row last selected (theme toggles reload the page).
        browser_state = pn.state.cache.get("session_browser_state") or {}
        with param.parameterized.discard_events(self.browser.cohort_sel):
            if browser_state.get("cohort") in self.browser.cohort_sel.options:
                self.browser.cohort_sel.value = browser_state["cohort"]
        self.browser.reload(select=browser_state.get("recording"))

    # ── Browser ↔ explore views ────────────────────────────────────────────────

    def _show_view(self, view):
        self._view = view
        if view == "browser":
            self._main[:] = [self.browser.view]
            self._sidebar[:] = [self.browser.sidebar]
        else:
            self._main[:] = [self._content]
            self._sidebar[:] = [self._explore_sidebar]
        state = pn.state.cache.get("habitat_last_state")
        if state is not None:
            state["view"] = view

    async def _explore_from_browser(self, cohort_label, session_id, animal_id):
        """Explore button: point the existing dropdowns at the row, then load."""
        if self.cohort_sel.value != cohort_label:
            self.cohort_sel.value = cohort_label  # triggers _update_sessions
        if session_id not in self.session_sel.options:
            self.session_sel.options = sorted(set(self.session_sel.options) | {session_id})
        self.session_sel.value = session_id       # triggers _update_animals
        if animal_id not in self.animal_sel.options:
            self.animal_sel.options = sorted(set(self.animal_sel.options) | {animal_id})
        self.animal_sel.value = animal_id
        self._show_view("explore")
        await self._load_session(cohort_label, session_id, animal_id)

    def _restore_data(self, cohort, session_id, animal_id):
        """Full data already loaded in this process (back → Explore, theme reload)."""
        data = pn.state.cache.get(_cache_key(cohort, session_id, animal_id))
        if data is None:
            return False
        self._ks_data = data["ks_data"]
        self._events = data["events"]
        self._raster_img = data["raster_img"]
        self._raster_bin = data.get("raster_bin", xc.DEFAULT_RASTER_BIN)
        self._t0 = data["t0"]
        self._t1 = data["t1"]
        self._pca = data.get("pca")
        self._cebra = data.get("cebra")
        self._cebra_pos = data.get("cebra_position")
        self._cebra_ev = data.get("cebra_events")
        self._cebra_evp = data.get("cebra_events_position")
        self._umap = data.get("umap")
        self._track_sig = data.get("track_sig")
        self._signature = data.get("signature")
        self._traj = data.get("trajectory")
        self._traj_status = None
        self._sync_traj_sources()
        self._x_range = None
        self._last_x_range = [self._t0, self._t1]
        return True

    def _set_status(self, text=None, kind="info"):
        self._status.object = text or ""
        self._status.alert_type = kind
        self._status.visible = bool(text)

    # ── Session / animal dropdowns ─────────────────────────────────────────────

    def _update_sessions(self, *args):
        cfg = CONFIG_OPTIONS[self.cohort_sel.value]
        try:
            manifest = _recordings(cfg)
            sessions = sorted(manifest["session"].unique().tolist())
        except Exception:
            sessions = []
        self.session_sel.options = sessions
        if sessions:
            self.session_sel.value = sessions[0]

    def _update_animals(self, *args):
        session = self.session_sel.value
        if not session:
            return
        cfg = CONFIG_OPTIONS[self.cohort_sel.value]
        try:
            manifest = _recordings(cfg)
            animals = sorted(
                manifest.loc[manifest["session"] == session, "animal"].tolist()
            )
        except Exception:
            animals = []
        self.animal_sel.options = animals
        if animals:
            self.animal_sel.value = animals[0]

    # ── Cache restore (theme-toggle page reloads) ──────────────────────────────

    def _try_restore_from_cache(self):
        state = pn.state.cache.get("habitat_last_state")
        if state is None:
            return

        cohort = state["cohort"]
        session_id = state["session"]
        animal_id = state["animal"]

        # Restore dropdowns (cohort → sessions → animals cascade automatically via watches)
        if cohort in self.cohort_sel.options and self.cohort_sel.value != cohort:
            self.cohort_sel.value = cohort  # triggers _update_sessions

        if session_id in self.session_sel.options:
            self.session_sel.value = session_id  # triggers _update_animals

        if animal_id in self.animal_sel.options:
            self.animal_sel.value = animal_id

        # Restore behavior/param widgets (safe: _on_behavior_change returns early if no data)
        if state.get("btype_label") in BTYPE_LABELS:
            self.btype_sel.value = state["btype_label"]
        if state.get("min_events") is not None:
            self.min_events_sl.value = state["min_events"]
        if state.get("pca_bin") is not None:
            self.pca_bin_sl.value = state["pca_bin"]
        if state.get("raster_bin") is not None:
            self.raster_bin_sl.value = state["raster_bin"]

        # Restore heavy data objects
        if not self._restore_data(cohort, session_id, animal_id):
            return
        self._refresh_behavior()
        if state.get("view") == "explore":
            self._show_view("explore")

    # ── Load: cached view first, full data in the background ──────────────────

    async def _on_load(self, event=None):
        await self._load_session(
            self.cohort_sel.value, self.session_sel.value, self.animal_sel.value)

    async def _load_session(self, cohort_label, session_id, animal_id):
        """Show the session as fast as possible, then make it fully interactive.

        1. Loaded earlier in this process -> instant.
        2. A disk cache at the default bins -> drawn immediately; the full
           data (Kilosort, events, sync) then loads in a worker thread.
        3. Otherwise -> full load in the worker thread, then draw and cache.
        The background load recomputes rastermap/PCA only if the source files
        changed since the cache was written.
        """
        self._load_token += 1
        token = self._load_token
        cohort = COHORT_NAMES[cohort_label]
        cfg = CONFIG_OPTIONS[cohort_label]
        raster_bin, pca_bin = self.raster_bin_sl.value, self.pca_bin_sl.value

        if self._restore_data(cohort_label, session_id, animal_id):
            self._set_status()
            self._refresh_behavior()
            return

        cached = (xc.load_view_cache(cohort, session_id, animal_id)
                  if xc.is_default_bins(raster_bin, pca_bin) else None)
        self._cebra, self._signature = None, None      # never carry over another session's
        self._cebra_pos, self._track_sig = None, None
        self._umap = self._cebra_ev = self._cebra_evp = None
        if cached is not None:
            self._ks_data = None
            self._events = cached["events"]
            self._signature = cached["signature"]
            self._cebra = xc.cached_cebra(cached, cached["signature"], xc.cebra_params(
                xc.DEFAULT_CEBRA_BIN, xc.DEFAULT_CEBRA_ITERATIONS, xc.DEFAULT_CEBRA_SEED))
            self._cebra_ev = xc.cached_cebra(cached, cached["signature"], xc.cebra_events_params(
                xc.DEFAULT_CEBRA_BIN, xc.DEFAULT_CEBRA_ITERATIONS, xc.DEFAULT_CEBRA_SEED,
                xc.DEFAULT_EVENTS_BEHAVIOR, xc.DEFAULT_EVENTS_MIN))
            self._umap = xc.cached_umap(cached, cached["signature"], xc.umap_params(
                xc.DEFAULT_UMAP_BIN, xc.DEFAULT_UMAP_NEIGHBORS, xc.DEFAULT_UMAP_MIN_DIST,
                xc.DEFAULT_UMAP_SEED))
            self._apply_products(cached)
            self._track_sig = cached.get("tracking_signature")
            if self._track_sig is not None and cached.get("trajectory"):
                self._cebra_pos = xc.cached_cebra(
                    cached, f"{cached['signature']}|{self._track_sig}",
                    xc.cebra_params(xc.DEFAULT_CEBRA_BIN, xc.DEFAULT_CEBRA_ITERATIONS,
                                    xc.DEFAULT_CEBRA_SEED, "position",
                                    cached["trajectory"].get("default")))
                self._cebra_evp = xc.cached_cebra(
                    cached, f"{cached['signature']}|{self._track_sig}",
                    xc.cebra_events_position_params(
                        xc.DEFAULT_CEBRA_BIN, xc.DEFAULT_CEBRA_ITERATIONS, xc.DEFAULT_CEBRA_SEED,
                        xc.DEFAULT_EVENTS_BEHAVIOR, xc.DEFAULT_EVENTS_MIN,
                        cached["trajectory"].get("default")))
            self._traj = cached.get("trajectory")
            self._traj_status = (None if "trajectory" in cached
                                 else "Loading tracking in the background…")
            self._sync_traj_sources()
            self._x_range = None
            saved = datetime.fromtimestamp(cached["saved_at"]).strftime("%Y-%m-%d %H:%M")
            self._set_status(
                f"Showing the cached view of {animal_id} · {session_id} (saved {saved}). "
                "Loading the full spike data in the background — a PCA bin-size change "
                "applies once it finishes.", "info")
            self._refresh_behavior()
        else:
            self._set_status()
            self._traj, self._traj_status = None, "Loading tracking…"
            self._sync_traj_sources()
            self._content[:] = [pn.pane.Alert(
                f"Loading {animal_id} · {session_id}: spikes, events and sync, then "
                "rastermap and PCA. Later visits open from cache.", alert_type="warning")]

        self._loading = True
        try:
            result = await asyncio.to_thread(
                _load_full, cfg, session_id, animal_id, raster_bin, pca_bin,
                cached["signature"] if cached else None)
        except Exception as e:
            logger.exception("loading %s %s failed", animal_id, session_id)
            if token == self._load_token:
                if cached is not None:
                    self._set_status(f"Background load failed ({e}); showing the cached "
                                     "view, which cannot re-bin.", "danger")
                else:
                    self._content[:] = [pn.pane.Alert(f"Failed to load: {e}",
                                                      alert_type="danger")]
            return
        finally:
            if token == self._load_token:
                self._loading = False
        if token != self._load_token:
            return          # the user moved to another session meanwhile

        self._ks_data = result["ks_data"]
        self._events = result["events"]
        products = result["products"]
        self._signature = result["signature"]
        self._update_cebra_controls()       # spikes are loaded: fitting is now possible
        if products is not None:
            # sources changed; save_view_cache drops these too
            self._cebra = self._cebra_pos = self._umap = self._cebra_ev = None
            self._cebra_evp = None
            self._apply_products(products)
            self._x_range = None     # t0/t1 may have moved; start from the full span
            await asyncio.to_thread(xc.save_view_cache, cohort, session_id, animal_id,
                                    products, self._events, result["signature"])

        pn.state.cache[_cache_key(cohort_label, session_id, animal_id)] = {
            "ks_data": self._ks_data,
            "events": self._events,
            "raster_img": self._raster_img,
            "raster_bin": self._raster_bin,
            "t0": self._t0,
            "t1": self._t1,
            "pca": self._pca,
            "cebra": self._cebra,
            "cebra_position": self._cebra_pos,
            "cebra_events": self._cebra_ev,
            "cebra_events_position": self._cebra_evp,
            "umap": self._umap,
            "track_sig": self._track_sig,
            "signature": self._signature,
            "trajectory": self._traj,
        }
        pn.state.cache["habitat_last_state"] = {
            "cohort": cohort_label,
            "session": session_id,
            "animal": animal_id,
            "btype_label": self.btype_sel.value,
            "min_events": self.min_events_sl.value,
            "pca_bin": self.pca_bin_sl.value,
            "raster_bin": self.raster_bin_sl.value,
            "view": "explore",
        }

        if cached is not None and products is not None:
            self._set_status("Source files changed since the cached view was saved; "
                             "rastermap and PCA were recomputed.", "warning")
        else:
            self._set_status()
        # Redraw when the image changed, or when a PCA bin change was deferred.
        if products is not None or cached is None or self._pca_needs_refit():
            self._refresh_behavior()

        # Stage 2: the focal animal's trajectory (reads tracking CSVs).
        track_sig = xc.tracking_signature(result["tracking_files"])
        if (cached is not None and products is None and "trajectory" in cached
                and cached.get("tracking_signature") == track_sig):
            return          # the cached trajectory is current
        if not result["tracking_files"]:
            traj = None
            self._traj_status = f"No tracking files for {session_id[:8]}."
        else:
            try:
                traj = await asyncio.to_thread(
                    xc.load_focal_trajectory, result["tracking_files"], result["sync"],
                    animal_id, self._t0, self._t1)
                self._traj_status = None
            except Exception as e:
                logger.exception("trajectory load failed")
                traj, self._traj_status = None, f"Tracking failed to load: {e}"
        if token != self._load_token:
            return
        self._traj = traj
        self._track_sig = track_sig if self._traj_status is None else None
        self._sync_traj_sources()
        data = pn.state.cache.get(_cache_key(cohort_label, session_id, animal_id))
        if data is not None:
            data["trajectory"] = traj
            data["track_sig"] = self._track_sig
        if self._traj_status is None:
            await asyncio.to_thread(xc.update_cached_trajectory, cohort, session_id,
                                    animal_id, traj, track_sig)
        self._update_traj_pane()
        self._update_cebra_controls()       # tracking is in: the position fit is possible

    def _apply_products(self, products):
        self._raster_img = products["raster_img"]
        self._raster_bin = products["raster_bin"]
        self._t0 = products["t0"]
        self._t1 = products["t1"]
        self._pca = products["pca"]
        self._last_x_range = [self._t0, self._t1]

    def _pca_needs_refit(self):
        return (self._pca is not None
                and abs(self._pca["pca_bin"] - self.pca_bin_sl.value) > 1e-9)

    # ── Refresh: rebuild timeline + PCA, reuse rastermap image ───────────────

    def _refresh_behavior(self):
        animal_id = self.animal_sel.value
        btype = _label_to_abbrev(self.btype_sel.value)

        # Preserve current zoom; fall back to full range on first load
        if self._x_range is not None:
            cur_start = float(self._x_range.start)
            cur_end = float(self._x_range.end)
        else:
            cur_start, cur_end = self._t0, self._t1

        if self._pca_needs_refit() and self._ks_data is not None:
            # Without spikes (cached view) the old trajectory stays until the
            # background load finishes; _load_session then redraws.
            self._pca = xc.fit_pca_trajectory(
                self._ks_data, self.pca_bin_sl.value, self._t0, self._t1)
            data = pn.state.cache.get(_cache_key(
                self.cohort_sel.value, self.session_sel.value, animal_id))
            if data is not None:
                data["pca"] = self._pca
        self._full_pop = _build_pop_data(
            self._events, self._pca, btype, animal_id, self.min_events_sl.value,
        )

        # ── Build Bokeh figures ────────────────────────────────────────────────
        # Timeline: owns the Range1d
        p_tl = self._make_timeline(btype, animal_id, cur_start, cur_end, self.min_events_sl.value)
        # Rastermap: shares the timeline's x_range (canonical Bokeh linking approach)
        p_rm = self._make_rastermap(p_tl.x_range)
        # Store reference for the periodic PCA-update callback
        self._x_range = p_tl.x_range
        self._last_x_range = [cur_start, cur_end]

        self._add_cursor_spans(p_tl, p_rm)

        # Both figures MUST live in a single Bokeh document (one pn.pane.Bokeh)
        bokeh_layout = bk_col(p_tl, p_rm, sizing_mode="stretch_width")

        self._cebra_pop = self._build_cebra_pop(self._cebra, "CEBRA", btype, animal_id)
        self._cebra_pos_pop = self._build_cebra_pop(self._cebra_pos, "CEBRA-pos", btype, animal_id)
        self._umap_pop = self._build_cebra_pop(self._umap, "UMAP", btype, animal_id)
        self._cebra_ev_pop = self._build_cebra_pop(self._cebra_ev, "CEBRA-ev", btype, animal_id)
        self._cebra_evp_pop = self._build_cebra_pop(self._cebra_evp, "CEBRA-ev+pos", btype,
                                                    animal_id)
        plotly_fig = self._compute_pca_fig(cur_start, cur_end)
        traj_fig = self._compute_traj_fig(cur_start, cur_end)
        cebra_fig = self._compute_cebra_fig("time", cur_start, cur_end)
        cebra_pos_fig = self._compute_cebra_fig("position", cur_start, cur_end)
        umap_fig = self._compute_cebra_fig("umap", cur_start, cur_end)
        cebra_ev_fig = self._compute_cebra_fig("events", cur_start, cur_end)
        cebra_evp_fig = self._compute_cebra_fig("events_position", cur_start, cur_end)

        # Always recreate panes — reassigning .object on an existing Bokeh pane
        # can cause document-isolation issues when shared Range1d objects change.
        self._bokeh_pane = pn.pane.Bokeh(bokeh_layout, sizing_mode="stretch_width")
        # link_figure=False: with a linked figure, every restyle the browser
        # applies is echoed back as `restyle_data`, re-applied to the Python
        # figure and re-sent — an endless loop that made the cursor crawl.
        # Cursor moves are sent straight to the browser model instead.
        self._plotly_pane = pn.pane.Plotly(plotly_fig, sizing_mode="stretch_width",
                                           link_figure=False)
        self._traj_pane = pn.pane.Plotly(traj_fig, sizing_mode="stretch_width",
                                         link_figure=False)
        self._cebra_pane = pn.pane.Plotly(cebra_fig, sizing_mode="stretch_width",
                                          link_figure=False)
        self._cebra_pos_pane = pn.pane.Plotly(cebra_pos_fig, sizing_mode="stretch_width",
                                              link_figure=False)
        self._umap_pane = pn.pane.Plotly(umap_fig, sizing_mode="stretch_width",
                                         link_figure=False)
        self._cebra_ev_pane = pn.pane.Plotly(cebra_ev_fig, sizing_mode="stretch_width",
                                             link_figure=False)
        self._cebra_evp_pane = pn.pane.Plotly(cebra_evp_fig, sizing_mode="stretch_width",
                                              link_figure=False)
        for pane in (self._plotly_pane, self._traj_pane, self._cebra_pane,
                     self._cebra_pos_pane, self._umap_pane, self._cebra_ev_pane,
                     self._cebra_evp_pane):
            pane.param.watch(self._on_plotly_hover, "hover_data")
        self._update_cebra_controls()
        # PCA, the four CEBRA variants and UMAP share one slot as tabs. dynamic=True renders only the
        # visible tab (a 3-D scene drawn while hidden gets a zero-size canvas);
        # cursor moves still update the hidden figure, so it is current when shown.
        embed_tabs = pn.Tabs(("PCA", self._plotly_pane), ("CEBRA", self._cebra_pane),
                             ("CEBRA (position)", self._cebra_pos_pane),
                             ("UMAP", self._umap_pane),
                             ("CEBRA (events)", self._cebra_ev_pane),
                             ("CEBRA (events + pos)", self._cebra_evp_pane),
                             dynamic=True, active=self._embed_tab,
                             sizing_mode="stretch_width")
        embed_tabs.param.watch(lambda e: setattr(self, "_embed_tab", e.new), "active")
        self._content[:] = [
            self._status, self._bokeh_pane, self._cursor_md,
            pn.Row(embed_tabs,
                   pn.Column(self.traj_src_sel, self._traj_pane, sizing_mode="stretch_width"),
                   sizing_mode="stretch_width"),
        ]

    # ── CEBRA (fitted on demand) ───────────────────────────────────────────────

    def _cebra_request(self):
        return xc.cebra_params(self.cebra_bin_sl.value, self.cebra_iter_sl.value,
                               xc.DEFAULT_CEBRA_SEED)

    def _update_cebra_controls(self):
        """Button/note state: needs the package, the spikes and, for position, tracking."""
        if self._cebra_fitting:
            return
        self.umap_btn.disabled = not (xc.umap_available() and self._ks_data is not None)
        self._umap_note.object = (
            f"_{xc.UMAP_INSTALL_HINT}_" if not xc.umap_available() else
            "_Available once the full spike data has loaded._" if self._ks_data is None else
            "_Unsupervised, on the same z-scored rates. Default-parameter fits are cached._")
        has_pkg = xc.cebra_available()
        has_ks = self._ks_data is not None
        has_traj = self._selected_traj() is not None
        self.cebra_btn.disabled = not (has_pkg and has_ks)
        self.cebra_pos_btn.disabled = not (has_pkg and has_ks and has_traj)
        self.cebra_ev_btn.disabled = not (has_pkg and has_ks and self._events is not None)
        self.cebra_evp_btn.disabled = not (has_pkg and has_ks and has_traj
                                           and self._events is not None)
        if not has_pkg:
            self._cebra_note.object = f"_{xc.CEBRA_INSTALL_HINT}_"
        elif not has_ks:
            self._cebra_note.object = "_Available once the full spike data has loaded._"
        elif not has_traj:
            self._cebra_note.object = ("_Time fit: minutes, cached. The position fit also "
                                       "needs tracking for this block._")
        else:
            self._cebra_note.object = (
                "_Time: unsupervised. Position: supervised by the focal rat's (x, y). "
                "Events: opponent label for 0-2 s after each event onset (behaviour type "
                "and min events from above); Events + position adds the rat's (x, y) and "
                "needs tracking. Fits take minutes; default-parameter fits are cached._")

    def _build_cebra_pop(self, fit, label, btype, animal_id):
        if fit is None or self._events is None:
            return None
        p = fit["params"]
        if p.get("mode") == "umap":
            detail = (f"UMAP · n_neighbors {p['n_neighbors']} · min_dist {p['min_dist']} · "
                      f"seed {p['seed']}")
        else:
            if p.get("mode") in ("events", "events_position"):
                both = " and (x, y)" if p["mode"] == "events_position" else ""
                what = (f"CEBRA-Behavior on opponent{both} after {p['behavior']} onset "
                        f"({p['window'][0]:g}-{p['window'][1]:g} s), {fit.get('n_events', '?')} "
                        f"events, {len(fit.get('opponents', []))} opponents, "
                        f"{fit.get('n_labeled_bins', '?')} labelled bins")
            else:
                what = ("CEBRA-Time" if p.get("mode", "time") == "time" else
                        f"CEBRA-Behavior on (x, y), {p['tracking']} tracking, "
                        f"{fit.get('n_tracked_bins', '?')} tracked bins")
            detail = (f"{what} · {p['max_iterations']} iterations · seed {p['seed']} · "
                      f"{fit['device']}")
        return _build_pop_data(
            self._events, fit, btype, animal_id, self.min_events_sl.value,
            label=label, subtitle=f"{detail} — axes have no intrinsic meaning")

    def _compute_cebra_fig(self, which, t_start, t_end):
        import plotly.graph_objects as go
        pop, fit = {"time": (self._cebra_pop, self._cebra),
                    "position": (self._cebra_pos_pop, self._cebra_pos),
                    "umap": (self._umap_pop, self._umap),
                    "events": (self._cebra_ev_pop, self._cebra_ev),
                    "events_position": (self._cebra_evp_pop, self._cebra_evp)}[which]
        name = {"time": "CEBRA", "position": "CEBRA-pos", "umap": "UMAP",
                "events": "CEBRA-events", "events_position": "CEBRA-events+pos"}[which]
        if pop is None:
            fig = go.Figure()
            if not (xc.umap_available() if which == "umap" else xc.cebra_available()):
                text = xc.UMAP_INSTALL_HINT if which == "umap" else xc.CEBRA_INSTALL_HINT
            elif self._cebra_fitting == which:
                text = f"Fitting {name}… this can take a minute or more."
            elif fit is not None:
                text = "No events to mark for this behaviour type."
            elif which == "umap":
                text = "Press Fit UMAP in the sidebar to compute the embedding."
            elif which == "events_position":
                text = ("Press Fit CEBRA (events + position) in the sidebar. It is supervised "
                        "by the opponent and the rat's (x, y) in the first 2 s after each event, "
                        "so only events inside the tracked stretch count.")
            elif which == "events":
                text = ("Press Fit CEBRA (events) in the sidebar. It is supervised by the "
                        "opponent in the first 2 s after each event of the selected "
                        "behaviour type.")
            elif which == "position":
                text = ("Press Fit CEBRA (position) in the sidebar. It is supervised by the "
                        "focal rat's tracked (x, y), so the block needs tracking.")
            else:
                text = "Press Fit CEBRA in the sidebar to compute the embedding."
            fig.add_annotation(text=text, xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=13))
            fig.update_layout(title=f"{name} trajectory", height=380,
                              margin=dict(l=0, r=0, t=40, b=0),
                              xaxis_visible=False, yaxis_visible=False)
            return fig
        return _make_pca_plotly(pop, t_start, t_end,
                                cursor=self._embedding_point_at(
                                    fit, self._cursor_t, fit["params"]["bin"]))

    async def _on_fit_cebra(self, event=None):
        await self._fit_cebra("time")

    async def _on_fit_cebra_position(self, event=None):
        await self._fit_cebra("position")

    async def _on_fit_cebra_events(self, event=None):
        await self._fit_cebra("events")

    async def _on_fit_cebra_events_position(self, event=None):
        await self._fit_cebra("events_position")

    async def _fit_cebra(self, which):
        if self._ks_data is None or self._cebra_fitting or not xc.cebra_available():
            return
        needs_traj = which in ("position", "events_position")
        traj = self._selected_traj() if needs_traj else None
        if needs_traj and traj is None:
            return
        token = self._load_token
        cohort_label, session_id = self.cohort_sel.value, self.session_sel.value
        animal_id = self.animal_sel.value
        params = self._cebra_request()
        btype = _label_to_abbrev(self.btype_sel.value)
        min_events = self.min_events_sl.value
        if which in ("events", "events_position"):
            if self._events is None:
                return
            params = xc.cebra_events_params(params["bin"], params["max_iterations"],
                                            params["seed"], btype, min_events)
        # What the fit is computed from, for the cache (position adds the tracking files).
        signature = self._signature
        if needs_traj:
            signature = (f"{self._signature}|{self._track_sig}"
                         if self._signature and self._track_sig else None)
        self._cebra_fitting = which
        self.cebra_btn.disabled = self.cebra_pos_btn.disabled = self.umap_btn.disabled = \
            self.cebra_ev_btn.disabled = self.cebra_evp_btn.disabled = True
        self._cebra_note.object = "_Fitting CEBRA…_"
        self._refresh_behavior()
        t_start = time.monotonic()
        try:
            if which == "time":
                fit = await asyncio.to_thread(
                    xc.fit_cebra_embedding, self._ks_data, params["bin"], self._t0, self._t1,
                    max_iterations=params["max_iterations"], seed=params["seed"])
                short = "fewer than 3 quality cells"
            elif which == "events_position":
                fit = await asyncio.to_thread(
                    xc.fit_cebra_events_position_embedding, self._ks_data, self._events, traj,
                    animal_id, btype, params["bin"], self._t0, self._t1, min_events=min_events,
                    max_iterations=params["max_iterations"], seed=params["seed"])
                short = (f"fewer than 3 quality cells, fewer than 2 opponents with "
                         f"{min_events}+ {btype} events inside the tracked stretch, or fewer "
                         f"than {xc.MIN_EVENT_BINS} labelled, tracked bins")
            elif which == "events":
                fit = await asyncio.to_thread(
                    xc.fit_cebra_events_embedding, self._ks_data, self._events, animal_id,
                    btype, params["bin"], self._t0, self._t1, min_events=min_events,
                    max_iterations=params["max_iterations"], seed=params["seed"])
                short = (f"fewer than 3 quality cells, fewer than 2 opponents with "
                         f"{min_events}+ {btype} events, or fewer than {xc.MIN_EVENT_BINS} "
                         "labelled bins")
            else:
                fit = await asyncio.to_thread(
                    xc.fit_cebra_position_embedding, self._ks_data, traj, params["bin"],
                    self._t0, self._t1,
                    max_iterations=params["max_iterations"], seed=params["seed"])
                short = (f"fewer than 3 quality cells or fewer than {xc.MIN_POSITION_BINS} "
                         "tracked bins")
        except Exception as e:
            logger.exception("CEBRA fit failed")
            fit, error = None, str(e)
        else:
            error = None if fit is not None else short
        self._cebra_fitting = False
        if token != self._load_token:
            return          # another session was loaded meanwhile; drop the result
        if error is not None:
            self._set_status(f"CEBRA fit failed: {error}", "danger")
        else:
            key = xc.CEBRA_KEYS[which]
            setattr(self, {"time": "_cebra", "position": "_cebra_pos", "events": "_cebra_ev",
                           "events_position": "_cebra_evp"}[which], fit)
            self._set_status(
                f"CEBRA ({which}) fit finished in {time.monotonic() - t_start:.0f} s "
                f"on {fit['device']}.", "success")
            data = pn.state.cache.get(_cache_key(cohort_label, session_id, animal_id))
            if data is not None:
                data[key] = fit
            if signature is not None:
                await asyncio.to_thread(
                    xc.update_cached_cebra, COHORT_NAMES[cohort_label], session_id,
                    animal_id, fit, signature)
        self._refresh_behavior()

    def _umap_request(self):
        return xc.umap_params(self.umap_bin_sl.value, self.umap_neighbors_sl.value,
                              self.umap_min_dist_sl.value, xc.DEFAULT_UMAP_SEED)

    async def _on_fit_umap(self, event=None):
        if self._ks_data is None or self._cebra_fitting or not xc.umap_available():
            return
        token = self._load_token
        cohort_label, session_id = self.cohort_sel.value, self.session_sel.value
        animal_id = self.animal_sel.value
        params = self._umap_request()
        signature = self._signature
        self._cebra_fitting = "umap"      # one embedding fit at a time
        self.cebra_btn.disabled = self.cebra_pos_btn.disabled = self.umap_btn.disabled = \
            self.cebra_ev_btn.disabled = self.cebra_evp_btn.disabled = True
        self._umap_note.object = "_Fitting UMAP…_"
        self._refresh_behavior()
        t_start = time.monotonic()
        try:
            fit = await asyncio.to_thread(
                xc.fit_umap_embedding, self._ks_data, params["bin"], self._t0, self._t1,
                n_neighbors=params["n_neighbors"], min_dist=params["min_dist"],
                seed=params["seed"])
            error = None if fit is not None else "fewer than 3 quality cells, or too few bins"
        except Exception as e:
            logger.exception("UMAP fit failed")
            fit, error = None, str(e)
        self._cebra_fitting = False
        if token != self._load_token:
            return          # another session was loaded meanwhile; drop the result
        if error is not None:
            self._set_status(f"UMAP fit failed: {error}", "danger")
        else:
            self._umap = fit
            self._set_status(f"UMAP fit finished in {time.monotonic() - t_start:.0f} s.",
                             "success")
            data = pn.state.cache.get(_cache_key(cohort_label, session_id, animal_id))
            if data is not None:
                data["umap"] = fit
            if signature is not None:
                await asyncio.to_thread(
                    xc.update_cached_umap, COHORT_NAMES[cohort_label], session_id,
                    animal_id, fit, signature)
        self._refresh_behavior()

    # ── Figure builders ────────────────────────────────────────────────────────

    def _make_timeline(self, btype, animal_id, cur_start, cur_end, min_events):
        df = _focal_events(self._events.events_data, btype, animal_id)
        df = df.dropna(subset=["ts_start_ephys", "initiator", "victim"]).reset_index(drop=True)

        # Every event is drawn; opponents below min_events (which the PCA does
        # not mark) are faded rather than hidden.
        opp_counts = df["opponent"].value_counts()
        df["alpha"] = np.where(df["opponent"].map(opp_counts) >= min_events, 0.85, 0.3)
        n_faded = int((df["alpha"] < 0.5).sum())

        colors = _rat_colors(df)
        all_rats = list(colors)
        rat_to_y = {r: i for i, r in enumerate(all_rats)}
        df["y_init"] = df["initiator"].map(rat_to_y).fillna(0).astype(float)
        df["y_vic"] = df["victim"].map(rat_to_y).fillna(0).astype(float)
        df["color"] = df["opponent"].map(colors)

        src = ColumnDataSource(
            df[["ts_start_ephys", "type", "initiator", "victim", "y_init", "y_vic",
                "color", "alpha"]]
        )
        n_rats = max(len(all_rats), 1)
        x_range = Range1d(start=cur_start, end=cur_end, bounds=(self._t0, self._t1))

        wz = WheelZoomTool(dimensions="width")
        p = figure(
            title=(f"{len(df)} {btype} events involving {animal_id} — {self.session_sel.value}"
                   + (f"  │  {n_faded} faded (opponent < {min_events} events, not in PCA)"
                      if n_faded else "")
                   + "  │  zoom/pan to filter PCA"),
            height=220, width=900,
            x_range=x_range,
            y_range=(-0.5, n_rats - 0.5),
            tools=[wz, "reset", "xpan"],
            active_scroll=wz,
            x_axis_label="Time (s, ephys clock)",
            sizing_mode="stretch_width",
        )
        p.yaxis.ticker = FixedTicker(ticks=list(range(n_rats)))
        p.yaxis.major_label_overrides = {i: r for r, i in rat_to_y.items()}
        p.add_tools(HoverTool(tooltips=[
            ("Time (s)", "@ts_start_ephys{0.1f}"),
            ("Type", "@type"),
            ("Initiator", "@initiator"),
            ("Victim", "@victim"),
        ]))
        p.segment(
            x0="ts_start_ephys", x1="ts_start_ephys",
            y0="y_init", y1="y_vic", source=src,
            line_color="grey", line_width=1, line_alpha=0.5,
        )
        p.scatter(
            x="ts_start_ephys", y="y_init", source=src,
            color="color", size=9, alpha="alpha",
        )
        p.scatter(
            x="ts_start_ephys", y="y_vic", source=src,
            fill_color="white", size=9, line_color="color", line_width=1.5,
            line_alpha="alpha",
        )
        return p

    def _make_rastermap(self, shared_x_range):
        img = self._raster_img
        n_rows = img.shape[0]
        # Colour limits from the full image, so zooming never rescales colours.
        vmin = float(np.nanpercentile(img, 2))
        vmax = float(np.nanpercentile(img, 98))
        if vmax <= vmin:
            vmax = vmin + 1.0
        mapper = LinearColorMapper(palette=Inferno256, low=vmin, high=vmax)

        wz = WheelZoomTool(dimensions="width")
        p = figure(
            title="Rastermap — quality cells sorted by activity similarity",
            height=280, width=900,
            x_range=shared_x_range,   # canonical linking: same object as timeline
            y_range=(0, n_rows),
            tools=[wz, "reset", "xpan"],
            active_scroll=wz,
            x_axis_label="Time (s, ephys clock)",
            y_axis_label="Neuron (sorted)",
            sizing_mode="stretch_width",
        )
        self._raster_src = ColumnDataSource(self._raster_data(
            float(shared_x_range.start), float(shared_x_range.end)))
        p.image(image="image", x="x", y=0, dw="dw", dh=n_rows,
                source=self._raster_src, color_mapper=mapper)
        return p

    def _raster_data(self, start, end):
        """Source columns for the visible window plus one window of margin per side."""
        span = end - start
        w = _raster_window(self._raster_img, self._t0, self._raster_bin,
                           start - span, end + span)
        return {"image": [w["image"]], "x": [w["x"]], "dw": [w["dw"]]}

    def _compute_pca_fig(self, t_start, t_end):
        import plotly.graph_objects as go
        if self._full_pop is None:
            fig = go.Figure()
            btype = _label_to_abbrev(self.btype_sel.value)
            fig.add_annotation(
                text=(
                    f"No '{btype}' events with "
                    f"≥{self.min_events_sl.value} trials per opponent "
                    f"involving {self.animal_sel.value}."
                ),
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=13),
            )
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
            return fig
        return _make_pca_plotly(self._full_pop, t_start, t_end,
                                cursor=self._pca_point_at(self._cursor_t))

    def _sync_traj_sources(self):
        """Point the source toggle at ``self._traj``'s formats (default selected)."""
        formats = xc.trajectory_formats(self._traj)
        names = {"apt_tqt": "APT", "mask_metrics": "manual"}
        options = {f"{names.get(k, k)} tracking ({len(v['t']):,} frames)": k
                   for k, v in formats.items()}
        default = (self._traj or {}).get("default") or next(iter(formats), None)
        with param.parameterized.discard_events(self.traj_src_sel):
            self.traj_src_sel.options = options
            if self.traj_src_sel.value not in options.values():
                self.traj_src_sel.value = default
        self.traj_src_sel.visible = len(options) > 1

    def _selected_traj(self):
        formats = xc.trajectory_formats(self._traj)
        fmt = self.traj_src_sel.value if self.traj_src_sel.value in formats else \
            (self._traj or {}).get("default") or next(iter(formats), None)
        return None if fmt is None else {**formats[fmt], "format": fmt}

    def _compute_traj_fig(self, t_start, t_end):
        animal_id = self.animal_sel.value
        btype = _label_to_abbrev(self.btype_sel.value)
        ev_df = None
        if self._events is not None:
            ev_df = _focal_events(self._events.events_data, btype, animal_id).dropna(
                subset=["ts_start_ephys", "initiator", "victim"])
        colors = (self._full_pop or {}).get("opp_colors") or (
            _rat_colors(ev_df) if ev_df is not None else {})
        return _make_trajectory_plotly(
            self._selected_traj(), ev_df, colors, self.min_events_sl.value, t_start, t_end,
            animal_id, btype, BehavioralEventsData.BEHAVIOR_TYPES, self._traj_status,
            other_sources=len(xc.trajectory_formats(self._traj)) > 1,
            cursor=self._traj_point_at(self._cursor_t))

    def _update_traj_pane(self):
        if self._traj_pane is None:
            return
        start, end = self._last_x_range
        if start is None:
            start, end = self._t0, self._t1
        self._traj_pane.object = self._compute_traj_fig(start, end)

    # ── Shared time cursor ─────────────────────────────────────────────────────

    def _add_cursor_spans(self, *figs):
        """A vertical line per Bokeh figure, moved together in the browser.

        The JS callback moves every line on mouse-move/pan with no server round
        trip; the location change then syncs to Python, where it drives the
        Plotly cursors (via ``_flush_cursor``).
        """
        t = self._cursor_t
        self._cursor_spans = [
            Span(location=t, dimension="height", line_color=CURSOR_COLOR,
                 line_width=1.5, line_alpha=0.9, visible=t is not None)
            for _ in figs]
        # Throttled (trailing edge): every location change is a sync message,
        # and an unthrottled sweep sends hundreds per second.
        move = CustomJS(args=dict(spans=self._cursor_spans, ms=CURSOR_JS_THROTTLE_MS), code="""
            const st = window.__habitatCursor || (window.__habitatCursor = {last: 0, timer: null});
            st.x = cb_obj.x;
            st.spans = spans;
            const apply = () => {
                st.last = Date.now(); st.timer = null;
                for (const s of st.spans) { s.location = st.x; s.visible = true; }
            };
            const wait = ms - (Date.now() - st.last);
            if (wait <= 0) { apply(); } else if (!st.timer) { st.timer = setTimeout(apply, wait); }
        """)
        for fig, span in zip(figs, self._cursor_spans):
            fig.add_layout(span)
            fig.js_on_event(bk_events.MouseMove, move)
            fig.js_on_event(bk_events.Pan, move)
        self._cursor_spans[0].on_change(
            "location", lambda attr, old, new: self._request_cursor(new, "bokeh"))

    def _request_cursor(self, t, source):
        if t is None or not np.isfinite(t) or float(t) == self._cursor_t:
            return      # includes the server's own span updates echoing back
        self._cursor_pending = float(t)
        self._cursor_source = source
        self._cursor_requested_at = time.monotonic()

    def _on_plotly_hover(self, event):
        """Hovering a PCA/trajectory point moves the cursor to that point's time."""
        for point in (event.new or {}).get("points", []):
            t = point.get("customdata")
            if isinstance(t, (list, tuple)):
                t = t[0] if t else None
            if isinstance(t, (int, float)):
                self._request_cursor(t, "plotly")
                return

    @staticmethod
    def _embedding_point_at(emb, t, bin_size):
        if t is None or emb is None:
            return None
        bc = emb["bin_centers"]
        i = int(np.clip(np.searchsorted(bc, t), 0, len(bc) - 1))
        if abs(bc[i] - t) > bin_size:
            return None             # outside the recording
        return emb["scores"][i]

    def _pca_point_at(self, t):
        return self._embedding_point_at(
            self._pca, t, self._pca["pca_bin"] if self._pca else 0.0)

    def _cebra_point_at(self, t, fit=None):
        fit = self._cebra if fit is None else fit
        return self._embedding_point_at(fit, t, fit["params"]["bin"] if fit else 0.0)

    def _traj_point_at(self, t):
        traj = self._selected_traj()
        if t is None or traj is None or len(traj["t"]) == 0:
            return None
        tt = traj["t"]
        i = int(np.clip(np.searchsorted(tt, t), 1, len(tt) - 1))
        i = i - 1 if abs(tt[i - 1] - t) < abs(tt[i] - t) else i
        if abs(tt[i] - t) > EVENT_SNAP_S:
            return None             # not tracked at this moment
        return traj["x"][i], traj["y"][i]

    @staticmethod
    def _move_plotly_cursor(pane, point, three_d=False):
        """Move the pane's cursor trace with a one-trace restyle, browser only."""
        fig = pane.object if pane is not None else None
        if fig is None:
            return
        for k, trace in enumerate(fig.data):
            if trace.name == CURSOR_NAME:
                coords = ([], [], []) if point is None else tuple([v] for v in point)
                update = {"x": [coords[0]], "y": [coords[1]]}
                if three_d:
                    update["z"] = [coords[2]]
                # Keep the Python figure current for the next full redraw ...
                trace.update(x=coords[0], y=coords[1], **({"z": coords[2]} if three_d else {}))
                # ... and send just this trace to the browser. The pane is
                # unlinked, so the browser's restyle echo is not re-sent.
                for model, _ in list(pane._models.values()):
                    model.restyle = {"data": update, "traces": [k]}
                return

    def _flush_cursor(self):
        t = self._cursor_pending
        if t is None:
            return
        now = time.monotonic()
        if t != self._cursor_t:
            self._cursor_t = t
            # Lines already show a browser-side move; only a hover in a Plotly
            # pane has to move them from here.
            if self._cursor_source == "plotly":
                for span in self._cursor_spans:
                    span.location = t
                    span.visible = True
            tracked = self._traj_point_at(t)
            note = "" if tracked is not None or self._traj is None else \
                " · not tracked at this moment"
            self._cursor_md.object = (f"Cursor **t = {t:.1f} s** (ephys clock){note} — "
                                      "hover or pan any plot to move it")
        if t == self._cursor_plotly_t:
            return
        settled = now - self._cursor_requested_at >= CURSOR_SETTLE_S
        due = now - self._cursor_plotly_at >= CURSOR_FOLLOW_S
        if not (settled or due):
            return
        self._cursor_plotly_t, self._cursor_plotly_at = t, now
        self._move_plotly_cursor(self._plotly_pane, self._pca_point_at(t), three_d=True)
        self._move_plotly_cursor(self._cebra_pane, self._cebra_point_at(t), three_d=True)
        self._move_plotly_cursor(self._cebra_pos_pane, self._cebra_point_at(t, self._cebra_pos),
                                 three_d=True)
        self._move_plotly_cursor(self._umap_pane, self._cebra_point_at(t, self._umap),
                                 three_d=True)
        self._move_plotly_cursor(self._cebra_ev_pane, self._cebra_point_at(t, self._cebra_ev),
                                 three_d=True)
        self._move_plotly_cursor(self._cebra_evp_pane, self._cebra_point_at(t, self._cebra_evp),
                                 three_d=True)
        self._move_plotly_cursor(self._traj_pane, self._traj_point_at(t))

    # ── Periodic callback: update PCA when zoom/pan changes ───────────────────

    def _check_range_update(self):
        if self._x_range is None or self._plotly_pane is None:
            return
        cur = [float(self._x_range.start), float(self._x_range.end)]
        if cur == self._last_x_range:
            return
        self._last_x_range = cur
        if self._raster_src is not None:
            self._raster_src.data = self._raster_data(cur[0], cur[1])
        self._plotly_pane.object = self._compute_pca_fig(cur[0], cur[1])
        if self._traj_pane is not None:
            self._traj_pane.object = self._compute_traj_fig(cur[0], cur[1])
        if self._cebra_pane is not None:
            self._cebra_pane.object = self._compute_cebra_fig("time", cur[0], cur[1])
        if self._cebra_pos_pane is not None:
            self._cebra_pos_pane.object = self._compute_cebra_fig("position", cur[0], cur[1])
        if self._umap_pane is not None:
            self._umap_pane.object = self._compute_cebra_fig("umap", cur[0], cur[1])
        if self._cebra_ev_pane is not None:
            self._cebra_ev_pane.object = self._compute_cebra_fig("events", cur[0], cur[1])
        if self._cebra_evp_pane is not None:
            self._cebra_evp_pane.object = self._compute_cebra_fig(
                "events_position", cur[0], cur[1])

    def _on_behavior_change(self, *args):
        # A cached view (no spikes yet) can still switch behaviour type.
        if self._raster_img is None or self._events is None:
            return
        # Keep cached widget state in sync so theme-toggle restores current settings
        state = pn.state.cache.get("habitat_last_state")
        if state is not None:
            state["btype_label"] = self.btype_sel.value
            state["min_events"] = self.min_events_sl.value
            state["pca_bin"] = self.pca_bin_sl.value
        self._refresh_behavior()

    # ── Layout ─────────────────────────────────────────────────────────────────

    @property
    def layout(self):
        return pn.template.FastListTemplate(
            title="Habitat Pipeline — Interactive",
            sidebar=[self._sidebar],
            main=[self._main],
            # Browser first: the table needs the width; ☰ (or Explore) opens it.
            collapsed_sidebar=True,
        )

    @property
    def _explore_sidebar(self):
        return pn.Column(
            self.back_btn,
            pn.layout.Divider(),
            "## Session",
            self.cohort_sel,
            self.session_sel,
            self.animal_sel,
            self.load_btn,
            pn.layout.Divider(),
            "## Behavioral Events",
            self.btype_sel,
            self.min_events_sl,
            pn.layout.Divider(),
            "## PCA",
            self.pca_bin_sl,
            pn.layout.Divider(),
            "## Rastermap",
            self.raster_bin_sl,
            pn.layout.Divider(),
            "## CEBRA",
            self.cebra_bin_sl,
            self.cebra_iter_sl,
            self.cebra_btn,
            self.cebra_pos_btn,
            self.cebra_ev_btn,
            self.cebra_evp_btn,
            self._cebra_note,
            pn.layout.Divider(),
            "## UMAP",
            self.umap_bin_sl,
            self.umap_neighbors_sl,
            self.umap_min_dist_sl,
            self.umap_btn,
            self._umap_note,
            pn.pane.Markdown(
                "_Zoom/pan top panels to update PCA and CEBRA._",
                styles={"font-size": "12px", "color": "#888"},
            ),
            width=280,
        )


HabitatApp().layout.servable()
