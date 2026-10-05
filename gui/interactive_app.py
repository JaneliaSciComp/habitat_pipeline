# Run from project root: panel serve gui/interactive_app.py --show
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import panel as pn
import param
from bokeh.layouts import column as bk_col
from bokeh.models import (
    ColumnDataSource, FixedTicker, HoverTool, LinearColorMapper,
    Range1d, WheelZoomTool,
)
from bokeh.palettes import Category10, Inferno256
from bokeh.plotting import figure

from ephys.decode_opponent_identity import align_spikes_to_events, extract_firing_rate_features
from ingestion.data_paths import DataStorageManager, get_animals_and_sessions
from ingestion.ephys_sync import DataSyncManager
from gui import explore_cache as xc
from gui.session_browser import SessionBrowser
from ingestion.kilosort_data_import import load_kilosort_data
from video.behavioral_events import BehavioralEventsData, load_behavioral_events

pn.extension("plotly", "tabulator")

PALETTE = Category10[10]
logger = logging.getLogger(__name__)

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


def _build_pop_data(events, pca_base, behavior_type, animal_id, min_events):
    """Lay one behaviour type's event markers over a precomputed PCA trajectory.

    ``pca_base`` is :func:`gui.explore_cache.fit_pca_trajectory`'s output; the
    fit itself is behaviour-independent, so changing the behaviour type or
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
        "var_explained": pca_base["var_explained"],
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
    events.synchronize_with_ephys(DataSyncManager(dsm, dio_channel=1), create_new_columns=True)

    if signature != cached_signature:
        products = xc.compute_view_products(ks_data, raster_bin, pca_bin)
    else:
        products = None
        ks_data.filter_cells_by_firing_patterns()   # what later re-binning expects
    return {"ks_data": ks_data, "events": events, "signature": signature,
            "products": products}


def _make_pca_plotly(pop_data_full, t_view_start, t_view_end):
    """Full continuous PCA trajectory with event markers filtered to the current view.

    Background: thin line colored by time (viridis) clipped to [t_view_start, t_view_end].
    Foreground: one marker-trace per opponent, showing only events in the view window.
    """
    import plotly.graph_objects as go

    scores = pop_data_full["scores"]
    bin_centers = pop_data_full["bin_centers"]
    var = pop_data_full["var_explained"] * 100
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
            ))
            n_events_shown += opp_mask.sum()

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
            xaxis_title=f"PC1 ({var[0]:.1f}%)",
            yaxis_title=f"PC2 ({var[1]:.1f}%)",
            zaxis_title=f"PC3 ({var[2]:.1f}%)",
        ),
        title=(
            f"PCA trajectory — {n_events_shown} events in view"
            f"  [{t_view_start:.0f}–{t_view_end:.0f} s]"
        ),
        height=460,
        legend=dict(title="Opponent", x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0),
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
        if cached is not None:
            self._ks_data = None
            self._events = cached["events"]
            self._apply_products(cached)
            self._x_range = None
            saved = datetime.fromtimestamp(cached["saved_at"]).strftime("%Y-%m-%d %H:%M")
            self._set_status(
                f"Showing the cached view of {animal_id} · {session_id} (saved {saved}). "
                "Loading the full spike data in the background — a PCA bin-size change "
                "applies once it finishes.", "info")
            self._refresh_behavior()
        else:
            self._set_status()
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
        if products is not None:
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

        # Both figures MUST live in a single Bokeh document (one pn.pane.Bokeh)
        bokeh_layout = bk_col(p_tl, p_rm, sizing_mode="stretch_width")

        plotly_fig = self._compute_pca_fig(cur_start, cur_end)

        # Always recreate panes — reassigning .object on an existing Bokeh pane
        # can cause document-isolation issues when shared Range1d objects change.
        self._bokeh_pane = pn.pane.Bokeh(bokeh_layout, sizing_mode="stretch_width")
        self._plotly_pane = pn.pane.Plotly(plotly_fig, sizing_mode="stretch_width")
        self._content[:] = [self._status, self._bokeh_pane, self._plotly_pane]

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
        return _make_pca_plotly(self._full_pop, t_start, t_end)

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
            pn.pane.Markdown(
                "_Zoom/pan top panels to update PCA._",
                styles={"font-size": "12px", "color": "#888"},
            ),
            width=280,
        )


HabitatApp().layout.servable()
