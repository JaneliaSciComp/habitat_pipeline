# Run from project root: panel serve gui/interactive_app.py --show
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import panel as pn
from bokeh.layouts import column as bk_col
from bokeh.models import (
    ColumnDataSource, FixedTicker, HoverTool, LinearColorMapper,
    Range1d, WheelZoomTool,
)
from bokeh.palettes import Category10, Inferno256
from bokeh.plotting import figure
from scipy.stats import zscore
from sklearn.decomposition import PCA

from ephys.decode_opponent_identity import align_spikes_to_events, extract_firing_rate_features
from ephys.rastermap_viz import bin_spikes_matrix
from ingestion.data_paths import DataStorageManager, get_animals_and_sessions
from ingestion.ephys_sync import DataSyncManager
from ingestion.kilosort_data_import import load_kilosort_data
from video.behavioral_events import BehavioralEventsData

pn.extension("plotly")

PALETTE = Category10[10]

# ── Per-process data cache (survives theme-toggle page reloads) ────────────────

def _cache_key(cohort: str, session_id: str, animal_id: str) -> str:
    return f"habitat_data__{cohort}__{session_id}__{animal_id}"
CONFIG_OPTIONS = {
    "Cohort 7 (default)": None,
    "Cohort 5": "cohort5_paths.json",
}
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

def _compute_spike_matrix(ks_data, quality_indices, t0, t1, bin_size_s):
    """Return (n_quality_cells, n_bins) firing-rate matrix for quality-filtered cells."""
    edges = np.arange(t0, t1 + bin_size_s, bin_size_s)
    n_bins = len(edges) - 1
    mat = np.zeros((len(quality_indices), n_bins), dtype=np.float64)
    for row, ci in enumerate(quality_indices):
        counts, _ = np.histogram(ks_data.spike_times_by_cell[ci], bins=edges)
        mat[row] = counts / bin_size_s
    return mat


def _fit_rastermap(fr_matrix):
    """Fit Rastermap on (n_cells, n_time_bins) matrix. Returns display image (float64, C-order)."""
    from rastermap import Rastermap
    n_cells = fr_matrix.shape[0]
    model = Rastermap(
        n_PCs=min(200, n_cells - 1),
        n_clusters=min(100, max(4, n_cells // 4)),
        normalize=True,
        mean_time=True,
        verbose=False,
        verbose_sorting=False,
    )
    model.fit(fr_matrix)
    # [::-1] gives negative strides — Bokeh image glyph requires C-contiguous float64
    img = np.ascontiguousarray(
        np.nan_to_num(model.X_embedding[::-1, :], nan=0.0, posinf=0.0, neginf=0.0),
        dtype=np.float64,
    )
    return img


def _build_pop_data(events, ks_data, behavior_type, animal_id, pca_bin, min_events, t0, t1):
    """Fit PCA on the full continuous recording (quality cells, raw FR).

    Opponent colors use the same PALETTE + all_rats index as the timeline plot.

    Returns dict with keys:
      scores        : ndarray (n_bins, 3)  — full trajectory in PC space
      bin_centers   : ndarray (n_bins,)   — time of each bin center (s)
      var_explained : ndarray (3,)
      ev_starts     : ndarray of event ts_start_ephys
      ev_opponents  : ndarray of opponent labels (str)
      opp_colors    : dict {opponent: hex_color}  — same mapping as timeline
      btype_map     : dict {abbrev: full_name}
    or None on failure.
    """
    # Full recording firing-rate matrix (quality-filtered cells, raw FR — no z-score)
    spks, bin_centers = bin_spikes_matrix(
        ks_data, bin_size=pca_bin, start_time=t0, end_time=t1, filtered_only=True,
    )
    n_cells, _ = spks.shape
    if n_cells < 3:
        return None

    X = np.nan_to_num(zscore(spks, axis=1), nan=0.0)
    pca = PCA(n_components=3)
    scores = pca.fit_transform(X.T)   # (n_bins, 3)

    # Gather events of the selected behavior type
    try:
        ev_starts, _, ev_labels = events.extract_opponent_labels(
            animal_of_interest=animal_id,
            behavior_type=behavior_type,
            min_events_per_class=min_events,
        )
    except Exception:
        ev_starts = ev_labels = np.array([])

    # Opponent colors: exact same logic as _make_timeline (including min_events filter)
    df = events.events_data
    df_filt = df[(df["type"] == behavior_type) &
                 ((df["initiator"] == animal_id) | (df["victim"] == animal_id))].copy()
    df_filt["opponent"] = df_filt.apply(
        lambda r: r["victim"] if r["initiator"] == animal_id else r["initiator"], axis=1
    )
    opp_counts = df_filt["opponent"].value_counts()
    valid_opps = opp_counts[opp_counts >= min_events].index
    df_filt = df_filt[df_filt["opponent"].isin(valid_opps)]
    all_rats = sorted(set(df_filt["initiator"].tolist() + df_filt["victim"].tolist()))
    opp_colors = {r: PALETTE[all_rats.index(r) % 10] for r in all_rats}

    return {
        "scores": scores,
        "bin_centers": bin_centers,
        "var_explained": pca.explained_variance_ratio_,
        "ev_starts": ev_starts,
        "ev_opponents": ev_labels,
        "btype_map": BehavioralEventsData.BEHAVIOR_TYPES,
        "opp_colors": opp_colors,
        "behavior_type": behavior_type,
    }


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

class HabitatApp:
    def __init__(self):
        self._ks_data = None
        self._events = None
        self._raster_img = None
        self._t0 = None
        self._t1 = None
        self._full_pop = None
        self._x_range = None      # the live Range1d shared by both Bokeh figures
        self._last_x_range = [None, None]
        self._bokeh_pane = None
        self._plotly_pane = None
        self._loading = False

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

        self.cohort_sel.param.watch(self._update_sessions, "value")
        self.session_sel.param.watch(self._update_animals, "value")
        self.load_btn.on_click(self._on_load)
        self.btype_sel.param.watch(self._on_behavior_change, "value")
        self.min_events_sl.param.watch(self._on_behavior_change, "value")
        self.pca_bin_sl.param.watch(self._on_behavior_change, "value")
        self._update_sessions()
        pn.state.add_periodic_callback(self._check_range_update, period=600)
        self._try_restore_from_cache()

    # ── Session / animal dropdowns ─────────────────────────────────────────────

    def _update_sessions(self, *args):
        cfg = CONFIG_OPTIONS[self.cohort_sel.value]
        try:
            manifest = get_animals_and_sessions(config_path=cfg)
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
            manifest = get_animals_and_sessions(config_path=cfg)
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
        data = pn.state.cache.get(_cache_key(cohort, session_id, animal_id))
        if data is None:
            return

        self._ks_data = data["ks_data"]
        self._events = data["events"]
        self._raster_img = data["raster_img"]
        self._t0 = data["t0"]
        self._t1 = data["t1"]
        self._last_x_range = [self._t0, self._t1]

        self._refresh_behavior()

    # ── Load: spike data + rastermap only ─────────────────────────────────────

    def _on_load(self, event):
        if self._loading:
            return
        self._loading = True
        cfg = CONFIG_OPTIONS[self.cohort_sel.value]
        animal_id = self.animal_sel.value
        session_id = self.session_sel.value

        self._content[:] = [pn.pane.Alert("Loading spike data…", alert_type="warning")]
        try:
            dsm = DataStorageManager(animal_id, session_id, config_path=cfg, auto_load=True)
            self._ks_data = load_kilosort_data(dsm)
            self._events = BehavioralEventsData(dsm, auto_load=True)
            sync = DataSyncManager(dsm, dio_channel=1, auto_load=True)
            self._events.synchronize_with_ephys(sync, create_new_columns=True)
        except Exception as e:
            self._content[:] = [pn.pane.Alert(f"Failed to load: {e}", alert_type="danger")]
            self._loading = False
            return

        self._content[:] = [pn.pane.Alert("Fitting Rastermap…", alert_type="warning")]
        try:
            filter_results = self._ks_data.filter_cells_by_firing_patterns()
            passed_ids = set(filter_results["passed_clusters"])
            ks_ids = list(self._ks_data.ks_ids)
            quality_indices = [i for i, cid in enumerate(ks_ids) if cid in passed_ids]

            quality_spikes = np.concatenate(
                [self._ks_data.spike_times_by_cell[i] for i in quality_indices]
            )
            self._t0 = float(quality_spikes.min())
            self._t1 = float(quality_spikes.max())

            fr_matrix = _compute_spike_matrix(
                self._ks_data, quality_indices, self._t0, self._t1, self.raster_bin_sl.value
            )
            self._raster_img = _fit_rastermap(fr_matrix)
        except Exception as e:
            self._content[:] = [pn.pane.Alert(f"Rastermap failed: {e}", alert_type="danger")]
            self._loading = False
            return

        self._x_range = None
        self._last_x_range = [self._t0, self._t1]

        # Persist loaded data so theme-toggle page reloads don't require a re-load
        pn.state.cache[_cache_key(self.cohort_sel.value, session_id, animal_id)] = {
            "ks_data": self._ks_data,
            "events": self._events,
            "raster_img": self._raster_img,
            "t0": self._t0,
            "t1": self._t1,
        }
        pn.state.cache["habitat_last_state"] = {
            "cohort": self.cohort_sel.value,
            "session": session_id,
            "animal": animal_id,
            "btype_label": self.btype_sel.value,
            "min_events": self.min_events_sl.value,
            "pca_bin": self.pca_bin_sl.value,
            "raster_bin": self.raster_bin_sl.value,
        }

        self._refresh_behavior()
        self._loading = False

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

        self._full_pop = _build_pop_data(
            self._events, self._ks_data, btype, animal_id,
            self.pca_bin_sl.value, self.min_events_sl.value,
            self._t0, self._t1,
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
        self._content[:] = [self._bokeh_pane, self._plotly_pane]

    # ── Figure builders ────────────────────────────────────────────────────────

    def _make_timeline(self, btype, animal_id, cur_start, cur_end, min_events):
        df = self._events.events_data.copy()
        df = df[df["type"] == btype]
        df = df[(df["initiator"] == animal_id) | (df["victim"] == animal_id)]
        df = df.dropna(subset=["ts_start_ephys", "initiator", "victim"]).reset_index(drop=True)

        # Count events per opponent; drop opponents below min_events threshold
        def get_opp(row):
            return row["victim"] if row["initiator"] == animal_id else row["initiator"]
        df["opponent"] = df.apply(get_opp, axis=1)
        opp_counts = df["opponent"].value_counts()
        valid_opps = opp_counts[opp_counts >= min_events].index
        df = df[df["opponent"].isin(valid_opps)].reset_index(drop=True)

        all_rats = sorted(set(df["initiator"].tolist() + df["victim"].tolist()))
        rat_to_y = {r: i for i, r in enumerate(all_rats)}
        df["y_init"] = df["initiator"].map(rat_to_y).fillna(0).astype(float)
        df["y_vic"] = df["victim"].map(rat_to_y).fillna(0).astype(float)

        df["color"] = df["opponent"].apply(lambda opp: PALETTE[all_rats.index(opp) % 10])

        src = ColumnDataSource(
            df[["ts_start_ephys", "type", "initiator", "victim", "y_init", "y_vic", "color"]]
        )
        n_rats = max(len(all_rats), 1)
        x_range = Range1d(start=cur_start, end=cur_end, bounds=(self._t0, self._t1))

        wz = WheelZoomTool(dimensions="width")
        p = figure(
            title=f"{btype} events involving {animal_id} — {self.session_sel.value}  │  zoom/pan to filter PCA",
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
            color="color", size=9, alpha=0.85,
        )
        p.scatter(
            x="ts_start_ephys", y="y_vic", source=src,
            fill_color="white", size=9, line_color="color", line_width=1.5,
        )
        return p

    def _make_rastermap(self, shared_x_range):
        img = self._raster_img
        n_rows, n_cols = img.shape
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
        p.image(
            image=[img],
            x=self._t0, y=0,
            dw=(self._t1 - self._t0), dh=n_rows,
            color_mapper=mapper,
        )
        return p

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
        if self._loading or self._x_range is None or self._plotly_pane is None:
            return
        cur = [float(self._x_range.start), float(self._x_range.end)]
        if cur == self._last_x_range:
            return
        self._last_x_range = cur
        self._plotly_pane.object = self._compute_pca_fig(cur[0], cur[1])

    def _on_behavior_change(self, *args):
        if self._ks_data is None or self._events is None:
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
        sidebar = pn.Column(
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
        return pn.template.FastListTemplate(
            title="Habitat Pipeline — Interactive",
            sidebar=[sidebar],
            main=[self._content],
        )


HabitatApp().layout.servable()
