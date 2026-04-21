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
from sklearn.decomposition import PCA

from ephys.decode_opponent_identity import align_spikes_to_events, extract_firing_rate_features
from ingestion.data_paths import DataStorageManager, get_animals_and_sessions
from ingestion.ephys_sync import DataSyncManager
from ingestion.kilosort_data_import import KilosortData
from video.behavioral_events import BehavioralEventsData

pn.extension("plotly")

PALETTE = Category10[10]
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


def _build_pop_data(events, ks_data, behavior_type, animal_id, pca_tw, pca_bin, min_events):
    """Build per-event firing-rate matrices for events involving animal_id.
    Returns dict {pop_data, event_starts, time_bins} or None."""
    try:
        ev_starts, ev_ends, ev_labels = events.extract_opponent_labels(
            animal_of_interest=animal_id,
            behavior_type=behavior_type,
            min_events_per_class=min_events,
        )
    except Exception:
        return None
    if len(ev_starts) == 0:
        return None

    edges = np.arange(pca_tw[0], pca_tw[1] + pca_bin, pca_bin)
    n_bins = len(edges) - 1
    time_bins = edges[:-1] + pca_bin / 2
    n_cells = len(ks_data.ks_ids)
    pop_data, event_starts = {}, {}

    for label in np.unique(ev_labels):
        mask = ev_labels == label
        starts = ev_starts[mask]
        event_starts[label] = starts
        mat = np.zeros((len(starts), n_cells, n_bins), dtype=np.float32)
        for ci, spikes in enumerate(ks_data.spike_times_by_cell):
            aligned = align_spikes_to_events(spikes, starts, pca_tw)
            fr = extract_firing_rate_features(aligned, pca_tw, pca_bin)
            nc = min(n_bins, fr.shape[1])
            mat[:, ci, :nc] = fr[:, :nc]
        pop_data[label] = mat

    return {"pop_data": pop_data, "event_starts": event_starts, "time_bins": time_bins}


def _make_pca_plotly(pop_data_full, event_starts_by_label, t_view_start, t_view_end):
    import plotly.graph_objects as go
    import plotly.express as px
    palette = px.colors.qualitative.Set1
    X_list, valid_labels, subset_data = [], [], {}

    for label, matrix in pop_data_full.items():
        starts = event_starts_by_label[label]
        mask = (starts >= t_view_start) & (starts <= t_view_end)
        if mask.sum() < 2:
            continue
        sub = matrix[mask]
        n_sel, n_cells, n_bins = sub.shape
        subset_data[label] = sub
        valid_labels.append(label)
        X_list.append(sub.transpose(0, 2, 1).reshape(n_sel * n_bins, n_cells))

    fig = go.Figure()
    if len(X_list) < 2:
        fig.add_annotation(
            text="Not enough events in view for PCA (need ≥2 events per opponent).",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=13),
        )
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
        return fig

    X = np.vstack(X_list)
    pca = PCA(n_components=3)
    X_red = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    idx = 0
    for i, label in enumerate(valid_labels):
        sub = subset_data[label]
        n_sel, n_cells, n_bins = sub.shape
        n_pts = n_sel * n_bins
        mean_traj = X_red[idx:idx + n_pts].reshape(n_sel, n_bins, 3).mean(axis=0)
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter3d(
            x=mean_traj[:, 0], y=mean_traj[:, 1], z=mean_traj[:, 2],
            mode="lines+markers", name=str(label),
            line=dict(color=color, width=5), marker=dict(size=3, color=color),
        ))
        fig.add_trace(go.Scatter3d(
            x=[mean_traj[0, 0]], y=[mean_traj[0, 1]], z=[mean_traj[0, 2]],
            mode="markers", showlegend=False,
            marker=dict(size=9, color=color, symbol="diamond"),
        ))
        idx += n_pts

    fig.update_layout(
        scene=dict(
            xaxis_title=f"PC1 ({var[0]:.1f}%)",
            yaxis_title=f"PC2 ({var[1]:.1f}%)",
            zaxis_title=f"PC3 ({var[2]:.1f}%)",
        ),
        title=f"Mean population trajectories — {sum(len(v) for v in subset_data.values())} events in view",
        height=420, legend=dict(x=0.01, y=0.99), margin=dict(l=0, r=0, t=40, b=0),
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
            value=BTYPE_LABELS[0],
        )
        self.min_events_sl = pn.widgets.IntSlider(
            name="Min events / opponent", value=3, start=1, end=30
        )
        self.pca_tw_start = pn.widgets.FloatSlider(
            name="PCA window start (s)", value=-1.0, start=-10.0, end=0.0, step=0.25
        )
        self.pca_tw_end = pn.widgets.FloatSlider(
            name="PCA window end (s)", value=2.0, start=0.0, end=10.0, step=0.25
        )
        self.pca_bin_sl = pn.widgets.FloatSlider(
            name="PCA bin size (s)", value=0.2, start=0.05, end=1.0, step=0.05
        )
        self.raster_bin_sl = pn.widgets.FloatSlider(
            name="Rastermap bin size (s)", value=0.1, start=0.02, end=1.0, step=0.02
        )

        self.cohort_sel.param.watch(self._update_sessions, "value")
        self.session_sel.param.watch(self._update_animals, "value")
        self.load_btn.on_click(self._on_load)
        self.btype_sel.param.watch(self._on_behavior_change, "value")
        self.min_events_sl.param.watch(self._on_behavior_change, "value")

        self._update_sessions()
        pn.state.add_periodic_callback(self._check_range_update, period=600)

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
            self._ks_data = KilosortData(dsm)
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

        self._refresh_behavior()
        self._loading = False

    # ── Refresh: rebuild timeline + PCA, reuse rastermap image ───────────────

    def _refresh_behavior(self):
        animal_id = self.animal_sel.value
        btype = _label_to_abbrev(self.btype_sel.value)
        pca_tw = (self.pca_tw_start.value, self.pca_tw_end.value)

        # Preserve current zoom; fall back to full range on first load
        if self._x_range is not None:
            cur_start = float(self._x_range.start)
            cur_end = float(self._x_range.end)
        else:
            cur_start, cur_end = self._t0, self._t1

        self._full_pop = _build_pop_data(
            self._events, self._ks_data, btype, animal_id,
            pca_tw, self.pca_bin_sl.value, self.min_events_sl.value,
        )

        # ── Build Bokeh figures ────────────────────────────────────────────────
        # Timeline: owns the Range1d
        p_tl = self._make_timeline(btype, animal_id, cur_start, cur_end)
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

    def _make_timeline(self, btype, animal_id, cur_start, cur_end):
        df = self._events.events_data.copy()
        df = df[df["type"] == btype]
        df = df[(df["initiator"] == animal_id) | (df["victim"] == animal_id)]
        df = df.dropna(subset=["ts_start_ephys", "initiator", "victim"]).reset_index(drop=True)

        all_rats = sorted(set(df["initiator"].tolist() + df["victim"].tolist()))
        rat_to_y = {r: i for i, r in enumerate(all_rats)}
        df["y_init"] = df["initiator"].map(rat_to_y).fillna(0).astype(float)
        df["y_vic"] = df["victim"].map(rat_to_y).fillna(0).astype(float)

        def opponent_color(row):
            opp = row["victim"] if row["initiator"] == animal_id else row["initiator"]
            return PALETTE[all_rats.index(opp) % 10]
        df["color"] = df.apply(opponent_color, axis=1)

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
        return _make_pca_plotly(
            self._full_pop["pop_data"],
            self._full_pop["event_starts"],
            t_start, t_end,
        )

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
            self.pca_tw_start,
            self.pca_tw_end,
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
