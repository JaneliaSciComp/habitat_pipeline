# Run from project root: panel serve gui/interactive_app.py --show
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import panel as pn
from bokeh.layouts import column as bk_col
from bokeh.models import (
    BoxSelectTool, ColumnDataSource, FixedTicker, HoverTool, Span,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure

from ephys.decode_opponent_identity import align_spikes_to_events, extract_firing_rate_features
from gui.cache import cache_path, load_cache
from ingestion.data_paths import DataStorageManager, get_animals_and_sessions
from ingestion.ephys_sync import DataSyncManager
from ingestion.kilosort_data_import import KilosortData
from video.behavioral_events import BehavioralEventsData

pn.extension()

CONFIG_OPTIONS = {
    "Cohort 7 (default)": None,
    "Cohort 5": "cohort5_paths.json",
}
PALETTE = Category10[10]
BTYPE_OPTIONS = {
    f"{k} — {v}": k for k, v in BehavioralEventsData.BEHAVIOR_TYPES.items()
}


# ── PETH computation ───────────────────────────────────────────────────────────

def _compute_peth_data(ks_data, cluster_id, event_times, time_window, bin_size):
    """Return a ColumnDataSource-compatible dict for one cell's PETH."""
    edges = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
    n_bins = len(edges) - 1
    centers = (edges[:-1] + edges[1:]) / 2
    zeros = {
        "x": centers,
        "y": np.zeros(n_bins),
        "y_upper": np.zeros(n_bins),
        "y_lower": np.zeros(n_bins),
    }

    if len(event_times) == 0:
        return zeros

    try:
        cell_idx = list(ks_data.ks_ids).index(cluster_id)
    except ValueError:
        return zeros

    spike_times = ks_data.spike_times_by_cell[cell_idx]
    aligned = align_spikes_to_events(spike_times, np.asarray(event_times), time_window)
    fr_matrix = extract_firing_rate_features(aligned, time_window, bin_size)
    # fr_matrix: (n_events, n_bins_computed) — trim/pad to n_bins
    n_cols = min(n_bins, fr_matrix.shape[1])
    mean_fr = np.zeros(n_bins)
    sem_fr = np.zeros(n_bins)
    mean_fr[:n_cols] = fr_matrix[:, :n_cols].mean(axis=0)
    if len(event_times) > 1:
        sem_fr[:n_cols] = fr_matrix[:, :n_cols].std(axis=0) / np.sqrt(len(event_times))

    return {
        "x": centers,
        "y": mean_fr,
        "y_upper": mean_fr + sem_fr,
        "y_lower": mean_fr - sem_fr,
    }


# ── App class ──────────────────────────────────────────────────────────────────

class HabitatApp:
    def __init__(self):
        # Loaded data
        self._ks_data = None
        self._events = None
        self._top_cells = []        # list of cluster_ids ordered by accuracy
        self._timeline_source = None
        self._peth_sources = {}     # {cluster_id: ColumnDataSource}
        self._all_event_times = None

        # Main content area — replaced after loading
        self._content = pn.Column(
            pn.pane.Alert(
                "Select a session and animal in the sidebar, "
                "then press **Load Session**.",
                alert_type="info",
            )
        )

        # ── Sidebar widgets ────────────────────────────────────────────────
        self.cohort_sel = pn.widgets.Select(
            name="Cohort", options=list(CONFIG_OPTIONS.keys())
        )
        self.session_sel = pn.widgets.Select(name="Session", options=[])
        self.animal_sel = pn.widgets.Select(name="Animal", options=[])

        self.btype_sel = pn.widgets.Select(
            name="Behavior type", options=BTYPE_OPTIONS, value="EC"
        )
        self.n_cells_sl = pn.widgets.IntSlider(
            name="Top N cells", value=5, start=1, end=20
        )
        self.tw_start_sl = pn.widgets.FloatSlider(
            name="PETH start (s)", value=-1.0, start=-10.0, end=0.0, step=0.25
        )
        self.tw_end_sl = pn.widgets.FloatSlider(
            name="PETH end (s)", value=2.0, start=0.0, end=10.0, step=0.25
        )
        self.bin_sl = pn.widgets.FloatSlider(
            name="Bin size (s)", value=0.2, start=0.05, end=2.0, step=0.05
        )
        self.load_btn = pn.widgets.Button(
            name="Load Session", button_type="primary", width=220
        )

        # ── Wire callbacks ─────────────────────────────────────────────────
        self.cohort_sel.param.watch(self._update_sessions, "value")
        self.session_sel.param.watch(self._update_animals, "value")
        self.btype_sel.param.watch(self._on_behavior_change, "value")
        self.load_btn.on_click(self._on_load)

        # Populate initial dropdowns
        self._update_sessions()

    # ── Session / animal population ────────────────────────────────────────────

    def _update_sessions(self, *args):
        config_path = CONFIG_OPTIONS[self.cohort_sel.value]
        try:
            manifest = get_animals_and_sessions(config_path=config_path)
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
        config_path = CONFIG_OPTIONS[self.cohort_sel.value]
        try:
            manifest = get_animals_and_sessions(config_path=config_path)
            animals = sorted(
                manifest.loc[manifest["session"] == session, "animal"].tolist()
            )
        except Exception:
            animals = []
        self.animal_sel.options = animals
        if animals:
            self.animal_sel.value = animals[0]

    # ── Load button ────────────────────────────────────────────────────────────

    def _on_load(self, event):
        self._content[:] = [pn.pane.Alert("Loading data…", alert_type="warning")]
        config_path = CONFIG_OPTIONS[self.cohort_sel.value]
        animal_id = self.animal_sel.value
        session_id = self.session_sel.value

        try:
            dsm = DataStorageManager(
                animal_id, session_id, config_path=config_path, auto_load=True
            )
            self._ks_data = KilosortData(dsm)
            self._events = BehavioralEventsData(dsm, auto_load=True)
            sync = DataSyncManager(dsm, dio_channel=1, auto_load=True)
            self._events.synchronize_with_ephys(sync, create_new_columns=True)
        except Exception as e:
            self._content[:] = [
                pn.pane.Alert(f"Failed to load session: {e}", alert_type="danger")
            ]
            return

        self._load_top_cells()
        self._build_plots()

    # ── Cell ranking ───────────────────────────────────────────────────────────

    def _load_top_cells(self):
        """Rank cells by decoding accuracy (from cache) or spike count (fallback)."""
        config_path = CONFIG_OPTIONS[self.cohort_sel.value]
        tw = (self.tw_start_sl.value, self.tw_end_sl.value)
        decode_params = dict(
            behavior_type=self.btype_sel.value,
            time_window=tw,
            time_bin_size=self.bin_sl.value,
            cv_folds=5,
            min_events_per_class=5,
        )
        pkl = cache_path(
            "decoding",
            self.animal_sel.value,
            self.session_sel.value,
            config_path,
            decode_params,
        )
        results = load_cache(pkl)
        n = self.n_cells_sl.value

        if results and "successful_cells" in results:
            cell_accs = [
                (cid, results["cell_results"][cid].get("accuracy", np.nan))
                for cid in results["successful_cells"]
            ]
            cell_accs = [(cid, acc) for cid, acc in cell_accs if not np.isnan(acc)]
            self._top_cells = [
                cid
                for cid, _ in sorted(cell_accs, key=lambda x: x[1], reverse=True)[:n]
            ]
        else:
            # Fallback: sort by total spike count (proxy for firing rate)
            ks = self._ks_data
            cell_counts = [
                (cid, len(ks.spike_times_by_cell[i]))
                for i, cid in enumerate(ks.ks_ids)
            ]
            self._top_cells = [
                cid
                for cid, _ in sorted(cell_counts, key=lambda x: x[1], reverse=True)[:n]
            ]

    # ── Plot construction ──────────────────────────────────────────────────────

    def _get_events_df(self):
        df = self._events.events_data.copy()
        btype = self.btype_sel.value
        if btype:
            df = df[df["type"] == btype]
        return df.dropna(subset=["ts_start_ephys"]).reset_index(drop=True)

    def _build_plots(self):
        events_df = self._get_events_df()

        if events_df.empty:
            self._content[:] = [
                pn.pane.Alert(
                    f"No {self.btype_sel.value!r} events with ephys timestamps "
                    "found for this session.",
                    alert_type="warning",
                )
            ]
            return

        # ── Rat y-axis ─────────────────────────────────────────────────────
        all_rats = sorted(
            set(events_df["initiator"].tolist() + events_df["victim"].tolist())
        )
        rat_to_y = {r: i for i, r in enumerate(all_rats)}
        events_df["y_init"] = events_df["initiator"].map(rat_to_y).fillna(0)
        events_df["y_vic"] = events_df["victim"].map(rat_to_y).fillna(0)

        # ── Color by event type ────────────────────────────────────────────
        etypes = sorted(events_df["type"].unique())
        color_map = {t: PALETTE[i % 10] for i, t in enumerate(etypes)}
        events_df["color"] = events_df["type"].map(color_map)

        self._timeline_source = ColumnDataSource(events_df)
        self._all_event_times = events_df["ts_start_ephys"].values

        # ── Timeline figure ────────────────────────────────────────────────
        p_tl = figure(
            title=(
                f"{self.btype_sel.value} events — "
                f"{self.animal_sel.value} / {self.session_sel.value}  "
                "│  box-select a time window to filter PETH"
            ),
            height=300,
            sizing_mode="stretch_width",
            tools="reset,wheel_zoom,pan",
            x_axis_label="Time (s, ephys clock)",
            y_range=(-0.5, len(all_rats) - 0.5),
        )
        box_select = BoxSelectTool(dimensions="width")
        p_tl.add_tools(box_select)
        p_tl.toolbar.active_drag = box_select

        p_tl.yaxis.ticker = FixedTicker(ticks=list(range(len(all_rats))))
        p_tl.yaxis.major_label_overrides = {i: r for r, i in rat_to_y.items()}

        p_tl.add_tools(
            HoverTool(
                tooltips=[
                    ("Time (s)", "@ts_start_ephys{0.1f}"),
                    ("Type", "@type"),
                    ("Initiator", "@initiator"),
                    ("Victim", "@victim"),
                ]
            )
        )

        # Segment: initiator → victim
        p_tl.segment(
            x0="ts_start_ephys", x1="ts_start_ephys",
            y0="y_init", y1="y_vic",
            source=self._timeline_source,
            line_color="grey", line_width=1, line_alpha=0.5,
        )
        # Initiator: filled circle (selection target)
        p_tl.circle(
            x="ts_start_ephys", y="y_init",
            source=self._timeline_source,
            color="color", size=9, alpha=0.85,
            selection_color="color", selection_alpha=1.0,
            nonselection_alpha=0.15,
        )
        # Victim: hollow circle
        p_tl.circle(
            x="ts_start_ephys", y="y_vic",
            source=self._timeline_source,
            color="white", size=9, alpha=0.9,
            line_color="color", line_width=1.5,
            selection_color="white", selection_alpha=1.0,
            nonselection_alpha=0.1,
        )

        # ── PETH figure ────────────────────────────────────────────────────
        tw = (self.tw_start_sl.value, self.tw_end_sl.value)
        p_peth = figure(
            title=(
                f"PETH — top {len(self._top_cells)} cells  "
                f"({len(self._all_event_times)} events)"
            ),
            height=350,
            sizing_mode="stretch_width",
            tools="reset,wheel_zoom,pan",
            x_axis_label="Time from event onset (s)",
            y_axis_label="Firing rate (Hz)",
        )
        p_peth.add_layout(
            Span(
                location=0, dimension="height",
                line_color="black", line_dash="dashed", line_width=1,
            )
        )

        self._peth_sources = {}
        for i, cid in enumerate(self._top_cells):
            color = PALETTE[i % 10]
            data = _compute_peth_data(
                self._ks_data, cid, self._all_event_times, tw, self.bin_sl.value
            )
            src = ColumnDataSource(data)
            self._peth_sources[cid] = src
            p_peth.varea(
                x="x", y1="y_lower", y2="y_upper",
                source=src, color=color, alpha=0.2,
            )
            p_peth.line(
                x="x", y="y", source=src,
                color=color, line_width=2,
                legend_label=f"cell {cid}",
            )

        if self._top_cells:
            p_peth.legend.location = "top_right"
            p_peth.legend.click_policy = "hide"

        # ── Selection callback (timeline → PETH) ───────────────────────────
        # Capture local references so the closure doesn't depend on self mutating
        peth_sources = self._peth_sources
        all_event_times = self._all_event_times
        ks_data_ref = self._ks_data

        def on_selection_change(attr, old, new):
            if new:
                sel_times = self._timeline_source.data["ts_start_ephys"][
                    np.array(new)
                ]
            else:
                sel_times = all_event_times

            n_events = len(sel_times)
            p_peth.title.text = (
                f"PETH — top {len(peth_sources)} cells  "
                f"({n_events} event{'s' if n_events != 1 else ''} selected)"
            )
            current_tw = (self.tw_start_sl.value, self.tw_end_sl.value)
            for cid, src in peth_sources.items():
                src.data = _compute_peth_data(
                    ks_data_ref, cid, sel_times, current_tw, self.bin_sl.value
                )

        self._timeline_source.selected.on_change("indices", on_selection_change)

        # Both figures share one Bokeh document via bk_col wrapper
        combined = bk_col(p_tl, p_peth, sizing_mode="stretch_width")
        self._content[:] = [pn.pane.Bokeh(combined)]

    # ── Behavior type change ───────────────────────────────────────────────────

    def _on_behavior_change(self, *args):
        if self._ks_data is not None and self._events is not None:
            self._load_top_cells()
            self._build_plots()

    # ── Layout ─────────────────────────────────────────────────────────────────

    @property
    def layout(self):
        sidebar = pn.Column(
            "## Session",
            self.cohort_sel,
            self.session_sel,
            self.animal_sel,
            pn.layout.Divider(),
            "## Parameters",
            self.btype_sel,
            self.n_cells_sl,
            self.tw_start_sl,
            self.tw_end_sl,
            self.bin_sl,
            pn.pane.Markdown(
                "_Tip: adjust params then press **Load Session** to apply._",
                styles={"font-size": "12px", "color": "#888"},
            ),
            pn.layout.Divider(),
            self.load_btn,
            width=280,
        )
        return pn.template.FastListTemplate(
            title="Habitat Pipeline — Interactive",
            sidebar=[sidebar],
            main=[self._content],
        )


HabitatApp().layout.servable()
