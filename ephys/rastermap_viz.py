"""
Rastermap Visualization Module

Provides functions to sort and visualize neural population activity using
Rastermap (Stringer et al., 2024, Nature Neuroscience).

Requires: pip install rastermap
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.kilosort_data_import import KilosortData
    from video.behavioral_events import BehavioralEventsData


def bin_spikes_matrix(ks_data: "KilosortData",
                      bin_size: float = 0.5,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      filtered_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert per-cell spike times into a neurons × time-bins firing-rate matrix.

    Parameters
    ----------
    ks_data : KilosortData
        Must have ``spike_times_by_cell`` populated.
    bin_size : float
        Bin width in seconds (default 500 ms).
    start_time : float, optional
        Start of the time range.  Defaults to the earliest spike.
    end_time : float, optional
        End of the time range.  Defaults to the latest spike.
    filtered_only : bool
        If True, include only cells that passed
        ``filter_cells_by_firing_patterns()``.  If ``filter_results``
        is not yet populated, runs it with default parameters
        automatically.

    Returns
    -------
    spks : np.ndarray, shape (n_cells, n_bins)
        Firing-rate matrix (spikes / s).
    bin_centers : np.ndarray, shape (n_bins,)
        Time of each bin center in seconds.
    """
    spike_times_list = ks_data.spike_times_by_cell

    if filtered_only:
        if not hasattr(ks_data, 'filter_results') or ks_data.filter_results is None:
            ks_data.filter_cells_by_firing_patterns()
        passed_ids = set(ks_data.filter_results['passed_clusters'])
        mask = [i for i, cid in enumerate(ks_data.ks_ids) if cid in passed_ids]
        spike_times_list = [spike_times_list[i] for i in mask]

    if start_time is None:
        start_time = min(
            (st[0] for st in spike_times_list if len(st) > 0), default=0.0
        )
    if end_time is None:
        end_time = max(
            (st[-1] for st in spike_times_list if len(st) > 0), default=1.0
        )

    bin_edges = np.arange(start_time, end_time + bin_size, bin_size)
    n_bins = len(bin_edges) - 1
    n_cells = len(spike_times_list)

    spks = np.zeros((n_cells, n_bins), dtype=np.float32)
    for i, st in enumerate(spike_times_list):
        if len(st) > 0:
            counts, _ = np.histogram(st, bins=bin_edges)
            spks[i] = counts / bin_size  # convert to Hz

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return spks, bin_centers


def run_rastermap(spks: np.ndarray,
                  n_PCs: int = 200,
                  n_clusters: int = 100,
                  locality: float = 0.75,
                  time_lag_window: int = 5,
                  **kwargs):
    """
    Fit a Rastermap model on a firing-rate matrix.

    Parameters
    ----------
    spks : np.ndarray, shape (n_cells, n_bins)
        Firing-rate (or z-scored) matrix.
    n_PCs, n_clusters, locality, time_lag_window
        Core Rastermap hyper-parameters (see rastermap docs).
    **kwargs
        Additional keyword arguments forwarded to ``Rastermap()``.

    Returns
    -------
    model : rastermap.Rastermap
        Fitted model.  Key attributes: ``isort``, ``embedding``,
        ``X_embedding``.
    """
    from rastermap import Rastermap

    model = Rastermap(
        n_PCs=n_PCs,
        n_clusters=n_clusters,
        locality=locality,
        time_lag_window=time_lag_window,
        **kwargs,
    ).fit(spks)
    return model


def plot_rastermap(ks_data: "KilosortData",
                   bin_size: float = 0.5,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   n_PCs: int = 200,
                   n_clusters: int = 100,
                   locality: float = 0.75,
                   time_lag_window: int = 5,
                   vmin: float = 0,
                   vmax: float = 1.5,
                   cmap: str = "gray_r",
                   figsize: Tuple[float, float] = (14, 6),
                   event_times: Optional[np.ndarray] = None,
                   event_color: str = "red",
                   **rastermap_kwargs) -> Tuple[plt.Figure, "object"]:
    """
    One-call rastermap visualization for a KilosortData session.

    Bins spike times, runs Rastermap, and displays the sorted activity
    heatmap.  Optionally overlays vertical lines at behavioural event times.

    Parameters
    ----------
    ks_data : KilosortData
        Kilosort session data with spike_times_by_cell.
    bin_size : float
        Bin width in seconds (default 500 ms).
    start_time, end_time : float, optional
        Time range to visualise.
    n_PCs, n_clusters, locality, time_lag_window
        Rastermap parameters.
    vmin, vmax : float
        Colour limits for the heatmap.
    cmap : str
        Matplotlib colormap name.
    figsize : tuple
        Figure size.
    event_times : np.ndarray, optional
        Times (in seconds) at which to draw vertical marker lines.
    event_color : str
        Colour of event marker lines.
    **rastermap_kwargs
        Extra arguments forwarded to ``Rastermap()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    model : rastermap.Rastermap
        The fitted model (for further inspection / re-use).
    """
    # 1. Bin spikes
    spks, bin_centers = bin_spikes_matrix(ks_data, bin_size, start_time, end_time)

    # 2. Fit rastermap
    model = run_rastermap(
        spks,
        n_PCs=min(n_PCs, spks.shape[0] - 1),  # can't exceed n_cells-1
        n_clusters=min(n_clusters, spks.shape[0]),
        locality=locality,
        time_lag_window=time_lag_window,
        **rastermap_kwargs,
    )

    # 3. Use the binned embedding produced by rastermap when available
    X_plot = model.X_embedding if hasattr(model, "X_embedding") and model.X_embedding is not None else spks[model.isort]

    # 4. Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        X_plot,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="auto",
        extent=[bin_centers[0], bin_centers[-1], X_plot.shape[0], 0],
    )

    # Overlay event markers
    if event_times is not None:
        for t in event_times:
            ax.axvline(t, color=event_color, linewidth=0.5, alpha=0.6)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neurons (rastermap sorted)")
    ax.set_title(
        f"Rastermap — {ks_data.animal_id} / {ks_data.session_id}  "
        f"({spks.shape[0]} cells, bin={bin_size*1000:.0f} ms)"
    )
    plt.tight_layout()
    return fig, model


def plot_rastermap_interactive(ks_data: "KilosortData",
                               bin_size: float = 0.5,
                               start_time: Optional[float] = None,
                               end_time: Optional[float] = None,
                               n_PCs: int = 200,
                               n_clusters: int = 100,
                               locality: float = 0.75,
                               time_lag_window: int = 5,
                               zmin: float = 0,
                               zmax: float = 1.5,
                               colorscale: str = "Greys",
                               width: int = 1200,
                               height: int = 500,
                               event_times: Optional[np.ndarray] = None,
                               event_color: str = "red",
                               **rastermap_kwargs):
    """
    Interactive rastermap visualization with horizontal zoom and pan.

    Same computation as ``plot_rastermap`` but rendered with Plotly so the
    user can zoom and pan along the time axis.

    Parameters
    ----------
    ks_data : KilosortData
        Kilosort session data with spike_times_by_cell.
    bin_size : float
        Bin width in seconds.
    start_time, end_time : float, optional
        Time range to visualise.
    n_PCs, n_clusters, locality, time_lag_window
        Rastermap parameters.
    zmin, zmax : float
        Colour limits for the heatmap.
    colorscale : str
        Plotly colorscale name (default ``"Greys"`` to mimic ``gray_r``).
    width, height : int
        Figure dimensions in pixels.
    event_times : np.ndarray, optional
        Times (in seconds) at which to draw vertical marker lines.
    event_color : str
        Colour of event marker lines.
    **rastermap_kwargs
        Extra arguments forwarded to ``Rastermap()``.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    model : rastermap.Rastermap
    """
    import plotly.graph_objects as go

    # 1. Bin spikes
    spks, bin_centers = bin_spikes_matrix(ks_data, bin_size, start_time, end_time)

    # 2. Fit rastermap
    model = run_rastermap(
        spks,
        n_PCs=min(n_PCs, spks.shape[0] - 1),
        n_clusters=min(n_clusters, spks.shape[0]),
        locality=locality,
        time_lag_window=time_lag_window,
        **rastermap_kwargs,
    )

    # 3. Sorted matrix
    X_plot = model.X_embedding if hasattr(model, "X_embedding") and model.X_embedding is not None else spks[model.isort]

    # 4. Interactive heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=X_plot,
            x=bin_centers,
            y=np.arange(X_plot.shape[0]),
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            colorbar=dict(title="Hz"),
        )
    )

    # Overlay event markers
    if event_times is not None:
        for t in np.asarray(event_times):
            fig.add_vline(x=t, line_color=event_color, line_width=1, opacity=0.6)

    fig.update_layout(
        title=(
            f"Rastermap — {ks_data.animal_id} / {ks_data.session_id}  "
            f"({spks.shape[0]} cells, bin={bin_size*1000:.0f} ms)"
        ),
        xaxis_title="Time (s)",
        yaxis_title="Neurons (rastermap sorted)",
        yaxis=dict(autorange="reversed"),
        width=width,
        height=height,
        dragmode="zoom",
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
        ),
    )

    return fig, model


def plot_rastermap_with_events(ks_data: "KilosortData",
                               behavior_data: "BehavioralEventsData",
                               animal_of_interest: str,
                               behavior_type: Optional[str] = None,
                               bin_size: float = 0.5,
                               time_window: Tuple[float, float] = (-2.0, 4.0),
                               filtered_only: bool = True,
                               n_PCs: int = 200,
                               n_clusters: int = 100,
                               locality: float = 0.75,
                               time_lag_window: int = 5,
                               vmin: float = 0,
                               vmax: float = 1.5,
                               cmap: str = "gray_r",
                               figsize: Tuple[float, float] = (14, 6),
                               **rastermap_kwargs) -> Tuple[plt.Figure, "object"]:
    """
    Rastermap visualisation of event-aligned population activity.

    Extracts a trial-averaged (or concatenated) firing-rate matrix centred on
    behavioural events, then sorts neurons with Rastermap.

    Parameters
    ----------
    ks_data : KilosortData
    behavior_data : BehavioralEventsData
        Must be synchronised with ephys timestamps already.
    animal_of_interest : str
        Animal id (e.g. '631').
    behavior_type : str, optional
        Behaviour abbreviation filter (e.g. 'EC').
    bin_size : float
        Bin width in seconds.
    time_window : tuple
        (pre, post) in seconds around each event start.
    n_PCs, n_clusters, locality, time_lag_window
        Rastermap parameters.
    vmin, vmax, cmap, figsize
        Plotting parameters.
    **rastermap_kwargs
        Extra arguments forwarded to ``Rastermap()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    model : rastermap.Rastermap
    """
    # Extract event times
    event_starts, _, opponent_labels = behavior_data.extract_opponent_labels(
        animal_of_interest=animal_of_interest,
        behavior_type=behavior_type,
        min_events_per_class=1,
    )

    if event_starts is None or len(event_starts) == 0:
        print("No events found.")
        return None, None

    # Sort events by opponent label so trials are grouped
    sort_idx = np.argsort(opponent_labels)
    event_starts = event_starts[sort_idx]
    opponent_labels = opponent_labels[sort_idx]

    # Build concatenated firing-rate matrix (all trials side by side)
    bin_edges = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
    n_bins = len(bin_edges) - 1

    spike_times_list = ks_data.spike_times_by_cell
    if filtered_only:
        if not hasattr(ks_data, 'filter_results') or ks_data.filter_results is None:
            ks_data.filter_cells_by_firing_patterns()
        passed_ids = set(ks_data.filter_results['passed_clusters'])
        cell_mask = [i for i, cid in enumerate(ks_data.ks_ids) if cid in passed_ids]
        spike_times_list = [spike_times_list[i] for i in cell_mask]

    n_cells = len(spike_times_list)
    n_events = len(event_starts)
    concat_matrix = np.zeros((n_cells, n_bins * n_events), dtype=np.float32)

    for trial_idx, event_t in enumerate(event_starts):
        col_start = trial_idx * n_bins
        col_end = col_start + n_bins
        for i, st in enumerate(spike_times_list):
            rel = st[(st >= event_t + time_window[0]) & (st < event_t + time_window[1])] - event_t
            if len(rel) > 0:
                counts, _ = np.histogram(rel, bins=bin_edges)
                concat_matrix[i, col_start:col_end] = counts / bin_size

    # Fit rastermap on concatenated data
    model = run_rastermap(
        concat_matrix,
        n_PCs=min(n_PCs, n_cells - 1),
        n_clusters=min(n_clusters, n_cells),
        locality=locality,
        time_lag_window=time_lag_window,
        **rastermap_kwargs,
    )

    X_plot = model.X_embedding if hasattr(model, "X_embedding") and model.X_embedding is not None else concat_matrix[model.isort]

    # Build x-axis: concatenated trial time
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_bins = n_bins * n_events

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        X_plot,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="auto",
        extent=[0, total_bins, X_plot.shape[0], 0],
    )

    # Draw vertical lines at event onset (t=0) within each trial
    for trial_idx in range(n_events):
        onset_bin = trial_idx * n_bins + np.searchsorted(bin_centers, 0)
        ax.axvline(onset_bin, color="red", linewidth=0.5, linestyle="--", alpha=0.6)

    # Draw opponent-colored trial boundaries and top color bar
    unique_opponents = np.unique(opponent_labels)
    cmap_tab = plt.cm.get_cmap("tab10", len(unique_opponents))
    opp_color_map = {opp: cmap_tab(i) for i, opp in enumerate(unique_opponents)}

    for trial_idx in range(n_events):
        col = opp_color_map[opponent_labels[trial_idx]]
        x0 = trial_idx * n_bins
        x1 = (trial_idx + 1) * n_bins
        # Color bar along the top
        ax.axhspan(-X_plot.shape[0] * 0.02, 0, xmin=x0 / total_bins, xmax=x1 / total_bins,
                    color=col, alpha=0.8, clip_on=False)

    # Draw trial boundary lines
    for trial_idx in range(1, n_events):
        boundary = trial_idx * n_bins
        ax.axvline(boundary, color="white", linewidth=0.3, alpha=0.4)

    # Legend for opponents
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=opp_color_map[opp], label=opp) for opp in unique_opponents]
    ax.legend(handles=legend_handles, loc="upper right", fontsize="small", title="Opponent")

    type_label = f" ({behavior_type})" if behavior_type else ""
    ax.set_xlabel("Concatenated trials (bins)")
    ax.set_ylabel("Neurons (rastermap sorted)")
    ax.set_title(
        f"Rastermap — event-aligned (concatenated){type_label}  "
        f"{ks_data.animal_id} / {ks_data.session_id}  "
        f"({n_cells} cells, {n_events} events, bin={bin_size*1000:.0f} ms)"
    )
    plt.tight_layout()
    return fig, model
