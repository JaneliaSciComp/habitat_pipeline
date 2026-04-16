"""
Location Decoding Module

Decode the position of self or other animals from neural population activity.
Uses binned firing rates as features and supports multiple decoder types
(Bayesian, linear, MLP).

Typical workflow
----------------
1. Align neural activity and tracking positions in a common time base.
2. Bin spikes into a firing-rate matrix; down-sample tracking to match.
3. Train a decoder (cross-validated) to predict (x, y) from firing rates.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.kilosort_data_import import KilosortData
    from video.tracking_import import VideoTrackingData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_filtered_spike_times(ks_data: "KilosortData",
                              filtered_only: bool = True) -> List[np.ndarray]:
    """Return spike-time arrays, optionally restricted to quality cells."""
    spike_times_list = ks_data.spike_times_by_cell
    if filtered_only:
        if not hasattr(ks_data, 'filter_results') or ks_data.filter_results is None:
            ks_data.filter_cells_by_firing_patterns()
        passed_ids = set(ks_data.filter_results['passed_clusters'])
        mask = [i for i, cid in enumerate(ks_data.ks_ids) if cid in passed_ids]
        spike_times_list = [spike_times_list[i] for i in mask]
    return spike_times_list


def build_binned_data(ks_data: "KilosortData",
                      tracking_data: "VideoTrackingData",
                      object_name: str,
                      bin_size: float = 0.5,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      filtered_only: bool = True,
                      ) -> Dict:
    """
    Align neural firing rates and animal positions into matching time bins.

    Parameters
    ----------
    ks_data : KilosortData
        Kilosort session with ``spike_times_by_cell`` populated.
    tracking_data : VideoTrackingData
        Must have timestamps loaded (``tracking.timestamps`` is not None).
    object_name : str
        Tracked object to decode (e.g. ``"rat631"``).
    bin_size : float
        Bin width in seconds.
    start_time, end_time : float, optional
        Restrict the time range.  Defaults to overlapping range of neural
        and tracking data.
    filtered_only : bool
        Use only quality-filtered cells.

    Returns
    -------
    dict with keys
        ``firing_rates`` : np.ndarray, shape (n_bins, n_cells)
        ``positions``    : np.ndarray, shape (n_bins, 2)  – (x, y)
        ``bin_centers``  : np.ndarray, shape (n_bins,)
        ``n_cells``      : int
        ``bin_size``     : float
    """
    spike_times_list = _get_filtered_spike_times(ks_data, filtered_only)

    # Get tracking trajectory
    traj = tracking_data.get_object_trajectory(object_name)
    if traj is None:
        raise ValueError(f"No trajectory found for object '{object_name}'.")

    # Use ephys-synchronised timestamps if available, otherwise raw timestamps
    if 'ephys_timestamps' in traj.columns:
        track_ts = traj['ephys_timestamps'].values
    elif hasattr(tracking_data, 'ephys_timestamps') and tracking_data.ephys_timestamps is not None:
        track_ts = tracking_data.ephys_timestamps[:len(traj)]
    else:
        raise ValueError("VideoTrackingData must be synchronised with ephys first "
                         "(call tracking.synchronize_with_ephys(sync_manager)).")
    track_x = traj['center_x'].values
    track_y = traj['center_y'].values

    # Determine overlapping time range
    neural_min = min((st[0] for st in spike_times_list if len(st) > 0), default=0.0)
    neural_max = max((st[-1] for st in spike_times_list if len(st) > 0), default=1.0)
    print(f"Neural data time range: {neural_min:.2f} to {neural_max:.2f} seconds")
    print(f"Tracking data time range: {track_ts[0]:.2f} to {track_ts[-1]:.2f} seconds")
    track_min, track_max = track_ts[0], track_ts[-1]
    if start_time is None:
        start_time = max(neural_min, track_min)
    if end_time is None:
        end_time = min(neural_max, track_max)

    bin_edges = np.arange(start_time, end_time + bin_size, bin_size)
    n_bins = len(bin_edges) - 1
    n_cells = len(spike_times_list)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Bin spikes → firing rates (n_bins, n_cells)
    firing_rates = np.zeros((n_bins, n_cells), dtype=np.float32)
    for j, st in enumerate(spike_times_list):
        if len(st) > 0:
            counts, _ = np.histogram(st, bins=bin_edges)
            firing_rates[:, j] = counts / bin_size

    # Bin positions → mean (x, y) per bin
    positions = np.zeros((n_bins, 2), dtype=np.float64)
    for b in range(n_bins):
        mask = (track_ts >= bin_edges[b]) & (track_ts < bin_edges[b + 1])
        if mask.any():
            positions[b, 0] = np.nanmean(track_x[mask])
            positions[b, 1] = np.nanmean(track_y[mask])
        else:
            positions[b] = np.nan

    # Drop bins with missing position data
    valid = ~np.isnan(positions[:, 0])
    firing_rates = firing_rates[valid]
    positions = positions[valid]
    bin_centers = bin_centers[valid]

    return {
        'firing_rates': firing_rates,
        'positions': positions,
        'bin_centers': bin_centers,
        'n_cells': n_cells,
        'bin_size': bin_size,
    }


# ---------------------------------------------------------------------------
# Single-cell location decoding
# ---------------------------------------------------------------------------

def decode_location_single_cell(spike_times: np.ndarray,
                                track_ts: np.ndarray,
                                track_x: np.ndarray,
                                track_y: np.ndarray,
                                bin_size: float = 0.5,
                                cv_folds: int = 5,
                                decoder: str = 'ridge',
                                ) -> Dict:
    """
    Decode (x, y) position from a single neuron's firing rate.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times in seconds for one cell.
    track_ts, track_x, track_y : np.ndarray
        Tracking timestamps and positions (same length).
    bin_size : float
        Bin width in seconds.
    cv_folds : int
        Number of cross-validation folds.
    decoder : str
        ``'ridge'`` (default) or ``'linear'``.

    Returns
    -------
    dict
        ``r2_mean``, ``r2_std``, ``r2_x``, ``r2_y``, ``cv_scores``,
        ``status``
    """
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.multioutput import MultiOutputRegressor

    # Determine time range
    t_min = max(spike_times[0] if len(spike_times) > 0 else track_ts[0], track_ts[0])
    t_max = min(spike_times[-1] if len(spike_times) > 0 else track_ts[-1], track_ts[-1])
    bin_edges = np.arange(t_min, t_max + bin_size, bin_size)
    n_bins = len(bin_edges) - 1

    # Firing rate
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    fr = (counts / bin_size).reshape(-1, 1)

    # Mean position per bin
    positions = np.zeros((n_bins, 2))
    for b in range(n_bins):
        mask = (track_ts >= bin_edges[b]) & (track_ts < bin_edges[b + 1])
        if mask.any():
            positions[b, 0] = np.nanmean(track_x[mask])
            positions[b, 1] = np.nanmean(track_y[mask])
        else:
            positions[b] = np.nan

    valid = ~np.isnan(positions[:, 0])
    fr = fr[valid]
    positions = positions[valid]

    if len(fr) < cv_folds * 2:
        return {'status': 'insufficient_data', 'r2_mean': np.nan}

    scaler = StandardScaler()
    fr_scaled = scaler.fit_transform(fr)

    if decoder == 'ridge':
        model = MultiOutputRegressor(Ridge(alpha=1.0))
    else:
        model = MultiOutputRegressor(LinearRegression())

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, fr_scaled, positions, cv=kf, scoring='r2')

    # Per-axis R²
    model.fit(fr_scaled, positions)
    pred = model.predict(fr_scaled)
    ss_res_x = np.sum((positions[:, 0] - pred[:, 0]) ** 2)
    ss_tot_x = np.sum((positions[:, 0] - positions[:, 0].mean()) ** 2)
    ss_res_y = np.sum((positions[:, 1] - pred[:, 1]) ** 2)
    ss_tot_y = np.sum((positions[:, 1] - positions[:, 1].mean()) ** 2)

    return {
        'r2_mean': float(np.mean(scores)),
        'r2_std': float(np.std(scores)),
        'r2_x': float(1 - ss_res_x / ss_tot_x) if ss_tot_x > 0 else np.nan,
        'r2_y': float(1 - ss_res_y / ss_tot_y) if ss_tot_y > 0 else np.nan,
        'cv_scores': scores,
        'status': 'success',
    }


# ---------------------------------------------------------------------------
# Population location decoding — Bayesian (2-D spatial bins)
# ---------------------------------------------------------------------------

def _build_tuning_curves(firing_rates: np.ndarray,
                         positions: np.ndarray,
                         x_edges: np.ndarray,
                         y_edges: np.ndarray,
                         smoothing_sigma: float = 1.0,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute occupancy-normalised mean firing rate per spatial bin per cell.

    Parameters
    ----------
    firing_rates : (n_time_bins, n_cells)
    positions : (n_time_bins, 2)
    x_edges, y_edges : 1-D bin edges
    smoothing_sigma : Gaussian smoothing (in spatial-bin units).  0 = none.

    Returns
    -------
    tuning : (n_x_bins, n_y_bins, n_cells) – mean Hz per spatial bin
    occupancy : (n_x_bins, n_y_bins) – number of time bins spent in each spatial bin
    """
    from scipy.ndimage import gaussian_filter

    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1
    n_cells = firing_rates.shape[1]

    # Assign each time bin to a spatial bin
    x_idx = np.digitize(positions[:, 0], x_edges) - 1
    y_idx = np.digitize(positions[:, 1], y_edges) - 1
    x_idx = np.clip(x_idx, 0, n_x - 1)
    y_idx = np.clip(y_idx, 0, n_y - 1)

    occupancy = np.zeros((n_x, n_y), dtype=np.float64)
    rate_sum = np.zeros((n_x, n_y, n_cells), dtype=np.float64)

    for t in range(len(positions)):
        xi, yi = x_idx[t], y_idx[t]
        occupancy[xi, yi] += 1
        rate_sum[xi, yi, :] += firing_rates[t]

    # Mean firing rate (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        tuning = rate_sum / occupancy[:, :, np.newaxis]
    tuning[~np.isfinite(tuning)] = 0.0

    # Optional Gaussian smoothing of tuning curves
    if smoothing_sigma > 0:
        for c in range(n_cells):
            tuning[:, :, c] = gaussian_filter(tuning[:, :, c], sigma=smoothing_sigma)

    return tuning, occupancy


def _bayesian_decode(firing_rates: np.ndarray,
                     tuning: np.ndarray,
                     occupancy: np.ndarray,
                     time_bin_size: float,
                     x_centers: np.ndarray,
                     y_centers: np.ndarray,
                     use_occupancy_prior: bool = True,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode position with a Poisson-likelihood Bayesian decoder.

    P(pos | spikes) ∝ P(spikes | pos) · P(pos)

    where P(spikes | pos) = ∏_i  (λ_i · Δt)^n_i · exp(-λ_i · Δt) / n_i!

    In log-space (dropping constant n_i! terms):
        log P ∝ Σ_i [ n_i · log(λ_i · Δt) − λ_i · Δt ] + log P(pos)

    Parameters
    ----------
    firing_rates : (n_time_bins, n_cells) – Hz
    tuning : (n_x, n_y, n_cells) – mean Hz per spatial bin
    occupancy : (n_x, n_y) – time-bin counts
    time_bin_size : float – seconds per time bin
    x_centers, y_centers : 1-D arrays of spatial bin centres
    use_occupancy_prior : bool – weight by occupancy as prior

    Returns
    -------
    predicted_pos : (n_time_bins, 2)
    posterior : (n_time_bins, n_x, n_y) – normalised posterior per time bin
    """
    n_time = firing_rates.shape[0]
    n_x, n_y, n_cells = tuning.shape

    # Expected spike counts per time bin: λ * Δt
    lam_dt = tuning * time_bin_size                     # (n_x, n_y, n_cells)
    log_lam_dt = np.log(np.clip(lam_dt, 1e-10, None))  # avoid log(0)

    # Prior
    if use_occupancy_prior:
        prior = occupancy / occupancy.sum()
        prior = np.clip(prior, 1e-10, None)
    else:
        prior = np.ones((n_x, n_y)) / (n_x * n_y)
    log_prior = np.log(prior)

    # Spike counts per time bin
    spike_counts = firing_rates * time_bin_size  # (n_time, n_cells)

    # Precompute the constant part: -sum(λ*Δt) over cells for each spatial bin
    neg_sum_lam = -lam_dt.sum(axis=2)  # (n_x, n_y)

    predicted_pos = np.zeros((n_time, 2))
    posterior = np.zeros((n_time, n_x, n_y))

    for t in range(n_time):
        n = spike_counts[t]  # (n_cells,)
        # log-likelihood: Σ_i n_i·log(λ_i·Δt) − λ_i·Δt  (per spatial bin)
        log_lik = np.tensordot(log_lam_dt, n, axes=([2], [0])) + neg_sum_lam
        log_post = log_lik + log_prior
        # Normalise in log-space for numerical stability
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= post.sum()
        posterior[t] = post

        # MAP estimate → centre of the winning bin
        best = np.unravel_index(post.argmax(), post.shape)
        predicted_pos[t, 0] = x_centers[best[0]]
        predicted_pos[t, 1] = y_centers[best[1]]

    return predicted_pos, posterior


def decode_location_population(ks_data: "KilosortData",
                               tracking: "VideoTrackingData",
                               object_name: str,
                               bin_size: float = 0.5,
                               n_spatial_bins: int = 20,
                               smoothing_sigma: float = 1.0,
                               use_occupancy_prior: bool = True,
                               start_time: Optional[float] = None,
                               end_time: Optional[float] = None,
                               filtered_only: bool = True,
                               cv_folds: int = 5,
                               ) -> Dict:
    """
    Bayesian decoding of (x, y) position from population firing rates.

    Discretises the arena into a 2-D grid of spatial bins.  For each
    cross-validation fold the decoder learns Poisson tuning curves on the
    training set and produces a posterior probability map on the test set.
    The decoded position is the MAP (maximum a-posteriori) bin centre.

    Parameters
    ----------
    ks_data : KilosortData
    tracking : VideoTrackingData
    object_name : str
        Animal whose position to decode (e.g. ``"rat631"`` for self,
        or another rat's name for other-decoding).
    bin_size : float
        Temporal bin width in seconds.
    n_spatial_bins : int
        Number of bins per spatial dimension (total grid = n² bins).
    smoothing_sigma : float
        Gaussian smoothing of tuning curves in spatial-bin units.
    use_occupancy_prior : bool
        If True, weight posterior by occupancy (empirical spatial prior).
    start_time, end_time : float, optional
        Time range.
    filtered_only : bool
        Use quality-filtered cells only.
    cv_folds : int
        Number of cross-validation folds.

    Returns
    -------
    dict
        ``median_error``, ``mean_error``, ``median_error_per_fold``,
        ``y_true``, ``y_pred``, ``posterior``,
        ``x_centers``, ``y_centers``, ``bin_centers``,
        ``n_cells``, ``n_bins``, ``status``
    """
    from sklearn.model_selection import KFold

    data = build_binned_data(ks_data, tracking, object_name,
                             bin_size, start_time, end_time, filtered_only)
    X = data['firing_rates']   # (n_time, n_cells)
    Y = data['positions']      # (n_time, 2)
    bin_centers = data['bin_centers']

    if len(X) < cv_folds * 2:
        return {'status': 'insufficient_data', 'median_error': np.nan}

    # Spatial grid (shared across folds, fitted to full position range)
    x_edges = np.linspace(np.nanmin(Y[:, 0]), np.nanmax(Y[:, 0]), n_spatial_bins + 1)
    y_edges = np.linspace(np.nanmin(Y[:, 1]), np.nanmax(Y[:, 1]), n_spatial_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Cross-validated decoding
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    Y_pred_all = np.full_like(Y, np.nan)
    posterior_all = np.zeros((len(X), n_spatial_bins, n_spatial_bins))
    fold_errors = []

    for train_idx, test_idx in kf.split(X):
        tuning, occupancy = _build_tuning_curves(
            X[train_idx], Y[train_idx], x_edges, y_edges, smoothing_sigma)

        pred, post = _bayesian_decode(
            X[test_idx], tuning, occupancy, bin_size,
            x_centers, y_centers, use_occupancy_prior)

        Y_pred_all[test_idx] = pred
        posterior_all[test_idx] = post

        err = np.sqrt(np.sum((Y[test_idx] - pred) ** 2, axis=1))
        fold_errors.append(float(np.median(err)))

    errors = np.sqrt(np.sum((Y - Y_pred_all) ** 2, axis=1))

    return {
        'median_error': float(np.median(errors)),
        'mean_error': float(np.mean(errors)),
        'median_error_per_fold': fold_errors,
        'y_true': Y,
        'y_pred': Y_pred_all,
        'posterior': posterior_all,
        'x_edges': x_edges,
        'y_edges': y_edges,
        'x_centers': x_centers,
        'y_centers': y_centers,
        'bin_centers': bin_centers,
        'n_cells': data['n_cells'],
        'n_bins': len(X),
        'n_spatial_bins': n_spatial_bins,
        'decoder': 'bayesian',
        'status': 'success',
    }


# ---------------------------------------------------------------------------
# Decode all animals (self + each opponent)
# ---------------------------------------------------------------------------

def decode_all_locations(ks_data: "KilosortData",
                         tracking: "VideoTrackingData",
                         bin_size: float = 0.5,
                         n_spatial_bins: int = 20,
                         smoothing_sigma: float = 1.0,
                         filtered_only: bool = True,
                         cv_folds: int = 5,
                         ) -> Dict[str, Dict]:
    """
    Decode the position of every tracked animal from the same neural data.

    Useful for comparing self-location decoding accuracy to other-animal
    decoding accuracy.

    Parameters
    ----------
    ks_data : KilosortData
    tracking : VideoTrackingData
    bin_size, n_spatial_bins, smoothing_sigma, filtered_only, cv_folds
        Forwarded to ``decode_location_population``.

    Returns
    -------
    dict
        Keyed by ``object_name``.  Each value is the result dict from
        ``decode_location_population``.
    """
    results = {}
    for obj_name in tracking.get_object_names():
        print(f"Decoding position of {obj_name} ...", end=" ", flush=True)
        res = decode_location_population(
            ks_data, tracking, obj_name,
            bin_size=bin_size, n_spatial_bins=n_spatial_bins,
            smoothing_sigma=smoothing_sigma,
            filtered_only=filtered_only, cv_folds=cv_folds,
        )
        err = res.get('median_error', np.nan)
        print(f"median error = {err:.1f}" if not np.isnan(err) else res.get('status', ''))
        results[obj_name] = res
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_decoding_results(result: Dict,
                          object_name: str = "",
                          figsize: Tuple[float, float] = (16, 5)):
    """
    Visualise Bayesian decoded vs actual position.

    Panels: (1) actual vs decoded trajectory, (2) mean posterior occupancy,
    (3) decoding error over time.

    Parameters
    ----------
    result : dict
        Output of ``decode_location_population``.
    object_name : str
        Label for the plot title.
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if result.get('status') != 'success':
        print(f"Cannot plot: {result.get('status')}")
        return None

    Y = result['y_true']
    Y_hat = result['y_pred']
    t = result['bin_centers']
    errors = np.sqrt(np.sum((Y - Y_hat) ** 2, axis=1))

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Actual vs predicted trajectory
    ax = axes[0]
    ax.plot(Y[:, 0], Y[:, 1], '.', color='gray', markersize=1, alpha=0.4, label='Actual')
    ax.plot(Y_hat[:, 0], Y_hat[:, 1], '.', color='tab:red', markersize=1, alpha=0.4, label='Decoded')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{object_name} – trajectory')
    ax.legend(markerscale=5)
    ax.set_aspect('equal')

    # 2. Mean posterior map
    ax = axes[1]
    mean_post = result['posterior'].mean(axis=0).T  # (n_y, n_x) for imshow
    ax.imshow(mean_post, origin='lower', aspect='auto',
              extent=[result['x_edges'][0], result['x_edges'][-1],
                      result['y_edges'][0], result['y_edges'][-1]],
              cmap='hot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Mean posterior')

    # 3. Error over time
    ax = axes[2]
    ax.plot(t, errors, color='tab:purple', alpha=0.6, linewidth=0.5)
    ax.axhline(result['median_error'], color='red', linestyle='--', linewidth=1,
               label=f'Median = {result["median_error"]:.1f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Euclidean error')
    ax.set_title('Decoding error')
    ax.legend()

    fig.suptitle(f'Bayesian location decoding – {object_name}  ({result["n_cells"]} cells, '
                 f'{result["n_bins"]} time bins, {result["n_spatial_bins"]}² spatial bins)',
                 fontsize=12)
    plt.tight_layout()
    return fig


def plot_all_decoding_summary(all_results: Dict[str, Dict],
                              recording_animal: str = "",
                              figsize: Tuple[float, float] = (8, 5)):
    """
    Bar chart comparing decoding median error across all animals.

    Parameters
    ----------
    all_results : dict
        Output of ``decode_all_locations``.
    recording_animal : str
        Name of the animal carrying the implant (highlighted in the plot).
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    names = []
    med_errors = []
    fold_errors = []
    for name, res in all_results.items():
        if res.get('status') == 'success':
            names.append(name)
            med_errors.append(res['median_error'])
            fold_errors.append(np.std(res['median_error_per_fold']))

    if not names:
        print("No successful decodings to plot.")
        return None

    # Normalise recording_animal for matching
    rec_norm = recording_animal if recording_animal.startswith('rat') else f'rat{recording_animal}'
    colors = ['tab:blue' if n == rec_norm else 'tab:gray' for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, med_errors, yerr=fold_errors, color=colors,
                  capsize=4, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Median decoding error (pixels)')
    ax.set_title('Bayesian location decoding: self vs others')

    # Label self
    for bar, name in zip(bars, names):
        if name == rec_norm:
            bar.set_label('Self (implanted)')
        else:
            bar.set_label('Other')
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    return fig
