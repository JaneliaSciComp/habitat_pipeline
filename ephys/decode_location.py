"""
Location Decoding Module

Decode the position of self or other animals from neural population activity
using a Poisson-likelihood Bayesian decoder over a 2-D grid of spatial bins.

Typical workflow
----------------
1. Align neural activity and tracking positions in a common time base.
2. Bin spikes into a firing-rate matrix; down-sample tracking to match.
3. Cross-validate: learn Poisson tuning curves on the training bins, decode a
   posterior position map on the held-out bins, and read off the estimate
   (posterior mean by default, or the MAP bin centre).
"""

import numpy as np
from typing import Optional, Tuple, Dict, TYPE_CHECKING

from ingestion.kilosort_data_import import _DEFAULT_QUALITY_THRESHOLDS

if TYPE_CHECKING:
    from ingestion.kilosort_data_import import KilosortData
    from video.tracking_import import VideoTrackingData


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def build_binned_data(ks_data: "KilosortData",
                      tracking_data: "VideoTrackingData",
                      object_name: str,
                      bin_size: float = 0.5,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      use_quality_cells: bool = True,
                      quality_thresholds: Optional[dict] = None,
                      verbose: bool = False,
                      ) -> Dict:
    """
    Align neural firing rates and animal positions into matching time bins.

    Parameters
    ----------
    ks_data : KilosortData
        Kilosort session with ``spike_times_by_cell`` populated.
    tracking_data : VideoTrackingData
        Must be ephys-synchronised (``synchronize_with_ephys`` already called).
    object_name : str
        Tracked object to decode (e.g. ``"rat631"``).
    bin_size : float
        Bin width in seconds.
    start_time, end_time : float, optional
        Restrict the time range.  Defaults to the overlap of neural and
        tracking data.
    use_quality_cells : bool
        Use only quality-filtered cells.
    quality_thresholds : dict, optional
        Forwarded to ``filter_cells_by_firing_patterns``; defaults to
        ``_DEFAULT_QUALITY_THRESHOLDS`` when ``use_quality_cells`` is True.
    verbose : bool
        Print the inferred neural/tracking time ranges.

    Returns
    -------
    dict with keys
        ``firing_rates`` : np.ndarray, shape (n_bins, n_cells)
        ``positions``    : np.ndarray, shape (n_bins, 2)  – (x, y)
        ``bin_centers``  : np.ndarray, shape (n_bins,)
        ``n_cells``      : int
        ``bin_size``     : float
    """
    if use_quality_cells:
        thresholds = dict(_DEFAULT_QUALITY_THRESHOLDS) if quality_thresholds is None else quality_thresholds
        _, spike_times_list = ks_data.get_filtered_cells_spike_times(**thresholds)
    else:
        spike_times_list = list(ks_data.spike_times_by_cell)

    # Get tracking trajectory
    traj = tracking_data.get_object_trajectory(object_name)
    if traj is None:
        raise ValueError(f"No trajectory found for object '{object_name}'.")

    # Use ephys-synchronised timestamps if available, otherwise raw timestamps
    if 'ephys_timestamps' in traj.columns:
        track_ts = traj['ephys_timestamps'].values
    elif getattr(tracking_data, 'ephys_timestamps', None) is not None:
        track_ts = tracking_data.ephys_timestamps[:len(traj)]
    else:
        raise ValueError("VideoTrackingData must be synchronised with ephys first "
                         "(call tracking.synchronize_with_ephys(sync_manager)).")
    track_x = traj['center_x'].values
    track_y = traj['center_y'].values

    # Determine overlapping time range
    neural_min = min((st[0] for st in spike_times_list if len(st) > 0), default=0.0)
    neural_max = max((st[-1] for st in spike_times_list if len(st) > 0), default=1.0)
    track_min, track_max = track_ts[0], track_ts[-1]
    if verbose:
        print(f"Neural data time range: {neural_min:.2f} to {neural_max:.2f} seconds")
        print(f"Tracking data time range: {track_min:.2f} to {track_max:.2f} seconds")
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

    # Bin positions → mean (x, y) per bin (vectorised by bin assignment)
    bin_idx = np.digitize(track_ts, bin_edges) - 1
    inside = (bin_idx >= 0) & (bin_idx < n_bins)
    positions = np.full((n_bins, 2), np.nan)
    for axis, vals in enumerate((track_x, track_y)):
        finite = inside & np.isfinite(vals)
        sums = np.bincount(bin_idx[finite], weights=vals[finite], minlength=n_bins)
        counts = np.bincount(bin_idx[finite], minlength=n_bins)
        with np.errstate(invalid='ignore'):
            positions[:, axis] = np.where(counts > 0, sums / counts, np.nan)

    # Drop bins with missing position data
    valid = ~np.isnan(positions[:, 0])
    return {
        'firing_rates': firing_rates[valid],
        'positions': positions[valid],
        'bin_centers': bin_centers[valid],
        'n_cells': n_cells,
        'bin_size': bin_size,
    }


# ---------------------------------------------------------------------------
# Bayesian population decoder (2-D spatial bins)
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
    occupancy : (n_x_bins, n_y_bins) – number of time bins spent in each bin
    """
    from scipy.ndimage import gaussian_filter

    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1
    n_cells = firing_rates.shape[1]

    # Assign each time bin to a spatial bin
    x_idx = np.clip(np.digitize(positions[:, 0], x_edges) - 1, 0, n_x - 1)
    y_idx = np.clip(np.digitize(positions[:, 1], y_edges) - 1, 0, n_y - 1)

    occupancy = np.zeros((n_x, n_y), dtype=np.float64)
    rate_sum = np.zeros((n_x, n_y, n_cells), dtype=np.float64)
    np.add.at(occupancy, (x_idx, y_idx), 1)
    np.add.at(rate_sum, (x_idx, y_idx), firing_rates)

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
                     estimate: str = 'expected',
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode position with a Poisson-likelihood Bayesian decoder (vectorised).

    P(pos | spikes) ∝ P(spikes | pos) · P(pos)

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
    estimate : {'expected', 'map'}
        ``'expected'`` returns the posterior mean (continuous, sub-bin
        resolution); ``'map'`` returns the maximum-a-posteriori bin centre.

    Returns
    -------
    predicted_pos : (n_time_bins, 2)
    posterior : (n_time_bins, n_x, n_y) – normalised posterior per time bin
    """
    n_time = firing_rates.shape[0]
    n_x, n_y, _ = tuning.shape

    # Expected spike counts per spatial bin: λ * Δt
    lam_dt = tuning * time_bin_size                      # (n_x, n_y, n_cells)
    log_lam_dt = np.log(np.clip(lam_dt, 1e-10, None))    # avoid log(0)
    neg_sum_lam = -lam_dt.sum(axis=2)                    # (n_x, n_y)

    # Prior
    if use_occupancy_prior:
        prior = np.clip(occupancy / occupancy.sum(), 1e-10, None)
    else:
        prior = np.full((n_x, n_y), 1.0 / (n_x * n_y))
    log_prior = np.log(prior)

    # Observed spike counts per time bin
    spike_counts = firing_rates * time_bin_size          # (n_time, n_cells)

    # Log-posterior for every (time, x, y) at once
    log_post = np.tensordot(spike_counts, log_lam_dt, axes=([1], [2]))  # (n_time, n_x, n_y)
    log_post += neg_sum_lam + log_prior

    # Normalise in log-space, per time bin, for numerical stability
    flat = log_post.reshape(n_time, -1)
    flat -= flat.max(axis=1, keepdims=True)
    np.exp(flat, out=flat)
    flat /= flat.sum(axis=1, keepdims=True)
    posterior = flat.reshape(n_time, n_x, n_y)

    if estimate == 'map':
        best = flat.argmax(axis=1)
        xi, yi = np.unravel_index(best, (n_x, n_y))
        predicted_pos = np.column_stack([x_centers[xi], y_centers[yi]])
    else:  # posterior mean (expected value)
        pred_x = posterior.sum(axis=2) @ x_centers
        pred_y = posterior.sum(axis=1) @ y_centers
        predicted_pos = np.column_stack([pred_x, pred_y])

    return predicted_pos, posterior


def _cv_decode(X: np.ndarray,
               Y: np.ndarray,
               x_edges: np.ndarray,
               y_edges: np.ndarray,
               x_centers: np.ndarray,
               y_centers: np.ndarray,
               bin_size: float,
               smoothing_sigma: float,
               use_occupancy_prior: bool,
               estimate: str,
               cv_folds: int,
               return_posterior: bool = True,
               ) -> Tuple[np.ndarray, Optional[np.ndarray], list]:
    """Cross-validated Bayesian decode of ``Y`` from ``X`` over a fixed grid.

    Returns ``(Y_pred, posterior, fold_median_errors)``.  When
    ``return_posterior`` is False the posterior is not accumulated (used for
    the null, which only needs errors) and ``posterior`` is None.
    """
    from sklearn.model_selection import KFold

    n_sb = len(x_centers)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    Y_pred = np.full_like(Y, np.nan)
    posterior = np.zeros((len(X), n_sb, n_sb)) if return_posterior else None
    fold_errors = []

    for train_idx, test_idx in kf.split(X):
        tuning, occupancy = _build_tuning_curves(
            X[train_idx], Y[train_idx], x_edges, y_edges, smoothing_sigma)
        pred, post = _bayesian_decode(
            X[test_idx], tuning, occupancy, bin_size,
            x_centers, y_centers, use_occupancy_prior, estimate)
        Y_pred[test_idx] = pred
        if return_posterior:
            posterior[test_idx] = post
        err = np.sqrt(np.sum((Y[test_idx] - pred) ** 2, axis=1))
        fold_errors.append(float(np.median(err)))

    return Y_pred, posterior, fold_errors


def _null_position_decode(X: np.ndarray,
                          Y: np.ndarray,
                          grid: tuple,
                          bin_size: float,
                          smoothing_sigma: float,
                          use_occupancy_prior: bool,
                          estimate: str,
                          cv_folds: int,
                          null: str,
                          n_shuffles: int,
                          ) -> Dict:
    """Decoding-error null obtained by breaking the rate↔position pairing.

    ``null='reverse'`` time-reverses the trajectory once (deterministic,
    preserves every marginal and the autocorrelation of both streams).
    ``null='shuffle'`` applies ``n_shuffles`` random circular shifts to the
    trajectory, yielding a distribution.  In both cases the decoder is trained
    and scored against the *scrambled* labels, so a real rate↔position
    relationship is required to beat it.
    """
    x_edges, y_edges, x_centers, y_centers = grid

    if null == 'reverse':
        variants = [Y[::-1]]
    elif null == 'shuffle':
        rng = np.random.default_rng(0)
        n = len(Y)
        lo, hi = max(1, n // 10), max(2, n - n // 10)
        shifts = rng.integers(lo, hi, size=n_shuffles)
        variants = [np.roll(Y, int(s), axis=0) for s in shifts]
    else:
        raise ValueError(f"Unknown null method '{null}' (use 'reverse' or 'shuffle').")

    medians = []
    for Y_null in variants:
        Y_pred, _, _ = _cv_decode(
            X, Y_null, x_edges, y_edges, x_centers, y_centers, bin_size,
            smoothing_sigma, use_occupancy_prior, estimate, cv_folds,
            return_posterior=False)
        err = np.sqrt(np.sum((Y_null - Y_pred) ** 2, axis=1))
        medians.append(float(np.median(err)))

    medians = np.asarray(medians)
    return {
        'null_method': null,
        'null_median_error': float(np.mean(medians)),
        'null_median_error_std': float(np.std(medians)),
        'null_median_errors': medians.tolist(),
    }


def decode_location(ks_data: "KilosortData",
                    tracking: "VideoTrackingData",
                    object_name: str,
                    bin_size: float = 0.5,
                    n_spatial_bins: int = 20,
                    smoothing_sigma: float = 1.0,
                    use_occupancy_prior: bool = True,
                    estimate: str = 'expected',
                    rate_smoothing_sigma: float = 0.0,
                    start_time: Optional[float] = None,
                    end_time: Optional[float] = None,
                    use_quality_cells: bool = True,
                    quality_thresholds: Optional[dict] = None,
                    cv_folds: int = 5,
                    null: Optional[str] = 'reverse',
                    n_shuffles: int = 100,
                    ) -> Dict:
    """
    Bayesian decoding of (x, y) position from population firing rates.

    Discretises the arena into a 2-D grid of spatial bins.  For each
    cross-validation fold the decoder learns Poisson tuning curves on the
    training set and produces a posterior probability map on the test set.

    Parameters
    ----------
    ks_data : KilosortData
    tracking : VideoTrackingData
    object_name : str
        Animal whose position to decode (e.g. ``"rat631"`` for self, or
        another rat's name for other-decoding).
    bin_size : float
        Temporal bin width in seconds.
    n_spatial_bins : int
        Number of bins per spatial dimension (total grid = n² bins).
    smoothing_sigma : float
        Gaussian smoothing of tuning curves in spatial-bin units.
    use_occupancy_prior : bool
        If True, weight posterior by occupancy (empirical spatial prior).
    estimate : {'expected', 'map'}
        Position read-out: posterior mean (default) or MAP bin centre.
    rate_smoothing_sigma : float
        If > 0, Gaussian-smooth firing rates along time (in time-bin units)
        before decoding to raise SNR.
    start_time, end_time : float, optional
        Time range.
    use_quality_cells : bool
        Use quality-filtered cells only.
    quality_thresholds : dict, optional
        Forwarded to the cell quality filter.
    cv_folds : int
        Number of cross-validation folds.
    null : {'reverse', 'shuffle', None}
        Baseline error from a broken rate↔position pairing.  ``'reverse'``
        (default) time-reverses the trajectory once; ``'shuffle'`` applies
        ``n_shuffles`` random circular shifts for a distribution; ``None``
        skips the null.
    n_shuffles : int
        Number of circular shifts when ``null='shuffle'``.

    Returns
    -------
    dict
        ``median_error``, ``mean_error``, ``median_error_per_fold``,
        ``y_true``, ``y_pred``, ``posterior``,
        ``x_edges``/``y_edges``/``x_centers``/``y_centers``, ``bin_centers``,
        ``n_cells``, ``n_bins``, ``n_spatial_bins``, ``parameters``, ``status``.
        When ``null`` is set: ``null_method``, ``null_median_error``,
        ``null_median_error_std``, ``null_median_errors``.
    """
    data = build_binned_data(ks_data, tracking, object_name,
                             bin_size, start_time, end_time,
                             use_quality_cells, quality_thresholds)
    X = data['firing_rates']   # (n_time, n_cells)
    Y = data['positions']      # (n_time, 2)
    bin_centers = data['bin_centers']

    if rate_smoothing_sigma > 0 and len(X) > 0:
        from scipy.ndimage import gaussian_filter1d
        X = gaussian_filter1d(X, sigma=rate_smoothing_sigma, axis=0)

    parameters = {
        'object_name': object_name,
        'bin_size': bin_size,
        'n_spatial_bins': n_spatial_bins,
        'smoothing_sigma': smoothing_sigma,
        'use_occupancy_prior': use_occupancy_prior,
        'estimate': estimate,
        'rate_smoothing_sigma': rate_smoothing_sigma,
        'cv_folds': cv_folds,
        'use_quality_cells': use_quality_cells,
        'quality_thresholds': quality_thresholds,
        'null': null,
        'n_shuffles': n_shuffles,
    }

    if len(X) < cv_folds * 2:
        return {'status': 'insufficient_data', 'median_error': np.nan,
                'parameters': parameters}

    # Spatial grid (shared across folds, fitted to full position range)
    x_edges = np.linspace(np.nanmin(Y[:, 0]), np.nanmax(Y[:, 0]), n_spatial_bins + 1)
    y_edges = np.linspace(np.nanmin(Y[:, 1]), np.nanmax(Y[:, 1]), n_spatial_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    grid = (x_edges, y_edges, x_centers, y_centers)

    # Cross-validated decoding
    Y_pred_all, posterior_all, fold_errors = _cv_decode(
        X, Y, *grid, bin_size, smoothing_sigma,
        use_occupancy_prior, estimate, cv_folds)
    errors = np.sqrt(np.sum((Y - Y_pred_all) ** 2, axis=1))

    result = {
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
        'parameters': parameters,
        'status': 'success',
    }

    # Null baseline (broken rate↔position pairing)
    if null is not None:
        result.update(_null_position_decode(
            X, Y, grid, bin_size, smoothing_sigma,
            use_occupancy_prior, estimate, cv_folds, null, n_shuffles))

    return result


# ---------------------------------------------------------------------------
# Decode all animals (self + each opponent)
# ---------------------------------------------------------------------------

def decode_all_locations(ks_data: "KilosortData",
                         tracking: "VideoTrackingData",
                         bin_size: float = 0.5,
                         n_spatial_bins: int = 20,
                         smoothing_sigma: float = 1.0,
                         use_occupancy_prior: bool = True,
                         estimate: str = 'expected',
                         rate_smoothing_sigma: float = 0.0,
                         use_quality_cells: bool = True,
                         quality_thresholds: Optional[dict] = None,
                         cv_folds: int = 5,
                         null: Optional[str] = 'reverse',
                         n_shuffles: int = 100,
                         ) -> Dict[str, Dict]:
    """
    Decode the position of every tracked animal from the same neural data.

    Useful for comparing self-location decoding accuracy to other-animal
    decoding accuracy.  All keyword arguments are forwarded to
    ``decode_location``.

    Returns
    -------
    dict
        Keyed by ``object_name``.  Each value is a ``decode_location`` result.
    """
    results = {}
    for obj_name in tracking.get_object_names():
        print(f"Decoding position of {obj_name} ...", end=" ", flush=True)
        res = decode_location(
            ks_data, tracking, obj_name,
            bin_size=bin_size, n_spatial_bins=n_spatial_bins,
            smoothing_sigma=smoothing_sigma, use_occupancy_prior=use_occupancy_prior,
            estimate=estimate, rate_smoothing_sigma=rate_smoothing_sigma,
            use_quality_cells=use_quality_cells, quality_thresholds=quality_thresholds,
            cv_folds=cv_folds, null=null, n_shuffles=n_shuffles,
        )
        err = res.get('median_error', np.nan)
        null_err = res.get('null_median_error', np.nan)
        if not np.isnan(err):
            msg = f"median error = {err:.1f}"
            if not np.isnan(null_err):
                msg += f"  (null = {null_err:.1f})"
            print(msg)
        else:
            print(res.get('status', ''))
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
        Output of ``decode_location``.
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
                              figsize: Tuple[float, float] = (9, 5)):
    """
    Bar chart comparing decoding median error across all animals.

    When the results carry a null baseline (``null_median_error``, produced by
    ``decode_location(..., null=...)``), a second set of grey bars shows the
    null error so each animal's decoding can be read against its own chance
    level.

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

    names, med_errors, fold_errors = [], [], []
    null_errors, null_stds = [], []
    for name, res in all_results.items():
        if res.get('status') == 'success':
            names.append(name)
            med_errors.append(res['median_error'])
            fold_errors.append(np.std(res['median_error_per_fold']))
            null_errors.append(res.get('null_median_error', np.nan))
            null_stds.append(res.get('null_median_error_std', 0.0))

    if not names:
        print("No successful decodings to plot.")
        return None

    # Normalise recording_animal for matching
    rec_norm = recording_animal if recording_animal.startswith('rat') else f'rat{recording_animal}'
    colors = ['tab:blue' if n == rec_norm else 'tab:gray' for n in names]
    has_null = np.any(np.isfinite(null_errors))

    # Null label reflects the method actually used (reverse vs shuffle)
    methods = {res['parameters'].get('null') for res in all_results.values()
               if res.get('status') == 'success' and 'parameters' in res}
    methods.discard(None)
    null_label = 'Null (shuffled)' if methods == {'shuffle'} else 'Null (reversed)'

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    width = 0.38 if has_null else 0.6
    offset = width / 2 if has_null else 0.0

    decoded_bars = ax.bar(x - offset, med_errors, width, yerr=fold_errors,
                          color=colors, capsize=4, edgecolor='black', linewidth=0.5)
    for bar, name in zip(decoded_bars, names):
        bar.set_label('Self (implanted)' if name == rec_norm else 'Other')

    if has_null:
        ax.bar(x + offset, null_errors, width, yerr=null_stds,
               color='lightgray', hatch='//', capsize=4,
               edgecolor='black', linewidth=0.5, label=null_label)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Median decoding error (pixels)')
    ax.set_title('Bayesian location decoding: self vs others')

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    return fig
