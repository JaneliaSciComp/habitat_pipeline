"""
Multi-animal session orchestrator.

All animals recorded in one RatCity session share a single ephys clock —
Kilosort spike times across animal directories are already comparable in
seconds — so building a population-level analysis across animals only
requires a shared time grid and per-animal binning. This module provides
``MultiAnimalSession``: a thin orchestrator that holds one
``DataStorageManager`` per animal, exposes lazily-loaded ``KilosortData``
instances, a single shared ``DataSyncManager``, and a single shared
``BehavioralEventsData`` for the session, and bins spikes onto a common
ephys-second grid via ``get_common_binned_rates``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ingestion.data_paths import DataStorageManager
from ingestion.ephys_sync import DataSyncManager
from ingestion.kilosort_data_import import KilosortData, load_kilosort_data
from video.behavioral_events import BehavioralEventsData, load_behavioral_events

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(".gui_cache") / "multi_animal"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_params(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _bin_spike_lists(
    spike_times_list: List[np.ndarray],
    bin_size_sec: float,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin a list of spike-time arrays into a (n_cells, n_bins) rate matrix.

    Mirrors :meth:`KilosortData.bin_spike_times` but takes the spike lists
    directly so that all animals share identical bin edges, regardless of
    which clusters pass per-animal quality filters.
    """
    bin_edges = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)
    n_bins = len(bin_edges) - 1
    n_cells = len(spike_times_list)
    matrix = np.zeros((n_cells, n_bins), dtype=np.float64)
    for i, st in enumerate(spike_times_list):
        if len(st) > 0:
            counts, _ = np.histogram(st, bins=bin_edges)
            matrix[i] = counts / bin_size_sec
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return matrix, bin_centers


def _gaussian_smooth(matrix: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Gaussian smoothing along the time (last) axis, per cell."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(matrix, sigma=sigma_bins, axis=1, mode="reflect")


# ---------------------------------------------------------------------------
# MultiAnimalSession
# ---------------------------------------------------------------------------

@dataclass
class MultiAnimalSession:
    """Orchestrator over multiple per-animal DataStorageManagers for one session.

    Parameters
    ----------
    session_id : str
        Session identifier shared by every animal in ``animal_ids``.
    animal_ids : list of str
        Animals recorded simultaneously in this session.
    config_path : str, optional
        Cohort config path passed to every ``DataStorageManager``.
    dio_channel : int
        DIO channel for the canonical ``DataSyncManager``.
    sync_from_animal : str, optional
        Which animal's DIO + pulse-log to build the canonical sync from.
        Defaults to ``animal_ids[0]``. All animals share one ephys clock, so
        the choice is cosmetic; consistency across animals can be verified
        via :meth:`verify_sync_consistency`.
    """

    session_id: str
    animal_ids: List[str]
    config_path: Optional[str] = None
    dio_channel: int = 1
    sync_from_animal: Optional[str] = None

    dsm_by_animal: Dict[str, DataStorageManager] = field(init=False, default_factory=dict)
    ks_by_animal: Dict[str, KilosortData] = field(init=False, default_factory=dict)
    _sync: Optional[DataSyncManager] = field(init=False, default=None, repr=False)
    _events: Optional[BehavioralEventsData] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.animal_ids:
            raise ValueError("animal_ids must be non-empty")
        for aid in self.animal_ids:
            dsm = DataStorageManager(aid, self.session_id, config_path=self.config_path)
            if dsm.session_id != self.session_id:
                raise ValueError(
                    f"DSM session_id ({dsm.session_id}) does not match the requested "
                    f"session_id ({self.session_id})"
                )
            self.dsm_by_animal[aid] = dsm
        if self.sync_from_animal is None:
            self.sync_from_animal = self.animal_ids[0]
        elif self.sync_from_animal not in self.dsm_by_animal:
            raise ValueError(
                f"sync_from_animal {self.sync_from_animal!r} not in animal_ids {self.animal_ids}"
            )

    # ----------------------------------------------------------------------
    # Lazy data accessors
    # ----------------------------------------------------------------------

    def get_ks(self, animal_id: str) -> KilosortData:
        """Load and cache a ``KilosortData`` for an animal."""
        if animal_id not in self.ks_by_animal:
            dsm = self.dsm_by_animal[animal_id]
            self.ks_by_animal[animal_id] = load_kilosort_data(dsm.get_kilosort_path())
        return self.ks_by_animal[animal_id]

    @property
    def sync(self) -> DataSyncManager:
        """Canonical ``DataSyncManager`` built from ``sync_from_animal``."""
        if self._sync is None:
            dsm = self.dsm_by_animal[self.sync_from_animal]
            self._sync = DataSyncManager(dsm, dio_channel=self.dio_channel)
        return self._sync

    @property
    def events(self) -> Optional[BehavioralEventsData]:
        """Shared session-level ``BehavioralEventsData``, ephys-synchronized.

        Returns ``None`` if the behavioral events fail to load.
        """
        if self._events is None:
            dsm = self.dsm_by_animal[self.sync_from_animal]
            try:
                self._events = load_behavioral_events(
                    dsm.get_behavioral_event_files(), session_id=self.session_id
                )
            except Exception as e:
                logger.warning("Failed to load behavioral events: %s", e)
                return None
            try:
                self._events.synchronize_with_ephys(self.sync, create_new_columns=True)
            except Exception as e:
                logger.warning("Failed to synchronize behavioral events with ephys: %s", e)
        return self._events

    # ----------------------------------------------------------------------
    # Common time grid
    # ----------------------------------------------------------------------

    def get_common_time_window(self) -> Tuple[float, float]:
        """Return ``(t_start, t_end)`` in ephys seconds.

        Because all animals share one ephys clock, this is simply
        ``(0.0, min_over_animals(duration_seconds))``.
        """
        durations = [self.get_ks(aid).duration_seconds for aid in self.animal_ids]
        return 0.0, float(min(durations))

    def get_common_binned_rates(
        self,
        bin_size_sec: float,
        t_start_ephys: Optional[float] = None,
        t_end_ephys: Optional[float] = None,
        filter_kwargs: Optional[dict] = None,
        smoothing_sigma_sec: Optional[float] = None,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Return ``(bin_centers_ephys, {animal_id: rates})`` on a shared grid.

        Each ``rates`` array has shape ``(n_cells_animal, n_bins)``. All
        animals share identical bin edges in ephys seconds, so the returned
        ``bin_centers_ephys`` applies to every entry in the dict.

        Parameters
        ----------
        bin_size_sec : float
            Width of each bin in seconds.
        t_start_ephys, t_end_ephys : float, optional
            Time window in ephys seconds. Defaults to the intersection of
            all animals' durations.
        filter_kwargs : dict, optional
            Forwarded to ``KilosortData.filter_cells_by_firing_patterns`` to
            choose quality cells per animal. Defaults to module-default
            thresholds.
        smoothing_sigma_sec : float, optional
            If set, Gaussian-smooth each cell's binned rate along the time
            axis with sigma in bin units ``= smoothing_sigma_sec / bin_size_sec``.
        use_cache : bool
            If True, read/write a pickle under ``.gui_cache/multi_animal/``
            keyed on all of the above arguments.
        """
        t0, t1 = self.get_common_time_window()
        if t_start_ephys is None:
            t_start_ephys = t0
        if t_end_ephys is None:
            t_end_ephys = t1
        if t_end_ephys <= t_start_ephys:
            raise ValueError(
                f"t_end_ephys ({t_end_ephys}) must be > t_start_ephys ({t_start_ephys})"
            )

        cache_file: Optional[Path] = None
        if use_cache:
            key = _hash_params({
                "session_id": self.session_id,
                "animal_ids": sorted(self.animal_ids),
                "bin_size_sec": bin_size_sec,
                "t_start": t_start_ephys,
                "t_end": t_end_ephys,
                "filter_kwargs": filter_kwargs or {},
                "smoothing_sigma_sec": smoothing_sigma_sec,
            })
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file = _CACHE_DIR / f"{self.session_id}_{key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    payload = pickle.load(f)
                return payload["bin_centers"], payload["rates_by_animal"]

        kwargs = filter_kwargs or {}
        rates_by_animal: Dict[str, np.ndarray] = {}
        bin_centers_ref: Optional[np.ndarray] = None
        for aid in self.animal_ids:
            ks = self.get_ks(aid)
            _, spike_times_list = ks.get_filtered_cells_spike_times(**kwargs)
            rates, bin_centers = _bin_spike_lists(
                spike_times_list, bin_size_sec, t_start_ephys, t_end_ephys,
            )
            if smoothing_sigma_sec is not None and smoothing_sigma_sec > 0:
                sigma_bins = smoothing_sigma_sec / bin_size_sec
                rates = _gaussian_smooth(rates, sigma_bins)
            if bin_centers_ref is None:
                bin_centers_ref = bin_centers
            elif not np.array_equal(bin_centers_ref, bin_centers):
                raise RuntimeError(
                    "Internal error: bin centers differ across animals; this should be unreachable"
                )
            rates_by_animal[aid] = rates

        if use_cache and cache_file is not None:
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {"bin_centers": bin_centers_ref, "rates_by_animal": rates_by_animal}, f,
                )

        return bin_centers_ref, rates_by_animal

    # ----------------------------------------------------------------------
    # Sync consistency
    # ----------------------------------------------------------------------

    def verify_sync_consistency(
        self,
        tol_slope_dev_from_one: float = 1e-4,
        min_r_squared: float = 0.999,
    ) -> Dict[str, dict]:
        """Build a sync from every animal and compare against the canonical.

        All animals share one ephys clock, so each animal's sync should map
        ephys seconds to behavior seconds with slope ~1 (the prompt's
        threshold) and the same intercept as the canonical sync. Emits
        warnings on mismatch; never raises. Returns a per-animal report.
        """
        canonical = self.sync
        report: Dict[str, dict] = {}
        for aid in self.animal_ids:
            dsm = self.dsm_by_animal[aid]
            try:
                s = DataSyncManager(dsm, dio_channel=self.dio_channel)
            except Exception as e:
                logger.warning("Could not build sync for animal %s: %s", aid, e)
                report[aid] = {"status": "error", "error": str(e)}
                continue
            r2 = s.mapping["r_squared"]
            slope_dev = abs(s.slope - 1.0)
            slope_diff = abs(s.slope - canonical.slope)
            intercept_diff = abs(s.intercept - canonical.intercept)
            ok = (
                slope_dev < tol_slope_dev_from_one
                and r2 > min_r_squared
                and slope_diff < tol_slope_dev_from_one
            )
            report[aid] = {
                "slope": s.slope,
                "intercept": s.intercept,
                "r_squared": r2,
                "slope_dev_from_one": slope_dev,
                "slope_diff_from_canonical": slope_diff,
                "intercept_diff_from_canonical": intercept_diff,
                "passed": ok,
            }
            if not ok:
                logger.warning(
                    "Sync inconsistency for animal %s: slope=%.6g r2=%.6g slope_diff=%.6g",
                    aid, s.slope, r2, slope_diff,
                )
        return report

    def __repr__(self) -> str:
        return (
            f"MultiAnimalSession(session_id={self.session_id!r}, "
            f"animal_ids={self.animal_ids}, sync_from={self.sync_from_animal})"
        )
