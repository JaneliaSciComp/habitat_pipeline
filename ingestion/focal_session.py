"""
Focal-session input loader.

Several single-focal analyses (partner-distance decoding, social place fields)
need exactly one implanted animal's ephys plus the *session* tracking (which
already contains every animal) — the partner / target animals contribute a
trajectory only and have no ephys. This module centralizes that loading so those
analyses do **not** build a :class:`~ingestion.multi_animal_session.MultiAnimalSession`
(which would demand a ``DataStorageManager`` / ``KilosortData`` per animal).

``load_focal_session_inputs`` builds only the **focal** animal's
:class:`~ingestion.data_paths.DataStorageManager` and returns the four objects
downstream code needs: the focal :class:`~ingestion.kilosort_data_import.KilosortData`,
the session :class:`~video.tracking_import.VideoTrackingData`, a
:class:`~ingestion.ephys_sync.DataSyncManager`, and the ``pixels_per_cm``
calibration (``None`` if uncalibrated). Tracking↔ephys / pixels→cm conversion is
then done via :func:`video.tracking_import.resolve_tracking_on_ephys_clock`.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from ingestion.data_paths import DataStorageManager
from ingestion.ephys_sync import DataSyncManager
from ingestion.kilosort_data_import import KilosortData, load_kilosort_data
from video.tracking_import import VideoTrackingData, load_tracking_data


class FocalSessionInputs(NamedTuple):
    """Loaded objects for a focal animal + its session tracking."""
    ks_focal: KilosortData
    tracking: VideoTrackingData
    sync: DataSyncManager
    pixels_per_cm: Optional[float]


def load_focal_session_inputs(session_id: str, focal: str, *,
                              config_path: Optional[str] = None,
                              dio_channel: int = 1) -> FocalSessionInputs:
    """Load a focal animal's ephys + the session tracking + clock sync.

    Builds only the **focal** animal's :class:`DataStorageManager` — no partner /
    target is ever loaded as ephys. The session tracking file (resolved through
    the focal DSM) already contains every animal's trajectory.

    Returns a :class:`FocalSessionInputs` ``(ks_focal, tracking, sync,
    pixels_per_cm)``.
    """
    dsm = DataStorageManager(focal, session_id, config_path=config_path)
    ks_focal = load_kilosort_data(dsm.get_kilosort_path())
    sync = DataSyncManager(dsm, dio_channel=dio_channel)
    tracking = load_tracking_data(dsm)
    return FocalSessionInputs(ks_focal, tracking, sync, dsm.get_pixels_per_cm())
