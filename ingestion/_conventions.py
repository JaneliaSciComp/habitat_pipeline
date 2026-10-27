"""
Facts about how RatCity data is named and encoded on disk.

One home for the constants and identifier rules that every loader needs, so a
convention is changed in one place rather than in each module that happens
to depend on it.
"""

import re
from pathlib import Path
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Ephys clock
# ---------------------------------------------------------------------------

#: Trodes acquisition rate. The pipeline's ephys clock is ``sample count / SAMPLE_RATE``.
SAMPLE_RATE = 30000.0

#: Kilosort reports a spike at its detection onset; shifting by this many
#: samples moves it to the template centre.
SPIKE_TEMPLATE_OFFSET = 31

# ---------------------------------------------------------------------------
# Recording / session identifiers
# ---------------------------------------------------------------------------

#: The timestamp embedded in a recording's on-disk prefix. Trodes writes
#: ``<YYYYmmdd>_<HHMMSS>`` and every artifact for one recording shares it.
RECORDING_STAMP_RE = re.compile(r'(20\d{6}_\d{6})')

#: A query that is exactly a recording stamp names one recording and nothing
#: else, so failing to find it must raise rather than fall back to the day's
#: primary recording.
FULL_RECORDING_ID_RE = re.compile(r'20\d{6}_\d{6}')


def parse_session_date(session_id: str) -> str:
    """The 8-digit ``YYYYMMDD`` prefix of a session id (``'20251210_110059'`` -> ``'20251210'``)."""
    if len(session_id) < 8:
        raise ValueError(f"Session ID too short to extract date: {session_id}")
    session_date = session_id[:8]
    if not session_date.isdigit():
        raise ValueError(f"Could not extract valid date from session_id: {session_id}")
    return session_date


# ---------------------------------------------------------------------------
# Animal identifiers
# ---------------------------------------------------------------------------

#: Behavioural-event columns that hold an animal id.
RAT_ID_COLUMNS = ('initiator', 'victim', 'winner', 'loser')


def canonical_rat_id(animal_id: str) -> str:
    """``'631'`` -> ``'rat631'``; an id that already starts with ``rat`` is unchanged."""
    return animal_id if animal_id.startswith('rat') else f"rat{animal_id}"


#: APT names rat 635 as ``rat4635``. The leading ``4`` is a tracker-side prefix,
#: not part of the animal id, and is stripped so object names match the
#: ``rat635`` / ``635`` used by the configs, kilosort directories and events.
_APT_ANIMAL_RE = re.compile(r'^rat4(\d{3})$', re.IGNORECASE)


def normalize_object_name(name: str) -> str:
    """``'rat4635'`` -> ``'rat635'``; any other name is returned unchanged."""
    match = _APT_ANIMAL_RE.match(str(name).strip())
    return f"rat{match.group(1)}" if match else str(name)


#: A ``cohort<N>`` path component (``cohort7``, ``cohort1vs2_exp1``, ...).
_COHORT_DIR_RE = re.compile(r'^cohort(\d+)', re.IGNORECASE)

#: First cohort whose animal ids are four digits starting with 4 (``rat4678``),
#: which :func:`normalize_object_name` would wrongly shorten to ``rat678``.
_FIRST_COHORT_WITHOUT_NAME_NORMALIZATION = 9


def cohort_from_path(path: Union[str, Path]) -> Optional[int]:
    """Cohort number from the deepest ``cohort<N>`` directory in ``path``, or None."""
    cohort = None
    for part in Path(path).parts:
        match = _COHORT_DIR_RE.match(part)
        if match:
            cohort = int(match.group(1))
    return cohort


def uses_name_normalization(path: Union[str, Path]) -> bool:
    """Whether names in a file at ``path`` should go through :func:`normalize_object_name`.

    True for cohorts before 9 and for paths naming no cohort (the old
    behaviour). False from cohort 9 on, whose real ids (``rat4678``) match the
    APT ``rat4<ddd>`` pattern.
    """
    cohort = cohort_from_path(path)
    return cohort is None or cohort < _FIRST_COHORT_WITHOUT_NAME_NORMALIZATION
