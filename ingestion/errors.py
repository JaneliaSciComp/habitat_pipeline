"""
Exceptions that say *which* piece of data is missing.

Every one subclasses :class:`FileNotFoundError`, which is what these paths
raised before, so existing ``except FileNotFoundError`` handlers still catch
them; new code can tell the cases apart:

- :class:`DataRootUnavailable` — a configured root (ephys, video, tracking,
  events) is not reachable at all. On this project that almost always means
  the ``//nearline`` share is not mounted, not that data is absent.
- :class:`RecordingNotFound` — the root is there, but no session / animal /
  recording matches the requested id.
- :class:`MissingCompanionFile` — the recording exists and its primary files
  are present, but a file needed to load it is not (e.g. the
  ``*.timestamps.dat`` beside ``kilosort4/``).
"""


class DataRootUnavailable(FileNotFoundError):
    """A configured data root does not exist or cannot be reached."""


class RecordingNotFound(FileNotFoundError):
    """No session, animal or recording matches the requested id."""


class MissingCompanionFile(FileNotFoundError):
    """A recording's primary files exist but a file needed to load it does not."""
