"""
Layer 0 of the AI-in-the-loop discovery platform: the two artifacts an agent
must consult *before* proposing or running an analysis.

- :mod:`discovery.hazards` + ``hazards.json`` — the traps that have silently
  corrupted a result in this project, each carrying an executable detector
  where one exists. The machine-readable mirror of ``CLAUDE.md``'s
  "Gotchas I keep tripping over".
- :mod:`discovery.capability_manifest` + ``capability_manifest.json`` — what
  data actually exists per session/animal/modality, so an untestable
  hypothesis fails at generation time rather than 40 minutes into an
  extraction.

See ``docs/AI_DISCOVERY_LOOP_DESIGN.md`` and ``HANDOFF.md``.

Deliberate import discipline: :mod:`discovery.capability_manifest` is the
*consult* path and must stay free of :mod:`ingestion` / :mod:`video` /
:mod:`ephys` imports so that consulting the manifest cannot reach the
``//nearline`` SMB share. The expensive probing lives in
:mod:`discovery.manifest_build`. A test enforces the separation. Nothing is
imported eagerly here for the same reason.
"""

__all__ = ['hazards', 'detectors', 'capability_manifest', 'requirements', 'manifest_build']
