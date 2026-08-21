"""
Self-contained local HTML reports over the lab notebook.

One document per hypothesis with a fixed eight-section structure, plus an index
over all of them. Everything is read from ``habitat_pipeline.db``; nothing is
uploaded anywhere and no section is fetched from a CDN, so a report stays
readable years later from the file alone.

See :mod:`reports.hypothesis_report`.
"""

__all__ = ['hypothesis_report']
