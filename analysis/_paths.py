"""Repo-root and results-directory helpers shared by every analysis.

Each analysis ``analysis/<name>/`` writes its outputs to ``results/<name>/``;
route every such path through :func:`results_dir` so a file moving one level
deeper cannot silently change where results land.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"


def results_dir(name: str) -> Path:
    """``results/<name>/`` for an analysis folder name (not created here)."""
    return RESULTS_ROOT / name
