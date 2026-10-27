"""Export the focal rat's (x, y) trajectory with event markers, as in the dashboard's
trajectory panel: path coloured by time, events coloured by opponent.

Reads the explore cache (nothing is refitted or reloaded from the share). By default it uses
the same tracking source and time range as the cached CEBRA-position fit, so the figure
matches ``export_embedding_video.py``.

    python scripts/export_trajectory_figure.py --animal rat631 --session 20251216_094334
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gui import explore_cache as xc  # noqa: E402
from export_embedding_video import load_events  # noqa: E402

#: An event is placed only if a tracked frame lies this close (as in the dashboard).
EVENT_SNAP_S = 1.0
FORMAT_NAMES = {"apt_tqt": "APT", "mask_metrics": "manual"}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--cohort", default="cohort7")
    p.add_argument("--session", required=True)
    p.add_argument("--animal", required=True)
    p.add_argument("--source", choices=sorted(FORMAT_NAMES), default=None,
                   help="tracking source (default: the one the cached CEBRA-position fit used)")
    p.add_argument("--events", default="EC", metavar="TYPE", help="'none' for no markers")
    p.add_argument("--min-events", type=int, default=3)
    p.add_argument("--t-start", type=float, default=None)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--size", type=int, default=900, help="figure width/height in pixels")
    p.add_argument("--dark", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    path = xc.cache_path(a.cohort, a.session, a.animal)
    if not path.exists():
        raise SystemExit(f"No explore cache at {path}. Open the block in the dashboard first.")
    with open(path, "rb") as f:
        entry = pickle.load(f)
    formats = xc.trajectory_formats(entry.get("trajectory"))
    if not formats:
        raise SystemExit("No tracking trajectory cached for this block.")
    pos_fit = entry.get("cebra_position")
    source = a.source or (pos_fit or {}).get("params", {}).get("tracking") \
        or entry["trajectory"].get("default")
    if source not in formats:
        raise SystemExit(f"Tracking source '{source}' not cached (have {sorted(formats)}).")
    traj = formats[source]
    t, x, y = (np.asarray(traj[k], float) for k in ("t", "x", "y"))

    lo, hi = (pos_fit["bin_centers"][[0, -1]] if pos_fit is not None else (t[0], t[-1]))
    lo = a.t_start if a.t_start is not None else lo
    hi = a.t_end if a.t_end is not None else hi
    view = (t >= lo) & (t <= hi)
    if view.sum() < 2:
        raise SystemExit(f"{FORMAT_NAMES[source]} tracking has no frames in [{lo:.0f}, {hi:.0f}] s.")
    tv, xv, yv = t[view], x[view], y[view]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, fg = ("#111111", "#dddddd") if a.dark else ("white", "#222222")
    fig, ax = plt.subplots(figsize=(a.size / 100, a.size / 100), dpi=100, facecolor=bg)
    ax.set_facecolor(bg)
    # Faint path, broken where tracking has a gap, then time-coloured samples.
    gap = np.r_[False, np.diff(tv) > 2.0 * np.median(np.diff(tv))]
    ax.plot(np.where(gap, np.nan, xv), np.where(gap, np.nan, yv), color="grey", lw=0.6, alpha=0.25)
    sc = ax.scatter(xv, yv, c=tv, cmap="viridis", s=3, alpha=0.5, vmin=lo, vmax=hi, linewidths=0)

    n_shown = n_untracked = 0
    if a.events.lower() != "none":
        ev_t, ev_opp, colors = load_events(path, a.animal, a.events, a.min_events)
        m = (ev_t >= lo) & (ev_t <= hi)
        ev_t, ev_opp = ev_t[m], ev_opp[m]
        idx = np.clip(np.searchsorted(t, ev_t), 1, len(t) - 1)
        idx = np.where(np.abs(t[idx - 1] - ev_t) < np.abs(t[idx] - ev_t), idx - 1, idx)
        snapped = np.abs(t[idx] - ev_t) <= EVENT_SNAP_S
        n_untracked = int((~snapped).sum())
        for o in np.unique(ev_opp[snapped]):
            k = snapped & (ev_opp == o)
            ax.scatter(x[idx[k]], y[idx[k]], s=70, c=colors.get(o, "grey"), edgecolors="black",
                       linewidths=0.6, label=o, zorder=10)
            n_shown += int(k.sum())
        if n_shown:
            leg = ax.legend(title="Opponent", loc="upper left", fontsize=8, title_fontsize=8,
                            facecolor=bg, labelcolor=fg, edgecolor="none")
            leg.get_title().set_color(fg)

    ax.set_aspect("equal")
    ax.invert_yaxis()   # image coordinates, as in the dashboard
    src = FORMAT_NAMES[source]
    ax.set_xlabel(f"x ({src} pixels)", color=fg)
    ax.set_ylabel(f"y ({src} pixels)", color=fg)
    ax.tick_params(colors=fg, labelsize=8)
    for s in ax.spines.values():
        s.set_color(fg)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Time (s)", color=fg)
    cb.ax.tick_params(colors=fg, labelsize=8)
    cb.solids.set_alpha(1.0)
    note = f" · {n_untracked} not tracked at event time" if n_untracked else ""
    ev_txt = "" if a.events.lower() == "none" else f" — {n_shown} {a.events} events{note}"
    ax.set_title(f"{a.animal} trajectory ({src}){ev_txt}\n[{lo:.0f}–{hi:.0f} s]",
                 color=fg, fontsize=11)
    fig.tight_layout()

    tag = "" if a.events.lower() == "none" else f"_{a.events}"
    out = a.out or ROOT / "results" / f"{a.animal}_{a.session}_trajectory_{src}{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=bg)
    print(f"wrote {out}  ({len(tv)} frames, {n_shown} events shown, {n_untracked} untracked)")


if __name__ == "__main__":
    main()
