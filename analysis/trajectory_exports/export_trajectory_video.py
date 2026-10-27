"""Export a video of the focal rat's (x, y) trajectory building up over time.

Playback runs through the tracked time range: the path is drawn up to the current moment
(coloured by time, with the last few seconds emphasised), event markers (coloured by opponent)
appear as they happen, and a ring marks the rat's current position. Reads the explore cache,
so nothing is refitted or reloaded from the share. By default it uses the same tracking
source as the cached CEBRA-position fit.

    python analysis/trajectory_exports/export_trajectory_video.py --animal rat631 --session 20251216_094334
    python analysis/trajectory_exports/export_trajectory_video.py --animal rat631 --session 20251216_094334 \
        --duration 90 --events F
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gui import explore_cache as xc  # noqa: E402
from analysis.trajectory_exports.export_embedding_video import load_events  # noqa: E402

EVENT_SNAP_S = 1.0       # an event is placed only if a tracked frame lies this close
FORMAT_NAMES = {"apt_tqt": "APT", "mask_metrics": "manual"}
MAX_PATH_POINTS = 25000  # path samples drawn; the full-rate track is only used for the cursor
TRAIL_S = 10.0           # length of the emphasised recent trail, in recording seconds
MAX_GAP_S = 2.0          # no cursor / no line across a tracking gap longer than this


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--cohort", default="cohort7")
    p.add_argument("--session", required=True)
    p.add_argument("--animal", required=True)
    p.add_argument("--source", choices=sorted(FORMAT_NAMES), default=None,
                   help="tracking source (default: the one the cached CEBRA-position fit used)")
    p.add_argument("--events", default="EC", metavar="TYPE", help="'none' for no markers")
    p.add_argument("--min-events", type=int, default=3)
    p.add_argument("--t-start", type=float, default=None, help="default: first tracked frame")
    p.add_argument("--t-end", type=float, default=None, help="default: last tracked frame")
    p.add_argument("--duration", type=float, default=60.0, help="video length in seconds")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--size", type=int, default=900, help="square frame size in pixels")
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
    t, x, y = (np.asarray(formats[source][k], float) for k in ("t", "x", "y"))
    src = FORMAT_NAMES[source]

    lo = a.t_start if a.t_start is not None else t[0]
    hi = a.t_end if a.t_end is not None else t[-1]
    keep = (t >= lo) & (t <= hi)
    if keep.sum() < 2:
        raise SystemExit(f"{src} tracking has no frames in [{lo:.0f}, {hi:.0f}] s.")
    t, x, y = t[keep], x[keep], y[keep]
    lo, hi = t[0], t[-1]

    # Path samples for drawing (strided); NaN-split where tracking has a gap.
    stride = max(1, int(np.ceil(len(t) / MAX_PATH_POINTS)))
    ts, xs, ys = t[::stride], x[::stride], y[::stride]

    # Events at the animal's tracked position at event start.
    ev_t = np.array([]); ev_x = ev_y = ev_t; ev_opp = np.array([]); colors = {}
    if a.events.lower() != "none":
        et, eo, colors = load_events(path, a.animal, a.events, a.min_events)
        m = (et >= lo) & (et <= hi)
        et, eo = et[m], eo[m]
        idx = np.clip(np.searchsorted(t, et), 1, len(t) - 1)
        idx = np.where(np.abs(t[idx - 1] - et) < np.abs(t[idx] - et), idx - 1, idx)
        ok = np.abs(t[idx] - et) <= EVENT_SNAP_S
        ev_t, ev_x, ev_y, ev_opp = et[ok], x[idx[ok]], y[idx[ok]], eo[ok]
        print(f"{len(ev_t)} {a.events} events placed, {int((~ok).sum())} not tracked at event time")

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, fg = ("#111111", "#dddddd") if a.dark else ("white", "#222222")
    fig, ax = plt.subplots(figsize=(a.size / 100, a.size / 100), dpi=100, facecolor=bg)
    ax.set_facecolor(bg)
    pad = 0.03 * max(np.ptp(x), np.ptp(y))
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.max() + pad, y.min() - pad)      # image coordinates, as in the dashboard
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({src} pixels)", color=fg)
    ax.set_ylabel(f"y ({src} pixels)", color=fg)
    ax.tick_params(colors=fg, labelsize=8)
    for s in ax.spines.values():
        s.set_color(fg)

    (line,) = ax.plot([], [], color="grey", lw=0.6, alpha=0.25)
    path_sc = ax.scatter([], [], c=[], cmap="viridis", s=3, alpha=0.5, vmin=lo, vmax=hi,
                         linewidths=0)
    (trail,) = ax.plot([], [], color=fg, lw=1.6, alpha=0.8, zorder=5)
    cursor = ax.scatter([], [], s=260, facecolors="none", edgecolors="#e53e3e", linewidths=2.2,
                        zorder=12)
    ev_sc = {}
    for o in np.unique(ev_opp):
        ev_sc[o] = (ax.scatter([], [], s=70, c=colors.get(o, "grey"), edgecolors="black",
                               linewidths=0.6, zorder=10, label=o), ev_opp == o)
    if ev_sc:
        leg = ax.legend(title="Opponent", loc="upper left", fontsize=8, title_fontsize=8,
                        facecolor=bg, labelcolor=fg, edgecolor="none")
        leg.get_title().set_color(fg)
    cb = fig.colorbar(path_sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Time (s)", color=fg)
    cb.ax.tick_params(colors=fg, labelsize=8)
    cb.solids.set_alpha(1.0)
    ax.set_title(f"{a.animal} trajectory ({src})", color=fg, fontsize=11)
    clock = ax.text(0.99, 0.01, "", transform=ax.transAxes, ha="right", va="bottom",
                    color=fg, fontsize=11, family="monospace")
    fig.tight_layout()

    n_frames = max(2, int(round(a.duration * a.fps)))
    out = a.out or ROOT / "results" / "trajectory_exports" / (
        f"{a.animal}_{a.session}_trajectory_{src}"
        f"{'' if a.events.lower() == 'none' else '_' + a.events}_buildup.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for i in range(n_frames):
            T = lo + (hi - lo) * i / (n_frames - 1)
            k = int(np.searchsorted(ts, T, side="right"))
            tk, xk, yk = ts[:k], xs[:k], ys[:k]
            gap = np.r_[False, np.diff(tk) > MAX_GAP_S * 1.0] if k else np.array([], bool)
            line.set_data(np.where(gap, np.nan, xk), np.where(gap, np.nan, yk))
            path_sc.set_offsets(np.c_[xk, yk] if k else np.empty((0, 2)))
            path_sc.set_array(tk)
            # Emphasised recent trail from the full-rate track.
            j1 = int(np.searchsorted(t, T, side="right"))
            j0 = int(np.searchsorted(t, T - TRAIL_S, side="left"))
            tx, ty = x[j0:j1].copy(), y[j0:j1].copy()
            if len(tx) > 1:
                g = np.r_[False, np.diff(t[j0:j1]) > MAX_GAP_S]
                tx[g] = np.nan; ty[g] = np.nan
            trail.set_data(tx, ty)
            # Cursor at the nearest tracked frame, hidden across a tracking gap.
            c = np.clip(j1 - 1, 0, len(t) - 1)
            cursor.set_offsets([[x[c], y[c]]] if abs(t[c] - T) <= MAX_GAP_S else np.empty((0, 2)))
            n_ev = 0
            for o, (sc, mask) in ev_sc.items():
                sel = mask & (ev_t <= T)
                sc.set_offsets(np.c_[ev_x[sel], ev_y[sel]] if sel.any() else np.empty((0, 2)))
                n_ev += int(sel.sum())
            clock.set_text(f"t = {T:7.1f} s" + (f"   {n_ev} {a.events}" if ev_sc else ""))
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            if writer is None:
                h, w = rgba.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
                if not writer.isOpened():
                    raise SystemExit(f"Could not open {out} for writing.")
            writer.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
            if i % 100 == 0:
                print(f"  frame {i}/{n_frames}", flush=True)
    finally:
        if writer is not None:
            writer.release()
        plt.close(fig)
    print(f"wrote {out}  ({n_frames} frames, {n_frames / a.fps:.1f} s, "
          f"covers {lo:.0f}-{hi:.0f} s of recording)")


if __name__ == "__main__":
    main()
