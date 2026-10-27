"""Export a video of a cached 3-D embedding slowly rotating about the z-axis.

Reads the fit the Panel dashboard already cached (``.gui_cache/explore/``), so nothing is
refitted. Frames are drawn with matplotlib (trajectory coloured by time, Viridis, like the
dashboard) and written with OpenCV, so no ffmpeg/kaleido is needed.

    python analysis/trajectory_exports/export_embedding_video.py --animal rat631 --session 20251216_094334
    python analysis/trajectory_exports/export_embedding_video.py --animal rat631 --session 20251216_094334 \
        --seconds-per-rev 60 --fps 30 --out cebra_pos.mp4
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gui import explore_cache as xc  # noqa: E402

#: --embedding choice -> key in the cache entry
SHELL = 0.04        # markers sit this fraction outside the unit sphere
BACK_ALPHA = 0.25   # opacity of markers on the far side of the sphere
EMBEDDINGS = {"cebra-pos": "cebra_position", "cebra": "cebra", "umap": "umap"}
TITLES = {"cebra_position": "CEBRA-Behavior (x, y)", "cebra": "CEBRA-Time", "umap": "UMAP"}
AXES = {"cebra_position": "CEBRA-pos", "cebra": "CEBRA", "umap": "UMAP"}


#: Category10, the palette the dashboard's timeline and embedding panels use
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def load_events(entry_path: Path, animal: str, btype: str, min_events: int):
    """(times, opponents, {opponent: colour}) for one event type, like the dashboard's markers."""
    with open(entry_path, "rb") as f:
        events = pickle.load(f).get("events")
    if events is None:
        raise SystemExit("No synced events in the cache entry.")
    ts, _, opp = events.extract_opponent_labels(
        animal_of_interest=animal, behavior_type=btype, min_events_per_class=min_events)
    df = events.events_data
    df = df[(df["type"] == btype) & ((df["initiator"] == animal) | (df["victim"] == animal))]
    df = df.dropna(subset=["ts_start_ephys", "initiator", "victim"])
    rats = sorted(set(df["initiator"].tolist() + df["victim"].tolist()))
    colors = {r: PALETTE[i % 10] for i, r in enumerate(rats)}
    return np.asarray(ts, float), np.asarray(opp), colors


def load_fit(cohort: str, session: str, animal: str, key: str) -> dict:
    path = xc.cache_path(cohort, session, animal)
    if not path.exists():
        raise SystemExit(f"No explore cache at {path}. Open the block in the dashboard first.")
    with open(path, "rb") as f:
        entry = pickle.load(f)
    fit = entry.get(key)
    if fit is None:
        raise SystemExit(f"'{key}' is not cached for this block. Press its Fit button in the "
                         f"dashboard first (cached keys: {[k for k in EMBEDDINGS.values() if entry.get(k)]}).")
    return fit


def render(fit: dict, out: Path, *, title: str, axis: str, events=None, fps: int, seconds_per_rev: float, revs: float,
           elev: float, stride: int, t_range, size: int, dark: bool, start_azim: float) -> int:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    scores, t = np.asarray(fit["scores"]), np.asarray(fit["bin_centers"])
    keep = np.ones(len(t), bool) if t_range is None else (t >= t_range[0]) & (t <= t_range[1])
    scores, t = scores[keep][::stride], t[keep][::stride]
    if len(scores) < 2:
        raise SystemExit("No points left after the time window / stride.")

    bg, fg = ("#111111", "#dddddd") if dark else ("white", "#222222")
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100, facecolor=bg)
    ax = fig.add_axes([0.03, 0.02, 0.88, 0.9], projection="3d", facecolor=bg)
    ax.computed_zorder = False   # honour zorder: markers always draw above the lines
    pts = scores.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap="viridis", linewidths=0.8, zorder=1, alpha=0.6 if events is None else 0.1)
    lc.set_array(t[:-1])
    lc.set_clim(t[0], t[-1])
    ax.add_collection3d(lc)
    n_events = 0
    markers = []
    if events is not None:
        ev_t, ev_opp, colors = events
        full, tt = np.asarray(fit["scores"]), np.asarray(fit["bin_centers"])
        # Same placement as the dashboard: the bin containing the event start.
        inside = (ev_t >= t[0]) & (ev_t <= t[-1])
        idx = np.searchsorted(tt, ev_t[inside]).clip(0, len(tt) - 1)
        # Pushed just outside the sphere the embedding lies on, so they sit on top of the lines.
        pos, who = full[idx] * (1.0 + SHELL), ev_opp[inside]
        n_events = len(pos)
        for o in np.unique(who):
            m = who == o
            sc = ax.scatter(pos[m, 0], pos[m, 1], pos[m, 2], s=30, c=colors.get(o, "grey"),
                            edgecolors="black", linewidths=0.5, depthshade=False, label=o,
                            zorder=10)
            markers.append((sc, matplotlib.colors.to_rgb(colors.get(o, "grey")), pos[m]))
        if n_events:
            ax.legend(title="Opponent", loc="upper left", fontsize=8, title_fontsize=8,
                      facecolor=bg, labelcolor=fg, edgecolor="none")
            ax.get_legend().get_title().set_color(fg)
    print(f"{n_events} event markers")
    lo, hi = scores.min(0), scores.max(0)
    if markers:
        allp = np.concatenate([m[2] for m in markers])
        lo, hi = np.minimum(lo, allp.min(0)), np.maximum(hi, allp.max(0))
    pad = 0.03 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    ax.set_box_aspect((1, 1, 1), zoom=0.95)
    ax.set_xlabel(f"{axis} 1", color=fg)
    ax.set_ylabel(f"{axis} 2", color=fg)
    ax.set_zlabel(f"{axis} 3", color=fg)
    ax.tick_params(colors=fg, labelsize=7)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0, 0, 0, 0))
    cb = fig.colorbar(lc, ax=ax, shrink=0.55, pad=-0.02)
    cb.solids.set_alpha(1.0)   # the bar would otherwise inherit the lines' transparency
    cb.set_label("Time (s)", color=fg)
    cb.ax.tick_params(colors=fg, labelsize=7)
    fig.suptitle(title, color=fg, fontsize=11, y=0.985)

    n_frames = int(round(revs * seconds_per_rev * fps))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not writer.isOpened():
        raise SystemExit(f"Could not open {out} for writing.")
    try:
        for i in range(n_frames):
            azim = start_azim + 360.0 * revs * i / n_frames
            ax.view_init(elev=elev, azim=azim)
            if markers:   # dim markers on the far side of the sphere (facing away from the camera)
                a, e = np.radians(azim), np.radians(elev)
                cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
                for sc, rgb, xyz in markers:
                    front = (xyz @ cam) >= 0
                    sc.set_facecolors([(*rgb, 1.0 if f else BACK_ALPHA) for f in front])
                    sc.set_edgecolors([(0, 0, 0, 1.0 if f else BACK_ALPHA) for f in front])
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
            if i % 50 == 0:
                print(f"  frame {i}/{n_frames}", flush=True)
    finally:
        writer.release()
        plt.close(fig)
    return n_frames


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--cohort", default="cohort7")
    p.add_argument("--session", required=True, help="recording id as used in the cache, e.g. 20251216_094334")
    p.add_argument("--animal", required=True, help="e.g. rat631")
    p.add_argument("--embedding", choices=sorted(EMBEDDINGS), default="cebra-pos")
    p.add_argument("--events", default="EC", metavar="TYPE",
                   help="behaviour type to mark (EC, F, ...); 'none' for no markers")
    p.add_argument("--min-events", type=int, default=3, help="min events per opponent")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds-per-rev", type=float, default=40.0, help="time for one full 360° turn")
    p.add_argument("--revs", type=float, default=1.0, help="number of turns (1 loops seamlessly)")
    p.add_argument("--elev", type=float, default=20.0, help="camera elevation in degrees")
    p.add_argument("--start-azim", type=float, default=-60.0)
    p.add_argument("--stride", type=int, default=2, help="draw every Nth bin (faster)")
    p.add_argument("--t-start", type=float, default=None)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--size", type=int, default=900, help="square frame size in pixels")
    p.add_argument("--dark", action="store_true")
    a = p.parse_args(argv)

    key = EMBEDDINGS[a.embedding]
    fit = load_fit(a.cohort, a.session, a.animal, key)
    t_range = None if a.t_start is None and a.t_end is None else (
        a.t_start if a.t_start is not None else -np.inf, a.t_end if a.t_end is not None else np.inf)
    events = None if a.events.lower() == "none" else load_events(
        xc.cache_path(a.cohort, a.session, a.animal), a.animal, a.events, a.min_events)
    tag = "" if events is None else f"_{a.events}"
    out = a.out or ROOT / "results" / "trajectory_exports" / f"{a.animal}_{a.session}_{a.embedding}{tag}_rotation.mp4"
    title = f"{TITLES[key]} — {a.animal} {a.session}"
    n = render(fit, out, title=title, axis=AXES[key], events=events, fps=a.fps, seconds_per_rev=a.seconds_per_rev, revs=a.revs,
               elev=a.elev, stride=a.stride, t_range=t_range, size=a.size, dark=a.dark,
               start_azim=a.start_azim)
    print(f"wrote {out}  ({n} frames, {n / a.fps:.1f} s)")


if __name__ == "__main__":
    main()
