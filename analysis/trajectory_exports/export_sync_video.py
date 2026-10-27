"""Side-by-side video: CEBRA-position trajectory (left) and the rat's (x, y) trajectory (right),
both building up over the same stretch of recording with the same events and a shared clock.

Left panel: 3-D CEBRA-position embedding, drawn up to the current time and slowly rotating about
the z-axis; a red ring marks the current bin and the last few seconds are emphasised.
Right panel: the (x, y) path, same as ``export_trajectory_video.py``.
Event markers (coloured by opponent) appear in both panels when they happen. Reads the explore
cache only; nothing is refitted. Playback covers the tracked time range, since the spatial panel
has nothing to show outside it.

    python analysis/trajectory_exports/export_sync_video.py --animal rat631 --session 20251216_094334
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
from analysis.trajectory_exports.export_embedding_video import BACK_ALPHA, SHELL, load_events  # noqa: E402
from analysis.trajectory_exports.export_trajectory_video import (EVENT_SNAP_S, FORMAT_NAMES, MAX_GAP_S,  # noqa: E402
                                     MAX_PATH_POINTS, TRAIL_S)


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
    p.add_argument("--revs", type=float, default=1.0, help="turns of the 3-D view over the video")
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--start-azim", type=float, default=-60.0)
    p.add_argument("--height", type=int, default=800, help="frame height in pixels (width is 2x)")
    p.add_argument("--dark", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    path = xc.cache_path(a.cohort, a.session, a.animal)
    if not path.exists():
        raise SystemExit(f"No explore cache at {path}. Open the block in the dashboard first.")
    with open(path, "rb") as f:
        entry = pickle.load(f)
    fit = entry.get("cebra_position")
    if fit is None:
        raise SystemExit("CEBRA (position) is not cached for this block. Fit it in the dashboard first.")
    formats = xc.trajectory_formats(entry.get("trajectory"))
    source = a.source or fit.get("params", {}).get("tracking") or entry["trajectory"].get("default")
    if source not in formats:
        raise SystemExit(f"Tracking source '{source}' not cached (have {sorted(formats)}).")
    t, x, y = (np.asarray(formats[source][k], float) for k in ("t", "x", "y"))
    src = FORMAT_NAMES[source]
    emb, bc = np.asarray(fit["scores"]), np.asarray(fit["bin_centers"])

    lo = a.t_start if a.t_start is not None else t[0]
    hi = a.t_end if a.t_end is not None else t[-1]
    lo, hi = max(lo, t[0], bc[0]), min(hi, t[-1], bc[-1])
    keep = (t >= lo) & (t <= hi)
    if keep.sum() < 2 or hi <= lo:
        raise SystemExit(f"No overlap between {src} tracking and the CEBRA fit in the window.")
    t, x, y = t[keep], x[keep], y[keep]
    stride = max(1, int(np.ceil(len(t) / MAX_PATH_POINTS)))
    ts, xs, ys = t[::stride], x[::stride], y[::stride]
    cm = (bc >= lo) & (bc <= hi)
    emb_v, bc_v = emb[cm], bc[cm]

    # One event set for both panels: placed in space (snapped to a tracked frame) and in CEBRA
    # space (the bin containing the event start), so the two panels always agree.
    ev_t = ev_x = ev_y = np.array([]); ev_opp = np.array([]); ev_pos = np.empty((0, 3)); colors = {}
    if a.events.lower() != "none":
        et, eo, colors = load_events(path, a.animal, a.events, a.min_events)
        m = (et >= lo) & (et <= hi)
        et, eo = et[m], eo[m]
        idx = np.clip(np.searchsorted(t, et), 1, len(t) - 1)
        idx = np.where(np.abs(t[idx - 1] - et) < np.abs(t[idx] - et), idx - 1, idx)
        ok = np.abs(t[idx] - et) <= EVENT_SNAP_S
        ev_t, ev_x, ev_y, ev_opp = et[ok], x[idx[ok]], y[idx[ok]], eo[ok]
        ev_pos = emb[np.searchsorted(bc, ev_t).clip(0, len(bc) - 1)] * (1.0 + SHELL)
        print(f"{len(ev_t)} {a.events} events placed, {int((~ok).sum())} not tracked at event time")

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    bg, fg = ("#111111", "#dddddd") if a.dark else ("white", "#222222")
    W, H = 2 * a.height, a.height
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=bg)

    # ── left: CEBRA-position, 3-D ──────────────────────────────────────────────
    ax3 = fig.add_axes([0.0, 0.03, 0.5, 0.86], projection="3d", facecolor=bg)
    ax3.computed_zorder = False     # markers always draw above the lines
    pts = emb_v.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection([], cmap="viridis", linewidths=0.8, alpha=0.3, zorder=1)
    lc.set_clim(lo, hi)
    ax3.add_collection3d(lc, autolim=False)
    (trail3,) = ax3.plot([], [], [], color=fg, lw=1.8, alpha=0.9, zorder=5)
    ring3 = ax3.scatter([], [], [], s=220, facecolors="none", edgecolors="#e53e3e",
                        linewidths=2.2, depthshade=False, zorder=12)
    ev3 = []
    for o in np.unique(ev_opp):
        sc = ax3.scatter([], [], [], s=30, c=colors.get(o, "grey"), edgecolors="black",
                         linewidths=0.5, depthshade=False, zorder=10)
        ev3.append((sc, matplotlib.colors.to_rgb(colors.get(o, "grey")), ev_opp == o))
    allp = np.concatenate([emb_v, ev_pos]) if len(ev_pos) else emb_v
    mn, mx = allp.min(0), allp.max(0)
    pad = 0.03 * (mx - mn)
    ax3.set_xlim(mn[0] - pad[0], mx[0] + pad[0])
    ax3.set_ylim(mn[1] - pad[1], mx[1] + pad[1])
    ax3.set_zlim(mn[2] - pad[2], mx[2] + pad[2])
    ax3.set_box_aspect((1, 1, 1), zoom=0.95)
    for lab, setter in zip("123", (ax3.set_xlabel, ax3.set_ylabel, ax3.set_zlabel)):
        setter(f"CEBRA-pos {lab}", color=fg)
    ax3.tick_params(colors=fg, labelsize=7)
    for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        pane.set_pane_color((0, 0, 0, 0))

    # ── right: spatial trajectory ──────────────────────────────────────────────
    ax2 = fig.add_axes([0.53, 0.12, 0.38, 0.74], facecolor=bg)
    pad2 = 0.03 * max(np.ptp(x), np.ptp(y))
    ax2.set_xlim(x.min() - pad2, x.max() + pad2)
    ax2.set_ylim(y.max() + pad2, y.min() - pad2)      # image coordinates, as in the dashboard
    ax2.set_aspect("equal")
    ax2.set_xlabel(f"x ({src} pixels)", color=fg)
    ax2.set_ylabel(f"y ({src} pixels)", color=fg)
    ax2.tick_params(colors=fg, labelsize=8)
    for s in ax2.spines.values():
        s.set_color(fg)
    (line2,) = ax2.plot([], [], color="grey", lw=0.6, alpha=0.25)
    path_sc = ax2.scatter([], [], c=[], cmap="viridis", s=3, alpha=0.5, vmin=lo, vmax=hi,
                          linewidths=0)
    (trail2,) = ax2.plot([], [], color=fg, lw=1.6, alpha=0.8, zorder=5)
    ring2 = ax2.scatter([], [], s=260, facecolors="none", edgecolors="#e53e3e", linewidths=2.2,
                        zorder=12)
    ev2 = []
    for o in np.unique(ev_opp):
        sc = ax2.scatter([], [], s=70, c=colors.get(o, "grey"), edgecolors="black",
                         linewidths=0.6, zorder=10, label=o)
        ev2.append((sc, ev_opp == o))
    if ev2:
        leg = ax2.legend(title="Opponent", loc="upper left", fontsize=8, title_fontsize=8,
                         facecolor=bg, labelcolor=fg, edgecolor="none")
        leg.get_title().set_color(fg)

    cax = fig.add_axes([0.945, 0.22, 0.012, 0.5])
    cb = fig.colorbar(path_sc, cax=cax)
    cb.set_label("Time (s)", color=fg)
    cb.ax.tick_params(colors=fg, labelsize=8)
    cb.solids.set_alpha(1.0)
    fig.text(0.25, 0.955, "CEBRA-Behavior (x, y)", color=fg, ha="center", fontsize=12)
    fig.text(0.72, 0.955, f"{a.animal} trajectory ({src})", color=fg, ha="center", fontsize=12)
    clock = fig.text(0.72, 0.045, "", color=fg, ha="center", fontsize=12, family="monospace")

    n_frames = max(2, int(round(a.duration * a.fps)))
    out = a.out or ROOT / "results" / "trajectory_exports" / (
        f"{a.animal}_{a.session}_cebra-pos_vs_space_{src}"
        f"{'' if a.events.lower() == 'none' else '_' + a.events}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    empty3, empty2 = np.empty((0, 3)), np.empty((0, 2))
    try:
        for i in range(n_frames):
            f01 = i / (n_frames - 1)
            T = lo + (hi - lo) * f01
            # left
            kb = int(np.searchsorted(bc_v, T, side="right"))
            lc.set_segments(segs[:max(kb - 1, 0)])
            lc.set_array(bc_v[:max(kb - 1, 0)])
            b0 = int(np.searchsorted(bc_v, T - TRAIL_S, side="left"))
            tr = emb_v[b0:kb]
            trail3.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])
            cur = emb_v[max(kb - 1, 0)]
            ring3._offsets3d = ([cur[0]], [cur[1]], [cur[2]])
            azim = a.start_azim + 360.0 * a.revs * f01
            ax3.view_init(elev=a.elev, azim=azim)
            az, el = np.radians(azim), np.radians(a.elev)
            cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            n_ev = 0
            for sc, rgb, mask in ev3:
                sel = mask & (ev_t <= T)
                pos = ev_pos[sel]
                sc._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2]) if len(pos) else ([], [], [])
                front = (pos @ cam) >= 0 if len(pos) else np.array([], bool)
                sc.set_facecolors([(*rgb, 1.0 if fr else BACK_ALPHA) for fr in front] or [(*rgb, 1)])
                sc.set_edgecolors([(0, 0, 0, 1.0 if fr else BACK_ALPHA) for fr in front] or [(0, 0, 0, 1)])
                n_ev += int(sel.sum())
            # right
            k = int(np.searchsorted(ts, T, side="right"))
            tk, xk, yk = ts[:k], xs[:k], ys[:k]
            gap = np.r_[False, np.diff(tk) > MAX_GAP_S] if k else np.array([], bool)
            line2.set_data(np.where(gap, np.nan, xk), np.where(gap, np.nan, yk))
            path_sc.set_offsets(np.c_[xk, yk] if k else empty2)
            path_sc.set_array(tk)
            j1 = int(np.searchsorted(t, T, side="right"))
            j0 = int(np.searchsorted(t, T - TRAIL_S, side="left"))
            tx, ty = x[j0:j1].copy(), y[j0:j1].copy()
            if len(tx) > 1:
                g = np.r_[False, np.diff(t[j0:j1]) > MAX_GAP_S]
                tx[g] = np.nan
                ty[g] = np.nan
            trail2.set_data(tx, ty)
            c = np.clip(j1 - 1, 0, len(t) - 1)
            ring2.set_offsets([[x[c], y[c]]] if abs(t[c] - T) <= MAX_GAP_S else empty2)
            for sc, mask in ev2:
                sel = mask & (ev_t <= T)
                sc.set_offsets(np.c_[ev_x[sel], ev_y[sel]] if sel.any() else empty2)
            clock.set_text(f"t = {T:7.1f} s" + (f"    {n_ev} {a.events} events" if ev2 else ""))

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
