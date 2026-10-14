"""Animated demo (GIF) for the video: true pose vs EKF estimate with its 2-sigma ellipse.
    python make_demo.py --seed 3     (convert to mp4 with ffmpeg if you want a video file)"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from src import config as C
from src.mission import run_trial
from main import ellipse
from src.config import Cfg

# A4-MUJOCO: added --backend and --push
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--backend", choices=["numpy", "mujoco"], default="numpy"); ap.add_argument("--push", type=float, default=-1.0)
a = ap.parse_args()
r = run_trial(a.seed, Cfg(backend=a.backend, push_time=a.push)); T = r["log"]; step = 4
fig, ax = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw=dict(width_ratios=[1.2, 1]))
m = ax[0]; m.set_aspect("equal"); m.set_xlim(-0.1, 5.1); m.set_ylim(-0.1, 5.1)
for cx, cy, rr in r["obstacles"]:
    m.add_patch(Circle((cx, cy), rr, color="gray", alpha=0.6))
m.plot(C.MARKERS[:, 0], C.MARKERS[:, 1], "ks", ms=5); m.plot(*C.DOCK_MARKER[:2], "gs", ms=9)
m.plot(*C.WAYPOINT_1, "b*", ms=12)
(lt,) = m.plot([], [], "k", lw=2, label="true"); (le,) = m.plot([], [], "r", lw=1.3, label="EKF")
(ld,) = m.plot([], [], "b--", lw=1, label="dead reckoning"); (rb,) = m.plot([], [], "ko", ms=7)
ell = [ellipse(m, (0, 0), np.eye(2) * 1e-6, 2.0, fc="none", ec="r") or None]
m.legend(loc="upper right", fontsize=8)
err = np.hypot(*(T["true"][:, :2] - T["est"][:, :2]).T)
sig = 2 * np.sqrt(np.array([P[0, 0] + P[1, 1] for P in T["P"]]))
e = ax[1]; e.set_xlim(0, T["t"][-1]); e.set_ylim(0, max(err.max(), sig[5:].max()) * 1.1)
(l1,) = e.plot([], [], "r", label="position error"); (l2,) = e.plot([], [], "k--", label=r"2$\sigma$")
e.set_xlabel("t [s]"); e.set_ylabel("[m]"); e.legend()

def draw(i):
    k = i * step
    lt.set_data(T["true"][:k + 1, 0], T["true"][:k + 1, 1]); le.set_data(T["est"][:k + 1, 0], T["est"][:k + 1, 1])
    ld.set_data(T["dr"][:k + 1, 0], T["dr"][:k + 1, 1]); rb.set_data([T["true"][k, 0]], [T["true"][k, 1]])
    for p in list(m.patches):
        if p.__class__.__name__ == "Ellipse":
            p.remove()
    ellipse(m, T["est"][k, :2], T["P"][k][:2, :2], 2.0, fc="none", ec="r", lw=1.5)
    l1.set_data(T["t"][:k + 1], err[:k + 1]); l2.set_data(T["t"][:k + 1], sig[:k + 1])
    m.set_title(f"t={T['t'][k]:.1f}s  markers in view: {T['n_vis'][k]}")
    return lt, le

anim = animation.FuncAnimation(fig, draw, frames=len(T["t"]) // step, interval=60)
tag = "" if a.backend == "numpy" else "_mujoco" + ("_push" if a.push >= 0 else "")   # A4-MUJOCO
anim.save(C.RESULTS_DIR / f"demo_seed{a.seed}.gif", writer=animation.PillowWriter(fps=15))
print("saved results/demo_seed%d.gif" % a.seed)
