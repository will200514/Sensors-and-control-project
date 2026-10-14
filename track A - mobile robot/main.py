
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import Cfg 
from matplotlib.patches import Circle, Ellipse
from src import config as C
from src.mission import run_trial


def ellipse(ax, mean, cov, n=2.0, **kw):
    w, V = np.linalg.eigh(cov); w = np.maximum(w, 1e-12)
    ax.add_patch(Ellipse(mean, 2 * n * np.sqrt(w[0]), 2 * n * np.sqrt(w[1]),
                         angle=np.degrees(np.arctan2(V[1, 0], V[0, 0])), **kw))


def plot_run(r, path):
    T = r["log"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    a = ax[0]
    a.imshow(r["grid"].T, origin="lower", extent=(0, 5, 0, 5), cmap="Greys", alpha=0.25)
    for cx, cy, rr in r["obstacles"]:
        a.add_patch(Circle((cx, cy), rr, color="tab:gray", alpha=0.6))
    a.plot(C.MARKERS[:, 0], C.MARKERS[:, 1], "ks", ms=5, label="ArUco markers")
    a.plot(*C.DOCK_MARKER[:2], "gs", ms=9, label="dock marker")
    a.plot(*C.WAYPOINT_1, "b*", ms=12, label="waypoint 1")
    a.plot(T["dr"][:, 0], T["dr"][:, 1], "b--", lw=1, label="dead reckoning")
    a.plot(T["true"][:, 0], T["true"][:, 1], "k", lw=2, label="true path")
    a.plot(T["est"][:, 0], T["est"][:, 1], "r", lw=1.2, label="EKF estimate")
    for i in range(0, len(T["est"]), 60):
        ellipse(a, T["est"][i, :2], T["P"][i][:2, :2], 3.0, fc="none", ec="r", lw=0.7, alpha=0.7)
    a.set_aspect("equal"); a.set_xlim(-0.1, 5.1); a.set_ylim(-0.1, 5.1)
    a.legend(fontsize=7, loc="upper right"); a.set_title(f"seed {r['seed']}: {r['reason']}, success={r['success']}")
    err = T["true"] - T["est"]; err[:, 2] = (err[:, 2] + np.pi) % (2 * np.pi) - np.pi
    sig = np.sqrt(np.array([P[0, 0] + P[1, 1] for P in T["P"]]))
    ax[1].plot(T["t"], np.hypot(err[:, 0], err[:, 1]), "r", label="actual position error")
    ax[1].plot(T["t"], 2 * sig, "k--", label=r"2$\sigma$ (x,y)")
    ax[1].plot(T["t"], np.hypot(*(T["true"][:, :2] - T["dr"][:, :2]).T), "b:", label="dead reckoning error")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("[m]"); ax[1].legend(fontsize=8); ax[1].set_title("estimate vs truth")
    ax[2].plot(T["t"], r["nees"], lw=0.8); ax[2].axhline(3, color="k", ls=":")
    ax[2].axhline(7.81, color="r", ls="--", lw=0.8); ax[2].set_yscale("log")
    ax[2].set_xlabel("t [s]"); ax[2].set_title(f"NEES (mean {r['nees_mean']:.2f}; ideal 3, 95% bound 7.8)")
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=3)
    # A4-MUJOCO: added --backend, --view, --push and passed them to run_trial
    ap.add_argument("--backend", choices=["numpy", "mujoco"], default="numpy")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--push", type=float, default=-1.0)
    a = ap.parse_args()
    r = run_trial(a.seed, Cfg(backend=a.backend, viewer=a.view, push_time=a.push))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
           if k in ("success", "reason", "collided", "dock_pos_err", "dock_head_err", "time", "rmse_pos",
                    "rmse_head", "rmse_pos_dr", "track_rmse", "nees_mean", "n_updates", "n_rejected", "plans")})
    # A4-MUJOCO: MuJoCo plots get a suffix so they do not overwrite the NumPy ones
    tag = "" if a.backend == "numpy" else "_mujoco" + ("_push" if a.push >= 0 else "")
    plot_run(r, C.RESULTS_DIR / f"run_seed{a.seed}{tag}.png")
