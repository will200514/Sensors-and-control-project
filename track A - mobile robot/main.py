"""Run MuJoCo and save diagnostic plots. macOS: mjpython main.py --seed 3."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from src import config as C
from src.config import Cfg
from src.mission import run_trial
from src.model.diffdrive import wrap
from tools.make_room import build_room


def ellipse(ax, mean, cov, n=2.0, **kwargs):
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, 1e-12)
    patch = Ellipse(
        mean, 2 * n * np.sqrt(values[0]), 2 * n * np.sqrt(values[1]),
        angle=np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])), **kwargs,
    )
    ax.add_patch(patch)
    return patch


def plot_run(result, path):
    if "log" not in result:
        print(f"No samples to plot: {result['reason']}")
        return
    data = result["log"]
    time = data["t"]
    error = data["true"] - data["est"]
    error[:, 2] = wrap(error[:, 2])
    covariance = data["P"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    route, position, heading, tracking, consistency, visibility = axes.flat

    route.imshow(result["grid"].T, origin="lower", extent=C.ROOM, cmap="Greys", alpha=0.25)
    for x, y, radius in result["obstacles"]:
        route.add_patch(Circle((x, y), radius, color="gray", alpha=0.6))
    route.plot(*C.MARKERS[0, :2], "ks", label="Marker 0")
    route.plot(*C.DOCK_MARKER[:2], "gs", label="Dock marker 1")
    for key, style, label in [("true", "k-", "Actual"), ("est", "r-", "EKF"), ("dr", "b--", "Odometry only")]:
        route.plot(data[key][:, 0], data[key][:, 1], style, label=label)
    for i in range(0, len(time), 60):
        ellipse(route, data["est"][i, :2], covariance[i, :2, :2], 2.0,
                fc="none", ec="r", alpha=0.4)
    route.set(xlim=(-0.1, 5.1), ylim=(-0.1, 5.1), xlabel="x [m]", ylabel="y [m]", title="Robot path and 2-sigma ellipses")
    route.set_aspect("equal")
    route.legend(fontsize=8)

    position.plot(time, np.linalg.norm(error[:, :2], axis=1), "r", label="EKF position error")
    position.plot(time, np.linalg.norm(data["true"][:, :2] - data["dr"][:, :2], axis=1), "b:", label="Odometry-only error")
    position.plot(time, 2 * np.sqrt(covariance[:, 0, 0] + covariance[:, 1, 1]), "k--", label="2 sqrt(Pxx + Pyy), scale only")
    position.set(title="Position error", ylabel="Error [m]")
    position.legend(fontsize=8)

    heading.plot(time, np.degrees(error[:, 2]), "r", label="Heading error")
    bound = 2 * np.degrees(np.sqrt(np.maximum(covariance[:, 2, 2], 0)))
    heading.fill_between(time, -bound, bound, color="gray", alpha=0.25, label="±2 sigma")
    heading.set(title="Heading error", ylabel="Error [degrees]")
    heading.legend(fontsize=8)

    distance = np.linalg.norm(data["true"][:, :2] - data["ref"][:, :2], axis=1)
    tracking.plot(time, distance)
    tracking.set(title="Distance to active reference point", ylabel="Distance [m]")

    consistency.plot(time, np.maximum(result["nees"], 1e-12), label="NEES")
    consistency.axhline(3, color="k", ls=":", label="Expected value: 3")
    consistency.axhline(7.815, color="r", ls="--", label="95th percentile, 3 DOF")
    consistency.set(yscale="log", title="Filter consistency (NEES)")
    consistency.legend(fontsize=8)

    visibility.step(time, data["n_vis"], where="post")
    visibility.set(title="Detected markers (latest camera frame)", ylabel="Count", yticks=[0, 1, 2], ylim=(-0.1, 2.1))
    for ax in [position, heading, tracking, consistency, visibility]:
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.2)
    fig.suptitle(f"MuJoCo seed {result['seed']} | {result['reason']} | success={result['success']} | dock error={result['dock_pos_err']:.3f} m")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--push", type=float, default=-1.0)
    parser.add_argument("--no-view", action="store_true", help="Run MuJoCo without the live viewer")
    args = parser.parse_args()
    build_room()  # Keep rendered markers synchronised with configuration.
    result = run_trial(args.seed, Cfg(viewer=not args.no_view, push_time=args.push))
    keys = ("success", "reason", "collided", "dock_pos_err", "dock_head_err", "time",
            "rmse_pos", "rmse_head", "rmse_pos_dr", "track_rmse", "nees_mean",
            "n_updates", "n_rejected", "n_recoveries", "plans")
    print({k: round(v, 4) if isinstance(v, (float, np.floating)) else v for k, v in result.items() if k in keys})
    tag = "_push" if args.push >= 0 else ""
    plot_run(result, C.RESULTS_DIR / f"run_seed{args.seed}_mujoco{tag}.png")


if __name__ == "__main__":
    main()

