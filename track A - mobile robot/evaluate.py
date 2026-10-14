"""Systematic evaluation on HELD-OUT seeds (1000+), with ablations. Writes results/summary.md and figures.
    python evaluate.py            (full run, ~1-2 min)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
from src.mission import run_trial
from src.config import Cfg
from src import config as C

SEEDS = range(1000, 1050)      # 50 held-out trials for the baseline
ABL_SEEDS = range(1000, 1030)  # 30 trials per ablation (same scenarios as the baseline's first 30)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def diagnose(r):
    if r["success"]:
        return "ok"
    if r["collided"]:
        return "collision"
    if r["reason"] != "docked":
        return r["reason"]
    if r["rmse_pos"] > 0.3:
        return "estimator lost"
    return "dock tolerance"


def summarise(name, R):
    n = len(R); k = sum(r["success"] for r in R)
    lo, hi = wilson(k, n)
    ne = np.concatenate([r["nees"] for r in R])
    return dict(name=name, n=n, succ=k / n, lo=lo, hi=hi,
                dock_med=np.median([r["dock_pos_err"] for r in R]) * 100,
                dock_max=np.max([r["dock_pos_err"] for r in R]) * 100,
                head_med=np.degrees(np.median([r["dock_head_err"] for r in R])),
                time=np.mean([r["time"] for r in R]),
                rmse=np.mean([r["rmse_pos"] for r in R]) * 100,
                rmse_dr=np.mean([r["rmse_pos_dr"] for r in R]) * 100,
                nees_med=np.median(ne), nees_in=np.mean(ne < 7.81),
                track=np.mean([r["track_rmse"] for r in R]) * 100,
                effort=np.mean([r["effort"] for r in R]),
                coll=sum(r["collided"] for r in R),
                recov=np.mean([r["n_recoveries"] for r in R]),
                fails=[(r["seed"], diagnose(r)) for r in R if not r["success"]])


def run(name, cfg, seeds):
    R = [run_trial(s, cfg) for s in seeds]
    print(f"{name:34s} done", flush=True)
    return R


if __name__ == "__main__":
    # A4-MUJOCO: "python evaluate.py mujoco N" runs only the baseline and a push trial on MuJoCo, N seeds each
    # (a MuJoCo trial takes about 20 s, the full ablation set would take hours). No arguments = original behaviour.
    if len(sys.argv) > 1 and sys.argv[1] == "mujoco":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        base = Cfg(backend="mujoco")
        rows = [summarise(name, run(name, cfg, range(1000, 1000 + n))) for name, cfg in
                [("MuJoCo baseline", base), ("MuJoCo, 6 N push at 12 s", replace(base, push_time=12.0))]]
        with open(C.RESULTS_DIR / "summary_mujoco.md", "w") as f:
            f.write("| config | n | success | dock pos med/max [cm] | est. RMSE [cm] | NEES median | collisions |\n|---|---|---|---|---|---|---|\n")
            for s in rows:
                f.write(f"| {s['name']} | {s['n']} | {s['succ']*100:.0f}% | {s['dock_med']:.1f} / {s['dock_max']:.1f} | "
                        f"{s['rmse']:.1f} | {s['nees_med']:.2f} | {s['coll']} |\n")
        print(open(C.RESULTS_DIR / "summary_mujoco.md").read())
        sys.exit()
    base = Cfg()
    configs = [
        ("baseline (submitted design)", base),
        ("no camera (odometry only)", replace(base, use_camera=False)),
        ("process noise Q x0.1", replace(base, q_scale=0.4)),
        ("process noise Q x10", replace(base, q_scale=40.0)),
        ("camera noise R x0.1", replace(base, r_scale=0.2)),
        ("camera noise R x10", replace(base, r_scale=20.0)),
        ("innovation gate off", replace(base, gate=None)),
        ("calibration error fx +5%", replace(base, fx_filter=530.0 * 1.05)),
        ("calibration error fx -5%", replace(base, fx_filter=530.0 * 0.95)),
        ("narrow camera FOV (30 deg)", replace(base, cam_fov_deg=30.0)),
        ("wheel slip bias +/-5% (stress)", replace(base, extra_disturbance=0.05)),
        ("7 obstacles (stress)", replace(base, n_obstacles=7)),
    ]
    results, rows = {}, []
    for i, (name, cfg) in enumerate(configs):
        R = run(name, cfg, SEEDS if i == 0 else ABL_SEEDS)
        results[name] = R; rows.append(summarise(name, R))

    # LQR weight sweep: control effort vs tracking error
    lqr_rows = []
    for rs in [0.01, 0.1, 1.0, 10.0, 100.0]:
        R = run(f"LQR R x{rs}", replace(base, lqr_r_scale=rs), ABL_SEEDS)
        lqr_rows.append((rs, summarise(f"LQR R x{rs}", R)))

    with open(C.RESULTS_DIR / "summary.md", "w") as f:
        f.write("## Baseline and ablations (held-out seeds)\n\n")
        f.write("| config | n | success [95% CI] | dock pos med/max [cm] | dock head med [deg] | time [s] | "
                "est. RMSE [cm] | dead-reck. RMSE [cm] | NEES median | frac NEES<7.81 | track RMSE [cm] | collisions |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for s in rows:
            f.write(f"| {s['name']} | {s['n']} | {s['succ']*100:.0f}% [{s['lo']*100:.0f}, {s['hi']*100:.0f}] | "
                    f"{s['dock_med']:.1f} / {s['dock_max']:.1f} | {s['head_med']:.1f} | {s['time']:.0f} | "
                    f"{s['rmse']:.1f} | {s['rmse_dr']:.1f} | {s['nees_med']:.2f} | {s['nees_in']:.2f} | "
                    f"{s['track']:.1f} | {s['coll']} |\n")
        f.write("\n## Failure diagnosis (seed: cause)\n\n")
        for s in rows:
            if s["fails"]:
                f.write(f"- **{s['name']}**: " + ", ".join(f"{a}: {b}" for a, b in s["fails"]) + "\n")
        f.write("\n## LQR weight trade-off (R scaled; Q fixed)\n\n| R scale | success | track RMSE [cm] | control effort mean(v^2+w^2) |\n|---|---|---|---|\n")
        for rs, s in lqr_rows:
            f.write(f"| {rs} | {s['succ']*100:.0f}% | {s['track']:.2f} | {s['effort']:.4f} |\n")
    print(open(C.RESULTS_DIR / "summary.md").read())

    # ---- figures ----
    B = results[configs[0][0]]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].hist([r["dock_pos_err"] * 100 for r in B], bins=15, color="tab:green")
    ax[0].axvline(5, color="k", ls="--"); ax[0].set_xlabel("final docking position error [cm]"); ax[0].set_title("docking accuracy (50 trials)")
    ne = np.concatenate([r["nees"] for r in B])
    ax[1].hist(np.clip(ne, 0, 30), bins=60, density=True, alpha=0.7, label="measured NEES")
    from scipy.stats import chi2
    xs = np.linspace(0.01, 30, 300); ax[1].plot(xs, chi2.pdf(xs, 3), "k", label=r"$\chi^2_3$ (ideal)")
    ax[1].set_xlabel("NEES (clipped at 30)"); ax[1].legend(); ax[1].set_title("estimator consistency")
    ax[2].barh([s["name"] for s in rows][::-1], [s["succ"] * 100 for s in rows][::-1], color="tab:blue")
    ax[2].set_xlabel("success rate [%]"); ax[2].set_title("ablations"); ax[2].tick_params(axis="y", labelsize=7)
    plt.tight_layout(); fig.savefig(C.RESULTS_DIR / "evaluation.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].loglog([s["effort"] for _, s in lqr_rows], [s["track"] for _, s in lqr_rows], "o-")
    for rs, s in lqr_rows:
        ax[0].annotate(f"R x{rs:g}", (s["effort"], s["track"]), fontsize=8)
    ax[0].set_xlabel("control effort mean(v²+ω²)"); ax[0].set_ylabel("tracking RMSE [cm]"); ax[0].set_title("LQR trade-off")
    ax[1].semilogx([rs for rs, _ in lqr_rows], [s["succ"] * 100 for _, s in lqr_rows], "o-")
    ax[1].set_xlabel("R scale"); ax[1].set_ylabel("success [%]")
    plt.tight_layout(); fig.savefig(C.RESULTS_DIR / "lqr_tradeoff.png", dpi=130); plt.close(fig)
