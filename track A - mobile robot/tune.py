"""Tune the estimator noise scales on a TUNING seed set (0-19). Evaluation uses held-out seeds
(evaluate.py), so the reported numbers are not fitted to the trials they are reported on.
Criterion: average NEES close to 3 (consistency), then lowest position RMSE."""
import numpy as np
from src.mission import run_trial
from src.config import Cfg

if __name__ == "__main__":
    rows = []
    print(f"{'q_scale':>8}{'r_scale':>8}{'succ':>7}{'NEES':>8}{'frac<7.81':>11}{'rmse[cm]':>10}{'head[deg]':>11}")
    for qs in [0.5, 1, 2, 4, 8]:
        for rs in [0.5, 1, 2, 4]:
            R = [run_trial(s, Cfg(q_scale=qs, r_scale=rs)) for s in range(20)]
            ne = np.concatenate([r["nees"] for r in R])
            print(f"{qs:8.1f}{rs:8.1f}{np.mean([r['success'] for r in R]):7.2f}{np.median(ne):8.2f}"
                  f"{np.mean(ne < 7.81):11.2f}{np.mean([r['rmse_pos'] for r in R]) * 100:10.1f}"
                  f"{np.degrees(np.mean([r['rmse_head'] for r in R])):11.2f}", flush=True)
