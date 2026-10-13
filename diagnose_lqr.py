"""Diagnose the LQR R x100 failures (README hypothesis: weak controller on the initial turn).
Ground truth is used ONLY to score. Run: python diagnose_lqr.py"""
import numpy as np
from dataclasses import replace
from src.mission import run_trial
from src.config import Cfg
from evaluate import ABL_SEEDS

base = Cfg()
for rs in (1.0, 100.0):
    print(f"--- LQR R x{rs}")
    for s in ABL_SEEDS:
        r = run_trial(s, replace(base, lqr_r_scale=rs))
        if r["success"] and rs == 1.0:
            continue
        if r["success"]:
            continue
        L = r["log"]; true = L["true"]; ref = L["ref"]
        d = np.hypot(*(true[:, :2] - ref[:, :2]).T)
        t_max = L["t"][np.nanargmax(d)]
        # time to first collision isn't logged; report where and when tracking error peaks
        init_turn = abs(true[0, 2] - np.arctan2(ref[0, 1] - true[0, 1], ref[0, 0] - true[0, 0]))
        init_turn = np.degrees(min(init_turn, 2*np.pi - init_turn))
        print(f"seed {s}: reason={r['reason']} collided={r['collided']} dockpos={r['dock_pos_err']*100:.1f}cm "
              f"dockhead={np.degrees(r['dock_head_err']):.1f}deg time={r['time']:.0f}s "
              f"peak track err={np.nanmax(d)*100:.0f}cm at t={t_max:.1f}s, initial heading offset~{init_turn:.0f}deg, "
              f"est rmse={r['rmse_pos']*100:.1f}cm")
