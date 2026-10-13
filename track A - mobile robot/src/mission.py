"""Mission loop: sense -> estimate -> plan -> track. The controller sees the EKF estimate only."""
import numpy as np
from . import config as C
from .config import Cfg
from .model.diffdrive import wrap, f_motion
from .estimation.ekf import EKF
from .perception.lidar import OccGrid
from .control.planner import inflated_map, astar, shortcut, smooth_path, make_trajectory
from .control.lqr_tracker import LQRTracker
from .sim.world import World

REPLAN_STEPS = 100        # periodic replan (5 s); earlier replans only if the path becomes blocked


def _plan(grid, x_est, goal_phase, cfg):
    blocked = inflated_map(grid, cfg.inflate)
    if goal_phase == 0:
        path = astar(blocked, grid, x_est[:2], C.WAYPOINT_1)
        if path is None:
            return None
        P = smooth_path(shortcut(path, blocked, grid), blocked, grid)
        return make_trajectory(P, cfg.v_ref, v_end=0.12)
    path = astar(blocked, grid, x_est[:2], C.LEADIN)
    if path is None:
        return None
    P = smooth_path(shortcut(path, blocked, grid), blocked, grid)
    P = np.vstack([P, np.linspace(C.LEADIN, C.DOCK_POSE[:2], 12)[1:]])    # straight docking lead-in
    return make_trajectory(P, cfg.v_ref, v_end=0.03)


def run_trial(seed, cfg=None, log=True):
    cfg = cfg or Cfg()
    rng = np.random.default_rng(seed)
    world = World(rng, cfg)
    sd = np.array([cfg.p0_pos, cfg.p0_pos, np.deg2rad(cfg.p0_head_deg)])
    x0 = world.truth() + rng.normal(0, sd)     # initial guess: start pose known to within P0
    ekf = EKF(x0, np.diag(sd ** 2), cfg)
    grid, trk = OccGrid(), LQRTracker(cfg)
    x_dr = x0.copy()                            # dead-reckoning baseline (logging only)
    phase, plan_locked, last_plan, hold, plans = 0, False, -10 ** 9, 0, 0
    force_replan = False
    L = dict(t=[], true=[], est=[], P=[], dr=[], ref=[], e=[], u=[], phase=[], n_vis=[]) if log else None

    def sense_camera():
        n = acc = 0
        for j, u_px, s_px in world.camera():
            acc += ekf.update_marker(C.MARKERS[j][:2], u_px, s_px); n += 1
        ekf.end_frame(n, acc)
        return n

    n_vis = sense_camera()
    grid.update(ekf.x, world.lidar())
    max_steps = int(C.MAX_TIME / C.DT)
    done_reason = "timeout"
    for k in range(max_steps):
        x_est = ekf.x
        # ---------------- phase logic (uses ESTIMATE only) ----------------
        if phase == 0 and np.linalg.norm(x_est[:2] - C.WAYPOINT_1) < 0.25:
            phase, trk.traj, last_plan = 1, None, -10 ** 9
        in_lead = phase == 1 and x_est[0] > C.LEADIN[0] - 0.15 and abs(x_est[1] - C.LEADIN[1]) < 0.12
        if in_lead:
            plan_locked = True
        # ---------------- (re)planning ----------------
        if trk.traj is None or (not plan_locked and (force_replan or k - last_plan >= REPLAN_STEPS)):
            force_replan = False
            traj = _plan(grid, x_est, phase, cfg)
            last_plan, plans = k, plans + 1
            if traj is not None:
                trk.set_trajectory(traj)
            elif trk.traj is None:
                done_reason = "no_path"; break
        # ---------------- control ----------------
        if trk.done() and phase == 1:
            hold += 1
            u_cmd, e = (0.0, 0.0), np.zeros(3)
            if hold > 30:
                done_reason = "docked"; break
        else:
            u_cmd, e = trk.command(x_est)
        odo = world.step(*u_cmd)
        # ---------------- estimation ----------------
        ekf.predict(odo)
        x_dr = f_motion(x_dr, odo, C.DT)
        if (k + 1) % C.CAM_EVERY == 0:
            n_vis = sense_camera()
        if (k + 1) % C.LIDAR_EVERY == 0:
            grid.update(ekf.x, world.lidar())
            if trk.traj is not None and not plan_locked:      # is the remaining path still free?
                bl = inflated_map(grid, cfg.inflate)
                rem = trk.traj[trk.idx::5, :2]
                ix, iy = grid.to_cell(rem[:, 0], rem[:, 1])
                force_replan = bool(bl[ix, iy].any())
        if world.collided and done_reason == "timeout":
            pass
        if log:
            L["t"].append((k + 1) * C.DT); L["true"].append(world.truth()); L["est"].append(ekf.x.copy())
            L["P"].append(ekf.P.copy()); L["dr"].append(x_dr.copy()); L["e"].append(e); L["u"].append(u_cmd)
            L["phase"].append(phase); L["n_vis"].append(n_vis)
            L["ref"].append(trk.traj[trk.idx, :3].copy() if trk.traj is not None else np.full(3, np.nan))

    xt = world.truth()                          # evaluation only from here on
    pos_err_dock = np.linalg.norm(xt[:2] - C.DOCK_POSE[:2])
    head_err_dock = abs(float(wrap(xt[2] - C.DOCK_POSE[2])))
    success = (done_reason == "docked" and pos_err_dock < C.DOCK_TOL_POS
               and head_err_dock < C.DOCK_TOL_HEAD and not world.collided)
    res = dict(seed=seed, success=bool(success), reason=done_reason, collided=world.collided,
               dock_pos_err=pos_err_dock, dock_head_err=head_err_dock, time=world.t,
               n_updates=ekf.n_updates, n_recoveries=ekf.n_recoveries, n_rejected=ekf.n_rejected, plans=plans,
               start=world.start, obstacles=world.obstacles, cfg=cfg)
    if log:
        T = {k_: np.array(v) for k_, v in L.items()}
        err = T["true"] - T["est"]; err[:, 2] = wrap(err[:, 2])
        nees = np.array([e_ @ np.linalg.solve(P_, e_) for e_, P_ in zip(err, T["P"])])
        dr_err = T["true"] - T["dr"]; dr_err[:, 2] = wrap(dr_err[:, 2])
        ref_err = np.hypot(*(T["true"][:, :2] - T["ref"][:, :2]).T)
        u = T["u"]
        res.update(log=T, nees=nees,
                   rmse_pos=float(np.sqrt(np.mean(np.sum(err[:, :2] ** 2, 1)))),
                   rmse_head=float(np.sqrt(np.mean(err[:, 2] ** 2))),
                   rmse_pos_dr=float(np.sqrt(np.mean(np.sum(dr_err[:, :2] ** 2, 1)))),
                   track_rmse=float(np.sqrt(np.nanmean(ref_err ** 2))),
                   effort=float(np.mean(u[:, 0] ** 2 + u[:, 1] ** 2)),
                   nees_mean=float(nees.mean()), grid=grid.occupied())
    return res
