"""Single-marker docking mission.

1. Search from a local starting pose of [0, 0, 0].
2. Stop and confirm the docking marker.
3. Initialise the EKF in the marker-centred docking frame.
4. Plan and track a docking approach using the estimated pose.

Simulation truth is used only for logging and evaluation.
"""

import numpy as np

from . import config as C
from .config import Cfg
from .model.diffdrive import wrap, f_motion
from .estimation.ekf import EKF
from .perception.docking_pose import estimate_dock_pose
from .perception.lidar import OccGrid
from .control.planner import (
    inflated_map,
    astar,
    shortcut,
    smooth_path,
    make_trajectory,
)
from .control.lqr_tracker import LQRTracker
from .sim.mujoco_backend import MujocoWorld
from .evaluation_frames import (
    relative_pose,
    display_pose,
    dock_origin,
)


REPLAN_STEPS = 100



# constructs a docking trajectory
# finds a route the the approach point
def _plan(grid, pose, cfg):
    """Plan to the marker-relative approach, then towards the dock."""
    blocked = inflated_map(grid, cfg.inflate)

    path = astar(
        blocked,
        grid,
        pose[:2],
        C.DOCK_APPROACH,
    )

    if path is None:
        return None

    path = shortcut(path, blocked, grid)
    path = smooth_path(path, blocked, grid)

    docking_line = np.linspace(
        C.DOCK_APPROACH,
        C.DOCK_TARGET[:2],
        30,
    )

    points = np.vstack([path, docking_line[1:]])

    return make_trajectory(
        points,
        cfg.v_ref,
        v_end=0.03,
    )



# obtains a usable camera observation

def _camera_pose(world, cfg):
    """Return one docking-marker pose and the detection count."""
    detections = world.camera()

    for marker_id, corners in detections:
        if marker_id == C.DOCK_MARKER_ID:
            observation = estimate_dock_pose(corners, cfg)
            return observation, len(detections)

    return None, len(detections)



# checks the start up measurements 
# This prevents the robot from defining its docking frame from just one potentially noisy image.
def _confirmed_pose(observations):
    """Combine agreeing stationary observations without overconfidence."""
    poses = np.array([obs.pose for obs in observations])

    # Unwrap headings around the first observation before averaging.
    poses[:, 2] = poses[0, 2] + wrap(
        poses[:, 2] - poses[0, 2]
    )

    pose = np.median(poses, axis=0)
    differences = poses - pose

    position_spread = np.linalg.norm(
        differences[:, :2],
        axis=1,
    )
    heading_spread = np.abs(differences[:, 2])

    if (
        position_spread.max() > 0.35
        or heading_spread.max() > np.deg2rad(15.0)
    ):
        return None

    covariance = np.mean(
        [obs.covariance for obs in observations],
        axis=0,
    )

    # Retain shared camera uncertainty and add observed scatter.
    covariance += (
        differences.T @ differences / len(observations)
    )

    pose[2] = wrap(pose[2])

    return pose, covariance

# This function records the information needed for your plots. It does not calculate movement commands.
def _record(
    data,
    world,
    ekf,
    dead_reckoning,
    tracker,
    localised,
    phase,
    command,
    error,
    n_visible,
):
    """Record matching coordinate frames. Evaluation only."""
    origin = dock_origin() if localised else world.start
    actual = world.truth()

    reference = (
        tracker.traj[tracker.idx, :3].copy()
        if tracker.traj is not None
        else np.full(3, np.nan)
    )

    # Rotate covariance only when displaying start-frame data
    # in the docking frame.
    angle = (
        0.0
        if localised
        else world.start[2] - dock_origin()[2]
    )

    c, s = np.cos(angle), np.sin(angle)

    rotation = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    sample = {
        "t": world.t,
        "true": relative_pose(actual, origin),
        "est": ekf.x.copy(),
        "P": ekf.P.copy(),
        "dr": dead_reckoning.copy(),
        "ref": reference,
        "e": np.asarray(error).copy(),
        "u": command,
        "n_vis": n_visible,
        "frame": "dock" if localised else "start",
        "phase": phase,
        "plot_true": relative_pose(actual, dock_origin()),
        "plot_est": display_pose(
            ekf.x, world.start, localised
        ),
        "plot_dr": display_pose(
            dead_reckoning, world.start, localised
        ),
        "plot_P": rotation @ ekf.P @ rotation.T,
    }

    for name, value in sample.items():
        data.setdefault(name, []).append(value)

# evaluates completed trial, This function runs after the control loop ends.
def _results(
    seed,
    cfg,
    world,
    ekf,
    grid,
    data,
    reason,
    acquisition_time,
    plans,
    phase,
):
    """Calculate success and plotting data. Evaluation only."""
    actual = relative_pose(world.truth(), dock_origin())

    position_error = np.linalg.norm(
        actual[:2] - C.DOCK_TARGET[:2]
    )
    heading_error = abs(
        float(wrap(actual[2] - C.DOCK_TARGET[2]))
    )

    success = (
        reason == "docked"
        and position_error < C.DOCK_TOL_POS
        and heading_error < C.DOCK_TOL_HEAD
        and not world.collided
    )

    xmin, xmax, ymin, ymax = C.ROOM

    room_corners = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin),
    ]

    room = np.array([
        relative_pose([x, y, 0.0], dock_origin())[:2]
        for x, y in room_corners
    ])

    obstacles = [
        (
            *relative_pose(
                [x, y, 0.0], dock_origin()
            )[:2],
            radius,
        )
        for x, y, radius in world.obstacles
    ]

    result = {
        "seed": seed,
        "cfg": cfg,
        "success": bool(success),
        "reason": reason,
        "collided": world.collided,
        "time": world.t,
        "dock_pos_err": float(position_error),
        "dock_head_err": heading_error,
        "n_updates": ekf.n_updates,
        "n_rejected": ekf.n_rejected,
        "n_recoveries": ekf.n_recoveries,
        "plans": plans,
        "localised": acquisition_time is not None,
        "acquisition_time": acquisition_time,
        "start": world.start.copy(),
        "obstacles": obstacles,
        "room": room,
        "grid": grid.occupied(),
        "grid_bounds": grid.bounds,
        "sensor_snapshots": {},
    }

    

    if not data:
        return result

    samples = {
        name: np.asarray(values)
        for name, values in data.items()
    }

    error = samples["true"] - samples["est"]
    error[:, 2] = wrap(error[:, 2])

    dr_error = samples["true"] - samples["dr"]
    dr_error[:, 2] = wrap(dr_error[:, 2])

    nees = np.array([
        e @ np.linalg.solve(P, e)
        for e, P in zip(error, samples["P"])
    ])

    dock_samples = samples["frame"] == "dock"

    tracking_error = np.linalg.norm(
        samples["true"][:, :2]
        - samples["ref"][:, :2],
        axis=1,
    )

    tracking_samples = (
        dock_samples & np.isfinite(tracking_error)
    )

    def rms_squared(values):
        if len(values) == 0:
            return float("nan")
        return float(np.sqrt(np.mean(values)))

    result.update({
        "log": samples,
        "nees": nees,
        "rmse_pos": rms_squared(
            np.sum(error[dock_samples, :2] ** 2, axis=1)
        ),
        "rmse_head": rms_squared(
            error[dock_samples, 2] ** 2
        ),
        "rmse_pos_dr": rms_squared(
            np.sum(dr_error[dock_samples, :2] ** 2, axis=1)
        ),
        "track_rmse": rms_squared(
            tracking_error[tracking_samples] ** 2
        ),
        "effort": float(np.mean(
            np.sum(samples["u"] ** 2, axis=1)
        )),
        "nees_mean": (
            float(nees[dock_samples].mean())
            if dock_samples.any()
            else float("nan")
        ),
    })

    return result



# runs the mission
# 1. Create the simulation and processing objects
# 2. Initialise mission state
# 3. Run the timed loop
# 4. Search for the marker
# 5. Stop and confirm - When a usable observation appears, the mission records when confirmation began.
# 6. Establish the docking frame
# 7. Apply later camera corrections
# 8. Reacquire a lost marker
# 9. Move, predict and map - applies the command, obtains wheel odometry and predicts the next pose.
# 10. Finish and release resources

def run_trial(seed, cfg=None, log=True):
    cfg = cfg or Cfg()
    rng = np.random.default_rng(seed)
    world = MujocoWorld(rng, cfg)

    try:
        # The robot's initial pose defines the starting frame.
        initial_sd = np.array([
            cfg.p0_pos,
            cfg.p0_pos,
            np.deg2rad(cfg.p0_head_deg),
        ])

        ekf = EKF(
            np.zeros(3),
            np.diag(initial_sd ** 2),
            cfg,
        )

        grid = OccGrid()
        tracker = LQRTracker(cfg)
        dead_reckoning = np.zeros(3)
        data = {}

        localised = False
        acquisition_time = None
        last_fix = -np.inf
        last_detection = -np.inf
        confirmation_start = None
        observations = []

        phase = "search"
        reason = "timeout"
        n_visible = 0
        plans = 0
        last_plan = -REPLAN_STEPS
        force_replan = False
        hold = 0

        # Defaults also support the earlier configuration file.
        required_samples = max(
            1, int(getattr(cfg, "initial_pose_samples", 5))
        )
        settle_time = getattr(
            cfg, "acquisition_settle_time", 0.3
        )

        

        if log:
            print(
                "Start frame: [0, 0, 0]. "
                "Searching for docking marker.",
                flush=True,
            )

            _record(
                data, world, ekf, dead_reckoning,
                tracker, False, phase,
                (0.0, 0.0), np.zeros(3), 0,
            )

        # Sense LiDAR immediately, including during search.
        ranges = world.lidar()

        for step in range(int(C.MAX_TIME / C.DT)):
            if (
                world.viewer is not None
                and not world.viewer.is_running()
            ):
                reason = "viewer_closed"
                break

            # ---------- Camera ----------
            if step % C.CAM_EVERY == 0:
                observation, n_visible = _camera_pose(
                    world, cfg
                )

                if not localised:
                    if observation is not None:
                        last_detection = world.t

                        if confirmation_start is None:
                            confirmation_start = world.t
                            observations.clear()

                        if (
                            world.t - confirmation_start
                            >= settle_time
                        ):
                            observations.append(observation)

                            if len(observations) >= required_samples:
                                confirmed = _confirmed_pose(
                                    observations
                                )

                                if confirmed is not None:
                                    pose, covariance = confirmed

                                    ekf.initialise_dock(
                                        pose, covariance
                                    )

                                    localised = True
                                    acquisition_time = world.t
                                    last_fix = world.t

                                    dead_reckoning = ekf.x.copy()

                                    # Start a fresh map in the dock frame.
                                    grid = OccGrid()
                                    ranges = world.lidar()
                                    grid.update(ekf.x, ranges)

                                    if log:
                                        print(
                                            f"Dock frame acquired at "
                                            f"{world.t:.2f}s: "
                                            f"{np.round(ekf.x, 3)}",
                                            flush=True,
                                        )
                                else:
                                    observations.pop(0)

                else:
                    valid = int(observation is not None)
                    accepted = False

                    if observation is not None:
                        accepted = ekf.update_pose(
                            observation.pose,
                            observation.covariance,
                        )

                    ekf.end_frame(valid, int(accepted))

                    if accepted:
                        last_fix = world.t

            # ---------- Choose movement ----------
            error = np.zeros(3)
            command = (0.0, cfg.search_w)

            if not localised:
                if world.t >= cfg.search_timeout:
                    reason = "marker_not_found"
                    break

                if (
                    confirmation_start is not None
                    and world.t - last_detection < 0.5
                ):
                    phase = "confirm"
                    command = (0.0, 0.0)
                else:
                    phase = "search"
                    confirmation_start = None
                    observations.clear()

            elif world.t - last_fix > cfg.marker_lost_timeout:
                phase = "reacquire"

                if (
                    world.t - last_fix
                    > cfg.marker_lost_timeout + cfg.search_timeout
                ):
                    reason = "marker_lost"
                    break

            else:
                distance = np.linalg.norm(
                    ekf.x[:2] - C.DOCK_TARGET[:2]
                )
                heading_error = float(
                    wrap(C.DOCK_TARGET[2] - ekf.x[2])
                )

                if distance < 0.025:
                    phase = "align"

                    command = (
                        0.0,
                        float(np.clip(
                            1.5 * heading_error,
                            -0.5,
                            0.5,
                        )),
                    )

                    if abs(heading_error) < np.deg2rad(3.0):
                        hold += 1
                    else:
                        hold = 0

                    if hold >= 20:
                        reason = "docked"
                        break

                else:
                    phase = "navigate"
                    hold = 0

                    in_approach = (
                        ekf.x[0] < C.DOCK_APPROACH[0] + 0.15
                        and abs(ekf.x[1]) < 0.15
                    )

                    replan_due = (
                        force_replan
                        or step - last_plan >= REPLAN_STEPS
                    )

                    if (
                        tracker.traj is None
                        or (not in_approach and replan_due)
                    ):
                        trajectory = _plan(grid, ekf.x, cfg)

                        last_plan = step
                        plans += 1
                        force_replan = False

                        if trajectory is not None:
                            tracker.set_trajectory(trajectory)

                        elif tracker.traj is None:
                            reason = "no_path"
                            break

                    command, error = tracker.command(ekf.x)

            # ---------- Move and predict ----------
            odometry = world.step(*command)

            ekf.predict(odometry)

            dead_reckoning = f_motion(
                dead_reckoning,
                odometry,
                C.DT,
            )

            # ---------- LiDAR ----------
            if (step + 1) % C.LIDAR_EVERY == 0:
                ranges = world.lidar()

                if localised:
                    grid.update(ekf.x, ranges)

                    if tracker.traj is not None:
                        blocked = inflated_map(
                            grid, cfg.inflate
                        )

                        remaining = tracker.traj[
                            tracker.idx::5, :2
                        ]

                        ix, iy = grid.to_cell(
                            remaining[:, 0],
                            remaining[:, 1],
                        )

                        force_replan = bool(
                            blocked[ix, iy].any()
                        )

            

            if log:
                _record(
                    data, world, ekf, dead_reckoning,
                    tracker, localised, phase,
                    command, error, n_visible,
                )

        return _results(
            seed, cfg, world, ekf, grid, data,
            reason, acquisition_time, plans, phase,
        )

    finally:
        world.close()