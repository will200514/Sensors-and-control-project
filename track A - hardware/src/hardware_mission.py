"""Real TurtleBot docking mission.

Uses real camera, wheel and LiDAR measurements.
Does not import MuJoCo.

Camera calibration, robot geometry and marker settings
are read from src/config.py.
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np
from rosbridge_api import RosbridgeRobot

from . import config as C
from .config import Cfg
from .model.diffdrive import wrap
from .estimation.ekf import EKF
from .perception.aruco_detector import (
    make_detector,
    detect_markers,
)
from .perception.docking_pose import estimate_dock_pose
from .perception.lidar import OccGrid, BEAM_ANGLES
from .control.planner import (
    inflated_map,
    astar,
    shortcut,
    smooth_path,
    make_trajectory,
)
from .control.lqr_tracker import LQRTracker


def _wheel_motion(message, units):
    """Convert measured wheel velocities into robot v and omega."""
    names = message.get("name", [])
    rates = message.get("velocity", [])

    if len(names) != len(rates):
        raise ValueError("Missing joint velocities.")

    values = dict(zip(names, rates))

    left = float(values["wheel_left_joint"])
    right = float(values["wheel_right_joint"])

    if not np.isfinite([left, right]).all():
        raise ValueError("Non-finite wheel velocities.")

    if units == "rad/s":
        left *= C.WHEEL_RADIUS
        right *= C.WHEEL_RADIUS

    return np.array([
        (left + right) / 2.0,
        (right - left) / C.TRACK,
    ])


def _scan_ranges(scan):
    """Bin real LiDAR readings into the mapper's fixed beam angles."""
    angles = np.asarray(scan.angles, dtype=float)
    ranges = np.asarray(scan.ranges, dtype=float)

    valid = (
        np.isfinite(angles)
        & np.isfinite(ranges)
        & (ranges >= scan.range_min)
        & (ranges <= scan.range_max)
        & (ranges <= C.LIDAR_MAX)
    )

    output = np.full(len(BEAM_ANGLES), np.nan)

    angles = angles[valid]
    ranges = ranges[valid]

    if len(ranges):
        # BEAM_ANGLES is equally spaced over [-pi, pi).
        bins = np.rint(
            (wrap(angles) + np.pi)
            * len(output)
            / (2 * np.pi)
        ).astype(int) % len(output)

        # Keep the closest return within each angular bin.
        for index in np.unique(bins):
            output[index] = ranges[bins == index].min()

    return output, angles, ranges


def _map_scan(grid, pose, scan):
    """Update occupancy using the estimated LiDAR pose."""
    ranges, _, _ = _scan_ranges(scan)

    sensor_pose = pose.copy()

    offset_x = getattr(C, "LIDAR_OFFSET_X", 0.0)
    offset_y = getattr(C, "LIDAR_OFFSET_Y", 0.0)
    heading = pose[2]

    sensor_pose[0] += (
        np.cos(heading) * offset_x
        - np.sin(heading) * offset_y
    )

    sensor_pose[1] += (
        np.sin(heading) * offset_x
        + np.cos(heading) * offset_y
    )

    sensor_pose[2] += getattr(C, "LIDAR_YAW", 0.0)

    grid.update(sensor_pose, ranges)


def _plan(grid, pose, cfg):
    """Plan to the approach point, then along the docking line."""
    blocked = inflated_map(grid, cfg.inflate)

    xmin, xmax, ymin, ymax = grid.bounds

    for point in (
        pose[:2],
        C.DOCK_APPROACH,
        C.DOCK_TARGET[:2],
    ):
        if not (
            xmin <= point[0] < xmax
            and ymin <= point[1] < ymax
        ):
            return None

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

    docking_distance = np.linalg.norm(
        C.DOCK_APPROACH - C.DOCK_TARGET[:2]
    )

    number_of_points = max(
        30,
        int(docking_distance / (grid.res / 2)) + 1,
    )

    docking_line = np.linspace(
        C.DOCK_APPROACH,
        C.DOCK_TARGET[:2],
        number_of_points,
    )

    ix, iy = grid.to_cell(
        docking_line[:, 0],
        docking_line[:, 1],
    )

    if blocked[ix, iy].any():
        return None

    points = np.vstack([
        path,
        docking_line[1:],
    ])

    return make_trajectory(
        points,
        cfg.v_ref,
        v_end=0.03,
    )


def _confirmed(observations):
    """Combine agreeing stationary marker observations."""
    poses = np.array([
        observation.pose
        for observation in observations
    ])

    poses[:, 2] = (
        poses[0, 2]
        + wrap(poses[:, 2] - poses[0, 2])
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
        or heading_spread.max() > np.deg2rad(15)
    ):
        return None

    covariance = np.mean(
        [
            observation.covariance
            for observation in observations
        ],
        axis=0,
    )

    covariance += (
        differences.T
        @ differences
        / len(poses)
    )

    pose[2] = wrap(pose[2])

    return pose, covariance


def run_hardware(
    robot_ip,
    cfg=None,
    wheel_velocity_units="m/s",
):
    cfg = cfg or Cfg(viewer=False)

    if wheel_velocity_units not in ("m/s", "rad/s"):
        raise ValueError(
            "Wheel velocity units must be m/s or rad/s."
        )

    # Conservative settings for the first hardware run.
    cfg.v_ref = min(cfg.v_ref, 0.06)
    cfg.inflate = max(
        cfg.inflate,
        C.ROBOT_RADIUS + 0.06,
    )

    forward_limit = min(C.V_MAX, 0.08)
    turning_limit = min(C.W_MAX, 0.35)

    detector = make_detector()

    initial_covariance = np.diag([
        cfg.p0_pos ** 2,
        cfg.p0_pos ** 2,
        np.deg2rad(cfg.p0_head_deg) ** 2,
    ])

    ekf = EKF(
        np.zeros(3),
        initial_covariance,
        cfg,
    )

    grid = OccGrid()
    tracker = LQRTracker(cfg)

    reason = "timeout"
    localised = False
    phase = "waiting_for_sensors"

    samples = []
    observations = []

    latest_scan = None
    mapped_scan = None

    velocity = np.zeros(2)
    joint_key = None

    last_joint = None
    last_scan = None
    last_image = None

    last_fix = -np.inf
    last_detection = -np.inf
    last_plan = -np.inf

    confirmation_start = None
    aligned_since = None

    force_replan = True
    plans = 0
    next_status = 0.0

    print(
        f"Connecting to {robot_ip}. "
        f"Wheel velocity units: {wheel_velocity_units}",
        flush=True,
    )

    print(
        "Using camera and geometry settings "
        "from src/config.py.",
        flush=True,
    )

    with RosbridgeRobot(robot_ip) as robot:
        robot.stop()

        started = time.monotonic()
        previous = started

        try:
            while time.monotonic() - started < C.MAX_TIME:
                loop_start = time.monotonic()
                elapsed = loop_start - started
                dt = loop_start - previous
                previous = loop_start

                if not robot.is_connected:
                    reason = "connection_lost"
                    break

                if dt > 0.5:
                    reason = "control_loop_stalled"
                    break

                # ---------- Wheel odometry ----------
                joints = robot.query_joint_states()

                if joints is not None:
                    stamp = joints.get(
                        "header", {}
                    ).get("stamp")

                    key = repr(stamp)

                    if stamp is None:
                        key = id(joints)

                    if key != joint_key:
                        velocity = _wheel_motion(
                            joints,
                            wheel_velocity_units,
                        )

                        joint_key = key
                        last_joint = loop_start

                if last_joint is not None:
                    ekf.predict(velocity, dt=dt)

                # ---------- LiDAR ----------
                scan = robot.query_laser_scan(
                    timeout=0,
                    wait_for_new=True,
                )

                if scan is not None:
                    _, _, valid_ranges = _scan_ranges(scan)

                    if len(valid_ranges):
                        latest_scan = scan
                        last_scan = loop_start

                # ---------- Camera ----------
                image = robot.query_image(
                    timeout=0,
                    wait_for_new=True,
                )

                if image is not None:
                    last_image = loop_start
                    observation = None

                    gray = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2GRAY,
                    )

                    detections = detect_markers(
                        gray,
                        detector,
                    )

                    for marker_id, corners in detections:
                        if marker_id == C.DOCK_MARKER_ID:
                            observation = estimate_dock_pose(
                                corners,
                                cfg,
                            )

                            if observation is not None:
                                break

                    if observation is not None:
                        last_detection = elapsed

                        if not localised:
                            if confirmation_start is None:
                                confirmation_start = elapsed

                            settled = (
                                elapsed - confirmation_start
                                >= cfg.acquisition_settle_time
                            )

                            stationary = (
                                abs(velocity[0]) < 0.02
                                and abs(velocity[1]) < 0.10
                            )

                            if settled and stationary:
                                observations.append(
                                    observation
                                )

                                required = max(
                                    1,
                                    cfg.initial_pose_samples,
                                )

                                observations = (
                                    observations[-required:]
                                )

                                if len(observations) >= required:
                                    confirmed = _confirmed(
                                        observations
                                    )

                                    if confirmed is not None:
                                        ekf.initialise_dock(
                                            *confirmed
                                        )

                                        localised = True
                                        last_fix = elapsed

                                        grid = OccGrid()
                                        mapped_scan = None

                                        print(
                                            "Dock frame acquired:",
                                            np.round(ekf.x, 3),
                                            flush=True,
                                        )

                        else:
                            accepted = ekf.update_pose(
                                observation.pose,
                                observation.covariance,
                            )

                            ekf.end_frame(
                                1,
                                int(accepted),
                            )

                            if accepted:
                                last_fix = elapsed

                    elif localised:
                        ekf.end_frame(0, 0)

                # ---------- Sensor freshness ----------
                fresh = (
                    last_joint is not None
                    and loop_start - last_joint <= 0.5
                    and last_scan is not None
                    and loop_start - last_scan <= 1.0
                    and last_image is not None
                    and loop_start - last_image <= 2.0
                )

                if not fresh:
                    robot.stop()

                    if elapsed > 10.0:
                        reason = "sensor_timeout"
                        break

                    time.sleep(C.DT)
                    continue

                # ---------- Occupancy mapping ----------
                if (
                    localised
                    and latest_scan is not mapped_scan
                ):
                    _map_scan(
                        grid,
                        ekf.x,
                        latest_scan,
                    )

                    mapped_scan = latest_scan

                    if tracker.traj is not None:
                        blocked = inflated_map(
                            grid,
                            cfg.inflate,
                        )

                        remaining = tracker.traj[
                            tracker.idx::5,
                            :2,
                        ]

                        ix, iy = grid.to_cell(
                            remaining[:, 0],
                            remaining[:, 1],
                        )

                        force_replan |= bool(
                            blocked[ix, iy].any()
                        )

                # ---------- Mission state ----------
                forward = 0.0
                turning = 0.0

                if not localised:
                    if elapsed >= cfg.search_timeout:
                        reason = "marker_pose_not_acquired"
                        break

                    confirming = (
                        confirmation_start is not None
                        and elapsed - last_detection < 0.5
                    )

                    if confirming:
                        phase = "confirm"

                    else:
                        phase = "search"

                        confirmation_start = None
                        observations.clear()

                        turning = 0.0

                elif (
                    elapsed - last_fix
                    > cfg.marker_lost_timeout
                ):
                    phase = "reacquire"

                    turning = 0.0

                    aligned_since = None

                    lost_limit = (
                        cfg.marker_lost_timeout
                        + cfg.search_timeout
                    )

                    if elapsed - last_fix > lost_limit:
                        reason = "marker_lost"
                        break

                else:
                    distance = np.linalg.norm(
                        ekf.x[:2] - C.DOCK_TARGET[:2]
                    )

                    heading_error = float(
                        wrap(
                            C.DOCK_TARGET[2] - ekf.x[2]
                        )
                    )

                    if distance < 0.025:
                        phase = "align"

                        turning = float(
                            np.clip(
                                1.5 * heading_error,
                                -turning_limit,
                                turning_limit,
                            )
                        )

                        aligned = (
                            abs(heading_error)
                            < np.deg2rad(3)
                            and elapsed - last_fix < 0.75
                        )

                        if aligned:
                            if aligned_since is None:
                                aligned_since = elapsed

                            if elapsed - aligned_since >= 1.0:
                                reason = "docked_estimated"
                                break

                        else:
                            aligned_since = None

                    else:
                        phase = "navigate"
                        aligned_since = None

                        in_approach = (
                            ekf.x[0]
                            < C.DOCK_APPROACH[0] + 0.15
                            and abs(ekf.x[1]) < 0.15
                        )

                        replan_due = (
                            tracker.traj is None
                            or force_replan
                            or (
                                not in_approach
                                and elapsed - last_plan > 5
                            )
                        )

                        if replan_due:
                            robot.stop()

                            trajectory = _plan(
                                grid,
                                ekf.x,
                                cfg,
                            )

                            plans += 1
                            last_plan = elapsed
                            force_replan = False

                            if trajectory is None:
                                reason = "no_path"
                                break

                            tracker.set_trajectory(
                                trajectory
                            )

                        command, _ = tracker.command(
                            ekf.x
                        )

                        forward, turning = command

                # ---------- Live obstacle check ----------
                _, scan_angles, scan_ranges = (
                    _scan_ranges(latest_scan)
                )

                forward_angles = wrap(
                    scan_angles
                    + getattr(C, "LIDAR_YAW", 0.0)
                )

                front_sector = (
                    abs(forward_angles)
                    < np.deg2rad(35)
                )

                stop_distance = max(
                    0.20,
                    C.ROBOT_RADIUS + 0.06,
                )

                if forward > 0:
                    no_front_readings = (
                        not front_sector.any()
                    )

                    obstacle_close = (
                        not no_front_readings
                        and scan_ranges[
                            front_sector
                        ].min() < stop_distance
                    )

                    if (
                        no_front_readings
                        or obstacle_close
                    ):
                        robot.stop()

                        reason = (
                            "front_obstacle_or_no_front_scan"
                        )
                        break

                # ---------- Send motor command ----------
                forward = float(
                    np.clip(
                        forward,
                        0.0,
                        forward_limit,
                    )
                )

                turning = float(
                    np.clip(
                        turning,
                        -turning_limit,
                        turning_limit,
                    )
                )

                robot.send_command(
                    forward,
                    turning,
                )

                samples.append((
                    elapsed,
                    phase,
                    *ekf.x,
                    *velocity,
                    forward,
                    turning,
                ))

                if elapsed >= next_status:
                    print(
                        f"{elapsed:5.1f}s "
                        f"{phase:10s} "
                        f"pose={np.round(ekf.x, 3)} "
                        f"v={forward:.3f} "
                        f"w={turning:.3f}",
                        flush=True,
                    )

                    next_status = elapsed + 1.0

                remaining_time = (
                    C.DT
                    - (time.monotonic() - loop_start)
                )

                time.sleep(
                    max(0.0, remaining_time)
                )

        finally:
            # Attempts a stop and disconnects.
            robot.cleanup()

            path = (
                Path(C.RESULTS_DIR)
                / "hardware_run.csv"
            )

            with path.open(
                "w",
                newline="",
            ) as stream:
                writer = csv.writer(stream)

                writer.writerow([
                    "time_s",
                    "phase",
                    "x_est",
                    "y_est",
                    "heading_est",
                    "v_measured",
                    "w_measured",
                    "v_command",
                    "w_command",
                ])

                writer.writerows(samples)

            print(
                f"Log saved: {path}",
                flush=True,
            )

    return {
        "reason": reason,
        "localised": localised,
        "plans": plans,
        "estimated_pose": ekf.x.tolist(),
        "estimated_docking_complete": (
            reason == "docked_estimated"
        ),
    }