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
from .live_view import LiveView  # live window


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


def _visible_standoff(margin=0.05):
    """Closest robot-centre distance at which the whole marker (plus its white
    border) stays inside the image, from the loaded calibration and geometry."""
    width, height = C.CAM_IMAGE_SIZE
    half = C.MARKER_L / 2 * 1.1                      # include some white border
    up = np.arctan(C.CY_PX / C.FY) - np.deg2rad(C.CAM_PITCH_DEG)
    down = np.arctan((height - C.CY_PX) / C.FY) + np.deg2rad(C.CAM_PITCH_DEG)
    side = np.arctan(min(C.CX_PX, width - C.CX_PX) / C.FX)
    rise = C.MARKER_HEIGHT + half - C.CAM_HEIGHT     # top edge above the lens
    drop = C.CAM_HEIGHT - (C.MARKER_HEIGHT - half)   # bottom edge below the lens
    need = half / np.tan(side)
    for extent, angle in ((rise, up), (drop, down)):
        if extent > 0:
            if angle <= 0.02:
                return np.inf
            need = max(need, extent / np.tan(angle))
    return need + C.CAM_OFFSET + margin


def _ramp(new, previous, step):
    """[SMOOTH NAV] Limit how fast a speed command can grow. Slowing down and
    stopping are never delayed; a change of direction passes through zero."""
    if new * previous >= 0 and abs(new) <= abs(previous):
        return new
    base = previous if new * previous > 0 else 0.0
    return base + float(np.clip(new - base, -step, step))


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
    show_view=False,
    record_view=False,
):
    cfg = cfg or Cfg(viewer=False)

    # Use the real camera calibration and measured geometry
    # instead of the MuJoCo values. Stops with a clear message if either file is
    # missing (run calibrate_real_camera.py, then measure_geometry.py).
    for line in C.load_hardware_calibration():
        print("Loaded", line, flush=True)
    cfg.fx_filter = C.FX

    # Fit a small camera-tilt correction (prior 1 deg) on hardware.
    cfg.fit_tilt_sd_deg = 1.0

    # [SMOOTH NAV] Lock the dock frame from the marker once, then drive the
    # planned route on wheel odometry. Later marker sightings do not move the
    # pose estimate, so the estimate (and the robot) no longer jumps when the
    # camera pose is noisy, and losing sight of the marker does not matter.
    cfg.fuse_camera_after_lock = False
    cfg.lookahead = 0.30              # steer for a point 30 cm along the path
    cfg.turn_in_place_enter = 0.70    # spin in place above ~40 deg ...
    cfg.turn_in_place_exit = 0.20     # ... until within ~11 deg, then drive
    cfg.turn_in_place_gain = 1.0
    cfg.turn_in_place_min_w = 0.10
    cfg.max_accel = 0.15              # m/s^2
    cfg.max_ang_accel = 0.8           # rad/s^2
    cfg.replan_period = None          # only replan when blocked or off the path
    cfg.off_path_replan = 0.35
    cfg.no_path_retries = 10          # wait ~1 s and replan, up to 10 times

    # [SMOOTH NAV] Robot-centre stop distance from the marker wall when the
    # camera is not used after lock (about 20 cm between bumper and wall).
    odometry_standoff = 0.40

    # Stop where the camera can still see the whole marker.
    # The 0.30 m simulation standoff works only because the MuJoCo camera sits at
    # marker height. A higher or tilted real camera loses the bottom of the marker
    # that close, so "align" never gets a fix and the run ends in "marker_lost".
    # [SMOOTH NAV] The camera-visibility limit only matters when camera fixes
    # are needed at the stop. With the frame locked once, use a fixed stop.
    if cfg.fuse_camera_after_lock:
        standoff = max(C.DOCK_STANDOFF, _visible_standoff())
    else:
        standoff = max(C.DOCK_STANDOFF, odometry_standoff)
    if not np.isfinite(standoff) or standoff > 1.5:
        raise SystemExit(
            f"The camera can only see the whole marker from {standoff:.2f} m away or more. "
            f"Camera lens {C.CAM_HEIGHT:.3f} m high, tilted {C.CAM_PITCH_DEG:+.1f} deg "
            f"(negative = up), marker centre {C.MARKER_HEIGHT:.3f} m. Move the marker so "
            "its centre is closer to where the camera points (see WALKTHROUGH.md), then "
            "rerun measure_geometry.py."
        )
    C.DOCK_TARGET = np.array([standoff, 0.0, np.pi])
    C.DOCK_APPROACH = np.array([max(C.DOCK_APPROACH[0], standoff + 0.5), 0.0])
    print(f"Docking stop {standoff:.2f} m from the marker wall "
          f"(approach point {C.DOCK_APPROACH[0]:.2f} m).", flush=True)

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


    # A real camera never matches the model as closely as the
    # simulator does. Allow a few pixels of reprojection error on hardware.
    # Tighten back towards 2 px once calibration and measurements are good.
    cfg.max_reprojection_px = max(cfg.max_reprojection_px, 5.0)
    rejected_print = -np.inf
    size_checked = False

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

    # Live camera / map window (main.py turns it on).
    view = LiveView(show_view, record_view)
    front_min = np.inf

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

    # [SMOOTH NAV] state
    previous_forward = 0.0
    previous_turning = 0.0
    no_path_count = 0
    waiting_for_path = False

    # Obstacle back-off state
    backoff_until = None
    backoffs = 0
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

                    # The calibration only holds at the image size
                    # it was made at. Stop if the camera stream size differs.
                    if not size_checked:
                        actual_size = (image.shape[1], image.shape[0])
                        if actual_size != tuple(C.CAM_IMAGE_SIZE):
                            reason = (
                                f"camera image {actual_size} differs from "
                                f"calibration {tuple(C.CAM_IMAGE_SIZE)}"
                            )
                            break
                        size_checked = True

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

                    view.camera_frame(image, detections, observation)

                    # Report a marker that is seen but whose pose is
                    # rejected, instead of failing silently in "search".
                    if (
                        observation is None
                        and detections
                        and elapsed - rejected_print > 1.0
                    ):
                        print(
                            f"{elapsed:5.1f}s marker ID "
                            f"{C.DOCK_MARKER_ID} seen but pose rejected "
                            "(run aruco_diag.py to see why)",
                            flush=True,
                        )
                        rejected_print = elapsed

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

                                        if not cfg.fuse_camera_after_lock:
                                            print(
                                                "Dock frame locked. Driving the "
                                                "planned route on wheel odometry.",
                                                flush=True,
                                            )

                        elif not cfg.fuse_camera_after_lock:
                            # [SMOOTH NAV] Frame already locked: the marker is
                            # only shown in the live view, not fused.
                            pass

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

                    elif localised and cfg.fuse_camera_after_lock:
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

                        # Rotate in place to look for the marker
                        # (was 0.0, so the robot never searched).
                        turning = cfg.search_w

                elif (
                    cfg.fuse_camera_after_lock
                    and elapsed - last_fix
                    > cfg.marker_lost_timeout
                ):
                    phase = "reacquire"

                    # Rotate in place to find the marker again.
                    turning = cfg.search_w

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

                    # [SMOOTH NAV] The tracker never reverses, so a small
                    # overshoot used to leave it stuck short of 2.5 cm. Also
                    # align once it is at or past the stop on the dock line.
                    passed_stop = (
                        ekf.x[0] <= C.DOCK_TARGET[0] + 0.01
                        and abs(ekf.x[1]) < 0.08
                    )

                    if distance < 0.04 or passed_stop or phase == "align":
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
                            and (
                                not cfg.fuse_camera_after_lock
                                or elapsed - last_fix < 0.75
                            )
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

                        # [SMOOTH NAV] No timed replanning: a new A* path from
                        # a slightly different pose gives a different first
                        # heading, which made the robot swing back and forth.
                        # Replan only when the path is blocked or the robot is
                        # well off it.
                        off_path = (
                            tracker.traj is not None
                            and not in_approach
                            and tracker.cross_track > cfg.off_path_replan
                        )

                        replan_due = (
                            tracker.traj is None
                            or force_replan
                            or waiting_for_path
                            or off_path
                            or (
                                cfg.replan_period is not None
                                and not in_approach
                                and elapsed - last_plan > cfg.replan_period
                            )
                        )

                        # Retry no more than once a second while waiting.
                        retry_ready = (
                            not waiting_for_path
                            or elapsed - last_plan >= 1.0
                        )

                        # [CHANGE] Wait until any back-off is finished.
                        if (
                            replan_due
                            and retry_ready
                            and backoff_until is None
                        ):
                            trajectory = _plan(
                                grid,
                                ekf.x,
                                cfg,
                            )

                            plans += 1
                            last_plan = elapsed
                            force_replan = False

                            if trajectory is None:
                                # [SMOOTH NAV] Wait and try again instead of
                                # ending the run on the first failed plan.
                                no_path_count += 1

                                if no_path_count > cfg.no_path_retries:
                                    reason = "no_path"
                                    break

                                print(
                                    f"{elapsed:5.1f}s no path, waiting "
                                    f"({no_path_count}/{cfg.no_path_retries})",
                                    flush=True,
                                )
                                waiting_for_path = True

                            else:
                                no_path_count = 0
                                waiting_for_path = False
                                tracker.set_trajectory(
                                    trajectory
                                )

                        if waiting_for_path or tracker.traj is None:
                            phase = "wait_path"
                            forward = 0.0
                            turning = 0.0

                        else:
                            command, _ = tracker.command(
                                ekf.x
                            )

                            forward, turning = command

                # ---------- Live obstacle check ----------
                # Reverse away from close obstacles instead of
                # ending the run, then force a replan around them.
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

                rear_sector = (
                    abs(wrap(forward_angles - np.pi))
                    < np.deg2rad(35)
                )

                stop_distance = max(
                    0.20,
                    C.ROBOT_RADIUS + 0.06,
                )

                front_min = (
                    scan_ranges[front_sector].min()
                    if front_sector.any()
                    else np.inf
                )

                # No rear readings means we cannot check behind, so do
                # not reverse blind.
                rear_clear = bool(
                    rear_sector.any()
                    and scan_ranges[rear_sector].min()
                    > stop_distance
                )

                if backoff_until is not None:
                    finished = (
                        elapsed >= backoff_until
                        or front_min >= cfg.backoff_clearance
                        or not rear_clear
                    )

                    if finished:
                        backoff_until = None
                        force_replan = True
                        forward = 0.0
                        turning = 0.0
                        phase = "backoff_done"

                    else:
                        forward = -cfg.backoff_speed
                        turning = 0.0
                        phase = "backoff"

                elif forward > 0:
                    if not front_sector.any():
                        # LiDAR fault: no data in front at all.
                        robot.stop()
                        reason = "no_front_scan"
                        break

                    if front_min < stop_distance:
                        backoffs += 1

                        if backoffs > cfg.max_backoffs:
                            robot.stop()
                            reason = "too_many_obstacle_backoffs"
                            break

                        print(
                            f"Obstacle {front_min:.2f} m ahead, "
                            f"backing off ({backoffs}/"
                            f"{cfg.max_backoffs})",
                            flush=True,
                        )

                        force_replan = True
                        turning = 0.0

                        if rear_clear:
                            backoff_until = (
                                elapsed + cfg.backoff_time
                            )
                            forward = -cfg.backoff_speed
                            phase = "backoff"

                        else:
                            # Boxed in behind too: just stop and
                            # let the planner find another way.
                            forward = 0.0
                            phase = "blocked"

                # ---------- Send motor command ----------
                forward = float(
                    np.clip(
                        forward,
                        -cfg.backoff_speed,  # was 0.0
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

                # [SMOOTH NAV] Ramp speeds so commands do not step.
                step_dt = min(max(dt, 0.0), 0.1)

                if cfg.max_accel is not None:
                    forward = _ramp(
                        forward,
                        previous_forward,
                        cfg.max_accel * step_dt,
                    )

                if cfg.max_ang_accel is not None:
                    step = cfg.max_ang_accel * step_dt
                    turning = previous_turning + float(
                        np.clip(turning - previous_turning, -step, step)
                    )

                previous_forward = forward
                previous_turning = turning

                robot.send_command(
                    forward,
                    turning,
                )

                # [NEW] Update the live window; q / Esc stops the run.
                if view.draw(
                    elapsed=elapsed,
                    phase=phase,
                    pose=ekf.x,
                    localised=localised,
                    grid=grid,
                    inflate=cfg.inflate,
                    traj=tracker.traj,
                    idx=tracker.idx,
                    scan=latest_scan,
                    forward=forward,
                    turning=turning,
                    front_min=front_min,
                    note=f"plans {plans}  back-offs {backoffs}",
                ):
                    robot.stop()
                    reason = "stopped_from_live_view"
                    break

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

            print(f"Result: {reason}", flush=True)
            view.close()

    return {
        "reason": reason,
        "localised": localised,
        "plans": plans,
        "estimated_pose": ekf.x.tolist(),
        "estimated_docking_complete": (
            reason == "docked_estimated"
        ),
    }