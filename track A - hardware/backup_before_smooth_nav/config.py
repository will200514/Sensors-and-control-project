"""Configuration for TurtleBot3 navigation and single-marker docking."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Output directory
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Timing
DT = 0.05                 # Control step: 20 Hz
CAM_EVERY = 2             # Camera update: 10 Hz
LIDAR_EVERY = 4           # LiDAR update: 5 Hz
MAX_TIME = 180.0          # Trial timeout [s]


# TurtleBot3 platform
TRACK = 0.286             # Distance between drive wheels [m]
ROBOT_RADIUS = 0.20      # Approximate footprint radius [m]
WHEEL_RADIUS = 0.033      # Wheel radius [m]

V_MAX = 0.20              # Maximum commanded forward speed [m/s]
W_MAX = 1.5               # Maximum commanded turning speed [rad/s]


# Wheel odometry uncertainty
SIGMA_ENC = 0.003         # Wheel encoder speed noise [m/s]
SIGMA_SLIP = 0.02         # Relative wheel slip per step
SIGMA_BIAS = 0.01         # Relative wheel bias per trial


# Simulation geometry
# These coordinates place objects in MuJoCo.
# They are not supplied to the robot for initial localisation.
ROOM = (0.0, 3.0, 0.0, 3.0)  # xmin, xmax, ymin, ymax

DOCK_MARKER_ID = 1

# Marker x, y, and outward-facing direction in the simulation.
MARKERS = np.array([
    [2.98, 0.6, np.pi],
])

DOCK_MARKER = MARKERS[-1]
DOCK_STANDOFF = 0.30       # Robot-centre stopping distance [m]

DOCK_POSE = np.array([
    DOCK_MARKER[0] - DOCK_STANDOFF,
    DOCK_MARKER[1],
    0.0,
])

LEADIN = np.array([
    DOCK_MARKER[0] - 1.0,
    DOCK_MARKER[1],
])

DOCK_TOL_POS = 0.05
DOCK_TOL_HEAD = np.deg2rad(5.0)


# Camera calibration and mounting
# Simulation values. On hardware these are replaced by load_hardware_calibration().
FX = 530.0
FY = 530.0
CX_PX = 320.0
CY_PX = 240.0
CAM_IMAGE_SIZE = (640, 480)   # width, height

# Rectified simulation camera: zero lens distortion.
CAM_DISTORTION = (0.0, 0.0, 0.0, 0.0, 0.0)

MARKER_L = 0.20           # Printed marker side length [m]
CAM_OFFSET = 0.06         # Camera ahead of robot origin [m]
CAM_HEIGHT = 0.12         # Camera optical-centre height [m]
MARKER_HEIGHT = 0.12      # Marker-centre height [m]
CAM_PITCH_DEG = 0.0       # Camera tilt, positive = looking down [deg]


#  Real-robot calibration files, written by the scripts:
#   calibrate_real_camera.py -> results/real_camera_calib.npz (FX, FY, CX, CY, distortion)
#   measure_geometry.py      -> results/hardware_geometry.json (marker size and height,
#                               camera height and tilt)
# hardware_mission.py, aruco_diag.py and measure_geometry.py call
# load_hardware_calibration(). The MuJoCo simulation never does, so it keeps the
# simulation values above.
CAMERA_CALIB_FILE = RESULTS_DIR / "real_camera_calib.npz"
GEOMETRY_FILE = RESULTS_DIR / "hardware_geometry.json"
HARDWARE_GEOMETRY_KEYS = ("MARKER_L", "MARKER_HEIGHT", "CAM_HEIGHT",
                          "CAM_OFFSET", "CAM_PITCH_DEG")


def load_hardware_calibration(require_camera=True, require_geometry=True):
    """Replace the simulation camera/geometry values with the real-robot files."""
    import json

    g = globals()
    report = []

    if CAMERA_CALIB_FILE.exists():
        data = np.load(CAMERA_CALIB_FILE)
        K = data["K"]
        g["FX"], g["FY"] = float(K[0, 0]), float(K[1, 1])
        g["CX_PX"], g["CY_PX"] = float(K[0, 2]), float(K[1, 2])
        g["CAM_DISTORTION"] = tuple(float(v) for v in data["dist"].ravel()[:5])
        g["CAM_IMAGE_SIZE"] = tuple(int(v) for v in data["size"])
        report.append(f"camera: {CAMERA_CALIB_FILE.name} "
                      f"(fx {FX:.1f}, fy {FY:.1f}, cx {CX_PX:.1f}, cy {CY_PX:.1f}, "
                      f"{CAM_IMAGE_SIZE[0]}x{CAM_IMAGE_SIZE[1]}, "
                      f"RMS {float(data['rms']):.2f} px)")
    elif require_camera:
        raise FileNotFoundError(
            f"{CAMERA_CALIB_FILE} not found. Run calibrate_real_camera.py first.")
    else:
        report.append("camera: NOT calibrated, using simulation values")

    if GEOMETRY_FILE.exists():
        values = json.loads(GEOMETRY_FILE.read_text())
        for key in HARDWARE_GEOMETRY_KEYS:
            if key in values:
                g[key] = float(values[key])
        report.append(f"geometry: {GEOMETRY_FILE.name} "
                      f"(marker {MARKER_L * 100:.1f} cm at {MARKER_HEIGHT:.3f} m, "
                      f"camera {CAM_HEIGHT:.3f} m high, {CAM_OFFSET:.3f} m forward, "
                      f"tilt {CAM_PITCH_DEG:+.1f} deg)")
    elif require_geometry:
        raise FileNotFoundError(
            f"{GEOMETRY_FILE} not found. Run measure_geometry.py first.")
    else:
        report.append("geometry: NOT measured, using simulation values")

    return report


# Navigation in the marker-centred docking frame
# Origin: floor point below marker.
# +x: out from wall; +y: along wall; +z: up.
DOCK_TARGET = np.array([
    DOCK_STANDOFF,
    0.0,
    np.pi,
])

DOCK_APPROACH = np.array([1.0, 0.0])

# Computational map extent, not a known map of the room.
LOCAL_MAP_BOUNDS = (-8.0, 8.0, -8.0, 8.0)


# Legacy range/bearing camera-model settings
SIGMA_PX = 1.0
SIGMA_SIZE_PX = 1.0
SIGMA_BEARING_FLOOR = np.deg2rad(0.5)


# LiDAR
LIDAR_BEAMS = 120
LIDAR_MAX = 3.5
LIDAR_SIGMA = 0.01


@dataclass
class Cfg:
    """Settings used by the estimator, controller and simulation."""

    # EKF
    use_camera: bool = True
    q_scale: float = 4.0
    r_scale: float = 2.0

    # Three-dimensional pose measurement: x, y, heading.
    # 99% chi-squared threshold; None disables gating.
    gate: float | None = 11.345

    fx_filter: float = FX

    # Small uncertainty because the initial pose defines the local frame.
    # Camera-derived uncertainty replaces this at dock initialisation.
    p0_pos: float = 1e-4
    p0_head_deg: float = 0.01

    # Search and initial marker confirmation
    search_w: float = 0.20         # was 0.35; slower spin = less blur, more frames on the marker
    search_timeout: float = 60.0   # was 40; one full turn now takes ~31 s
    # was 8.0. Large obstacles can hide the marker for a while; the
    # robot keeps driving on wheel odometry before it starts spinning to look.
    marker_lost_timeout: float = 20.0
    initial_pose_samples: int = 5
    acquisition_settle_time: float = 0.3

    # Corner-based camera pose uncertainty
    corner_sigma_px: float = 1.0
    camera_pos_floor: float = 0.015
    camera_head_floor_deg: float = 1.0
    max_reprojection_px: float = 2.0
    # Prior std of the fitted camera-tilt correction [deg].
    # None = no tilt fit (simulation, original behaviour). Hardware sets 1.0.
    fit_tilt_sd_deg: float | None = None

    # LQR: acceptable errors and control corrections
    e_max: tuple = (0.05, 0.05, 0.10)   # weights on pose error
    u_max: tuple = (0.05, 0.5)          # weights for penalties on control correctiobs

    # weighting relating error to effort
    lqr_q_scale: float = 2.0  # error
    lqr_r_scale: float = 1.0  # effort

    # Planning
    v_ref: float = 0.18
    inflate: float = 0.24

    # Obstacle back-off: when something is too close in front,
    # reverse instead of ending the run, then replan around it.
    backoff_speed: float = 0.05        # reverse speed [m/s]
    backoff_time: float = 3.0          # longest single reverse [s]
    backoff_clearance: float = 0.40    # stop reversing once front gap exceeds this [m]
    max_backoffs: int = 6              # give up after this many back-offs

    # Simulation and display
    n_obstacles: int = 2
    viewer: bool = False
    show_sensors: bool = True

    # Optional disturbance; negative disables it.
    push_time: float = -1.0