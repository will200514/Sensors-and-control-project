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
TRACK = 0.160             # Distance between drive wheels [m]
ROBOT_RADIUS = 0.105      # Approximate footprint radius [m]
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
FX = 530.0
CX_PX = 320.0
CY_PX = 240.0

# Rectified simulation camera: zero lens distortion.
CAM_DISTORTION = (0.0, 0.0, 0.0, 0.0, 0.0)

MARKER_L = 0.20           # Printed marker side length [m]
CAM_OFFSET = 0.06         # Camera ahead of robot origin [m]
CAM_HEIGHT = 0.12         # Camera optical-centre height [m]
MARKER_HEIGHT = 0.12      # Marker-centre height [m]


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
    search_w: float = 0.35
    search_timeout: float = 40.0
    marker_lost_timeout: float = 8.0
    initial_pose_samples: int = 5
    acquisition_settle_time: float = 0.3

    # Corner-based camera pose uncertainty
    corner_sigma_px: float = 1.0
    camera_pos_floor: float = 0.015
    camera_head_floor_deg: float = 1.0
    max_reprojection_px: float = 2.0

    # LQR: acceptable errors and control corrections
    e_max: tuple = (0.05, 0.05, 0.10)   # weights on pose error
    u_max: tuple = (0.05, 0.5)          # weights for penalties on control correctiobs

    # weighting relating error to effort
    lqr_q_scale: float = 2.0  # error
    lqr_r_scale: float = 1.0  # effort

    # Planning
    v_ref: float = 0.18
    inflate: float = 0.24

    # Simulation and display
    n_obstacles: int = 2
    viewer: bool = False
    show_sensors: bool = True

    # Optional disturbance; negative disables it.
    push_time: float = -1.0