"""Central configuration for Track A (TurtleBot3 MuJoCo navigation and docking).

Every number here is either a TurtleBot3 Burger datasheet value, a sensor assumption, or a
tuning choice. Tuning choices are justified in README.md.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np



# Output folder, anchored to this project (works whatever folder you run from)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---- timing -------------------------------------------------------------------------
DT = 0.05            # control / physics step [s] (20 Hz)
CAM_EVERY = 2        # camera at 10 Hz
LIDAR_EVERY = 4      # LiDAR at 5 Hz
MAX_TIME = 180.0     # trial timeout [s]

# ---- platform (TurtleBot3 Burger) -------------------------------------------------------
TRACK = 0.160        # wheel separation b [m] (difference between the drive wheels)
ROBOT_RADIUS = 0.105 # footprint radius [m]
V_MAX = 0.20         # commanded linear speed limit [m/s] (hardware max 0.22)
W_MAX = 1.5          # commanded turn-rate limit [rad/s] (hardware max 2.84)
WHEEL_RADIUS = 0.033 # [m]  # A4-MUJOCO: added, used to convert wheel speeds for the MuJoCo motors



# ---- wheel odometry ------------------------------------------------------------------
SIGMA_ENC = 0.003    # encoder speed noise per wheel [m/s]
SIGMA_SLIP = 0.02    # random relative wheel slip per step
SIGMA_BIAS = 0.01    # fixed relative wheel-radius / slip bias per wheel (per trial)

# ---- world -------------------------------------------------------------------------------
ROOM = (0.0, 3.0, 0.0, 3.0)   # xmin, xmax, ymin, ymax
# (x, y, direction the marker face looks into the room)
MARKERS = np.array([
    [2.98, 0.6, np.pi],  # ID 1: docking marker
])
DOCK_MARKER_ID = 1  # ArUco ID; independent of its index in the simulation array
DOCK_MARKER = MARKERS[-1]
DOCK_STANDOFF = 0.30                                   # final distance robot centre -> marker [m]
DOCK_POSE = np.array([DOCK_MARKER[0] - DOCK_STANDOFF, DOCK_MARKER[1], 0.0])
LEADIN = np.array([DOCK_MARKER[0] - 1.0, DOCK_MARKER[1]])               # start of the straight docking lea
DOCK_TOL_POS = 0.05
DOCK_TOL_HEAD = np.deg2rad(5.0)

# Camera settings used by MuJoCo detection and the EKF
FX = 530.0
CX_PX = 320.0
CY_PX = 240.0
# Rectified simulation camera: no lens distortion. Calibrate these for a real camera.
CAM_DISTORTION = (0.0, 0.0, 0.0, 0.0, 0.0)
MARKER_L = 0.20  # 20 cm: detectable from the 4-5 m starting region at 640x480
CAM_OFFSET = 0.06
CAM_HEIGHT = 0.12
MARKER_HEIGHT = 0.12
# Dock frame D: origin on floor below marker; +x out of wall, +y along wall, +z up.
# These are marker-relative offsets, NOT the dock's simulator/world location.
DOCK_TARGET = np.array([DOCK_STANDOFF, 0.0, np.pi])
DOCK_APPROACH = np.array([1.0, 0.0])
LOCAL_MAP_BOUNDS = (-8.0, 8.0, -8.0, 8.0)


SIGMA_PX = 1.0
SIGMA_SIZE_PX = 1.0
SIGMA_BEARING_FLOOR = np.deg2rad(0.5)

# LiDAR settings
LIDAR_BEAMS = 120
LIDAR_MAX = 3.5
LIDAR_SIGMA = 0.01

@dataclass
class Cfg:
    """Per-trial tunables. Shared estimator, controller and MuJoCo settings."""
    # estimator
    use_camera: bool = True
    q_scale: float = 4.0            # multiplies the derived process covariance
<<<<<<< HEAD
    r_scale: float = 1.0            # multiplies the derived camera covariance
    gate: float = 9.21              # chi2(2) 99% innovation gate (None = off)
    fx_filter: float = FX           # focal length the filter believes (calibration error ablation)
    p0_pos: float = 0.5
    p0_head_deg: float = 10
=======
    r_scale: float = 2.0            # multiplies the derived camera covariance
    gate: float = 11.345            # chi2(3) 99% full-pose innovation gate (None = off)
    fx_filter: float = FX           # focal length the filter believes (calibration error ablation)
    # Local origin is defined at startup, not drawn from simulation truth.
    p0_pos: float = 1e-4
    p0_head_deg: float = 0.01
    search_w: float = 0.35
    search_timeout: float = 40.0
    marker_lost_timeout: float = 8.0
    corner_sigma_px: float = 1.0
    camera_pos_floor: float = 0.015
    camera_head_floor_deg: float = 1.0
    max_reprojection_px: float = 2.0
>>>>>>> 3b98b09 (added cam and lidar markers, origin starts at robot then moves to aruco marker, smaller room, less obstacles)
    # controller (Bryson's rule: Q = 1/e_max^2, R = 1/u_max^2)
    e_max: tuple = (0.05, 0.05, 0.10)   # x [m], y [m], theta [rad]
    u_max: tuple = (0.05, 0.5)          # delta v [m/s], delta omega [rad/s]
    lqr_q_scale: float = 2
    lqr_r_scale: float = 1
    # planner
    v_ref: float = 0.18
    inflate: float = 0.24
    # world randomisation
    n_obstacles: int = 2
    viewer: bool = False            # MuJoCo live 3D view (macOS: run with mjpython)
    push_time: float = -1.0         # time [s] of a 6 N sideways push, -1 = off

