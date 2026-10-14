"""Central configuration for Track A (TurtleBot3 visual waypoint navigation + docking).

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
TRACK = 0.160        # wheel separation b [m]
ROBOT_RADIUS = 0.105 # footprint radius [m]
V_MAX = 0.20         # commanded linear speed limit [m/s] (hardware max 0.22)
W_MAX = 1.5          # commanded turn-rate limit [rad/s] (hardware max 2.84)
MOTOR_TAU = 0.12     # first-order wheel-speed lag [s] (unmodelled by the estimator/controller)
WHEEL_RADIUS = 0.033 # [m]  # A4-MUJOCO: added, used to convert wheel speeds for the MuJoCo motors

# ---- camera (pinhole, 640x480, ~62 deg HFOV) -------------------------------------------------
FX = 530.0           # focal length [px]
CX_PX = 320.0        # principal point u [px]
MARKER_L = 0.10      # ArUco side length [m]
CAM_OFFSET = 0.06    # camera is mounted this far ahead of the robot centre [m]
CAM_HEIGHT = 0.12    # [m]  # A4-MUJOCO: added, markers are mounted at camera height
CAM_MAX_RANGE = 3.5
CAM_MIN_RANGE = 0.15
MARKER_MAX_INCIDENCE = np.deg2rad(60.0)
SIGMA_PX = 1.0       # corner-detection noise on marker centre column [px]
SIGMA_SIZE_PX = 1.0  # noise on apparent marker size [px]
SIGMA_BEARING_FLOOR = np.deg2rad(0.5)  # pose-estimation floor not explained by pixel noise

# ---- LiDAR ------------------------------------------------------------------------------
LIDAR_BEAMS = 120
LIDAR_MAX = 3.5
LIDAR_SIGMA = 0.01

# ---- wheel odometry ------------------------------------------------------------------
SIGMA_ENC = 0.003    # encoder speed noise per wheel [m/s]
SIGMA_SLIP = 0.02    # random relative wheel slip per step
SIGMA_BIAS = 0.01    # fixed relative wheel-radius / slip bias per wheel (per trial)

# ---- world -------------------------------------------------------------------------------
ROOM = (0.0, 5.0, 0.0, 5.0)   # xmin, xmax, ymin, ymax
# (x, y, direction the marker face looks into the room)
MARKERS = np.array([
    [1.0, 0.02, np.pi / 2], [2.5, 0.02, np.pi / 2], [3.9, 0.02, np.pi / 2],          # bottom wall
    [0.02, 1.0, 0.0], [0.02, 2.5, 0.0], [0.02, 4.0, 0.0],                            # left wall
    [1.0, 4.98, -np.pi / 2], [2.5, 4.98, -np.pi / 2], [4.0, 4.98, -np.pi / 2],       # top wall
    [4.98, 2.0, np.pi], [4.98, 3.2, np.pi], [4.98, 4.2, np.pi],                      # right wall
    [4.98, 0.8, np.pi],                                                              # DOCK marker (last)
])
DOCK_MARKER = MARKERS[-1]
DOCK_STANDOFF = 0.30                                   # final distance robot centre -> marker [m]
DOCK_POSE = np.array([DOCK_MARKER[0] - DOCK_STANDOFF, DOCK_MARKER[1], 0.0])
LEADIN = np.array([3.1, DOCK_MARKER[1]])               # start of the straight docking lead-in
WAYPOINT_1 = np.array([2.5, 2.5])                      # intermediate visual waypoint
DOCK_TOL_POS = 0.05
DOCK_TOL_HEAD = np.deg2rad(5.0)


@dataclass
class Cfg:
    """Per-trial tunables. Defaults are the submitted design; ablations override fields."""
    # estimator
    use_camera: bool = True
    q_scale: float = 4.0            # multiplies the derived process covariance
    r_scale: float = 2.0            # multiplies the derived camera covariance
    gate: float = 9.21              # chi2(2) 99% innovation gate (None = off)
    fx_filter: float = FX           # focal length the filter believes (calibration error ablation)
    cam_fov_deg: float = 62.2
    p0_pos: float = 0.10
    p0_head_deg: float = 5.0
    # controller (Bryson's rule: Q = 1/e_max^2, R = 1/u_max^2)
    e_max: tuple = (0.05, 0.05, 0.10)   # x [m], y [m], theta [rad]
    u_max: tuple = (0.05, 0.5)          # delta v [m/s], delta omega [rad/s]
    lqr_q_scale: float = 1.0
    lqr_r_scale: float = 1.0
    # planner
    v_ref: float = 0.18
    inflate: float = 0.24
    # world randomisation
    n_obstacles: int = 4
    extra_disturbance: float = 0.0  # extra relative wheel-slip bias (stress test)
    # A4-MUJOCO: added. Which simulator to run, the live viewer, and an optional sideways push (Part E).
    backend: str = "numpy"          # "numpy" (original) or "mujoco"
    viewer: bool = False            # MuJoCo live 3D view (macOS: run with mjpython)
    push_time: float = -1.0         # time [s] of a 6 N sideways push, -1 = off
