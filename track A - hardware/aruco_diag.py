"""Live ArUco diagnostic for the real TurtleBot camera.

Run from the "track A - hardware" folder with the robot on:
    python aruco_diag.py --robot-ip 192.168.1.210

Robot stays still. For every frame (about 1 per second) it prints:
  - image size and whether it matches CX_PX / CY_PX in src/config.py
  - which ArUco IDs are seen, in which dictionary (catches a wrong printed dictionary)
  - for ID 1: free solvePnP distance (compare with a tape measure), the level-floor fit RMS that
    estimate_dock_pose() uses to accept or reject the pose, and, if rejected, how far the marker
    sits above/below where the configured camera height, marker height and tilt say it should.
It also saves the latest annotated frame to results/aruco_diag.png.

Uses results/real_camera_calib.npz and results/hardware_geometry.json if they exist
(simulation values otherwise, which is useful before calibrating). Does not drive the robot.
"""
import argparse
import copy
import time

import cv2
import numpy as np

from src import config as C
from src.config import Cfg
from src.perception.aruco_detector import make_detector
from src.perception.docking_pose import (camera_matrix, estimate_dock_pose, marker_corners,
                                          project_corners)

DICTS = ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_7X7_50", "DICT_ARUCO_ORIGINAL"]


def other_dictionaries(gray):
    found = []
    for name in DICTS[1:]:
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)),
                                      cv2.aruco.DetectorParameters())
        _, ids, _ = det.detectMarkers(gray)
        if ids is not None:
            found.append((name, ids.ravel().tolist()))
    return found


def analyse(image, detector, cfg):
    lines = []
    h, w = image.shape[:2]
    match = "OK" if (w, h) == tuple(C.CAM_IMAGE_SIZE) else "MISMATCH, recalibrate at this size"
    lines.append(f"image {w}x{h}, calibration {C.CAM_IMAGE_SIZE[0]}x{C.CAM_IMAGE_SIZE[1]} -> {match}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    vis = image.copy()
    if ids is None:
        lines.append(f"no DICT_4X4_50 markers ({len(rejected)} square candidates rejected)")
        other = other_dictionaries(gray)
        if other:
            lines.append(f"  BUT found markers in other dictionaries: {other} -> reprint as DICT_4X4_50 ID 1")
        return lines, vis
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    lines.append(f"DICT_4X4_50 IDs seen: {ids.ravel().tolist()}")
    for c, j in zip(corners, ids.ravel()):
        if int(j) != C.DOCK_MARKER_ID:
            continue
        px = c.reshape(4, 2).astype(np.float64)
        side = np.linalg.norm(px - np.roll(px, -1, axis=0), axis=1).mean()
        lines.append(f"ID {j}: mean side {side:.1f} px")

        ok, rv, tv = cv2.solvePnP(marker_corners(), px, camera_matrix(cfg.fx_filter),
                                  np.array(C.CAM_DISTORTION), flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            lines.append(f"  free PnP distance to marker: {np.linalg.norm(tv):.3f} m "
                         f"(tape-measure this; a big mismatch means MARKER_L or FX is wrong)")
            cv2.drawFrameAxes(vis, camera_matrix(cfg.fx_filter), np.array(C.CAM_DISTORTION),
                              rv, tv, C.MARKER_L / 2)

        loose = copy.copy(cfg)
        loose.max_reprojection_px = 1e9
        obs = estimate_dock_pose(px, loose)
        if obs is None:
            lines.append("  level-floor fit: no valid candidate (marker behind camera, too small or too close)")
        else:
            verdict = "ACCEPTED" if obs.reprojection_px <= cfg.max_reprojection_px else "REJECTED"
            lines.append(f"  level-floor fit RMS {obs.reprojection_px:.2f} px "
                         f"(limit {cfg.max_reprojection_px}) -> {verdict}, "
                         f"pose {np.round(obs.pose, 3)}")
            resid = px - project_corners(obs.pose, cfg.fx_filter)
            dv = resid[:, 1].mean()
            if verdict == "REJECTED":
                lines.append(f"  marker sits {abs(dv):.1f} px {'lower' if dv > 0 else 'higher'} in the image "
                             f"than config allows (~{np.degrees(np.arctan(abs(dv) / cfg.fx_filter)):.1f} deg): "
                             f"rerun measure_geometry.py")
    return lines, vis


def main():
    from rosbridge_api import RosbridgeRobot

    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.1.210")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    for line in C.load_hardware_calibration(require_camera=False, require_geometry=False):
        print(line)
    detector, cfg = make_detector(), Cfg()
    cfg.fx_filter = C.FX
    cfg.fit_tilt_sd_deg = 1.0        # same as hardware_mission.py
    cfg.max_reprojection_px = 5.0    # same as hardware_mission.py
    out = C.RESULTS_DIR / "aruco_diag.png"
    with RosbridgeRobot(args.robot_ip) as robot:
        robot.stop()
        end, last = time.monotonic() + args.seconds, 0.0
        while time.monotonic() < end:
            image = robot.query_image(timeout=1.0, wait_for_new=True)
            if image is None:
                print("no camera frame in 1 s", flush=True)
                continue
            if time.monotonic() - last < 1.0:
                continue
            last = time.monotonic()
            lines, vis = analyse(image, detector, cfg)
            print("\n".join(lines), "\n", flush=True)
            cv2.imwrite(str(out), vis)
    print(f"Last annotated frame: {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
