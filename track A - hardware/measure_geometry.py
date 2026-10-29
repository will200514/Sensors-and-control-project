"""Measure camera height and tilt from the docking marker, then save them for the robot code.

Run from the "track A - hardware" folder AFTER calibrate_real_camera.py:
    python measure_geometry.py --robot-ip 192.168.1.210 --marker-size 0.20 --marker-height 0.12 --distance 0.70

Setup:
  1. Mount the printed ArUco DICT_4X4_50 ID 1 marker upright on the wall/dock.
  2. With a ruler, measure the BLACK square edge (no white border)   -> --marker-size  [m]
     and the floor-to-marker-centre height                           -> --marker-height [m]
  3. Put the robot facing the marker squarely, camera lens about 0.5 to 0.8 m from the
     marker face, and tape-measure that lens-to-marker distance      -> --distance [m] (check only)
  4. Run the script. The robot does not move.

It averages 40 frames of solvePnP (calibrated camera) and writes results/hardware_geometry.json with
MARKER_L, MARKER_HEIGHT, CAM_HEIGHT, CAM_PITCH_DEG and CAM_OFFSET. hardware_mission.py and
aruco_diag.py load that file automatically.
"""
import argparse
import json
import time

import cv2
import numpy as np

from src import config as C
from src.perception.aruco_detector import make_detector, detect_markers
from src.perception.docking_pose import camera_matrix


def camera_in_marker_frame(corners, marker_size):
    """Camera height above marker centre [m], downward tilt [deg], distance [m], RMS [px]."""
    h = marker_size / 2
    obj = np.array([[-h, h, 0.], [h, h, 0.], [h, -h, 0.], [-h, -h, 0.]])
    K, dist = camera_matrix(), np.array(C.CAM_DISTORTION)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, corners, K, dist,
                                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
    best = None
    for rv, tv, err in zip(rvecs, tvecs, np.ravel(errs)):
        if not (np.isfinite(rv).all() and np.isfinite(tv).all()) or tv[2, 0] <= 0:
            continue
        if best is None or err < best[2]:
            best = (rv, tv, float(err))
    if best is None:
        return None
    rv, tv, err = best
    R = cv2.Rodrigues(rv)[0]
    p = -R.T @ tv.ravel()                  # camera centre in marker frame (x right, y up, z out)
    optical = R.T @ np.array([0., 0., 1.])  # camera viewing direction in marker frame
    tilt_down = -np.degrees(np.arcsin(np.clip(optical[1], -1, 1)))
    return p[1], tilt_down, p[2], err


def main():
    from rosbridge_api import RosbridgeRobot

    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.1.210")
    ap.add_argument("--marker-size", type=float, required=True,
                    help="black square edge [m], measured with a ruler")
    ap.add_argument("--marker-height", type=float, required=True,
                    help="floor to marker centre [m]")
    ap.add_argument("--distance", type=float, default=None,
                    help="tape-measured camera lens to marker face [m], used as a check")
    ap.add_argument("--cam-offset", type=float, default=C.CAM_OFFSET,
                    help=f"camera lens forward of wheel-axle centre [m] (default {C.CAM_OFFSET})")
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()

    for line in C.load_hardware_calibration(require_geometry=False):
        print(line)

    detector = make_detector()
    rows = []
    with RosbridgeRobot(args.robot_ip) as robot:
        robot.stop()
        deadline = time.monotonic() + 60.0
        while len(rows) < args.frames and time.monotonic() < deadline:
            image = robot.query_image(timeout=1.0, wait_for_new=True)
            if image is None:
                continue
            if image.shape[1::-1] != tuple(C.CAM_IMAGE_SIZE):
                raise SystemExit(f"Camera image is {image.shape[1]}x{image.shape[0]} but the "
                                 f"calibration was {C.CAM_IMAGE_SIZE}. Recalibrate.")
            found = detect_markers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), detector)
            if not found:
                print("marker ID 1 not seen", flush=True)
                continue
            result = camera_in_marker_frame(found[0][1], args.marker_size)
            if result is not None:
                rows.append(result)
                print(f"{len(rows):2d}/{args.frames}", flush=True)

    if len(rows) < 10:
        raise SystemExit("Too few marker frames. Check the marker is in view and lit.")

    rows = np.array(rows)
    dh, tilt, dist, err = np.median(rows, axis=0)
    spread = rows[:, :2].std(axis=0)
    print(f"\ncamera {dh:+.3f} m relative to marker centre (std {spread[0]:.3f})")
    print(f"camera tilt {tilt:+.2f} deg down (std {spread[1]:.2f})")
    print(f"PnP distance {dist:.3f} m, PnP error {err:.2f} px")
    if args.distance is not None:
        print(f"tape distance {args.distance:.3f} m -> scale error "
              f"{100 * (dist - args.distance) / args.distance:+.1f} % "
              "(over 3 % means re-check --marker-size or the camera calibration)")

    geometry = {
        "MARKER_L": args.marker_size,
        "MARKER_HEIGHT": args.marker_height,
        "CAM_HEIGHT": round(float(args.marker_height + dh), 4),
        "CAM_PITCH_DEG": round(float(tilt), 2),
        "CAM_OFFSET": args.cam_offset,
    }
    C.GEOMETRY_FILE.write_text(json.dumps(geometry, indent=2))
    print(f"\nSaved {C.GEOMETRY_FILE}:\n{json.dumps(geometry, indent=2)}")
    print("Sanity check CAM_HEIGHT against a tape measure (floor to lens), within about 2 cm.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
