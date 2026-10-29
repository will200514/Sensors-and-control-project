"""Checkerboard calibration of the real TurtleBot camera through rosbridge.

Run from the "track A - hardware" folder:
    python calibrate_real_camera.py --robot-ip 192.168.1.210 --cols 8 --rows 6 --square 0.020 --auto

--cols/--rows are INNER corners (count where black squares touch, not the squares).
The 41014 A4 board (9 x 7 squares, 20 mm) has 8 x 6 inner corners: the defaults.

Click on the camera window first so it receives key presses (not the terminal).
  SPACE  capture the current frame (only works while the banner says BOARD FOUND)
  A      toggle auto-capture: grabs a frame every 1.5 s while the board is found and has moved
  C      calibrate (needs at least 12 captures) and save
  Q/ESC  quit
With --auto, auto-capture starts on, and it calibrates by itself after --target captures.
Results are saved to results/real_camera_calib.npz, captures to results/calib_images/.
"""
import argparse
import time

import cv2
import numpy as np

from src import config as C


def calibrate(obj, img, size):
    # Board views from a robot camera are mostly face-on, which cannot separate lens
    # distortion and image centre from focal length. Fit fx, fy only (centre at the
    # image middle, no distortion); the tilt fit and measure_geometry.py absorb the rest.
    w, h = size
    K0 = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], float)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST
             | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, K0, np.zeros(5), flags=flags)
    print(f"\n{len(img)} views, RMS reprojection {rms:.3f} px, image {size[0]}x{size[1]}")
    print(f"fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}")
    print(f"distortion {np.round(dist.ravel()[:5], 5).tolist()}")
    if rms > 1.0:
        print("RMS above 1 px: retake with the board flat, still and well lit.")
    np.savez(C.RESULTS_DIR / "real_camera_calib.npz", K=K, dist=dist, rms=rms,
             size=np.array(size))
    print(f"Saved {C.RESULTS_DIR / 'real_camera_calib.npz'}\n", flush=True)


def main():
    from rosbridge_api import RosbridgeRobot

    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.1.210")
    ap.add_argument("--cols", type=int, default=8, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down")
    ap.add_argument("--square", type=float, default=0.020, help="square size [m]")
    ap.add_argument("--auto", action="store_true", help="auto-capture and auto-calibrate")
    ap.add_argument("--target", type=int, default=20, help="captures before auto-calibrate")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    board = np.array([[i * args.square, j * args.square, 0]
                      for j in range(args.rows) for i in range(args.cols)], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    save_dir = C.RESULTS_DIR / "calib_images"
    save_dir.mkdir(exist_ok=True)

    obj, img, size = [], [], None
    auto, last_capture, last_corners = args.auto, 0.0, None
    found_once = False

    print("Click the 'calibration' window so it gets the key presses.", flush=True)
    with RosbridgeRobot(args.robot_ip) as robot:
        robot.stop()
        while robot.is_connected:
            frame = robot.query_image(timeout=1.0, wait_for_new=True)
            if frame is None:
                print("no camera frame in 1 s", flush=True)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
            if not ok:
                # The sector-based detector copes better with blur and lighting.
                ok, corners = cv2.findChessboardCornersSB(gray, pattern)
            vis = frame.copy()
            small = False
            if ok:
                # Refinement window must be smaller than a square, or corners snap to
                # the wrong edge (this broke the first calibration on far-away boards).
                grid = corners.reshape(args.rows, args.cols, 2)
                square = min(np.linalg.norm(np.diff(grid, axis=1), axis=2).min(),
                             np.linalg.norm(np.diff(grid, axis=0), axis=2).min())
                win = max(2, int(0.4 * square))
                corners = cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), crit)
                cv2.drawChessboardCorners(vis, pattern, corners, ok)
                small = square < 15
                if small:
                    ok = False
                if not found_once:
                    print("Board found.", flush=True)
                    found_once = True

            if ok:
                banner = "BOARD FOUND - press SPACE"
            elif small:
                banner = "BOARD TOO FAR - bring it closer (squares < 15 px)"
            else:
                banner = f"NO BOARD ({args.cols}x{args.rows} inner corners?)"
            colour = (0, 200, 0) if ok else (0, 0, 255)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(vis, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(vis, f"captures {len(img)}   auto {'ON' if auto else 'off'} (A)   "
                             f"calibrate (C)   {size[0]}x{size[1]}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("calibration", vis)
            key = cv2.waitKey(30) & 0xFF

            capture = False
            if key == ord(" "):
                if ok:
                    capture = True
                else:
                    print("SPACE pressed but no board in this frame: hold it still, fully in view.",
                          flush=True)
            elif auto and ok and time.monotonic() - last_capture > 1.5:
                moved = (last_corners is None or
                         np.linalg.norm((corners - last_corners).reshape(-1, 2), axis=1).mean() > 25)
                capture = moved

            if capture:
                obj.append(board)
                img.append(corners)
                last_capture, last_corners = time.monotonic(), corners.copy()
                cv2.imwrite(str(save_dir / f"view_{len(img):02d}.png"), frame)
                print(f"captured {len(img)}", flush=True)
                if auto and args.auto and len(img) >= args.target:
                    calibrate(obj, img, size)
                    break

            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                auto = not auto
                print(f"auto-capture {'ON: move the board around slowly' if auto else 'off'}",
                      flush=True)
            if key == ord("c"):
                if len(img) < 12:
                    print(f"need at least 12 captures (have {len(img)})", flush=True)
                else:
                    calibrate(obj, img, size)
            elif key not in (255, ord(" "), ord("a"), ord("q"), 27):
                print(f"key {key} ignored", flush=True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
