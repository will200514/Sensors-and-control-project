import time

import cv2
from rosbridge_api import RosbridgeRobot

from src import config as C
from src.config import Cfg
from src.perception.aruco_detector import make_detector
from src.perception.docking_pose import estimate_dock_pose


def main():
    detector = make_detector()
    cfg = Cfg()

    print("Expected dictionary: DICT_4X4_50")
    print("Expected marker ID:", C.DOCK_MARKER_ID)
    print("Configured marker size:", C.MARKER_L, "metres")
    print("Configured camera height:", C.CAM_HEIGHT)
    print("Configured marker centre height:", C.MARKER_HEIGHT)

    with RosbridgeRobot("192.168.1.210") as robot:
        robot.stop()

        deadline = time.monotonic() + 15.0
        last_print = -1.0

        while time.monotonic() < deadline:
            image = robot.query_image(
                timeout=0.5,
                wait_for_new=True,
            )

            if image is None:
                continue

            now = time.monotonic()

            if now - last_print < 1.0:
                continue

            last_print = now

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Read all IDs in the configured dictionary,
            # before the project's docking-ID filter.
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is None:
                print("No ArUco markers detected.", flush=True)
                continue

            print("Detected IDs:", ids.ravel().tolist(), flush=True)

            for marker_corners, marker_id in zip(
                corners, ids.ravel()
            ):
                if int(marker_id) != C.DOCK_MARKER_ID:
                    continue

                observation = estimate_dock_pose(
                    marker_corners.reshape(4, 2),
                    cfg,
                )

                if observation is None:
                    print(
                        "Correct ID detected, but pose rejected.",
                        flush=True,
                    )
                else:
                    print(
                        "Pose accepted:",
                        observation.pose,
                        "Reprojection error:",
                        observation.reprojection_px,
                        flush=True,
                    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMarker check ended.")