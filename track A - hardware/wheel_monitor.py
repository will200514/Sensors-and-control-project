"""Display measured wheel motion during manual driving."""

import time

from rosbridge_api import RosbridgeRobot
from src.model.diffdrive import joint_states_to_body
from src import config as C


def main():

    print("Connecting to 192.168.1.210...", flush=True)
    with RosbridgeRobot("192.168.1.210") as robot:
        print("Monitoring wheel motion. Press Ctrl+C to exit.")

        while robot.is_connected:
            joints = robot.query_joint_states()

            if joints is not None:
                print("Raw wheel velocities:", joints.get("velocity"))
                print("Wheel radius:", C.WHEEL_RADIUS, "Track:", C.TRACK)
                v, omega = joint_states_to_body(joints)

                print(
                    f"Forward: {v:+.3f} m/s | "
                    f"Turning: {omega:+.3f} rad/s",
                    flush=True,
                )

            time.sleep(0.2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")