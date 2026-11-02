"""Hardware-only TurtleBot docking entry point."""
" python main.py --robot-ip 192.168.1.206 "
import argparse
from src.hardware_mission import run_hardware



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-ip', required=True)
    # This robot's /joint_states wheel velocities are already in
    # m/s: spinning at 0.35 rad/s gives about -0.049 / +0.049 per wheel (= 0.35 * 0.143 m).
    # Reading them as rad/s shrank odometry about 30 times (first real run, 28 Oct).
    parser.add_argument('--wheel-velocity-units', choices=['m/s', 'rad/s'], default='m/s')
    # [NEW] Live camera / ArUco / map / LiDAR window
    parser.add_argument('--no-view', action='store_true', help='run without the live window')
    parser.add_argument('--record', action='store_true', help='also save results/live_view.mp4')
    args = parser.parse_args()
    try:
        print(run_hardware(args.robot_ip, wheel_velocity_units=args.wheel_velocity_units,
                           show_view=not args.no_view, record_view=args.record))
    except KeyboardInterrupt:
        print('\nInterrupted. Stop command attempted during cleanup.')


if __name__ == '__main__':
    main()
