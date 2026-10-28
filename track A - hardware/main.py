"""Hardware-only TurtleBot docking entry point."""
import argparse
from src.hardware_mission import run_hardware


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-ip', required=True)
    parser.add_argument('--wheel-velocity-units', choices=['m/s', 'rad/s'], default='m/s')
    args = parser.parse_args()
    try:
        print(run_hardware(args.robot_ip, wheel_velocity_units=args.wheel_velocity_units))
    except KeyboardInterrupt:
        print('\nInterrupted. Stop command attempted during cleanup.')


if __name__ == '__main__':
    main()
