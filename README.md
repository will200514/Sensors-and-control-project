# Sensors-and-control-project

41014 Sensors and Control for Mechatronic Systems, Integrated Project, Track A.

A TurtleBot3 in MuJoCo searches for an ArUco-marked charging station, localises with an EKF (wheel odometry + camera pose fixes), maps obstacles with a 2D LiDAR, and docks using an LQR trajectory tracker that only uses the estimated pose.

## Resources
- Project page: https://will200514.github.io/Sensors-and-control-project/
- Demo video: https://youtu.be/eys-GpOjkgE
- Simulation code: [`track A - mobile robot/`](track%20A%20-%20mobile%20robot/)
- Real TurtleBot3 code: [`track A - hardware/`](track%20A%20-%20hardware/)
- Validation notes: [`track A - mobile robot/VALIDATION.md`](track%20A%20-%20mobile%20robot/VALIDATION.md)

## Group information
Group number: 28

Group composition:
- Yani Hickey Berry
- Will Munro
- Pauras Londhe
- Mahmoud Abdallah

## Install info
Python 3.12, tested with MuJoCo 3.10.0 and OpenCV 4.x.
```bash
conda create -n ir_env python=3.12
conda activate ir_env
cd "track A - mobile robot"
python -m pip install -r requirements.txt
```
Dependencies: numpy, scipy, matplotlib, pillow, pytest, mujoco==3.10.0, opencv-python>=4.7,<5.

## Run commands
All commands are run from inside `track A - mobile robot/`.

Run one docking trial with the live MuJoCo viewer (macOS needs `mjpython` for the viewer):
```bash
mjpython main.py --seed 3
```
The robot rotates until it detects marker ID 1, switches to the dock frame, plans a path through the LiDAR map and drives to 0.30 m in front of the marker. Plots of estimate vs ground truth, NEES and detections are saved to `results/`.

Without the viewer (any OS):
```bash
python main.py --seed 3 --no-view
```

Unit tests (model, Jacobians, controllability, frame transforms, detector):
```bash
python -m pytest -q
```

Quick evaluation (baseline and push-disturbance trials):
```bash
python evaluate.py mujoco 1
```
`python evaluate.py mujoco 20` reproduces the 20-seed baseline and push results on the project page. Running `python evaluate.py` with no arguments runs the full sweep and takes a long time.
