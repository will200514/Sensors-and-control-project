# mjpython "/Users/willmunro/ir-env/Sensors-and-control-project/track b - manipulator/ur3_mujoco.py"
# allows me to run the mujoco environment on my mac.
import mujoco
import mujoco.viewer
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from PIL import Image


# ----------------------------------
# 1. PROJECT PATHS
# ----------------------------------

# Location of this Python script
script_dir = Path(__file__).resolve().parent

# Parent project directory
project_dir = script_dir.parent

# Original UR3e model
robot_path = (
    project_dir
    / "universal_robots_ur3e"
    / "ur3e.xml"
)

print("Robot XML:", robot_path)

if not robot_path.exists():
    raise FileNotFoundError(
        f"Robot XML not found: {robot_path}"
    )

# ----------------------------------
# 2. LOAD UR3E robot XML file
# ----------------------------------

tree = ET.parse(robot_path)
root = tree.getroot()

# ----------------------------------
# WRIST-MOUNTED RGB-D CAMERA
# adds the RGB-D camera to the wrist_3_joint body of the UR3e robot which is essentially
# the end-effector of the robot. The camera is mounted 8cm above the wrist_3_joint 
# ----------------------------------

wrist_body = None  # creates the variable wrist_body and sets it to None, which will be used to store the body element of the wrist_3_joint if found

for body in root.iter("body"):  # iterates over all the body elements in the XML tree to find the body element that contains the wrist_3_joint
    for joint in body.findall("joint"):
        if joint.get("name") == "wrist_3_joint":
            wrist_body = body
            break

    if wrist_body is not None:
        break

if wrist_body is None:
    raise ValueError("Could not find the wrist_3_joint body")

ET.SubElement(   # creates the camera element as a child of the wrist_3_joint body in the XML tree. The camera is positioned 8cm above the wrist_3_joint and has a field of view of 60 degrees.
    wrist_body,
    "camera",
    {
        "name": "wrist_camera",
        "pos": "0 0 0.08",
        "quat": "0.7071 0 -0.7071 0", # rotation of camera view relative to wrist. 
        "fovy": "100"
    }
)

# ----------------------------------
# VISIBLE CAMERA HOUSING
# ----------------------------------

ET.SubElement(
    wrist_body,
    "geom",
    {
        "name": "camera_housing",
        "type": "box",
        "pos": "0 0 0.08",
        "size": "0.025 0.035 0.015",
        "rgba": "0.1 0.1 0.1 1",
        "contype": "0",
        "conaffinity": "0",
        "mass": "0.001"
    }
)

print("Final wrist body:", wrist_body.get("name"))


# ----------------------------------
# 3. ADD ENVIRONMENT
# include the ground, light, a table, a cube and the robot to the worldbody of the XML
# ----------------------------------

worldbody = root.find("worldbody")

if worldbody is None:
    raise ValueError("Robot XML has no worldbody")

# Ground plane
ET.SubElement(
    worldbody,
    "geom",
    {
        "name": "ground",
        "type": "plane",
        "size": "2 2 0.1",
        "rgba": "0.85 0.85 0.85 1"
    }
)

# Overhead light
ET.SubElement(
    worldbody,
    "light",
    {
        "name": "overhead_light",
        "pos": "0 0 3",
        "dir": "0 0 -1"
    }
)

# Table
table = ET.SubElement(
    worldbody,
    "body",
    {
        "name": "table",
        "pos": "0.35 0 0.05"
    }
)

ET.SubElement(
    table,
    "geom",
    {
        "name": "table_surface",
        "type": "box",
        "size": "0.2 0.2 0.025",
        "rgba": "0.6 0.45 0.3 1"
    }
)

# Target cube
cube = ET.SubElement(
    worldbody,
    "body",
    {
        "name": "target_cube",
        "pos": "0.35 0 0.10"
    }
)

ET.SubElement(cube, "freejoint")

ET.SubElement(
    cube,
    "geom",
    {
        "name": "cube_geom",
        "type": "box",
        "size": "0.025 0.025 0.025",    # sizer is half-lengths, so this is a 5cm cube
        "rgba": "1 0 0 1",
        "mass": "0.1"
    }
)

# ----------------------------------
# 4. LOAD MUJOCO MODEL
# ----------------------------------

# Convert XML tree to string
xml_string = ET.tostring(
    root,
    encoding="unicode"
)

# Tell MuJoCo where mesh assets are
assets_dir = robot_path.parent / "assets"

# Load model from XML string
model = mujoco.MjModel.from_xml_string(
    xml_string,
    assets={
        path.relative_to(robot_path.parent).as_posix():
        path.read_bytes()
        for path in assets_dir.rglob("*")
        if path.is_file()
    }
)



data = mujoco.MjData(model)


# ----------------------------------
# INITIAL ROBOT POSE
# ----------------------------------

start_q = np.array([
    0.0,      # shoulder_pan
    -1.57,    # shoulder_lift
    -2.58,    # elbow
    0.565,    # wrist_1
    0.0,      # wrist_2
    0.0       # wrist_3
])

joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

# Set initial joint positions
for name, angle in zip(joint_names, start_q):
    data.joint(name).qpos[0] = angle
    data.joint(name).qvel[0] = 0.0

# Set actuator controls
# Assumes the actuators are position-controlled
data.ctrl[:6] = start_q

# Update simulation
mujoco.mj_forward(model, data)


# ----------------------------------
# CAMERA RENDERER
# creates a renderer for the wrist-mounted camera and captures an RGB image and a depth image from the camera's perspective. The RGB image is captured first, followed by the depth image. The shapes of both images and the depth range are printed to the console.
# ----------------------------------

renderer = mujoco.Renderer(  # creates a renderer object that will be used to render images from the wrist-mounted camera. The renderer is initialized with the MuJoCo model and the desired image dimensions (height and width).
    model,
    height=480,
    width=640
)

# Capture RGB image
renderer.update_scene(
    data,
    camera="wrist_camera"
)

rgb_image = renderer.render()

# Capture depth image
renderer.enable_depth_rendering()

renderer.update_scene(
    data,
    camera="wrist_camera"
)

depth_image = renderer.render()

renderer.disable_depth_rendering()




print("RGB image shape:", rgb_image.shape)
print("Depth image shape:", depth_image.shape)
print("Depth range:", depth_image.min(), depth_image.max())

print("Robot initialised at starting pose!")

print("Simulation loaded!")
print("Bodies:", model.nbody)
print("Joints:", model.njnt)
print("Actuators:", model.nu)


image_path = script_dir / "wrist_camera_rgb.png"

Image.fromarray(rgb_image).save(image_path)

print("RGB camera image saved to:", image_path)

# ----------------------------------
# 5. RUN SIMULATION
# ----------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        remaining = (
            model.opt.timestep
            - (time.time() - step_start)
        )

        if remaining > 0:
            time.sleep(remaining)


