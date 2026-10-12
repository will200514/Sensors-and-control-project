import mujoco
import mujoco.viewer
import time

# Path to your robot's MJCF file
model_path = "universal_robots_ur3e/scene.xml"

# Load robot
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Open interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)