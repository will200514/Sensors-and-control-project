"""Placeholder for the MuJoCo TurtleBot3 backend.

NOT IMPLEMENTED / NOT TESTED. Implement the same interface as src.sim.world.World:
    step(v_cmd, w_cmd) -> (v_odo, w_odo)   # set wheel velocity actuators, mj_step, read wheel encoders
    camera()  -> [(marker_index, u_px, s_px), ...]   # render the camera + cv2.aruco detection
    lidar()   -> np.ndarray(LIDAR_BEAMS)             # mj_ray from the LiDAR site
    truth()   -> pose from mjData (evaluation only)
The TurtleBot3 model is the robotis_mujoco_menagerie model linked in the Week 11 slides.
"""
class MujocoWorld:  # pragma: no cover
    def __init__(self, *a, **k):
        raise NotImplementedError("MuJoCo is not installed in the environment this was written in.")
