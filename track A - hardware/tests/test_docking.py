"""Geometry, frame transition and estimator checks for single-marker navigation."""
import numpy as np
import pytest
from dataclasses import replace
from src.config import Cfg
from src import config as C
from src.perception.docking_pose import project_corners, estimate_dock_pose
from src.estimation.ekf import EKF, odom_covariance
from src.evaluation_frames import relative_pose, compose_pose
from src.perception.lidar import OccGrid
from src.control.lqr_tracker import LQRTracker
from src.control.planner import astar, inflated_map
from src.mission import _plan
from src.perception.aruco_detector import make_detector, detect_markers
import cv2


@pytest.mark.parametrize('pose', [[2.,0.,np.pi], [4.,2.,-2.6], [.3,0.,np.pi], [1.2,-.5,2.8]])
def test_corner_pose_round_trip(pose):
    obs = estimate_dock_pose(project_corners(pose), Cfg())
    assert obs is not None
    d = obs.pose - pose
    d[2] = (d[2]+np.pi)%(2*np.pi)-np.pi
    assert np.max(np.abs(d)) < 1e-5
    assert np.linalg.eigvalsh(obs.covariance).min() > 0


def test_camera_covariance_scale_and_tiny_marker():
    pose = [2.,1.,-2.7]
    corners = project_corners(pose)
    a = estimate_dock_pose(corners, Cfg(r_scale=1.))
    b = estimate_dock_pose(corners, Cfg(r_scale=4.))
    assert np.allclose(b.covariance, 4*a.covariance)
    assert estimate_dock_pose(np.array([[0,0],[2,0],[2,2],[0,2]]), Cfg()) is None


def test_local_origin_and_acquisition_not_double_counted():
    ekf = EKF(np.zeros(3), np.eye(3)*1e-8, Cfg())
    assert np.array_equal(ekf.x, np.zeros(3))
    obs = estimate_dock_pose(project_corners([2.,.6,-2.9]), Cfg())
    ekf.initialise_dock(obs.pose, obs.covariance)
    assert np.array_equal(ekf.P, obs.covariance)
    assert ekf.n_updates == 0
    assert ekf.update_pose(obs.pose, obs.covariance)
    assert np.allclose(ekf.P, obs.covariance/2)


def test_covariance_sign_from_independent_wheel_noise():
    v,w=.15,.6
    vl,vr=v-.5*w*C.TRACK,v+.5*w*C.TRACK
    wheel_var=C.SIGMA_ENC**2 + (C.SIGMA_SLIP**2+C.SIGMA_BIAS**2)*np.square([vl,vr])
    J=np.array([[.5,.5],[-1/C.TRACK,1/C.TRACK]])
    assert np.allclose(odom_covariance(v,w), J@np.diag(wheel_var)@J.T)


def test_frame_round_trip_and_negative_grid_coordinates():
    origin=[4.2,1.3,2.7]; pose=[2.,-1.,-.8]
    assert np.allclose(relative_pose(compose_pose(origin,pose),origin),pose)
    grid=OccGrid()
    ij=grid.to_cell(-2.1,-3.4)
    p=grid.cell_to_world(*ij)
    assert np.linalg.norm(np.array(p)-[-2.1,-3.4]) <= np.sqrt(2)*grid.res
    route=astar(inflated_map(grid,.24),grid,[3.,2.],[1.88,0.])
    assert route is not None
    assert np.linalg.norm(np.array(route[-1])-[1.88,0.]) < .08


def test_stationary_robot_cannot_finish_trajectory_by_index():
    tracker=LQRTracker(Cfg())
    T=np.column_stack([np.linspace(0,2,200), np.zeros((200,2)), np.full(200,.1), np.zeros(200)])
    tracker.set_trajectory(T)
    for _ in range(300):
        tracker.command(np.zeros(3))
    assert tracker.idx < 20
    assert not tracker.done()


def test_planner_does_not_use_simulator_dock_coordinates(monkeypatch):
    grid=OccGrid(); pose=np.array([4.,2.,-2.8]); cfg=Cfg()
    a=_plan(grid,pose,cfg)
    monkeypatch.setattr(C,'DOCK_POSE',np.array([100.,200.,0.]))
    monkeypatch.setattr(C,'LEADIN',np.array([-100.,200.]))
    monkeypatch.setattr(C,'MARKERS',np.array([[100.,200.,0.]]))
    b=_plan(grid,pose,cfg)
    assert np.array_equal(a,b)
    assert np.allclose(b[-1,:2], C.DOCK_TARGET[:2])



def test_detector_only_keeps_docking_id():
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, expected in [(0,0),(C.DOCK_MARKER_ID,1)]:
        image=np.full((300,300),255,np.uint8)
        image[50:250,50:250]=cv2.aruco.generateImageMarker(dictionary,marker_id,200)
        detections=detect_markers(image,make_detector())
        assert len(detections)==expected
        if expected:
            assert detections[0][1].shape==(4,2)

