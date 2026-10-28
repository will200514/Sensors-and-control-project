"""Simulation-only frame transforms for scoring/plots. Never used for navigation."""
import numpy as np
from . import config as C
from .model.diffdrive import wrap


def relative_pose(pose, origin):
    c, s = np.cos(origin[2]), np.sin(origin[2])
    d = np.asarray(pose)[:2] - np.asarray(origin)[:2]
    return np.array([c*d[0] + s*d[1], -s*d[0] + c*d[1], wrap(pose[2]-origin[2])])


def compose_pose(origin, pose):
    c, s = np.cos(origin[2]), np.sin(origin[2])
    return np.array([origin[0]+c*pose[0]-s*pose[1], origin[1]+s*pose[0]+c*pose[1],
                     wrap(origin[2]+pose[2])])


def dock_origin():
    return C.DOCK_MARKER.copy()


def display_pose(pose, start, localised):
    world_pose = compose_pose(start, pose) if not localised else compose_pose(dock_origin(), pose)
    return relative_pose(world_pose, dock_origin())
