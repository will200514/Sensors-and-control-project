"""Generate random starting poses and obstacles for the simulation."""

import numpy as np
from .. import config as C

# allows seeds to reporoduce a layout
def sample_scenario(rng, cfg):
    if not 0 <= cfg.n_obstacles <= 8:
        raise ValueError("n_obstacles must be between 0 and 8 (room slots).")
    start = np.array([
        rng.uniform(0.35, 0.8),
        rng.uniform(2.05, 2.6),
        -np.pi/2,
    ])

    lead_end = C.DOCK_POSE[:2]
    obstacles = []
    tries = 0

    while len(obstacles) < cfg.n_obstacles and tries < 500:
        tries += 1

        r = rng.uniform(0.12, 0.18)
        c = np.array([
            rng.uniform(0.4, 2.6),
            rng.uniform(0.2, 2.6),
        ])

        # Keep obstacles clear of the docking approach
        seg = lead_end - C.LEADIN
        t = np.clip(
            np.dot(c - C.LEADIN, seg) / np.dot(seg, seg),
            0,
            1,
        )
        d_lead = np.linalg.norm(C.LEADIN + t * seg - c)

        # Simulation layout constraint: leave the initial docking marker visible.
        # This geometry is never passed to localisation or navigation.
        sight = C.DOCK_MARKER[:2] - start[:2]
        ts = np.clip(np.dot(c-start[:2], sight) / np.dot(sight, sight), 0., 1.)
        d_sight = np.linalg.norm(start[:2] + ts*sight - c)

        # Keep clear of the start and other obstacles
        ok = (
            np.linalg.norm(c - start[:2]) > r + 0.35
            and d_sight > r + 0.18
            and d_lead > r + 0.28
            and np.linalg.norm(c - C.LEADIN) > r + 0.35
            and all(
                np.linalg.norm(c - np.array(o[:2])) > r + o[2] + 0.3
                for o in obstacles
            )
        )

        if ok:
            obstacles.append((c[0], c[1], r))

    return start, obstacles