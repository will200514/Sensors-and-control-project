"""Generate random starting poses and obstacles for the simulation."""

import numpy as np
from .. import config as C


def sample_scenario(rng, cfg):
    if not 0 <= cfg.n_obstacles <= 8:
        raise ValueError("n_obstacles must be between 0 and 8 (room slots).")
    start = np.array([
        rng.uniform(0.4, 1.2),
        rng.uniform(3.3, 4.6),
        rng.uniform(-np.pi, np.pi),
    ])

    lead_end = C.DOCK_POSE[:2]
    obstacles = []
    tries = 0

    while len(obstacles) < cfg.n_obstacles and tries < 500:
        tries += 1

        r = rng.uniform(0.12, 0.25)
        c = np.array([
            rng.uniform(0.8, 4.2),
            rng.uniform(0.5, 4.5),
        ])

        # Keep obstacles clear of the docking approach
        seg = lead_end - C.LEADIN
        t = np.clip(
            np.dot(c - C.LEADIN, seg) / np.dot(seg, seg),
            0,
            1,
        )
        d_lead = np.linalg.norm(C.LEADIN + t * seg - c)

        # Keep clear of the start and other obstacles
        ok = (
            np.linalg.norm(c - start[:2]) > r + 0.5
            and d_lead > r + 0.45
            and np.linalg.norm(c - C.LEADIN) > r + 0.6
            and all(
                np.linalg.norm(c - np.array(o[:2]))
                > r + o[2] + 0.55
                for o in obstacles
            )
        )

        if ok:
            obstacles.append((c[0], c[1], r))

    return start, obstacles
