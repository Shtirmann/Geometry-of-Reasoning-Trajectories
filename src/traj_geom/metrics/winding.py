"""Winding number of a 2D path — how many turns it makes about a center.

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/00_smoke_extract.ipynb, cell-5).
TASK: Sum the signed, pi-wrapped angle increments of the path about a center
    (default = centroid) and divide by 2*pi to get the number of turns.
I/O: points_2d [T, 2] (+ optional center) -> float (signed number of turns).

NOTE: only meaningful on "loop" shapes (see shapes/gate.py). h_0 is a random
    init outlier — analyse winding on the path with the first state dropped.
"""

from __future__ import annotations

import numpy as np


def winding_number(points_2d: np.ndarray, center: np.ndarray | None = None) -> float:
    """Compute the signed winding number of a 2D path about a center.

    Args:
        points_2d: Ordered path of shape [T, 2].
        center: Point to wind about; defaults to the centroid of ``points_2d``.

    Returns:
        Signed number of full turns (e.g. ~+1.0 for one counter-clockwise loop).
    """
    pts = np.asarray(points_2d, dtype=float)
    c = pts.mean(axis=0) if center is None else np.asarray(center, dtype=float)
    v = pts - c
    ang = np.arctan2(v[:, 1], v[:, 0])
    d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi  # wrap into (-pi, pi]
    return float(d.sum() / (2 * np.pi))
