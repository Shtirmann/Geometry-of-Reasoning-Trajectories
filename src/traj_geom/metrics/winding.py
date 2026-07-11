"""Winding number of a 2D path, and the trajectory-level winding helper.

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/01_mvp_h2.ipynb, winding_number / winding_of).
TASK: signed turns of a 2D path about a center, plus winding_of that PCA-projects
    a full trajectory to 2D (after a burn-in) and winds it.
I/O: winding_number(pts [T,2], center?) -> float ; winding_of(traj [T,H], burn) -> float.

NOTE: the PCA-projection sign is arbitrary, so experiments compare |winding|.
    h_0 is a random-init outlier -> drop the first `burn` states before winding.
"""

from __future__ import annotations

import numpy as np

from traj_geom.metrics.projection import pca_to_2d


def winding_number(points_2d: np.ndarray, center: np.ndarray | None = None) -> float:
    """Compute the signed winding number of a 2D path about a center.

    Args:
        points_2d: Ordered path of shape [T, 2].
        center: Point to wind about; defaults to the centroid of ``points_2d``.

    Returns:
        Signed number of full turns (e.g. ~+1.0 for one counter-clockwise loop).
    """
    pts = np.asarray(points_2d, dtype=float)
    c = pts.mean(0) if center is None else np.asarray(center, dtype=float)
    v = pts - c
    ang = np.arctan2(v[:, 1], v[:, 0])
    d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi  # wrap into (-pi, pi]
    return float(d.sum() / (2 * np.pi))


def winding_of(traj: np.ndarray, burn: int = 4) -> float:
    """PCA-project a trajectory to 2D (after a burn-in) and return its winding.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        burn: Number of initial states to drop (h_0 init arc is an outlier).

    Returns:
        The winding number of the burned-in, PCA-projected path.
    """
    return winding_number(pca_to_2d(traj[burn:]))
