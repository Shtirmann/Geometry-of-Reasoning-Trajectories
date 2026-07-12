"""Persistent homology H1 of a trajectory (Vietoris-Rips) — the curator's metric.

OWNER: Extraction+Winding
STATUS: implemented (protocol metric; run via scripts/run_homology.py).
TASK: detect a genuine topological loop (an H1 hole) in a latent path and score
    its strength, normalised by the point-cloud diameter so it is scale-free.
I/O: points [T, dim] -> (H1 diagram, max persistence / diameter).

NOTE: requires the `tda` extra (ripser). The normalisation by cloud diameter makes
    the score comparable across trajectories of different spatial extent.
"""

from __future__ import annotations

import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist


def h1_persistence(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Vietoris-Rips H1 barcodes and the diameter-normalised max persistence.

    Args:
        points: Point cloud / trajectory of shape [T, dim].

    Returns:
        ``(dgm1, max_persist_norm)`` — the H1 persistence diagram and the maximum
        bar length divided by the cloud diameter (0.0 if there are no H1 features).
    """
    pts = np.asarray(points, dtype=float)
    dgm1 = ripser(pts, maxdim=1)["dgms"][1]
    diam = pdist(pts).max() if len(pts) > 1 else 1.0
    if len(dgm1) == 0:
        return dgm1, 0.0
    return dgm1, float((dgm1[:, 1] - dgm1[:, 0]).max() / (diam + 1e-9))
