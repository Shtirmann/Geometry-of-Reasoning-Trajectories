"""Loop-gate: classify a trajectory as settle / loop / drift.

OWNER: Shapes+Gate
STATUS: implemented (from notebooks/01_mvp_h2.ipynb, classify_shape).
TASK: classify the shape of a latent path so winding is only trusted on loops.
I/O: traj [T, hidden_dim] -> "settle" | "loop" | "drift".

LOGIC: settle if the LAST step is tiny (path converged); else loop if any far-apart
    (index gap > 3) pair of points comes back close relative to the path's extent;
    else drift.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from traj_geom.metrics.dynamics import step_norms


def classify_shape(
    traj: np.ndarray, settle_frac: float = 0.1, return_frac: float = 0.25
) -> str:
    """Classify the geometric shape of a trajectory.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        settle_frac: Last-step norm below this fraction of the max ⇒ "settle".
        return_frac: Min far-pair distance below this fraction of the max ⇒ "loop".

    Returns:
        ``"settle"``, ``"loop"``, or ``"drift"``.
    """
    s = step_norms(traj)
    if s[-1] < settle_frac * s.max():
        return "settle"
    d = squareform(pdist(traj))
    n = len(traj)
    mask = np.abs(np.subtract.outer(range(n), range(n))) > 3
    return "loop" if d[mask].min() / (d.max() + 1e-9) < return_frac else "drift"
