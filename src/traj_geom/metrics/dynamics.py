"""Step-size dynamics of a trajectory: per-step displacement and settling time.

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/01_mvp_h2.ipynb, step_norms / steps_to_settle).
TASK: measure how fast the recurrent path converges — the "effective compute"
    signal that turned out to carry the MVP result.
I/O: step_norms(traj [T,H]) -> [T-1] ; steps_to_settle(traj, frac) -> int.
"""

from __future__ import annotations

import numpy as np


def step_norms(traj: np.ndarray) -> np.ndarray:
    """Return the L2 norm of each recurrent step ``||h_{t+1} - h_t||``.

    Args:
        traj: Trajectory of shape [T, hidden_dim].

    Returns:
        Array of shape [T-1] with the per-step displacement norms.
    """
    return np.linalg.norm(np.diff(traj, axis=0), axis=1)


def steps_to_settle(traj: np.ndarray, frac: float = 0.1) -> int:
    """First step at which the displacement drops below ``frac`` of its max.

    This is our proxy for effective compute: how many recurrent steps the model
    spends before the path stops moving.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        frac: Fraction of the max step norm below which the path is "settled".

    Returns:
        Index of the first settled step, or the number of steps if it never settles.
    """
    s = step_norms(traj)
    below = np.where(s < frac * s.max())[0]
    return int(below[0]) if len(below) else len(s)


def contraction_rate(traj: np.ndarray) -> float:
    """Mean log-change of step size along the path. <0 = the path is contracting.

    Args:
        traj: Trajectory of shape [T, hidden_dim].

    Returns:
        Mean of ``diff(log(step_norms))``, or NaN if fewer than 3 steps survive the floor.
    """
    s = step_norms(traj)
    s = s[s > 1e-9]
    return float(np.mean(np.diff(np.log(s)))) if len(s) >= 3 else np.nan
