"""Objective convergence metrics for a latent trajectory (settle vs loop/drift).

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/01_mvp.ipynb).
TASK: model-free, sign-robust descriptors of how a path approaches (or fails to
    approach) a fixed point — the diagnostics behind the "everything settles" result.
I/O: traj [T, dim] -> float; path_independence takes two trajectories.

`step_norms` is reused from metrics.dynamics (single source of truth).
"""

from __future__ import annotations

import numpy as np

from traj_geom.metrics.dynamics import step_norms


def convergence_rate(traj: np.ndarray) -> float:
    """Log-slope of ``||h_t - h_final||`` over t. <0 = exponential pull to a fixed point.

    Args:
        traj: Trajectory of shape [T, dim].

    Returns:
        The fitted log-slope, or NaN if too few points remain above the floor.
    """
    hf = traj[-1]
    d = np.linalg.norm(traj[:-1] - hf, axis=1)
    d = d[d > 1e-9]
    return float(np.polyfit(np.arange(len(d)), np.log(d), 1)[0]) if len(d) > 3 else np.nan


def consecutive_step_cosine(traj: np.ndarray) -> float:
    """Mean cosine of adjacent steps (2nd half). <0 = oscillatory convergence.

    Follows Pappone et al. / Movahedi et al.: a negative mean cosine means the path
    zig-zags inward rather than gliding.

    Args:
        traj: Trajectory of shape [T, dim].

    Returns:
        Mean adjacent-step cosine over the second half, or NaN if too short.
    """
    dd = np.diff(traj, axis=0)
    c = [
        float(np.dot(dd[i], dd[i + 1]) / (np.linalg.norm(dd[i]) * np.linalg.norm(dd[i + 1]) + 1e-9))
        for i in range(len(dd) - 1)
    ]
    return float(np.mean(c[len(c) // 2:])) if len(c) > 2 else np.nan


def drift_to_loop_ratio(traj: np.ndarray) -> float:
    """``||h_final - h_0|| / mean step``. >>1 = a transient arc to a far point, not a loop.

    Pappone et al.'s drift-to-loop ratio (DLR).

    Args:
        traj: Trajectory of shape [T, dim].

    Returns:
        The DLR (dimensionless).
    """
    return float(np.linalg.norm(traj[-1] - traj[0]) / (step_norms(traj).mean() + 1e-9))


def path_independence(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    """Log-slope of the gap between two trajectories from different inits.

    <0 = the two paths converge to a common point (path-independence, Geiping et al.).

    Args:
        traj_a: First trajectory of shape [T, dim].
        traj_b: Second trajectory of shape [T, dim].

    Returns:
        The fitted log-slope of ``||a_t - b_t||``, or NaN if too few points remain.
    """
    d = np.linalg.norm(traj_a - traj_b, axis=1)
    d = d[d > 1e-9]
    return float(np.polyfit(np.arange(len(d)), np.log(d), 1)[0]) if len(d) > 3 else np.nan
