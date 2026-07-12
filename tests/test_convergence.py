"""Validation of the convergence metrics on synthetic paths with known behaviour."""

from __future__ import annotations

import numpy as np

from traj_geom.metrics.convergence import (
    consecutive_step_cosine,
    convergence_rate,
    drift_to_loop_ratio,
)


def _damping_spiral(n: int = 60, decay: float = 0.9) -> np.ndarray:
    t = np.arange(n)
    r = decay**t
    ang = 0.5 * t
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)


def _circle(n: int = 60) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def test_damping_spiral_converges() -> None:
    """A spiral collapsing to the origin has a negative convergence rate."""
    assert convergence_rate(_damping_spiral()) < 0


def test_circle_metrics_reasonable() -> None:
    """A closed circle glides (positive step cosine) and barely drifts (DLR ~ 1)."""
    c = _circle()
    assert consecutive_step_cosine(c) > 0
    assert drift_to_loop_ratio(c) < 3.0
