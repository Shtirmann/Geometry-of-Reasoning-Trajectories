"""classify_shape checks on synthetic trajectories with known ground truth.

These exercise the real notebook gate on purpose-built paths:
  * settle — steps decay to ~0 (last step tiny),
  * loop   — a closed circle returns near its start,
  * drift  — a short straight path never returns and never settles.
"""

from __future__ import annotations

import numpy as np

from traj_geom.shapes.gate import classify_shape


def test_classify_settle() -> None:
    """A path whose step size decays to ~0 is 'settle'."""
    mags = 0.7 ** np.arange(20)  # geometrically shrinking steps
    traj = np.stack([np.cumsum(mags), np.zeros(20)], axis=1)
    assert classify_shape(traj) == "settle"


def test_classify_loop() -> None:
    """A closed circle (constant step, returns to start) is 'loop'."""
    t = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    assert classify_shape(circle) == "loop"


def test_classify_drift() -> None:
    """A short straight path (never returns, never settles) is 'drift'."""
    xs = np.linspace(0.0, 1.0, 8)
    line = np.stack([xs, np.zeros(8)], axis=1)
    assert classify_shape(line) == "drift"
