"""Validation of the H1 persistence metric (skipped if ripser is not installed)."""

import numpy as np
import pytest

pytest.importorskip("ripser")

from traj_geom.metrics.homology import h1_persistence  # noqa: E402  (after importorskip)


def test_circle_has_h1() -> None:
    """A genuine circle has a strong H1 feature."""
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    _, mp = h1_persistence(np.c_[np.cos(t), np.sin(t)])
    assert mp > 0.3  # real loop -> noticeable persistence


def test_line_no_h1() -> None:
    """A straight line has no meaningful H1 feature."""
    _, mp = h1_persistence(np.c_[np.linspace(0, 1, 40), np.zeros(40)])
    assert mp < 0.1
