"""winding_number sanity checks on synthetic paths with known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.metrics.winding import winding_number


def _circle(n: int = 400) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def test_winding_number_circle_is_one() -> None:
    """One full counter-clockwise circle winds ~ +1."""
    assert winding_number(_circle()) == pytest.approx(1.0, abs=0.02)


def test_winding_number_reversed_circle_is_minus_one() -> None:
    """Reversing traversal flips the winding sign to ~ -1."""
    assert winding_number(_circle()[::-1]) == pytest.approx(-1.0, abs=0.02)


def test_winding_number_two_turns() -> None:
    """A path wrapping twice winds ~ +2."""
    t = np.linspace(0.0, 4.0 * np.pi, 800, endpoint=False)
    two = np.stack([np.cos(t), np.sin(t)], axis=1)
    assert winding_number(two) == pytest.approx(2.0, abs=0.02)
