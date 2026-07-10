"""Synthetic 2D paths with known ground-truth labels for testing gate & winding.

OWNER: Shapes+Gate
STATUS: stub — implement me.
TASK: Generate canonical 2D paths whose shape/winding is known analytically,
    so gate.py and winding.py can be validated WITHOUT the model.
I/O: shape params -> points_2d [T, 2].

Ground-truth labels:
    circle    -> "loop",   winding number == `turns`
    spiral_in -> "settle", winding > 0 but path converges inward
    spiral_out-> "drift",  winding > 0 but path diverges outward
    line      -> "drift",  winding number ~ 0
"""

from __future__ import annotations

import numpy as np


def circle(turns: float = 1.0, n: int = 200, radius: float = 1.0, noise: float = 0.0) -> np.ndarray:
    """Closed circular path. Ground-truth: shape="loop", winding=`turns`.

    Args:
        turns: Number of full revolutions.
        n: Number of points.
        radius: Circle radius.
        noise: Std of optional Gaussian jitter.

    Returns:
        Path of shape [n, 2].
    """
    raise NotImplementedError(
        "TODO(Shapes+Gate): theta in [0, 2*pi*turns], (radius*cos, radius*sin) + noise."
    )


def spiral_in(n: int = 200, noise: float = 0.0) -> np.ndarray:
    """Inward spiral. Ground-truth: shape="settle" (converges to center).

    Args:
        n: Number of points.
        noise: Std of optional Gaussian jitter.

    Returns:
        Path of shape [n, 2].
    """
    raise NotImplementedError(
        "TODO(Shapes+Gate): radius decreasing to ~0 while angle advances; add noise."
    )


def spiral_out(n: int = 200, noise: float = 0.0) -> np.ndarray:
    """Outward spiral. Ground-truth: shape="drift" (diverges outward).

    Args:
        n: Number of points.
        noise: Std of optional Gaussian jitter.

    Returns:
        Path of shape [n, 2].
    """
    raise NotImplementedError(
        "TODO(Shapes+Gate): radius increasing from ~0 while angle advances; add noise."
    )


def line(n: int = 200, noise: float = 0.0) -> np.ndarray:
    """Straight path. Ground-truth: shape="drift", winding ~ 0.

    Args:
        n: Number of points.
        noise: Std of optional Gaussian jitter.

    Returns:
        Path of shape [n, 2].
    """
    raise NotImplementedError(
        "TODO(Shapes+Gate): points along a straight segment + noise."
    )
