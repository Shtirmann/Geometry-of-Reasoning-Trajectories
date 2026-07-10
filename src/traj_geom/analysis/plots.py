"""Plots for the MVP: winding-vs-depth scatter and 2D PCA trajectory.

OWNER: Data+Analysis
STATUS: stub — implement me.
TASK: Produce the two figures the "done" criterion needs — a scatter of
    winding number against depth, and a 2D PCA path plot for inspection.
I/O: arrays / trajectory -> matplotlib Figure (saved or returned).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def scatter_winding_vs_depth(
    w: np.ndarray, d: np.ndarray, save_path: str | None = None
) -> Any:
    """Scatter winding number against reasoning depth.

    Args:
        w: Winding numbers, shape [N].
        d: Reasoning depths, shape [N].
        save_path: If given, write the figure there.

    Returns:
        The matplotlib Figure.
    """
    raise NotImplementedError(
        "TODO(Data+Analysis): scatter d (x) vs w (y), label axes, optional save."
    )


def plot_pca_trajectory(points_2d: np.ndarray, save_path: str | None = None) -> Any:
    """Plot a single 2D PCA-projected trajectory path.

    Args:
        points_2d: Projected path of shape [T, 2].
        save_path: If given, write the figure there.

    Returns:
        The matplotlib Figure.
    """
    raise NotImplementedError(
        "TODO(Data+Analysis): line-plot the ordered [T, 2] path, mark start/end, "
        "optional save."
    )
