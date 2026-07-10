"""Project a high-dimensional trajectory into 2D for winding analysis.

OWNER: Extraction+Winding
STATUS: stub — implement me.
TASK: Reduce states [T, hidden_dim] to [T, 2] with sklearn PCA (2 components),
    preserving step order so the 2D path can be wound.
I/O: states [T, hidden_dim] -> points_2d [T, 2].
"""

from __future__ import annotations

import numpy as np


def pca_to_2d(states: np.ndarray) -> np.ndarray:
    """Project a trajectory to its first two principal components.

    Args:
        states: Trajectory array of shape [T, hidden_dim].

    Returns:
        The projected path of shape [T, 2], in original step order.
    """
    raise NotImplementedError(
        "TODO(Extraction+Winding): fit sklearn.decomposition.PCA(n_components=2) "
        "on states and return the [T, 2] transform."
    )
