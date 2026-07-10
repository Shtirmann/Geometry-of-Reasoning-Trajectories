"""Winding number of a 2D path — how many turns it makes about a center.

OWNER: Extraction+Winding
STATUS: stub — implement me.
TASK: Sum the signed, pi-wrapped angle increments of the path about a center
    (default = centroid of the points) and divide by 2*pi to get turns.
I/O: points_2d [T, 2] (+ optional center) -> float (signed number of turns).

NOTE: wrap each angle delta into (-pi, pi] before summing so a full loop
    contributes +/-1.0; sign encodes direction. Only meaningful on "loop"
    shapes (see shapes/gate.py).
"""

from __future__ import annotations

import numpy as np


def winding_number(points_2d: np.ndarray, center: np.ndarray | None = None) -> float:
    """Compute the signed winding number of a 2D path about a center.

    Args:
        points_2d: Ordered path of shape [T, 2].
        center: Point to wind about; defaults to the centroid of ``points_2d``.

    Returns:
        Signed number of full turns (e.g. ~+1.0 for one counter-clockwise loop).
    """
    raise NotImplementedError(
        "TODO(Extraction+Winding): center points (default centroid), take angles "
        "atan2(dy, dx), sum pi-wrapped consecutive diffs, divide by 2*pi."
    )
