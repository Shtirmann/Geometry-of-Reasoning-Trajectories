"""Loop-gate: classify a trajectory as settle / loop / drift.

OWNER: Shapes+Gate
STATUS: stub — implement me.
TASK: Classify the shape of a latent path so that winding number is only
    computed on genuine loops.
I/O: states [T, hidden_dim] -> one of "settle" | "loop" | "drift".

NOTE: "settle" if the final step is tiny (path converged); "loop" if the
    path returns near an earlier point (revisits a neighbourhood); else
    "drift". Winding number is only meaningful on "loop".
"""

from __future__ import annotations

import numpy as np


def classify_shape(states: np.ndarray) -> str:
    """Classify the geometric shape of a trajectory.

    Args:
        states: Trajectory array of shape [T, hidden_dim].

    Returns:
        ``"settle"``, ``"loop"``, or ``"drift"``.
    """
    raise NotImplementedError(
        "TODO(Shapes+Gate): return 'settle' if last-step displacement is tiny; "
        "'loop' if any later point comes back near an earlier one; else 'drift'."
    )
