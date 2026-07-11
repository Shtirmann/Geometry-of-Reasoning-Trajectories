"""Spearman and length-controlled partial-Spearman correlations.

OWNER: Data+Analysis
STATUS: implemented (from notebooks/01_mvp_h2.ipynb, spearman / partial_spearman).
TASK: rank correlation of a metric vs a task variable, plus a partial correlation
    that regresses out a confounder z (prompt length L) on ranks.
I/O: spearman(x, y) -> (rho, p) ; partial_spearman(x, y, z) -> (rho, p).

NOTE: partial correlation is done on ranks (rankdata) with a linear residualisation
    against z — scipy only, no pingouin needed.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation.

    Args:
        x: First variable, shape [N].
        y: Second variable, shape [N].

    Returns:
        ``(rho, p_value)``.
    """
    r = spearmanr(x, y)
    return float(r[0]), float(r[1])


def partial_spearman(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """Partial Spearman correlation of x and y, controlling for z.

    Ranks x, y, z; linearly residualises the x- and y-ranks against the z-rank;
    then correlates the residuals.

    Args:
        x: First variable, shape [N].
        y: Second variable, shape [N].
        z: Confounder to control for (e.g. prompt length L), shape [N].

    Returns:
        ``(rho, p_value)`` of the z-controlled correlation.
    """
    xr, yr, zr = rankdata(x), rankdata(y), rankdata(z)
    rx = xr - np.polyval(np.polyfit(zr, xr, 1), zr)
    ry = yr - np.polyval(np.polyfit(zr, yr, 1), zr)
    r = spearmanr(rx, ry)
    return float(r[0]), float(r[1])
