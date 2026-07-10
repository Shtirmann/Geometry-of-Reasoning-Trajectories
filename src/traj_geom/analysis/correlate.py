"""Correlate winding number w with reasoning depth d (the MVP H2 test).

OWNER: Data+Analysis
STATUS: stub — implement me.
TASK: Spearman correlation of winding vs depth, plus a partial correlation
    that controls for the prompt-length confounder L.
I/O: arrays (w, d[, length]) -> (rho, p_value).

NOTE: use pingouin for the partial correlation (control for length). H2 is
    supported if rho(w, d) > 0 with small p AND it survives the length control.
"""

from __future__ import annotations

import numpy as np


def spearman(w: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation between winding and depth.

    Args:
        w: Winding numbers, shape [N].
        d: Reasoning depths, shape [N].

    Returns:
        ``(rho, p_value)``.
    """
    raise NotImplementedError(
        "TODO(Data+Analysis): return scipy.stats.spearmanr(w, d) as (rho, p)."
    )


def partial_spearman(
    w: np.ndarray, d: np.ndarray, length: np.ndarray
) -> tuple[float, float]:
    """Spearman correlation of winding vs depth, controlling for prompt length.

    Args:
        w: Winding numbers, shape [N].
        d: Reasoning depths, shape [N].
        length: Prompt lengths L (confounder to partial out), shape [N].

    Returns:
        ``(rho, p_value)`` of the partial (length-controlled) correlation.
    """
    raise NotImplementedError(
        "TODO(Data+Analysis): pingouin.partial_corr(x=w, y=d, covar=length, "
        "method='spearman') -> (rho, p)."
    )
