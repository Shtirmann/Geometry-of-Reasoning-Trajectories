"""Validation of the per-level Spearman helper."""

from __future__ import annotations

import pandas as pd
import pytest

from traj_geom.analysis.correlate import spearman_by_level


def _levels_df(levels: tuple[int, ...], slope: float) -> pd.DataFrame:
    """Several noisy rows per level whose group-means are strictly monotone in x."""
    rows = [{"x": lvl, "y": slope * lvl + s} for lvl in levels for s in range(4)]
    return pd.DataFrame(rows)


def test_spearman_by_level_monotone_increasing() -> None:
    """Strictly increasing group-means give rho=+1.0, the right N and crit."""
    df = _levels_df((2, 4, 6, 8, 10, 12), slope=2.0)
    rho, n, crit = spearman_by_level(df, "x", "y")
    assert rho == pytest.approx(1.0)
    assert n == 6
    assert crit == 0.886


def test_spearman_by_level_monotone_decreasing() -> None:
    """Strictly decreasing group-means give rho=-1.0 with N counted correctly."""
    df = _levels_df((1, 2, 3, 4, 5), slope=-3.0)
    rho, n, crit = spearman_by_level(df, "x", "y")
    assert rho == pytest.approx(-1.0)
    assert n == 5
    assert crit == 1.000
