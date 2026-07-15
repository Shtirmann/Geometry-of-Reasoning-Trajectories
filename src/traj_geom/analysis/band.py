"""Within-band depth regression + stratified bootstrap CI (the band-test).

OWNER: Data+Analysis
STATUS: implemented (real-data two-scale fix).
TASK: isolate the depth effect from prompt length by regressing a metric on the
    within-band depth indicator ``hi`` while controlling for ``seq_len`` (and,
    pooled, for band identity), then bootstrap a CI by resampling within
    (band x hi) strata.
I/O: depth_coef(df, y, controls, group) -> float ;
    bootstrap_ci(df, y, nboot) -> (est, lo95, hi95, p).

The band design pairs adjacent depths (e.g. 2 vs 3) matched on length, so ``hi``
(0=low, 1=high depth of the pair) carries the depth contrast net of length.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def depth_coef(
    df: pd.DataFrame,
    y: str,
    controls: tuple[str, ...] = ("seq_len",),
    group: str | None = None,
) -> float:
    """OLS coefficient on the within-band depth indicator ``hi``.

    Fits ``y ~ 1 + hi + controls [+ group dummies]`` and returns the ``hi`` coef.

    Args:
        df: Band data with an ``hi`` column and the control/group columns.
        y: Metric column to regress (e.g. ``"accel_content"``).
        controls: Continuous controls to include (default: ``seq_len``).
        group: Optional categorical column (e.g. ``"adj"``) added as dummies
            (first level dropped) — use when pooling across bands.

    Returns:
        The estimated coefficient on ``hi``.
    """
    cols = [np.ones(len(df)), df["hi"].to_numpy(float)]
    for c in controls:
        cols.append(df[c].to_numpy(float))
    if group is not None:
        for lv in sorted(df[group].unique())[1:]:  # drop first level as reference
            cols.append((df[group] == lv).to_numpy(float))
    x = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(x, df[y].to_numpy(float), rcond=None)
    return float(beta[1])  # coefficient on hi


def bootstrap_ci(
    df: pd.DataFrame,
    y: str,
    nboot: int = 5000,
    controls: tuple[str, ...] = ("seq_len",),
    group: str = "adj",
    strata: tuple[str, ...] = ("adj", "hi"),
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Stratified bootstrap CI for the within-band depth coefficient.

    Resamples rows with replacement WITHIN each ``strata`` group (default the
    (band x hi) cells) so the design is preserved, then refits ``depth_coef``.

    Args:
        df: Band data.
        y: Metric column to regress.
        nboot: Number of bootstrap resamples.
        controls: Continuous controls passed to :func:`depth_coef`.
        group: Categorical column for band dummies in the pooled fit.
        strata: Columns whose cells define the resampling blocks.
        seed: RNG seed for reproducibility.

    Returns:
        ``(est, lo95, hi95, p)`` — point estimate, 95% percentile CI, and the
        two-sided bootstrap p-value against 0.
    """
    est = depth_coef(df, y, controls, group)
    rng = np.random.default_rng(seed)
    blocks = [g.index.to_numpy() for _, g in df.groupby(list(strata))]
    boots = np.empty(nboot)
    for b in range(nboot):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in blocks])
        boots[b] = depth_coef(df.loc[take], y, controls, group)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return est, float(lo), float(hi), float(p)
