"""Validation of the two-scale real-data metrics, band bootstrap, and depth loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traj_geom.analysis.band import bootstrap_ci
from traj_geom.data.loaders import pararule_depth_of
from traj_geom.metrics.two_scale import compute_two_scale_metrics, two_scale_allpos

BAND_CSV = Path(__file__).resolve().parent.parent / "results" / "band.csv"


def test_two_scale_allpos_equals_per_position() -> None:
    """Vectorized all-position metrics match the per-position (David's) computation."""
    rng = np.random.default_rng(0)
    traj = rng.standard_normal((12, 5, 8))  # [steps, seq_len, dim]
    out = two_scale_allpos(traj)
    for p in range(traj.shape[1]):
        ref = compute_two_scale_metrics(traj[:, p, :])
        assert out["mean_accel"][p] == pytest.approx(ref["mean_acceleration"], rel=1e-9, abs=1e-9)
        assert out["mean_orth"][p] == pytest.approx(ref["mean_orthogonality"], rel=1e-6, abs=1e-9)


@pytest.mark.skipif(not BAND_CSV.exists(), reason="results/band.csv not present")
def test_bootstrap_band_content_signal() -> None:
    """Length-matched content acceleration rises ~+1.30/depth with a CI clear of 0."""
    band = pd.read_csv(BAND_CSV)
    est, lo, hi, p = bootstrap_ci(band, "accel_content", nboot=5000)
    assert est == pytest.approx(1.30, abs=0.05)
    assert 1.0 <= lo <= 1.15
    assert 1.45 <= hi <= 1.6
    assert p < 1e-4


@pytest.mark.skipif(not BAND_CSV.exists(), reason="results/band.csv not present")
def test_bootstrap_band_answer_null() -> None:
    """Answer-token acceleration shows no depth effect: its CI includes 0."""
    band = pd.read_csv(BAND_CSV)
    _, lo, hi, _ = bootstrap_ci(band, "accel_answer", nboot=5000)
    assert lo < 0 < hi


def test_pararule_depth_of_qdep_and_id_fallback() -> None:
    """Depth comes from meta['QDep'], falling back to the '-D<d>-' tag in the id."""
    for d in (2, 3, 4, 5):
        assert pararule_depth_of({"meta": {"QDep": str(d)}, "id": "x"}) == d
        assert pararule_depth_of({"meta": {}, "id": f"NegationRule-Animal-D{d}-1"}) == d
    with pytest.raises(ValueError):
        pararule_depth_of({"meta": {}, "id": "no-depth-here"})
