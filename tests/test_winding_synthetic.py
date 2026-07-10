"""Winding number sanity checks on synthetic paths with known ground truth.

Skipped until Shapes+Gate implements synthetic.py and Extraction+Winding
implements winding.py. The body is real: unskip once both are done.
"""

from __future__ import annotations

import pytest

from traj_geom.metrics.winding import winding_number
from traj_geom.shapes import synthetic


@pytest.mark.skip(reason="stub — implement synthetic.circle and winding_number first")
def test_circle_winding_matches_turns() -> None:
    """One full circle has winding number ~ 1; two turns ~ 2."""
    one = winding_number(synthetic.circle(turns=1.0, n=400))
    two = winding_number(synthetic.circle(turns=2.0, n=400))
    assert one == pytest.approx(1.0, abs=0.05)
    assert two == pytest.approx(2.0, abs=0.05)


@pytest.mark.skip(reason="stub — implement synthetic.line and winding_number first")
def test_line_has_zero_winding() -> None:
    """A straight path does not wind."""
    w = winding_number(synthetic.line(n=400))
    assert abs(w) == pytest.approx(0.0, abs=0.05)


@pytest.mark.skip(reason="stub — implement synthetic.circle and winding_number first")
def test_winding_sign_flips_with_direction() -> None:
    """Reversing traversal flips the winding sign."""
    pts = synthetic.circle(turns=1.0, n=400)
    assert winding_number(pts) == pytest.approx(-winding_number(pts[::-1]), abs=0.05)


@pytest.mark.skip(reason="stub — implement synthetic paths and winding_number first")
def test_winding_invariant_to_point_count() -> None:
    """Winding of one circle is ~1 regardless of sampling density."""
    coarse = winding_number(synthetic.circle(turns=1.0, n=50))
    fine = winding_number(synthetic.circle(turns=1.0, n=1000))
    assert coarse == pytest.approx(fine, abs=0.05)
