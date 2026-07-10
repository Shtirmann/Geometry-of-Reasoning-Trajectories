"""Loop-gate classification checks on synthetic paths with known ground truth.

Skipped until Shapes+Gate implements gate.py and synthetic.py. The body is
real: unskip once both are done.
"""

from __future__ import annotations

import pytest

from traj_geom.shapes import synthetic
from traj_geom.shapes.gate import classify_shape


@pytest.mark.skip(reason="stub — implement synthetic.circle and classify_shape first")
def test_circle_is_loop() -> None:
    """A closed circle is classified as a loop."""
    assert classify_shape(synthetic.circle(turns=1.0, n=400)) == "loop"


@pytest.mark.skip(reason="stub — implement synthetic.spiral_in and classify_shape first")
def test_spiral_in_settles() -> None:
    """An inward spiral converges and is classified as settle."""
    assert classify_shape(synthetic.spiral_in(n=400)) == "settle"


@pytest.mark.skip(reason="stub — implement synthetic.line and classify_shape first")
def test_line_drifts() -> None:
    """A straight path neither settles nor loops -> drift."""
    assert classify_shape(synthetic.line(n=400)) == "drift"


@pytest.mark.skip(reason="stub — implement synthetic.spiral_out and classify_shape first")
def test_spiral_out_drifts() -> None:
    """An outward spiral diverges -> drift."""
    assert classify_shape(synthetic.spiral_out(n=400)) == "drift"
