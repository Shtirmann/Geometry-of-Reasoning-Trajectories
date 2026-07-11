"""Contrast TEASER: winding of an easy vs a hard prompt across seeds.

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/00_smoke_extract.ipynb, cell-6).

WARNING: this is a TEASER, NOT a test of H2. It uses n=1 prompt per class (one
    easy, one hard) and reports winding mean +/- std over seeds. A real H2 test
    needs a depth-graded dataset, an init burn-in, and the loop-gate. In this
    teaser winding lives in the settle regime where the random-init noise
    dominates (std > mean), so the numbers are only a smoke signal.

Requires the `model` extra (torch + transformers) and a GPU.
"""

from __future__ import annotations

import numpy as np

from traj_geom.extraction.hook import extract_trajectory
from traj_geom.extraction.model import load_huginn
from traj_geom.metrics.projection import pca_to_2d
from traj_geom.metrics.winding import winding_number
from traj_geom.types import Trajectory

PROMPTS = {
    "easy": "Q: 2+3=? A:",
    "hard": (
        "Q: Alice has 3 apples, buys 4 more, gives 2 to Bob, then doubles "
        "what she has. How many? A:"
    ),
}
N_SEEDS = 5


def winding_of(traj: Trajectory) -> float:
    """PCA the trajectory to 2D, drop the init-dominated first state, then wind."""
    return winding_number(pca_to_2d(traj.states)[1:])


def main() -> None:
    """Load Huginn and print mean +/- std winding for each prompt class."""
    model, tok = load_huginn()
    for name, prompt in PROMPTS.items():
        ws = [
            winding_of(extract_trajectory(model, tok, prompt, seed=s))
            for s in range(N_SEEDS)
        ]
        print(f"{name:4s}  winding = {np.mean(ws):.3f} +/- {np.std(ws):.3f}")


if __name__ == "__main__":
    main()
