"""Contrast TEASER: winding of an easy vs a hard prompt across seeds.

OWNER: Extraction+Winding
STATUS: implemented (superseded by run_counting / run_dissociation).

WARNING: this is a TEASER, NOT a test of H2. It uses n=1 prompt per class (one
    easy, one hard) and reports |winding| mean +/- std over seeds. Both prompts
    settle, so the numbers are only a smoke signal — see the real experiments.

Requires the `model` extra (torch + transformers) and a GPU.
Run: python -m scripts.run_contrast
"""

from __future__ import annotations

import numpy as np

from traj_geom.metrics.winding import winding_of

PROMPTS = {
    "easy": "Q: 2+3=? A:",
    "hard": (
        "Q: Alice has 3 apples, buys 4 more, gives 2 to Bob, then doubles "
        "what she has. How many? A:"
    ),
}
N_SEEDS = 5


def main() -> None:
    """Load Huginn and print mean +/- std |winding| for each prompt class."""
    from traj_geom.extraction.hook import extract_trajectory
    from traj_geom.extraction.model import load_huginn

    model, tok = load_huginn()
    for name, prompt in PROMPTS.items():
        ws = [
            abs(winding_of(extract_trajectory(model, tok, prompt, seed=s)))
            for s in range(N_SEEDS)
        ]
        print(f"{name:4s}  |winding| = {np.mean(ws):.3f} +/- {np.std(ws):.3f}")


if __name__ == "__main__":
    main()
