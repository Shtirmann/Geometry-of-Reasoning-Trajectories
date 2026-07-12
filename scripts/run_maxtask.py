"""Experiment — running-maximum task: does the state-holding signature generalise?

Reproduces results/maxtask.csv. Unlike counting/switch, steps-to-settle does NOT
rise with length here (rho negative) — the effect is not universal. OWNER: Data+Analysis.

Run: uv run python -m scripts.run_maxtask
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model
from traj_geom.analysis.correlate import fmt_by_level, spearman
from traj_geom.metrics.dynamics import steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.synthetic import make_max_task

N_OPS = (4, 8, 16, 24, 32, 48)
N_SEEDS = 8


def compute() -> pd.DataFrame:
    """Extract a trajectory per max-task and score its geometry."""
    from traj_geom.extraction.hook import extract_trajectory

    model, tok = load_model()
    rows = []
    for n_ops in tqdm(N_OPS, desc="maxtask"):
        for s in range(N_SEEDS):
            t = make_max_task(n_ops, seed=s)
            tr = extract_trajectory(model, tok, t["prompt"], num_steps=64, seed=0)
            rows.append(
                {
                    "n_ops": n_ops,
                    "steps_settle": steps_to_settle(tr),
                    "winding": abs(winding_of(tr, 4)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Report steps-to-settle vs length for the running-maximum task."""
    mx = cached("maxtask.csv", compute)
    # Canonical: per-level Spearman + N.
    print("maxtask | steps~n_ops   [per-level]:", fmt_by_level(mx, "n_ops", "steps_settle"))
    print("maxtask | winding~n_ops [per-level]:", fmt_by_level(mx, "n_ops", "winding"))
    # Secondary (per-row):
    print("maxtask | steps~n_ops   [per-row]:", spearman(mx["n_ops"], mx["steps_settle"]))
    print("maxtask | winding~n_ops [per-row]:", spearman(mx["n_ops"], mx["winding"]))


if __name__ == "__main__":
    main()
