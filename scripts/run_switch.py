"""Experiment F — parity switch task: generalisation beyond counting (H3).

Reproduces results/switch.csv. A light-on/off parity task also needs state-holding;
steps-to-settle rises with n_ops just as for counting, showing the effect is not
specific to arithmetic. OWNER: Data+Analysis.

Run: python -m scripts.run_switch
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model
from traj_geom.analysis.correlate import spearman
from traj_geom.metrics.dynamics import steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.synthetic import make_switch_task

N_OPS = (4, 8, 16, 24, 32, 48)
N_SEEDS = 10


def compute() -> pd.DataFrame:
    """Extract a trajectory per switch task and score its geometry."""
    from traj_geom.extraction.hook import extract_trajectory

    model, tok = load_model()
    rows = []
    for n_ops in tqdm(N_OPS, desc="switch"):
        for s in range(N_SEEDS):
            t = make_switch_task(n_ops, seed=s)
            tr = extract_trajectory(model, tok, t["prompt"], num_steps=64, seed=0)
            rows.append(
                {
                    "n_ops": n_ops,
                    "winding": abs(winding_of(tr, burn=4)),
                    "steps_settle": steps_to_settle(tr),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Report winding/steps vs parity length."""
    sw = cached("switch.csv", compute)
    print("switch | winding~n_ops:", spearman(sw["n_ops"], sw["winding"]))
    print("switch | steps~n_ops:", spearman(sw["n_ops"], sw["steps_settle"]))


if __name__ == "__main__":
    main()
