"""Two-scale on real Huginn — counting track/local, length-matched.

Reproduces results/counting_two_scale.csv (+ counting_map.npy). Same track/local
design as the H3 dissociation, but scored with two-scale acceleration. Result: content
acceleration does NOT track count length for either variant (null). Cached: analysis-only
without GPU. OWNER: Data+Analysis.

Run: uv run python -m scripts.run_two_scale_counting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts._common import RESULTS_DIR, cached, load_model
from traj_geom.analysis.correlate import fmt_by_level

N_OPS = (4, 8, 16, 24, 32, 48)
N_SEEDS = 3
NUM_STEPS = 32
N_BINS = 10
MAP_PATH = RESULTS_DIR / "counting_map.npy"


def _posmap(mean_accel: np.ndarray, nbins: int = N_BINS) -> np.ndarray:
    idx = np.linspace(0, len(mean_accel), nbins + 1).astype(int)
    return np.array([mean_accel[idx[i]:idx[i + 1]].mean() for i in range(nbins)])


def compute() -> pd.DataFrame:
    """Real pipeline: all-position two-scale metrics for counting track/local."""
    from traj_geom.extraction.hook import extract_trajectory_allpos
    from traj_geom.metrics.two_scale import compute_two_scale_metrics, two_scale_allpos
    from traj_geom.shapes.synthetic import make_variants

    model, tok = load_model()
    rows, maps = [], []
    for n_ops in tqdm(N_OPS, desc="n_ops"):
        for seed in range(N_SEEDS):
            v = make_variants(n_ops, seed=seed)
            for kind in ("track", "local"):
                traj, seq_len = extract_trajectory_allpos(
                    model, tok, v[kind], num_steps=NUM_STEPS, seed=0
                )
                m = two_scale_allpos(traj)
                ans = compute_two_scale_metrics(traj[:, -1, :])
                rows.append(
                    {
                        "n_ops": n_ops,
                        "kind": kind,
                        "seed": seed,
                        "seq_len": seq_len,
                        "accel_answer": float(m["mean_accel"][-1]),
                        "orth_answer": float(m["mean_orth"][-1]),
                        "accel_content": float(m["mean_accel"][:-1].mean()),
                        "orth_content": float(m["mean_orth"][:-1].mean()),
                        "exit_answer": ans["exit_step_by_acceleration"],
                        "settle_answer": ans["exit_step_by_norm"],
                    }
                )
                maps.append(_posmap(m["mean_accel"]))
    np.save(MAP_PATH, np.array(maps))
    return pd.DataFrame(rows)


def main() -> None:
    """Report per-level content-acceleration vs count length for track and local."""
    df = cached("counting_two_scale.csv", compute)
    for kind in ("track", "local"):
        s = df[df.kind == kind]
        print(f"{kind} content~n_ops [per-level]:", fmt_by_level(s, "n_ops", "accel_content"))
        print(f"{kind} answer ~n_ops [per-level]:", fmt_by_level(s, "n_ops", "accel_answer"))


if __name__ == "__main__":
    main()
