"""Two-scale on real Huginn — PARARULE depth 2-5, all positions.

Reproduces results/pararule_depth.csv (+ pararule_depth_map.npy). Reports the
per-level Spearman of content/answer acceleration vs depth (+N), the length-controlled
partial correlation, and the mean per-position acceleration map. On the full depth range
depth and seq_len are collinear, so the partial is unreliable (see run_two_scale_depth
for the length-matched band test). Cached: analysis-only without GPU. OWNER: Data+Analysis.

Run: uv run python -m scripts.run_two_scale_real
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts._common import RESULTS_DIR, cached, load_model
from traj_geom.analysis.correlate import fmt_by_level, partial_spearman

DEPTHS = (2, 3, 4, 5)
N_PER_DEPTH = 30
NUM_STEPS = 32
N_BINS = 10
MAP_PATH = RESULTS_DIR / "pararule_depth_map.npy"


def _posmap(mean_accel: np.ndarray, nbins: int = N_BINS) -> np.ndarray:
    """Bin a per-position acceleration profile into ``nbins`` relative-position bins."""
    idx = np.linspace(0, len(mean_accel), nbins + 1).astype(int)
    return np.array([mean_accel[idx[i]:idx[i + 1]].mean() for i in range(nbins)])


def compute() -> pd.DataFrame:
    """Real pipeline: all-position two-scale metrics for PARARULE depths 2-5."""
    from traj_geom.data.loaders import load_pararule_real
    from traj_geom.extraction.hook import extract_trajectory_allpos
    from traj_geom.metrics.two_scale import compute_two_scale_metrics, two_scale_allpos

    model, tok = load_model()
    rows, maps = [], []
    for depth in tqdm(DEPTHS, desc="depth"):
        for seed, row in enumerate(load_pararule_real(depth, n=N_PER_DEPTH)):
            traj, seq_len = extract_trajectory_allpos(
                model, tok, row["prompt"], num_steps=NUM_STEPS, seed=0
            )
            m = two_scale_allpos(traj)
            ans = compute_two_scale_metrics(traj[:, -1, :])
            rows.append(
                {
                    "depth": depth,
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
    """Report per-level depth correlations, the length-partial, and the position map."""
    df = cached("pararule_depth.csv", compute)
    print("accel_content by depth:", df.groupby("depth")["accel_content"].mean().round(2).tolist())
    print("content~depth [per-level]:", fmt_by_level(df, "depth", "accel_content"))
    print("answer ~depth [per-level]:", fmt_by_level(df, "depth", "accel_answer"))
    print(
        "content~depth [per-row, partial|seq_len]:",
        partial_spearman(df["accel_content"], df["depth"], df["seq_len"]),
    )
    if MAP_PATH.exists():
        m = np.load(MAP_PATH)
        print(f"position map {m.shape}: mean accel per bin = {m.mean(0).round(2).tolist()}")


if __name__ == "__main__":
    main()
