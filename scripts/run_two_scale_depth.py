"""Two-scale MAIN result — length-matched adjacent-depth band on real Huginn.

Reproduces results/band.csv: adjacent depth pairs (2-3, 3-4, 4-5) matched on prompt
length, so the within-band indicator ``hi`` (0=low, 1=high depth) carries the depth
contrast net of length. Reports the depth coefficient on CONTENT-token acceleration
(the real signal) and ANSWER-token acceleration (null), with a stratified bootstrap CI.

Cached: if results/band.csv exists (shipped real run), only the analysis is printed —
no GPU needed. OWNER: Data+Analysis.

Run: uv run python -m scripts.run_two_scale_depth
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model
from traj_geom.analysis.band import bootstrap_ci, depth_coef

BANDS = ((2, 3), (3, 4), (4, 5))
N_LOAD = 200          # examples loaded per depth before length-matching
N_PER_CELL = 60       # matched examples kept per (band, depth)
NUM_STEPS = 32


def _length_match(low: list[dict], high: list[dict], n: int) -> list[tuple[int, dict]]:
    """Pair low/high-depth examples with identical seq_len; return up to n per side."""
    by_len: dict[int, list[dict]] = {}
    for r in high:
        by_len.setdefault(r["seq_len"], []).append(r)
    out: list[tuple[int, dict]] = []
    for r in low:
        bucket = by_len.get(r["seq_len"])
        if bucket:
            out.append((0, r))
            out.append((1, bucket.pop()))
        if len(out) >= 2 * n:
            break
    return out


def compute() -> pd.DataFrame:
    """Real pipeline: extract all-position trajectories over length-matched bands."""
    from traj_geom.data.loaders import load_pararule_real
    from traj_geom.extraction.hook import extract_trajectory_allpos
    from traj_geom.metrics.two_scale import compute_two_scale_metrics, two_scale_allpos

    model, tok = load_model()
    rows = []
    for low_d, high_d in tqdm(BANDS, desc="bands"):
        adj = f"{low_d}-{high_d}"
        pools = {d: list(load_pararule_real(d, n=N_LOAD)) for d in (low_d, high_d)}
        for d in (low_d, high_d):
            for r in pools[d]:
                r["seq_len"] = int(tok(r["prompt"], return_tensors="pt").input_ids.shape[1])
        for hi, row in _length_match(pools[low_d], pools[high_d], n=N_PER_CELL):
            traj, seq_len = extract_trajectory_allpos(
                model, tok, row["prompt"], num_steps=NUM_STEPS, seed=0
            )
            m = two_scale_allpos(traj)
            ans = compute_two_scale_metrics(traj[:, -1, :])  # answer-token exit/settle
            rows.append(
                {
                    "adj": adj,
                    "hi": hi,
                    "seed": 0,
                    "seq_len": seq_len,
                    "accel_answer": float(m["mean_accel"][-1]),
                    "orth_answer": float(m["mean_orth"][-1]),
                    "accel_content": float(m["mean_accel"][:-1].mean()),
                    "orth_content": float(m["mean_orth"][:-1].mean()),
                    "exit_answer": ans["exit_step_by_acceleration"],
                    "settle_answer": ans["exit_step_by_norm"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Report the within-band depth coefficient + bootstrap CI for content vs answer."""
    band = cached("band.csv", compute)
    for y in ("accel_content", "accel_answer"):
        est, lo, hi, p = bootstrap_ci(band, y, nboot=5000)
        tag = "CONTENT" if y == "accel_content" else "ANSWER "
        print(
            f"{tag} depth-coef|len (pooled, +band): "
            f"{est:+.3f}  95%CI[{lo:+.3f}, {hi:+.3f}]  p={p:.2e}"
        )
        per_band = {
            b: round(depth_coef(band[band.adj == b], y, ("seq_len",), None), 3)
            for b in sorted(band.adj.unique())
        }
        print(f"         per-band coef|len: {per_band}")


if __name__ == "__main__":
    main()
