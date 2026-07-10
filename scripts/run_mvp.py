"""CLI: run the full MVP H2 pipeline end-to-end and report (rho, p).

OWNER: Data+Analysis
STATUS: stub — implement me.
TASK: Load saved trajectories -> loop-gate keep loops -> PCA to 2D -> winding
    number -> Spearman rho(w, d) + partial (length-controlled) -> scatter plot.
I/O: --trajectories dir (+ --config) -> printed (rho, p) and saved figure.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the MVP runner."""
    parser = argparse.ArgumentParser(description="Run the H2 winding-vs-depth MVP.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--trajectories", default="data/trajectories", help="Directory of saved trajectories."
    )
    parser.add_argument("--out", default="data/figures", help="Output directory for figures.")
    return parser


def main() -> None:
    """Entry point for the MVP runner."""
    _ = build_parser().parse_args()
    raise NotImplementedError(
        "TODO(Data+Analysis): load trajectories, classify_shape -> keep loops, "
        "pca_to_2d, winding_number, spearman/partial_spearman(w, d, length), "
        "scatter_winding_vs_depth; print (rho, p)."
    )


if __name__ == "__main__":
    main()
