"""CLI: extract latent trajectories from Huginn and save them to disk.

OWNER: Extraction+Winding
STATUS: stub — implement me.
TASK: Wire config -> load_huginn -> load_depth_dataset -> extract_trajectory
    per (example, seed) -> Trajectory.save. This is glue over the stubs.
I/O: --config YAML + --out dir -> written .npz trajectories.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the extraction CLI."""
    parser = argparse.ArgumentParser(description="Extract Huginn latent trajectories.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--out", default="data/trajectories", help="Output directory.")
    return parser


def main() -> None:
    """Entry point for the extraction CLI."""
    _ = build_parser().parse_args()
    raise NotImplementedError(
        "TODO(Extraction+Winding): load config, model and dataset, loop over "
        "(example, seed), extract_trajectory, and save each Trajectory to --out."
    )


if __name__ == "__main__":
    main()
