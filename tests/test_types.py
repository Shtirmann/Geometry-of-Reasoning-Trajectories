"""Round-trip test for the shared Trajectory contract (NOT skipped)."""

from __future__ import annotations

import numpy as np

from traj_geom.types import Trajectory


def test_trajectory_save_load_roundtrip(tmp_path) -> None:
    """A saved Trajectory reloads to an identical object."""
    rng = np.random.default_rng(0)
    states = rng.standard_normal((33, 16)).astype(np.float32)
    traj = Trajectory(
        states=states,
        token_index=-1,
        seed=7,
        depth=3,
        seq_len=42,
        prompt_id="prompt-abc",
    )

    path = tmp_path / "traj.npz"
    traj.save(str(path))
    loaded = Trajectory.load(str(path))

    np.testing.assert_array_equal(loaded.states, traj.states)
    assert loaded.token_index == traj.token_index
    assert loaded.seed == traj.seed
    assert loaded.depth == traj.depth
    assert loaded.seq_len == traj.seq_len
    assert loaded.prompt_id == traj.prompt_id
