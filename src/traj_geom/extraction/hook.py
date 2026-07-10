"""Capture the latent path h_0..h_r of one token across recurrent unrolls.

OWNER: Extraction+Winding
STATUS: stub — implement me.
TASK: Hook the recurrent loop in Huginn's modeling_*.py so that every unroll
    step t writes the hidden state h_t of the selected token into a buffer,
    then pack the buffer into a Trajectory.
I/O: (model, prompt, num_steps, seed, token_index) -> Trajectory with
    states of shape [num_steps + 1, hidden_dim].

NOTE: output_hidden_states does NOT give the per-unroll states — register a
    forward hook on the recurrent block. Set the recurrent init from `seed`
    (the start h_0 is random) so runs are reproducible.
"""

from __future__ import annotations

from typing import Any

from traj_geom.types import Trajectory


def extract_trajectory(
    model: Any,
    prompt: str,
    num_steps: int,
    seed: int,
    token_index: int = -1,
) -> Trajectory:
    """Run the prompt through ``num_steps`` recurrent unrolls and record the path.

    Args:
        model: A model handle from :func:`traj_geom.extraction.model.load_huginn`.
        prompt: Input prompt text.
        num_steps: Number of recurrent unrolls r (yields r + 1 states h_0..h_r).
        seed: Seed for the random recurrent initialisation h_0.
        token_index: Prompt token position to trace; -1 = last (answer) token.

    Returns:
        A :class:`Trajectory` whose ``states`` has shape [num_steps + 1, hidden_dim].
    """
    raise NotImplementedError(
        "TODO(Extraction+Winding): seed the recurrent init, hook the unroll loop, "
        "collect h_t for token_index into states [num_steps+1, hidden_dim], "
        "build a Trajectory."
    )
