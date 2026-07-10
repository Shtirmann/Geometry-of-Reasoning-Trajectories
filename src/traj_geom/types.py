"""Shared data contract for the whole project: the `Trajectory` dataclass.

This is the ONE fully implemented module. Every teammate codes against
`Trajectory` and, above all, against its `states` array of shape
[num_steps + 1, hidden_dim] — the latent path h_0 .. h_r of a single token.

Because synthetic paths (a circle, a spiral) and real extracted hidden
states share the exact same `states` layout, gate/winding/projection code
can be written and tested WITHOUT the model, then run unchanged on real
Huginn trajectories. Do not break this contract without telling everyone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """A single latent reasoning path through recurrent depth.

    Attributes:
        states: Path h_0 .. h_r as an array of shape [num_steps + 1, hidden_dim].
        token_index: Position (in the prompt) of the analysed token; -1 = last.
        seed: Seed of the random recurrent initialisation h_0 (reproducibility).
        depth: d, the reasoning depth of the task the prompt encodes.
        seq_len: L, prompt length in tokens (a confounder to control for).
        prompt_id: Stable identifier of the source prompt.
    """

    states: np.ndarray
    token_index: int
    seed: int
    depth: int
    seq_len: int
    prompt_id: str

    def save(self, path: str) -> None:
        """Persist the trajectory to ``path`` as a compressed ``.npz`` archive.

        Args:
            path: Destination file path (``.npz`` extension recommended).
        """
        np.savez(
            path,
            states=np.asarray(self.states),
            token_index=self.token_index,
            seed=self.seed,
            depth=self.depth,
            seq_len=self.seq_len,
            prompt_id=self.prompt_id,
        )

    @classmethod
    def load(cls, path: str) -> Trajectory:
        """Load a trajectory previously written by :meth:`save`.

        Args:
            path: Path to a ``.npz`` archive produced by :meth:`save`.

        Returns:
            The reconstructed :class:`Trajectory`.
        """
        with np.load(path, allow_pickle=False) as data:
            return cls(
                states=data["states"],
                token_index=int(data["token_index"]),
                seed=int(data["seed"]),
                depth=int(data["depth"]),
                seq_len=int(data["seq_len"]),
                prompt_id=str(data["prompt_id"]),
            )
