"""Capture the latent path of one token across recurrent unrolls.

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/00_smoke_extract.ipynb, cell-3/cell-6).
TASK: Hook Huginn's recurrent core_block so every unroll writes the hidden
    state of the selected token into a buffer, then pack it into a Trajectory.
I/O: (model, tok, prompt, num_steps, seed, token_index, device) -> Trajectory
    with states of shape [num_steps, hidden_dim].

GOTCHAS (hard-won — keep):
  * Hook `model.transformer.core_block[-1]`. Call `_forward_hooks.clear()` and
    wrap the forward in try/finally, else a hook left by a crashed run DOUBLES
    the captures (bug: 64 states instead of 32).
  * `num_steps` must be a plain int, NOT a torch.tensor — len() of a 0-d tensor
    crashes the forward.
  * h_0 is a RANDOM init: torch.manual_seed(seed) before the forward for
    reproducibility. It is an outlier; downstream winding is analysed without
    the first captured state.
  * Use `model(...)` (forward), NOT generate — generate would decode tokens.
  * states has shape [num_steps, hidden_dim]: the hook records the num_steps
    core_block OUTPUTS (h_1..h_r); the random init h_0 is not among them.
"""

from __future__ import annotations

from typing import Any

import torch

from traj_geom.constants import DEFAULT_NUM_STEPS
from traj_geom.types import Trajectory


def extract_trajectory(
    model: Any,
    tok: Any,
    prompt: str,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: int = 0,
    token_index: int = -1,
    device: str = "cuda",
) -> Trajectory:
    """Run ``prompt`` through ``num_steps`` recurrent unrolls and record the path.

    Args:
        model: A model handle from :func:`traj_geom.extraction.model.load_huginn`.
        tok: The matching tokenizer.
        prompt: Input prompt text.
        num_steps: Number of recurrent unrolls r (plain int, not a tensor).
        seed: Seed for the random recurrent initialisation h_0.
        token_index: Prompt token position to trace; -1 = last (answer) token.
        device: Torch device string.

    Returns:
        A :class:`Trajectory` whose ``states`` has shape [num_steps, hidden_dim].
        ``depth`` is set to -1 (unknown here; filled in by the dataset pipeline).
    """
    core = model.transformer.core_block[-1]
    core._forward_hooks.clear()  # drop hooks left by any crashed run (else 64 vs 32)

    latents: list[torch.Tensor] = []
    handle = core.register_forward_hook(
        lambda m, i, o: latents.append(o.detach().float().cpu())
    )
    try:
        torch.manual_seed(seed)  # h_0 is random -> seed for reproducibility
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model(input_ids=ids, num_steps=num_steps)  # forward, NOT generate
    finally:
        handle.remove()  # removed even if the forward raised

    states = torch.stack([latent[0, token_index, :] for latent in latents]).numpy()
    return Trajectory(
        states=states,
        token_index=token_index,
        seed=seed,
        depth=-1,
        seq_len=int(ids.shape[1]),
        prompt_id=prompt,
    )
