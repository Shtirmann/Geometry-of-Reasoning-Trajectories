"""Load the Huginn recurrent-depth transformer for inference.

OWNER: Extraction+Winding
STATUS: stub — implement me.
TASK: Load Huginn (custom architecture, trust_remote_code=True) onto the
    target device/dtype and return a ready-to-run model handle.
I/O: model_id (+ device, dtype) -> loaded model object.

NOTE: Huginn is a custom architecture — you MUST pass trust_remote_code=True.
    Its per-unroll hidden states are NOT exposed via output_hidden_states; the
    recurrent loop must be hooked directly (see hook.py). Loading here only
    prepares the model; capturing h_t happens in extraction/hook.py.
"""

from __future__ import annotations

from typing import Any


def load_huginn(model_id: str, device: str = "cuda", dtype: str = "bf16") -> Any:
    """Load the Huginn model for inference.

    Args:
        model_id: HuggingFace id of the Huginn checkpoint.
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
        dtype: Compute dtype tag, e.g. ``"bf16"``.

    Returns:
        The loaded model object (ready for hooked inference).
    """
    raise NotImplementedError(
        "TODO(Extraction+Winding): load Huginn with trust_remote_code=True, "
        "move to device/dtype, set eval mode."
    )
