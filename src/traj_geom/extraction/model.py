"""Load the Huginn recurrent-depth transformer for inference.

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/00_smoke_extract.ipynb, cell-2).
TASK: Load Huginn (custom architecture, trust_remote_code=True) at the pinned
    revision onto the target device/dtype and return (model, tokenizer).
I/O: (device, dtype) -> (model, tok).

GOTCHA: transformers must be 4.50–4.53 and the revision must be pinned — see
    traj_geom.constants. Per-unroll hidden states are NOT exposed via
    output_hidden_states; capture them by hooking core_block (see hook.py).
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from traj_geom.constants import MODEL_ID, MODEL_REVISION


def load_huginn(
    device: str = "cuda", dtype: torch.dtype = torch.bfloat16
) -> tuple[Any, Any]:
    """Load the Huginn model and tokenizer at the pinned revision.

    Args:
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
        dtype: Compute dtype (default ``torch.bfloat16``).

    Returns:
        ``(model, tokenizer)`` — model moved to ``device`` and set to eval mode.
    """
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    return model, tok
