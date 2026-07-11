"""Hard-won constants for the Huginn extraction pipeline. Do NOT edit values blindly.

Every value here was fixed by debugging the smoke notebook — treat them as the
"as-is" reproducibility guarantee.
"""

from __future__ import annotations

# --- Model identity (pin BOTH; revision pins the trust_remote_code model code) ---
MODEL_ID = "tomg-group-umd/huginn-0125"
MODEL_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"

# --- Environment window (GOTCHA) ---
# transformers must be 4.50–4.53. Outside this window the custom modeling code breaks:
#   4.49-  : no `device` in _get_initial_cache_position
#   4.54+  : key_cache became a read-only property
#   5.x    : tied-weights loading expects a dict
# Pin 4.53.3 + MODEL_REVISION — both are mandatory.
TRANSFORMERS_VERSION = "4.53.3"

# --- Model geometry ---
HIDDEN_DIM = 5280            # dim of every h_t captured from core_block
DEFAULT_NUM_STEPS = 32       # recurrent unrolls r

# --- Cost model ---
# ~2.6 s / trajectory on a T4 at num_steps=32 (measured in the notebook).
# Use for run-size estimates: N trajectories ≈ N * 2.6 s.
SECONDS_PER_TRAJECTORY_T4 = 2.6
