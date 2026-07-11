"""Synthetic reasoning-task generators (state-holding probes for H3).

OWNER: Shapes+Gate
STATUS: implemented (from notebooks/01_mvp_h2.ipynb, make_* task builders).
TASK: generate prompts whose "reasoning depth" is a controlled integer, so
    effective compute can be regressed against it.
I/O: (n_ops, seed) -> dict with a prompt (and, for variants, matched prompts).

The counting / switch tasks force the model to HOLD a running state: to answer,
it cannot just read the last token, it must accumulate. `make_variants` is the
length-matched control — `track` and `local` share an identical body and differ
only in the question, isolating state-holding from raw prompt length.
"""

from __future__ import annotations

import random


def make_counting_task(n_ops: int, seed: int = 0) -> dict:
    """Running +/-1 sum. To answer, the model must hold a counter.

    Args:
        n_ops: Number of +/-1 operations (the reasoning depth).
        seed: Seed for the random op sequence.

    Returns:
        ``{"prompt", "n_ops", "answer"}``.
    """
    rng = random.Random(seed)
    ops = [rng.choice([1, -1]) for _ in range(n_ops)]
    text = (
        "Start at 0. "
        + " ".join("Add 1." if o > 0 else "Subtract 1." for o in ops)
        + " Final total? A:"
    )
    return {"prompt": text, "n_ops": n_ops, "answer": sum(ops)}


def make_variants(n_ops: int, seed: int = 0) -> dict:
    """Length-matched pair: identical body, different question (the killer control).

    Args:
        n_ops: Number of +/-1 operations (the reasoning depth).
        seed: Seed for the random op sequence.

    Returns:
        ``{"track", "local"}`` — ``track`` needs accumulation, ``local`` needs
        only the last instruction; both have the same token length.
    """
    rng = random.Random(seed)
    ops = [rng.choice([1, -1]) for _ in range(n_ops)]
    body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1." for o in ops)
    return {
        "track": body + " Final total? A:",  # needs accumulation
        "local": body + " What was the last instruction? A:",  # needs only the last step
    }


def make_switch_task(n_ops: int, seed: int = 0) -> dict:
    """Light on/off parity. Generalisation test beyond arithmetic counting.

    Args:
        n_ops: Number of flip/wait steps (the reasoning depth).
        seed: Seed for the random flip sequence.

    Returns:
        ``{"prompt", "answer"}`` where answer is "on"/"off" by flip parity.
    """
    rng = random.Random(seed)
    flips = [rng.choice([0, 1]) for _ in range(n_ops)]
    body = "Light is off. " + " ".join("Flip." if f else "Wait." for f in flips)
    return {"prompt": body + " Is the light on? A:", "answer": "on" if sum(flips) % 2 else "off"}
