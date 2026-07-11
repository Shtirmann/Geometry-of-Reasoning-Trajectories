"""Shared helpers for experiment scripts: results/ caching and lazy model loading.

Mirrors the notebook's `cached()` — compute once, reuse the CSV. When a result
CSV already exists (shipped from Kaggle in results/), the heavy compute is skipped
and no GPU / model extra is needed to re-run the analysis and figures.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def cached(name: str, compute_fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Return ``results/<name>`` if it exists, else compute, save and return it.

    Args:
        name: CSV filename under results/.
        compute_fn: Zero-arg callable producing the DataFrame on a cache miss.

    Returns:
        The cached or freshly computed DataFrame.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    if path.exists():
        print(f"loaded {path}")
        return pd.read_csv(path)
    df = compute_fn()
    df.to_csv(path, index=False)
    print(f"computed+saved {path}")
    return df


def load_model() -> tuple[Any, Any]:
    """Lazily import and load Huginn (needs the `model` extra and a GPU)."""
    from traj_geom.extraction.model import load_huginn

    return load_huginn()
