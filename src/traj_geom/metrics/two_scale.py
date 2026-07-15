"""Two-Scale Latent Dynamics metrics from Pappone et al., NeurIPS 2025"""

from typing import Any

import numpy as np


def two_scale_allpos(traj: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized all-position two-scale metrics for a [steps, seq_len, dim] trajectory.

    Equivalent to running the per-position metrics (compute_two_scale_metrics on each
    column ``traj[:, p, :]``) but computed in one pass over all positions.

    Args:
        traj: Trajectory of shape [num_steps, seq_len, dim].

    Returns:
        dict with:
          ``mean_accel``  [seq_len]  — mean acceleration ||Δ(k)-Δ(k-1)|| per position;
          ``mean_orth``   [seq_len]  — mean consecutive-step cosine per position;
          ``stepnorm``    [steps-1, seq_len] — per-step displacement norms.
    """
    d = np.diff(traj, axis=0)
    sn = np.linalg.norm(d, axis=2)
    a = np.linalg.norm(np.diff(d, axis=0), axis=2)
    num = (d[1:] * d[:-1]).sum(2)
    den = np.linalg.norm(d[1:], axis=2) * np.linalg.norm(d[:-1], axis=2) + 1e-12
    return dict(mean_accel=a.mean(0), mean_orth=(num / den).mean(0), stepnorm=sn)

def compute_step_deltas(trajectory: np.ndarray) -> np.ndarray:
    """Compute Δ(k) = h(k+1) - h(k)"""
    return np.diff(trajectory, axis=0)

def compute_acceleration(trajectory: np.ndarray) -> np.ndarray:
    """Compute acceleration a(k) = ||Δ(k) - Δ(k-1)||₂"""
    deltas = compute_step_deltas(trajectory)
    if len(deltas) < 2:
        return np.array([])
    second_diff = np.diff(deltas, axis=0)
    acceleration = np.linalg.norm(second_diff, axis=1)
    return acceleration

def compute_step_orthogonality(trajectory: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between consecutive step vectors"""
    deltas = compute_step_deltas(trajectory)
    if len(deltas) < 2:
        return np.array([])

    similarities = []
    for i in range(len(deltas) - 1):
        norm1 = np.linalg.norm(deltas[i])
        norm2 = np.linalg.norm(deltas[i+1])
        if norm1 > 0 and norm2 > 0:
            sim = np.dot(deltas[i], deltas[i+1]) / (norm1 * norm2)
        else:
            sim = 0.0
        similarities.append(sim)

    return np.array(similarities)

def compute_two_scale_metrics(
    trajectory: np.ndarray, threshold_percentile: float = 10.0
) -> dict[str, Any]:
    """Compute all two-scale dynamics metrics from Pappone et al."""
    deltas = compute_step_deltas(trajectory)
    step_norms = np.linalg.norm(deltas, axis=1)
    acceleration = compute_acceleration(trajectory)
    orthogonality = compute_step_orthogonality(trajectory)

    if len(acceleration) > 0:
        threshold_acc = (threshold_percentile / 100.0) * np.max(acceleration)
        below_threshold = acceleration < threshold_acc
        exit_by_acceleration = None
        for i in range(len(below_threshold) - 1):
            if below_threshold[i] and below_threshold[i+1]:
                exit_by_acceleration = i
                break
    else:
        exit_by_acceleration = None

    if len(step_norms) > 0:
        threshold_norm = (threshold_percentile / 100.0) * np.max(step_norms)
        exit_by_norm = (
            np.argmax(step_norms < threshold_norm)
            if np.any(step_norms < threshold_norm)
            else None
        )
    else:
        exit_by_norm = None

    return {
        'acceleration': acceleration.tolist() if len(acceleration) > 0 else [],
        'orthogonality': orthogonality.tolist() if len(orthogonality) > 0 else [],
        'step_norms': step_norms.tolist() if len(step_norms) > 0 else [],
        'exit_step_by_acceleration': exit_by_acceleration,
        'exit_step_by_norm': exit_by_norm,
        'max_acceleration': np.max(acceleration) if len(acceleration) > 0 else None,
        'max_step_norm': np.max(step_norms) if len(step_norms) > 0 else None,
        'mean_acceleration': np.mean(acceleration) if len(acceleration) > 0 else None,
        'mean_orthogonality': np.mean(orthogonality) if len(orthogonality) > 0 else None,
        'num_steps': len(trajectory)
    }

def compute_two_scale_for_dataset(
    trajectories: list[np.ndarray], metadata: list[dict]
) -> list[dict]:
    """Compute two-scale metrics for a dataset of trajectories"""
    results = []
    for traj, meta in zip(trajectories, metadata, strict=False):
        metrics = compute_two_scale_metrics(traj)
        metrics.update(meta)
        results.append(metrics)
    return results
