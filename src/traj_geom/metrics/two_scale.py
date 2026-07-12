"""Two-Scale Latent Dynamics metrics from Pappone et al., NeurIPS 2025"""

import numpy as np
from typing import Dict, Any, Optional, List

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

def compute_two_scale_metrics(trajectory: np.ndarray, threshold_percentile: float = 10.0) -> Dict[str, Any]:
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
        exit_by_norm = np.argmax(step_norms < threshold_norm) if np.any(step_norms < threshold_norm) else None
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

def compute_two_scale_for_dataset(trajectories: List[np.ndarray], metadata: List[Dict]) -> List[Dict]:
    """Compute two-scale metrics for a dataset of trajectories"""
    results = []
    for traj, meta in zip(trajectories, metadata):
        metrics = compute_two_scale_metrics(traj)
        metrics.update(meta)
        results.append(metrics)
    return results
