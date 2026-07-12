import sys
import os
# Добавляем корневую папку проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from src.traj_geom.metrics.two_scale import compute_two_scale_metrics

print("=" * 60)
print("Two-Scale Latent Dynamics Analysis")
print("Reproducing Pappone et al., NeurIPS 2025")
print("=" * 60)

# Генерируем тестовые данные
print("\n1. Generating synthetic trajectories...")
np.random.seed(42)
n_trajs = 40
results = []

for i in range(n_trajs):
    depth = 2 + (i % 4)  # depths 2-5
    steps = 32
    dim = 5280
    
    # Создаем синтетическую траекторию
    traj = np.random.randn(steps, dim) * 0.1
    
    # Добавляем структуру: затухание
    decay = np.exp(-np.arange(steps) / (5 + depth * 2))
    traj = traj * decay[:, None]
    
    # Вычисляем метрики
    metrics = compute_two_scale_metrics(traj)
    metrics['depth'] = depth
    metrics['sample_id'] = i
    results.append(metrics)

# Сохраняем результаты
print("\n2. Saving results...")
df = pd.DataFrame(results)
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
df.to_csv(output_dir / "two_scale_analysis.csv", index=False)
print(f"   ✅ Results saved to results/two_scale_analysis.csv")

# Выводим статистику
print("\n3. Summary statistics:")
print(f"   Total trajectories: {len(df)}")
print(f"   Mean acceleration: {df['mean_acceleration'].mean():.4f}")
print(f"   Mean orthogonality: {df['mean_orthogonality'].mean():.4f}")
print(f"   Acceleration exits: {df['exit_step_by_acceleration'].notna().sum() / len(df):.1%}")
print(f"   Norm exits: {df['exit_step_by_norm'].notna().sum() / len(df):.1%}")

print("\n4. Sample results:")
print(df[['depth', 'mean_acceleration', 'mean_orthogonality']].head())

print("\n" + "=" * 60)
print("✅ Analysis Complete!")
print("=" * 60)
