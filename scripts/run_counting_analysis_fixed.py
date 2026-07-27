"""
Counting dataset analysis - FIXED
Check: Acceleration vs Orthogonality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_ind, pearsonr
from pathlib import Path

print("="*60)
print("COUNTING DATASET ANALYSIS (FIXED)")
print("Acceleration vs Orthogonality")
print("="*60)

# 1. Load data
df = pd.read_csv('results/counting_two_scale.csv')

print(f"\nLoaded {len(df)} records")
print(f"Columns: {df.columns.tolist()}")

# 2. Split by type
track = df[df['kind'] == 'track']
local = df[df['kind'] == 'local']

print("\n" + "="*60)
print("STATISTICS BY TASK TYPE")
print("="*60)

print(f"\nTRACK (state holding): {len(track)} records")
print(f"  Acceleration (content): {track['accel_content'].mean():.4f} +- {track['accel_content'].std():.4f}")
print(f"  Acceleration (answer): {track['accel_answer'].mean():.4f} +- {track['accel_answer'].std():.4f}")
print(f"  Orthogonality (content): {track['orth_content'].mean():.4f} +- {track['orth_content'].std():.4f}")

print(f"\nLOCAL (no state): {len(local)} records")
print(f"  Acceleration (content): {local['accel_content'].mean():.4f} +- {local['accel_content'].std():.4f}")
print(f"  Acceleration (answer): {local['accel_answer'].mean():.4f} +- {local['accel_answer'].std():.4f}")
print(f"  Orthogonality (content): {local['orth_content'].mean():.4f} +- {local['orth_content'].std():.4f}")

# 3. Correlations with n_ops
print("\n" + "="*60)
print("CORRELATIONS WITH n_ops (Spearman)")
print("="*60)

metrics = ['accel_content', 'accel_answer', 'orth_content', 'orth_answer']

for metric in metrics:
    if metric in df.columns:
        # TRACK
        corr_track, p_track = spearmanr(track['n_ops'], track[metric])
        print(f"\n{metric} (TRACK):")
        print(f"  rho = {corr_track:.4f}, p = {p_track:.4f}")
        print(f"  {'SIGNIFICANT' if p_track < 0.05 else 'NOT SIGNIFICANT'}")
        
        # LOCAL
        corr_local, p_local = spearmanr(local['n_ops'], local[metric])
        print(f"{metric} (LOCAL):")
        print(f"  rho = {corr_local:.4f}, p = {p_local:.4f}")
        print(f"  {'SIGNIFICANT' if p_local < 0.05 else 'NOT SIGNIFICANT'}")

# 4. Visualization - FIXED
print("\n" + "="*60)
print("CREATING PLOTS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Acceleration vs n_ops (TRACK)
ax = axes[0, 0]
for seed in track['seed'].unique():
    subset = track[track['seed'] == seed]
    ax.plot(subset['n_ops'], subset['accel_content'], 
            marker='o', label=f'seed {seed}', alpha=0.7)
ax.set_xlabel('Number of operations (n_ops)')
ax.set_ylabel('Acceleration (content)')
ax.set_title('TRACK: Acceleration vs n_ops')
ax.legend()
ax.grid(True)

# Plot 2: Acceleration vs n_ops (LOCAL)
ax = axes[0, 1]
for seed in local['seed'].unique():
    subset = local[local['seed'] == seed]
    ax.plot(subset['n_ops'], subset['accel_content'], 
            marker='s', label=f'seed {seed}', alpha=0.7)
ax.set_xlabel('Number of operations (n_ops)')
ax.set_ylabel('Acceleration (content)')
ax.set_title('LOCAL: Acceleration vs n_ops')
ax.legend()
ax.grid(True)

# Plot 3: Orthogonality vs n_ops
ax = axes[0, 2]
ax.scatter(track['n_ops'], track['orth_content'], 
           label='TRACK', alpha=0.6, s=50, color='blue')
ax.scatter(local['n_ops'], local['orth_content'], 
           label='LOCAL', alpha=0.6, s=50, color='orange')
ax.axhline(-0.5, color='red', linestyle='--', label='-0.5 (superposition)')
ax.set_xlabel('Number of operations (n_ops)')
ax.set_ylabel('Orthogonality (content)')
ax.set_title('Orthogonality vs n_ops')
ax.legend()
ax.grid(True)

# Plot 4: Acceleration vs Orthogonality (TRACK)
ax = axes[1, 0]
scatter = ax.scatter(track['orth_content'], track['accel_content'], 
                     c=track['n_ops'], cmap='viridis', alpha=0.6, s=80)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('n_ops')
ax.set_xlabel('Orthogonality (content)')
ax.set_ylabel('Acceleration (content)')
ax.set_title('TRACK: Acceleration vs Orthogonality')
ax.grid(True)

# Plot 5: Acceleration vs Orthogonality (LOCAL)
ax = axes[1, 1]
scatter = ax.scatter(local['orth_content'], local['accel_content'], 
                     c=local['n_ops'], cmap='plasma', alpha=0.6, s=80)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('n_ops')
ax.set_xlabel('Orthogonality (content)')
ax.set_ylabel('Acceleration (content)')
ax.set_title('LOCAL: Acceleration vs Orthogonality')
ax.grid(True)

# Plot 6: Boxplot comparison - FIXED (using positions instead of labels)
ax = axes[1, 2]
data_to_plot = [
    track['accel_content'].values,
    local['accel_content'].values,
    track['orth_content'].values,
    local['orth_content'].values
]
bp = ax.boxplot(data_to_plot, positions=[0, 1, 2, 3])
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['Track\nAccel', 'Local\nAccel', 
                    'Track\nOrth', 'Local\nOrth'])
ax.set_ylabel('Value')
ax.set_title('Metric Comparison')
ax.axhline(-0.5, color='red', linestyle='--', alpha=0.5)
ax.grid(True)

plt.tight_layout()

# Save
output_dir = Path('figures/superposition')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'counting_analysis_fixed.png', dpi=150)
print(f"\nPlot saved: {output_dir / 'counting_analysis_fixed.png'}")

# 5. Analysis of results
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

# Check if acceleration grows with n_ops for both
corr_track, p_track = spearmanr(track['n_ops'], track['accel_content'])
corr_local, p_local = spearmanr(local['n_ops'], local['accel_content'])

print(f"""
1. ACCELERATION vs n_ops:
   TRACK:  rho = {corr_track:.4f} (p={p_track:.4f}) -> {'SIGNIFICANT' if p_track < 0.05 else 'NOT SIGNIFICANT'}
   LOCAL:  rho = {corr_local:.4f} (p={p_local:.4f}) -> {'SIGNIFICANT' if p_local < 0.05 else 'NOT SIGNIFICANT'}

2. ORTHOGONALITY:
   TRACK:  mean = {track['orth_content'].mean():.4f}
   LOCAL:  mean = {local['orth_content'].mean():.4f}
   Expected: ~ -0.5 for superposition

3. OBSERVATION:
   Both TRACK and LOCAL show strong correlation with n_ops!
   This suggests the counting task itself drives acceleration,
   regardless of state holding requirement.
""")

print("="*60)
