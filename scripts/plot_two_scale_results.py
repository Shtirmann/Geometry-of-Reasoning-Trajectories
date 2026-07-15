import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Загружаем результаты
df = pd.read_csv('results/two_scale_analysis.csv')

# Создаем фигуру
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Acceleration by depth
ax1 = axes[0, 0]
sns.boxplot(data=df, x='depth', y='mean_acceleration', ax=ax1)
ax1.set_title('Mean Acceleration by Depth')
ax1.set_xlabel('Depth')
ax1.set_ylabel('Mean Acceleration')

# 2. Orthogonality by depth
ax2 = axes[0, 1]
sns.boxplot(data=df, x='depth', y='mean_orthogonality', ax=ax2)
ax2.set_title('Mean Orthogonality by Depth')
ax2.set_xlabel('Depth')
ax2.set_ylabel('Mean Orthogonality')
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)

# 3. Exit steps distribution
ax3 = axes[1, 0]
df_clean = df.dropna(subset=['exit_step_by_acceleration'])
if len(df_clean) > 0:
    sns.histplot(data=df_clean, x='exit_step_by_acceleration', bins=15, ax=ax3)
ax3.set_title('Exit Step Distribution (Acceleration)')
ax3.set_xlabel('Step Number')
ax3.set_ylabel('Count')

# 4. Acceleration vs Orthogonality
ax4 = axes[1, 1]
scatter = ax4.scatter(df['mean_acceleration'], df['mean_orthogonality'],
                     c=df['depth'], cmap='viridis', alpha=0.7)
ax4.set_xlabel('Mean Acceleration')
ax4.set_ylabel('Mean Orthogonality')
ax4.set_title('Acceleration vs Orthogonality')
plt.colorbar(scatter, ax=ax4, label='Depth')

plt.tight_layout()

# Сохраняем
fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / 'two_scale_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Figure saved to figures/two_scale_results.png")
