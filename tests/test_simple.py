import sys
import os
# Добавляем корневую папку проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.traj_geom.metrics.two_scale import compute_step_orthogonality

# Простой тест
traj = np.array([[0,0], [1,0], [1,1], [0,1], [0,0]])
orth = compute_step_orthogonality(traj)
print(f"Orthogonality: {orth}")
assert np.allclose(orth, 0, atol=1e-5), "Test failed"
print("✅ Test passed")
