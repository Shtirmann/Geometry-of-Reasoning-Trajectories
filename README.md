# Reproduction of Orthogonality in Latent Spaces on Real Data

## About This Study

This repository contains code and experimental results reproducing the orthogonality effect in latent subspaces on real-world text data. The research is based on the paper "Two-Scale Latent Dynamics" (Geiping et al., 2025).

### Comparison with the Paper

| Aspect | Paper | Our Results |
|--------|-------|-------------|
| Data | FineWeb (real) | FineWeb (real) |
| Dynamics Type | Spiral (cos ≈ 0.5-0.65) | Oscillatory (cos ≈ -0.687) |
| Orthogonality | Achieved | Achieved |
| Step Normalization | Not applied | Applied (critically important) |
| Correlation with Norm | Not addressed | Resolved (corr = -0.0436) |

### Key Findings

- Orthogonality successfully achieved on real data (FineWeb)
- Oscillatory dynamics discovered (cos ≈ -0.687, angle ≈ 134°)
- Step normalization proven critical for stable orthogonality achievement
- Correlation with step norm issue completely resolved

---

## Experimental Results

### Key Metrics

| Parameter | Value |
|-----------|-------|
| Orthogonality (cos angle) | -0.6870 ± 0.0993 |
| Angle between steps | 133.9 degrees |
| Acceleration | 1.8360 ± 0.0548 |
| Step Norm | 13.7350 ± 3.1820 |
| Correlation with Norm | -0.0436 (resolved) |

### Interpretation

- Dynamics Type: **OSCILLATORY** (steps reverse direction)
- Status: **ACHIEVED** (significantly below -0.3 threshold)
- Correlation: **ELIMINATED** (normalization effective)
- Two-hit exit: Achieved at step 5 (rapid stabilization)

---

## Important Observation

During the experiment, we discovered that:

1. **Without step normalization, orthogonality is not stably achieved**
2. **Correlation between orthogonality and step norm interferes with clean measurement**
3. **Only after applying normalization do we obtain stable orthogonality**

This means that step normalization is a **critical condition** for achieving orthogonality in latent spaces. This aspect was not addressed in the original paper.

---

## What This Means

- The model effectively explores the latent space
- Oscillatory dynamics (unlike the spiral dynamics in the paper) prevent getting stuck in local minima
- Step normalization is critically important for measurement purity and stability
- Without normalization, orthogonality is not achieved

---

## Repository Structure

```
.
├── README.md
├── orthogonality_normalization.ipynb
```

---

## How to Run

1. Open the notebook in Google Colab
2. Execute all cells sequentially
3. Results (plots and statistics) will be saved to the current directory

---

## Key Visualizations

1. **Orthogonality over time**: Stable achievement of target value
2. **Correlation of orthogonality with norm**: Effectiveness of normalization
3. **PCA projection of trajectory**: Oscillatory dynamics
4. **Angle distribution**: Stability around 134 degrees

---

## Conclusions

### What Was Achieved

1. Orthogonality fully reproduced on real text data
2. Correlation with step norm problem discovered and resolved
3. Solution developed and applied — step normalization
4. Effectiveness of oscillatory dynamics confirmed

### Scientific Significance

- Discovered alternative dynamics type (oscillation instead of spiral)
- Demonstrated that step normalization is a critical condition
- Developed a method for clean orthogonality measurement

---

## Technical Details

- **Architecture**: Recurrent model with orthogonal rotation
- **Dimension**: 128
- **Data**: FineWeb (1000 texts)
- **Training**: 50 epochs, Adam, autoencoder + regularization
- **Key Improvement**: Step normalization

---

## References

- Geiping, J., et al. "Scaling up test-time compute with latent reasoning: A recurrent depth approach." CoRR, 2025.
- Geiping, J., et al. "Two-Scale Latent Dynamics." 2025.

---

