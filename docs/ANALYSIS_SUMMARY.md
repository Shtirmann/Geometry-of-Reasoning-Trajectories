# Two-Scale Latent Dynamics: Real-Data Results

## Overview

Replacement of synthetic trajectory generator with real-Huginn inference pipeline for two-scale dynamics analysis (Pappone et al., NeurIPS 2025). Synthetic placeholder produced artifactual results (accel rho=0.997) due to hand-coded decay 5+2*depth. Real data pipeline removes this artifact.

## Method

- Band analysis: compare adjacent depth levels with matched sequence length
- Depth bands: 2-3, 3-4, 4-5 (each with seq_len ~200)
- Tokens: CONTENT (meaningful) vs ANSWER (control)
- Metrics: acceleration (second difference) and orthogonality (cosine similarity)

## Results

### Acceleration (primary)

| Token | Depth-coef (pooled) | 95% CI | p-value |
|-------|---------------------|--------|---------|
| CONTENT | +1.297 | [1.055, 1.548] | < 0.0001 |
| ANSWER | -0.044 | [-0.604, 0.528] | 0.862 |

Per-band CONTENT coefficients:
- 2-3: +1.565
- 3-4: +1.266
- 4-5: +1.184

Conclusion: CONTENT acceleration increases with depth. ANSWER control shows no effect.

### Orthogonality (secondary)

| Token | Mean | Correlation with adj | p-value |
|-------|------|---------------------|---------|
| CONTENT | 0.1376 | +0.139 | 0.008 |
| ANSWER | 0.3888 | -0.004 | 0.946 |

Per-band CONTENT orthogonality:
- 2-3: 0.1132 +/- 0.1264
- 3-4: 0.1486 +/- 0.1348
- 4-5: 0.1510 +/- 0.1321

Conclusion: CONTENT orthogonality shows weak increase with depth. ANSWER remains stable.

## Files

- src/traj_geom/metrics/two_scale.py - metric implementations
- src/traj_geom/analysis/band.py - length-matched band regression
- scripts/run_two_scale_depth.py - main analysis script
- results/band.csv - output data
- figures/orthogonality_real.png - orthogonality visualization

## References

Pappone et al., NeurIPS 2025. arXiv:2509.23314
