# Two-Scale Latent Dynamics on real Huginn

Facts and numbers only. Metrics from Pappone et al. (arXiv:2509.23314); model
Huginn-0125 (arXiv:2502.05171), inference-only, single GPU, one init seed.

## Integrity

- `scripts/run_two_scale_analysis.py` is a **SYNTHETIC METRIC CHECK ONLY**:
  trajectories are `np.random.randn` with an injected `5+2*depth` exponential decay.
  The deck figures `accel rho=0.997`, `orth=-0.5` came from this generator — not a model.
- Real-data results below: `scripts/run_two_scale_depth.py`, `run_two_scale_real.py`,
  `run_two_scale_counting.py` (metric code in `metrics/two_scale.py` is unchanged/reused).

## Length-matched band — MAIN (`results/band.csv`)

Adjacent depth pairs matched on `seq_len`; `hi` = high-depth indicator; coefficient of
`hi` in `accel ~ hi + seq_len [+ band dummies]`; bootstrap resampled within (band × hi),
nboot=5000, seed=0.

| token | depth-coef \| len (pooled) | 95% CI | p |
|---|---|---|---|
| content | **+1.30** | [+1.05, +1.55] | <1e-4 |
| answer | +0.04 | [−0.60, +0.53] | 0.86 |

Per-band content coef|len: `2-3` +1.565, `3-4` +1.266, `4-5` +1.184 (n=120 each).

## Full-range depth (`results/pararule_depth.csv`, `pararule_depth_map.npy`)

- `accel_content` by depth 2–5: 18.00, 18.42, 18.53, 18.89.
- per-level content~depth: rho=+1.00 (N=4). depth and `seq_len` collinear on the full
  range → partial|seq_len unreliable (per-row partial rho=+0.61). Use the band test.
- position map (10 bins), mean accel:
  [13.36, 17.81, 18.15, 19.03, 19.48, 19.58, 19.51, 19.55, 19.42, 19.08].

## Counting track/local (`results/counting_two_scale.csv`)

- content~n_ops per-level: track rho=+0.94 (N=6), local rho=+1.00 (N=6) — both rise
  (length-confounded, not length-matched across n_ops); no track-vs-local dissociation.
- answer~n_ops per-level: track rho=−0.71, local rho=−0.43 (both n.s.).

## Caveats

- David's earlier evidence (accel rho=0.997 / orth=−0.5) was a random-noise artifact.
- Answer-token acceleration does not track depth.
- Real content signal appears only at matched length (band test); 'depth' co-varies with
  logical rule-count, not necessarily reasoning hop-count; high depths under-sampled.
- One init seed; one model (Huginn-0125).

## References

- Pappone et al. Two-Scale Latent Dynamics for Recurrent-Depth Transformers. NeurIPS 2025. arXiv:2509.23314.
- Geiping et al. Scaling up Test-Time Compute with Latent Reasoning. NeurIPS 2025. arXiv:2502.05171.
