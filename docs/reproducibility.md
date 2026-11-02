# Reproducibility

## Simulation generator

`spadag-generate-reversal` implements the data-generating mechanism of the
manuscript's regional-reversal simulation study: a regional-reversal graph
field on the regular 36-vertex mesh, `Uniform(0.5, 1.5)` coefficients, a
linear structural equation model with iid uniform locations on the unit
square, one global coefficient multiplier calibrated to a target median
population signal-to-noise ratio, and per-column standardization. The
manuscript's mesh-sensitivity study refits these datasets on other inference
meshes; its no-reversal study uses a different generator that is not part of
this package. The seeds of the 150-dataset reversal cohort (five cells
`P50_N500_S1`, `P25_N500_S1`, `P75_N500_S1`, `P50_N250_S1`, `P50_N1000_S1`,
30 replicates each) are tracked in
[`artifacts/data/reversal_cohort_seeds.json`](../artifacts/data/reversal_cohort_seeds.json)
together with the SHA-256 of each dataset's `gamma.npy`, so any cohort dataset
can be regenerated without distributing the bundles:

```bash
spadag-generate-reversal --out runs/P25_N500_S1_rep001 --cohort-cell P25_N500_S1 --replicate 1 --no-kl
spadag-prepare --x runs/P25_N500_S1_rep001/bundle/X.npy --locations runs/P25_N500_S1_rep001/bundle/obs_points.npy \
  --outdir runs/P25_N500_S1_rep001/inference --mesh-type regular --domain 0 1 0 1
```

For `p = 50`, the `n = 500` and `n = 250` datasets are re-standardized
prefixes of the corresponding `n = 1000` dataset, which is why the example
bundle's `metadata.json` lists two generator commands.

Recovery on an inference mesh that differs from the generating mesh is
evaluated with `spadag-evaluate --eval-points sobol|observed|both`, which pools
location-wise true and false positives and false negatives and reports the
representational ceiling of the inference mesh.

## Checkpoint compatibility

`--checkpoint` writes a versioned pickle containing the complete sampler and
Python/R RNG states. Compatibility is based on a SHA-256 fingerprint of the
five validated inference arrays, so an unchanged bundle can be moved between
directories or operating systems. A same-shaped bundle with different values
is rejected. A checkpoint also records the Python, NumPy and SciPy versions
and the SHA-256 of the sampler modules, and resuming it under different
versions is refused.

## Software versions

Exact continuation from a checkpoint and byte-for-byte regeneration of the
cohort arrays are tested with Python 3.13.9, NumPy 2.5.4, SciPy 1.18.1,
R 4.5.2, `BayesLogit` 2.4 and `BayesGPfit` 1.1.0. Bit-for-bit reproducibility
requires matching software versions and BLAS thread counts; the manuscript's
chains used one thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1`). NumPy's random streams can change between feature
releases, so the generator's output for a given seed is tied to the NumPy
version.

The R packages are external system dependencies and are not installed by the
wheel. The tests run with

```bash
python -m pip install '.[test,prepare,diagnostics]'
pytest
```

## Example figure

[`example_recovery.png`](example_recovery.png) in this directory was drawn
with [`plot_example_recovery.py`](plot_example_recovery.py) from the
manuscript's 50,000-sweep chain for the example dataset. Running the default protocol on the example bundle produces a
`summary.npz` in the same format, which the script accepts through
`--summary`.
