# Data format

## Inference bundle

The prepared schema-v1 bundle contains exactly five required NumPy arrays:

| File | Shape | Contract |
| --- | --- | --- |
| `X.npy` | `(n, p)` | finite observed variables, each column centered and scaled |
| `Phi_obs.npy` | `(n, H)` | finite, nonnegative P1 weights; every row sums to one |
| `mesh_points.npy` | `(H, 2)` | finite 2D mesh coordinates |
| `mesh_triangles.npy` | `(T, 3)` | integer vertex indices in `[0, H)`; no degenerate triangles |
| `A_kl.npy` | `(H, L)` | finite K–L design with `L > 0` |

No manifest or simulation truth is required for inference. The loader reads
only these five files and rejects inconsistent dimensions, invalid geometry,
non-finite values, or unsupported dtypes. A bundle built with any other
triangulation workflow can be passed to `spadag` directly, as long as it
contains these five arrays. `spadag-prepare` additionally writes
`preparation_manifest.json` with the mesh and K–L settings, the SHA-256 of
each array and the bundle's data fingerprint.

## Simulation truth

Simulation truth is a separate schema used only by `spadag-evaluate` and
`spadag-diagnose`: `edge_triangle.npy` with shape `(T, p, p)`, where
`edge_triangle[t, j, k] = 1` means that k → j is present in triangle `t`, and
optionally `metadata.json` with a `reversal_motifs` list that names, for each
reversal pair, the triangles in which each direction is true. Both commands
take `--truth` as the directory that holds these files; `spadag-diagnose` also
accepts the `.npy` file itself.

## Example dataset

[`artifacts/data/example_reversal_p50_n500`](../artifacts/data/example_reversal_p50_n500)
(about 1.4 MB) is a regional-reversal simulation with n = 500 observations,
p = 50 variables, a regular mesh with H = 36 vertices and T = 50 triangles, and
an L = 66 K–L design. It is replicate 1 of the `P50_N500_S1` cell of the
manuscript's simulation cohort, and `metadata.json` records how to regenerate
it. The true triangle graphs are read only by `spadag-evaluate`.

Use the bundle's `SHA256SUMS` file to verify the byte-level integrity of its
tracked files:

```bash
cd artifacts/data/example_reversal_p50_n500 && sha256sum -c SHA256SUMS
```
