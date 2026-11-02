# Running SpaDAG

This page covers the full sampling protocol, checkpoints, multi-chain
diagnostics, and preparing a bundle from your own data. The
[README](../README.md) has the installation and a short example;
[data-format.md](data-format.md) describes the files that `spadag` reads, and
[reproducibility.md](reproducibility.md) the simulation generator, checkpoint
compatibility and software versions.

## Default protocol

The CLI defaults are the settings used throughout the manuscript: 50,000
sweeps per chain, of which the first 25,000 are discarded as burn-in and every
fifth subsequent state is retained for the posterior summaries (5,000 draws),
initialization at the empty graph, 50 maximal component-reversal attempts per
sweep, and the 66-term K–L expansion of the prepared bundle. The manuscript's
simulation results come from one chain per dataset, each with its own seed,
and its data analysis from four chains with distinct seeds. A full chain with
checkpoints is

```bash
spadag \
  --data artifacts/data/example_reversal_p50_n500 \
  --trace-dir runs/chain1/trace \
  --checkpoint runs/chain1/checkpoint.pkl \
  --seed 1
```

`--initialization prior-random` starts from Bernoulli(0.5) indicators
restricted to local DAGs, and `--m-rev 0` disables the reversal move, which
gives the ordinary Gibbs sampler with Pólya–Gamma inclusion updates. The prior
hyperparameters (`--v0`, `--a-b`, `--b-b`, `--a-tau`, `--b-tau`,
`--common-gp-var-prior`, `--mu-prior-mean`, `--mu-prior-sd`) and the trace
thinning intervals are listed by `spadag --help`.

## Working model of the sampler

The structural model of the manuscript places a point mass at zero on
inactive coefficients, and its identifiability result (Theorem 1) concerns
that population model. To keep the coefficient updates Gaussian, the sampler
replaces the point mass with a Gaussian spike of fixed variance `--v0`
(default `1e-3`), uses the product of nodewise Gaussian regression densities
as a working likelihood, and targets the corresponding working posterior. The
local graphs are still defined by the indicators, so every sampled triangle
graph is acyclic, and the reported PIPs are posterior probabilities under
this working model.

## Checkpoints and resuming

`--checkpoint` writes a pickle every 250 sweeps (`--checkpoint-every`) and
keeps immutable snapshots every 5,000 sweeps (`--milestone-every`). To extend
a chain, pass its checkpoint to both `--checkpoint` and `--resume` with a
larger `--iters` and every other setting, including `--seed`, unchanged; the
checkpoint records the settings and refuses a different one.

The trace directory stores online triangle PIP and coefficient moments, small
diagnostic streams and packed indicator states; retained draws are written to
append-only chunk files linked by SHA-256, so the checkpoint holds no draw
history. The trace directory and the checkpoint must be moved together.
Pickle can execute code while loading, so resume only checkpoints that you
created or obtained from a trusted source.

## Output

`summary.npz` in the trace directory holds the posterior summaries.
`edge_prob` has shape `(T, p, p)` and holds the posterior inclusion
probability of each directed edge in each triangle, `edge_prob[t, j, k]` for
the edge k → j. `gamma_prob` has shape `(p, p, H)` and holds the nodal
indicator probabilities, `b_mean` the posterior mean of the nodal
coefficients, and `kept` the number of retained sweeps behind these
summaries. `--out` writes a copy of the summaries to a path of your choice.

## Multi-chain diagnostics

Run `spadag-diagnose` on two or more chain trace directories with distinct
seeds to compute rank-normalized split R-hat, bulk/tail ESS and between-chain
differences in edge probabilities; `--truth` (the same directory as for
`spadag-evaluate`) is optional:

```bash
spadag-diagnose runs/chain1/trace runs/chain2/trace \
  --truth artifacts/data/example_reversal_p50_n500 \
  --out runs/diagnostics.json
```

The report is written as JSON and as a Markdown summary next to it. It
requires the `diagnostics` extra (ArviZ and psutil).

## Use your own observations and locations

Most users provide only two files:

- `X.npy`: a finite observed-data matrix with shape `(n, p)` and `p > 1`;
- `obs_points.npy`: finite two-dimensional coordinates with shape `(n, 2)`.

Rows must align: row `i` in both files must describe the same observation.
`X` is the matrix that enters the Gaussian working likelihood, with each
column centered and scaled across observations; the default priors are set for
this scale, and neither `spadag-prepare` nor the sampler standardizes the
input. Count data need a variance-stabilizing transformation first, for
example the Pearson residuals used for the manuscript's tissue section. First
prepare a deterministic bundle, then run inference:

```bash
# 1. Triangulate the domain and prepare the inference bundle.
#    Defaults: observation-supported polygon mesh in the input coordinates and
#    the manuscript's K-L design (L = 66, max_range = 1, poly_degree = 10).
spadag-prepare \
  --x analysis/X.npy \
  --locations analysis/obs_points.npy \
  --outdir analysis/bundle \
  --target-tris 75 --min-obs-per-triangle 6 --max-obs-per-triangle 40

# 2. Run posterior inference on that fixed bundle.
spadag \
  --data analysis/bundle \
  --trace-dir analysis/chain1/trace \
  --checkpoint analysis/chain1/checkpoint.pkl \
  --seed 1
```

Preparation and inference are separate steps because the mesh and the K–L
design are analysis choices that determine checkpoint compatibility.
`spadag-prepare` records them in `preparation_manifest.json`, and `spadag`
reads only the resulting bundle.

The default preparation path is the one used for the manuscript's
tissue-section analysis: `--mesh-type polygon` meshes an observation-supported
polygon domain (concave hull with holes) in the input coordinates, optionally
bounding the observations per triangle with
`--min-obs-per-triangle`/`--max-obs-per-triangle`, and the K–L design is the
full degree-10 expansion (`--L 66 --max-range 1 --poly-degree 10`). `L` may
exceed the number of mesh vertices because the sampler keeps every coefficient
coordinate under its proper Gaussian prior. `--mesh-type regular` builds the
deterministic `5 x 5` triangulation and `--mesh-type cdt` a constrained
Delaunay triangulation over the observation bounding box; add
`--domain 0 1 0 1` to mesh the known unit-square domain instead, which
reproduces the simulation meshes. Advanced users may instead supply
`--mesh-points` and `--mesh-triangles` for an arbitrary triangulation, and
`--a-kl` for an existing K–L design. See `spadag-prepare --help` for the
complete interface.
