"""Generate regional-reversal simulation bundles.

``spadag-generate-reversal`` implements the data-generating mechanism of the
manuscript's simulation studies: the regional-reversal graph field
and ``Uniform(0.5, 1.5)`` coefficients of :mod:`spadag.sim.reversal_dgp`, one
global coefficient multiplier calibrated so that the median population
child--triangle SNR equals the target, iid uniform observation locations on
the unit square, a linear SEM with unit raw innovation variance, and per-column
standardization of the generated data.

For a given seed the generator reproduces the manuscript's simulation cohort
exactly; ``artifacts/data/reversal_cohort_seeds.json`` lists the cohort seeds
and the hashes of the resulting arrays.  Seeds are derived as follows:

* dataset master seed ``SeedSequence([cohort_master, p, replicate])``;
* graph, coefficient, location and innovation seeds are the four children of
  ``SeedSequence(master)``;
* ``p = 50`` replicates are generated with ``n = 1000`` and the ``n = 500`` and
  ``n = 250`` datasets are re-standardized prefixes of the same observations;
  ``p = 25`` and ``p = 75`` are generated directly with ``n = 500``.

The K-L design ``A_kl`` is not part of the data-generating mechanism.  It is
copied from ``--kl-reference`` (an existing bundle on the same mesh) or built
with ``BayesGPfit`` through R from ``--L/--a/--b/--max-range/--poly-degree``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np

from spadag.acyclic import is_dag, triangle_edge_tensor
from spadag.cpdag import dag_to_cpdag
from spadag.mcmc_data import load_inference_data
from spadag.p1_geometry import p1_integration_weights
from spadag.sim import sem
from spadag.mesh.basis import basis_matrix
from spadag.mesh.builders import build_regular_mesh
from spadag.sim.reversal_dgp import (
    PartitionConfig,
    cohort_config,
    draw_coefficients,
    draw_partitioned_graph,
    reversal_motifs_for_evaluation,
)

COHORT_MASTER = 2026092301
COHORT_CELLS = ("P50_N500_S1", "P25_N500_S1", "P75_N500_S1", "P50_N250_S1", "P50_N1000_S1")
COHORT_PARENT_N = {50: 1000, 25: 500, 75: 500}
DEFAULT_KL = dict(L=66, a=0.01, b=1.0, max_range=1.0, poly_degree=10)
SCHEMA_VERSION = "reversal-cohort-v2"
KL_ARRAYS = ("A_kl", "eigvals", "hyper_mu", "hyper_a", "hyper_b", "hyper_max_range", "hyper_L")


def _write_array_checksums(outdir: Path) -> None:
    """Write ``SHA256SUMS`` for every ``.npy`` file in ``outdir``."""
    paths = sorted(outdir.glob("*.npy"), key=lambda path: (path.name.casefold(), path.name))
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in paths]
    (outdir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seeds_from_master(master: int) -> list[int]:
    """Graph, coefficient, location and innovation seeds of one dataset."""
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(int(master)).spawn(4)]


def cohort_master_seed(p: int, replicate: int, cohort_master: int = COHORT_MASTER) -> int:
    """Dataset master seed of the simulation cohort."""
    return int(np.random.SeedSequence([int(cohort_master), int(p), int(replicate)]).generate_state(1)[0])


def parse_cell(cell: str) -> tuple[int, int]:
    """``P50_N500_S1`` -> ``(50, 500)``."""
    try:
        p_part, n_part, snr_part = cell.split("_")
        assert p_part[0] == "P" and n_part[0] == "N" and snr_part == "S1"
        return int(p_part[1:]), int(n_part[1:])
    except (ValueError, AssertionError) as exc:
        raise ValueError(f"unrecognized cohort cell {cell!r}; expected e.g. P50_N500_S1") from exc


def calibrate_verified_scale(base, triangles, edge, sigma_raw, target_snr: float = 1.0):
    """Calibrate the global coefficient multiplier to the target SNR median.

    The multiplier is computed with order-5 quadrature and kept if a
    higher-order integration reproduces the target median within ``1e-7``;
    otherwise it is recalibrated at the higher order.  The full SNR vector
    must also agree with the next order within ``1e-3``.  Failure raises
    instead of redrawing the dataset.
    """
    multiplier, population = sem.calibrate_global_scale(base, triangles, edge, sigma_raw, target_snr, order=5)
    attempts = []
    high_order = None
    for order, check_order in ((5, 9), (9, 13), (13, 17)):
        if order != 5:
            population = high_order
            if abs(population.median - target_snr) >= 1e-7:
                multiplier, population = sem.calibrate_global_scale(
                    base, triangles, edge, sigma_raw, target_snr, order=order)
        high_order = sem.population_child_triangle_snr(
            base * multiplier, triangles, edge, sigma_raw, order=check_order)
        error = float(np.max(np.abs(population.values - high_order.values)))
        attempts.append(dict(order=order, check_order=check_order, max_difference=error,
                             median=population.median, check_median=high_order.median))
        if error < 1e-3 and abs(high_order.median - target_snr) < 1e-7:
            if population.covariance_min_eigenvalue <= 0 or high_order.covariance_min_eigenvalue <= 0:
                raise RuntimeError("population covariance not positive definite")
            if max(population.covariance_condition_max, high_order.covariance_condition_max) > sem.MAX_COVARIANCE_CONDITION:
                raise RuntimeError("population covariance condition gate failed")
            return multiplier, population, high_order, dict(order=order, check_order=check_order, attempts=attempts)
    raise RuntimeError(f"SNR quadrature failed without reseeding: {attempts}")


def _kl_arrays(mesh, kl_reference: Path | None, kl: dict | None, rscript: str | None):
    """Return ``(arrays, description)`` for the K-L design of the bundle."""
    if kl_reference is not None:
        kl_reference = Path(kl_reference)
        for name, value in (("mesh_points", mesh.points), ("mesh_triangles", mesh.triangles)):
            if not np.array_equal(np.load(kl_reference / f"{name}.npy", allow_pickle=False), value):
                raise ValueError("--kl-reference mesh does not match the generating mesh")
        A_kl = np.load(kl_reference / "A_kl.npy", allow_pickle=False)
        eigvals_path = kl_reference / "eigvals.npy"
        eigvals = np.load(eigvals_path, allow_pickle=False) if eigvals_path.is_file() else np.full(A_kl.shape[1], np.nan)
        hyper = {}
        for name in ("hyper_mu", "hyper_a", "hyper_b", "hyper_max_range", "hyper_L"):
            path = kl_reference / f"{name}.npy"
            if path.is_file():
                hyper[name] = np.load(path, allow_pickle=False)
        description = dict(source="reference", reference=str(kl_reference.resolve()),
                           A_kl_sha256=sha256_file(kl_reference / "A_kl.npy"), L=int(A_kl.shape[1]))
        arrays = dict(A_kl=A_kl, eigvals=eigvals, **hyper)
        arrays.setdefault("hyper_mu", np.array([0.0]))
        return arrays, description
    if kl is None:
        return None, dict(source="none")
    from spadag.spatial_gp import build_kl_design

    design = build_kl_design(mesh.points, L=kl["L"], a=kl["a"], b=kl["b"], max_range=kl["max_range"],
                             poly_degree=kl.get("poly_degree"), rscript=rscript)
    arrays = dict(A_kl=design.A, eigvals=design.eigvals, hyper_mu=np.array([0.0]),
                  hyper_a=np.array([float(kl["a"])]), hyper_b=np.array([float(kl["b"])]),
                  hyper_max_range=np.array([float(kl["max_range"])]), hyper_L=np.array([int(kl["L"])]))
    description = dict(source="bayesgpfit", **{k: kl[k] for k in ("L", "a", "b", "max_range", "poly_degree")})
    return arrays, description


def generate_dataset(
    out: Path,
    *,
    n: int,
    p: int,
    master_seed: int,
    config: PartitionConfig | None = None,
    target_snr: float = 1.0,
    kl_reference: Path | None = None,
    kl: dict | None = None,
    rscript: str | None = None,
    record: dict | None = None,
) -> dict:
    """Generate one validated bundle at ``out/bundle``; ``out`` must not exist."""
    start = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = cohort_config(p) if config is None else config
    if config.p != p:
        raise ValueError("config.p must equal p")
    mesh = build_regular_mesh(5)
    graph_seed, coefficient_seed, location_seed, innovation_seed = seeds_from_master(master_seed)
    gamma, edge, graph_report = draw_partitioned_graph(mesh.points, mesh.triangles, graph_seed, config)
    if not graph_report["target_reached"]:
        raise RuntimeError("graph failed its density target; the generator never reseeds")
    base = draw_coefficients(mesh.points, mesh.triangles, gamma, edge, coefficient_seed)
    sigma_raw = np.ones(p)
    multiplier, population, high_order, quadrature = calibrate_verified_scale(
        base, mesh.triangles, edge, sigma_raw, target_snr)
    raw_b = base * multiplier
    locations = np.random.default_rng(location_seed).uniform(0., 1., size=(n, 2))
    phi_sparse, tri_ids = basis_matrix(mesh, locations)
    phi = phi_sparse.toarray()
    assert np.all(tri_ids >= 0)
    X_raw, innovations_raw, beta_raw, orders = sem.simulate_sem(
        phi, tri_ids, edge, raw_b, sigma_raw, seed=innovation_seed)
    weights, _ = p1_integration_weights(mesh.points, mesh.triangles)
    std = sem.standardize_sem(X_raw, innovations_raw, raw_b, phi, weights, sigma_raw)
    std_snr = sem.population_child_triangle_snr(
        std["b_coeffs"], mesh.triangles, edge, std["sigma_noise"], order=quadrature["order"])
    q = np.full_like(base, 0.5)  # placeholder: this design has no true inclusion field
    q[np.arange(p), np.arange(p)] = 0
    arrays = dict(
        mesh_points=mesh.points, mesh_triangles=mesh.triangles,
        boundary_segments=mesh.constraints.segments, obs_points=locations,
        Phi_obs=phi, obs_triangle_id=tri_ids, q_node=q, eta_node=np.zeros_like(base),
        gamma=gamma.astype(np.int64), b_coeffs=std["b_coeffs"], beta_obs=std["beta_obs"],
        a0=std["a0"], a_spatial=std["a_spatial"], alpha_obs=std["alpha_obs"],
        X=std["X"], innovations=std["innovations"], sigma_noise=std["sigma_noise"],
        tau_alpha=np.ones(p), parents=np.ones((p, p), dtype=np.int32) - np.eye(p, dtype=np.int32),
        edge_triangle=edge.astype(np.int64), topological_order_triangle=orders,
        topological_order_obs=orders[tri_ids], integration_weights=weights,
        truth_defined_mask=~np.eye(p, dtype=bool), truth_q_defined=np.array([0], dtype=np.int8),
        base_b_coeffs_raw=base, b_coeffs_raw=raw_b, beta_obs_raw=beta_raw, X_raw=X_raw,
        innovations_raw=innovations_raw, sigma_noise_raw=sigma_raw,
        column_mean=std["column_mean"], column_sd=std["column_sd"],
        population_snr_values=population.values, population_snr_slot_indices=population.slot_indices,
        coefficient_multiplier=np.array([multiplier]),
        seed=np.array([master_seed], dtype=np.int64),
        location_seed=np.array([location_seed]), innovation_seed=np.array([innovation_seed]),
    )
    kl_arrays, kl_description = _kl_arrays(mesh, kl_reference, kl, rscript)
    if kl_arrays is not None:
        arrays.update(kl_arrays)
    bundle = out / "bundle"
    bundle.mkdir()
    for name, value in arrays.items():
        if name != "eigvals":
            assert np.all(np.isfinite(value)), name
        np.save(bundle / f"{name}.npy", value, allow_pickle=False)
    compelled = np.stack([dag_to_cpdag(g)[0] for g in edge])
    reconstruction_error = float(np.max(np.abs(
        std["X"] - std["alpha_obs"] - np.einsum("ijk,ik->ij", std["beta_obs"], std["X"]) - std["innovations"])))
    solved = np.linalg.solve(np.eye(p)[None] - beta_raw, innovations_raw[..., None])[..., 0]
    checks = dict(
        raw_linear_solve_error=float(np.max(np.abs(solved - X_raw))),
        standardized_reconstruction_error=reconstruction_error,
        standardization_snr_invariance_error=float(np.max(np.abs(population.values - std_snr.values))),
        quadrature_check_snr_median=high_order.median,
        quadrature_snr_max_difference=float(np.max(np.abs(population.values - high_order.values))),
        X_mean_max_abs=float(np.abs(std["X"].mean(axis=0)).max()),
        X_sd_max_deviation=float(np.abs(std["X"].std(axis=0) - 1).max()),
        phi_partition_error=float(np.abs(phi.sum(axis=1) - 1).max()),
        max_local_indegree=int(edge.sum(axis=2).max()),
        observations_per_triangle=np.bincount(tri_ids, minlength=len(edge)).tolist(),
    )
    assert abs(population.median - target_snr) < 1e-7
    assert checks["raw_linear_solve_error"] < 1e-10
    assert reconstruction_error < 1e-10 and checks["standardization_snr_invariance_error"] < 1e-9
    assert checks["X_mean_max_abs"] < 1e-10 and checks["X_sd_max_deviation"] < 1e-10
    assert checks["phi_partition_error"] < 1e-10
    assert checks["max_local_indegree"] <= 2
    assert checks["quadrature_snr_max_difference"] < 1e-3
    for motif in graph_report["reversal_motifs"]:
        assert compelled[motif["left_triangles"], motif["b"], motif["a"]].all()
        assert compelled[motif["right_triangles"], motif["a"], motif["b"]].all()
    metadata = dict(
        version=SCHEMA_VERSION, truth_schema_version=3,
        purpose="regional-reversal graph field with localized unshielded-collider reversals; not prior predictive",
        observation_design="iid Uniform unit square; one draw without count selection",
        n=n, p=p, H=len(mesh.points), T=len(edge), mesh_type="regular-unit-square", n_grid=5,
        q_truth_defined=False,
        q_node_role="schema placeholder, no q truth; never initialize inference from truth",
        target_population_snr=target_snr, achieved_population_snr_median=population.median,
        population_snr_quantiles=sem.quantile_summary(population.values), coefficient_multiplier=multiplier,
        snr_quadrature=quadrature, covariance_condition_max=population.covariance_condition_max,
        coefficient_rule="pair-shared equiprobable sign; active vertex iid Uniform(.5,1.5); inactive exact zero; one global multiplier calibrated to the population SNR median",
        raw_baseline=0, raw_noise_sd=1, standardization=dict(ddof=0),
        kl_design=kl_description,
        seeds=dict(master=int(master_seed), graph=graph_seed, coefficient=coefficient_seed,
                   observations=location_seed, innovations=innovation_seed),
        graph_validation=graph_report, validation=checks,
        compelled_triangle_edge_count=int(compelled.sum()),
        compelled_fraction=float(compelled.sum() / edge.sum()),
        reversal_motifs=reversal_motifs_for_evaluation(graph_report["reversal_motifs"]),
        reversal_motif_cores=graph_report["reversal_motifs"],
        validation_gates_passed=True,
    )
    if record is not None:
        metadata["cohort_dataset"] = record
    write_json(bundle / "metadata.json", metadata)
    if kl_arrays is not None:
        loaded = load_inference_data(bundle)
        checks["loader_candidate_pairs"] = len(loaded.pairs)
        assert len(loaded.pairs) == p * (p - 1)
        write_json(bundle / "metadata.json", metadata)
    (bundle / "generation_command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    _write_array_checksums(bundle)
    write_json(out / "generation_result.json", dict(
        status="generated_validated", seconds=time.perf_counter() - start, bundle=str(bundle.resolve()),
        graph=graph_report, validation=checks, snr_median=population.median,
        coefficient_multiplier=multiplier, posterior_mcmc_started=False))
    return metadata


def derive_prefix(parent_out: Path, out: Path, n: int, record: dict | None = None) -> dict:
    """Re-standardize the first ``n`` raw observations of a completed parent."""
    parent = Path(parent_out) / "bundle"
    metadata = json.loads((parent / "metadata.json").read_text(encoding="utf-8"))
    if not 0 < n < metadata["n"]:
        raise ValueError("prefix size must be smaller than the parent sample size")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    bundle = out / "bundle"
    bundle.mkdir()
    load = lambda name: np.load(parent / f"{name}.npy", allow_pickle=False)
    observation_arrays = {"obs_points", "Phi_obs", "obs_triangle_id", "beta_obs_raw",
                          "X_raw", "innovations_raw", "topological_order_obs"}
    std = sem.standardize_sem(load("X_raw")[:n], load("innovations_raw")[:n], load("b_coeffs_raw"),
                               load("Phi_obs")[:n], load("integration_weights"), load("sigma_noise_raw"))
    for path in parent.glob("*.npy"):
        name = path.stem
        if name in std:
            np.save(bundle / path.name, std[name], allow_pickle=False)
        elif name in observation_arrays:
            np.save(bundle / path.name, np.load(path, allow_pickle=False)[:n], allow_pickle=False)
        else:
            shutil.copyfile(path, bundle / path.name)
    metadata["n"] = n
    metadata["paired_parent"] = str(parent.resolve())
    metadata["observation_design"] = "first n of the same replicate's iid Uniform observations; re-standardized per n"
    metadata["validation"] = dict(parent_generation_checks="see paired parent; not subset-specific")
    if record is not None:
        metadata["cohort_dataset"] = record
    write_json(bundle / "metadata.json", metadata)
    (bundle / "generation_command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    _write_array_checksums(bundle)
    write_json(out / "generation_result.json", dict(status="derived_prefix", parent=str(parent.resolve()), n=n))
    return metadata


def verify_bundle(bundle: Path, target_snr: float = 1.0) -> dict:
    """Reload a saved bundle and validate the structural, SEM and numerical contracts."""
    bundle = Path(bundle)
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    p, n = metadata["p"], metadata["n"]
    load = lambda name: np.load(bundle / f"{name}.npy", allow_pickle=False)
    for path in bundle.glob("*.npy"):
        if path.stem != "eigvals" and not np.isfinite(np.load(path, allow_pickle=False)).all():
            raise ValueError(f"non-finite values in {path.name}")
    gamma, edge, triangles = load("gamma"), load("edge_triangle"), load("mesh_triangles")
    assert gamma.shape == (p, p, 36) and edge.shape == (50, p, p)
    assert np.isin(gamma, [0, 1]).all() and np.isin(edge, [0, 1]).all()
    assert not np.diagonal(gamma, axis1=0, axis2=1).any()
    assert np.array_equal(triangle_edge_tensor(gamma, triangles), edge)
    assert all(is_dag(g) for g in edge)
    assert not np.any(edge & edge.transpose(0, 2, 1))
    assert edge.sum(axis=2).max() <= 2
    union = edge.any(axis=0)
    reversal_count = int(np.triu(union & union.T, 1).sum())
    assert reversal_count == 2 * (p // 5) and not is_dag(union)
    compelled = np.stack([dag_to_cpdag(g)[0] for g in edge])
    for motif in metadata["reversal_motif_cores"]:
        assert compelled[motif["left_triangles"], motif["b"], motif["a"]].all()
        assert compelled[motif["right_triangles"], motif["a"], motif["b"]].all()
    braw, base = load("b_coeffs_raw"), load("base_b_coeffs_raw")
    assert np.all(base[gamma == 0] == 0) and np.all(braw[gamma == 0] == 0)
    assert np.all((abs(base[gamma != 0]) >= .5) & (abs(base[gamma != 0]) <= 1.5))
    for j, k in np.argwhere(gamma.any(axis=2)):
        assert np.unique(np.sign(base[j, k, gamma[j, k].astype(bool)])).size == 1
    assert np.array_equal(braw, base * load("coefficient_multiplier")[0])
    X, raw, eps = load("X"), load("X_raw"), load("innovations_raw")
    phi, beta_raw = load("Phi_obs"), load("beta_obs_raw")
    assert X.shape == (n, p) and phi.shape == (n, 36)
    assert np.allclose(phi.sum(1), 1, atol=1e-12, rtol=0)
    assert np.allclose(beta_raw, np.einsum("ih,jkh->ijk", phi, braw), atol=1e-12, rtol=0)
    linear_error = float(np.max(abs(np.linalg.solve(np.eye(p)[None] - beta_raw, eps[..., None])[..., 0] - raw)))
    std_error = float(np.max(abs(X - load("alpha_obs") - np.einsum("ijk,ik->ij", load("beta_obs"), X) - load("innovations"))))
    assert linear_error < 1e-10 and std_error < 1e-10
    assert np.max(abs(X.mean(0))) < 1e-10 and np.max(abs(X.std(0) - 1)) < 1e-10
    assert np.allclose(X, (raw - raw.mean(0)) / raw.std(0), atol=1e-12, rtol=0)
    quadrature = metadata["snr_quadrature"]
    snr_check = sem.population_child_triangle_snr(
        braw, triangles, edge, load("sigma_noise_raw"), order=quadrature["check_order"])
    snr_std = sem.population_child_triangle_snr(
        load("b_coeffs"), triangles, edge, load("sigma_noise"), order=quadrature["order"],
        compute_covariance_diagnostics=False)
    snr5 = load("population_snr_values")
    qerror = float(np.max(abs(snr_check.values - snr5)))
    std_snr_error = float(np.max(abs(snr_std.values - snr5)))
    assert abs(float(np.median(snr5)) - target_snr) < 1e-7
    assert qerror < 1e-3 and std_snr_error < 1e-9
    result = dict(passed=True, raw_linear_solve_error=linear_error, standardized_reconstruction_error=std_error,
                  quadrature_snr_max_difference=qerror, standardization_snr_invariance_error=std_snr_error,
                  reversal_pairs=reversal_count, gamma_sha256=sha256_file(bundle / "gamma.npy"))
    counts = np.bincount(load("obs_triangle_id"), minlength=50)
    result.update(empty_triangles=int((counts == 0).sum()),
                  observation_count_min=int(counts.min()), observation_count_max=int(counts.max()))
    if (bundle / "A_kl.npy").is_file():
        loaded = load_inference_data(bundle)
        assert loaded.n == n and loaded.p == p and len(loaded.pairs) == p * (p - 1)
        result["loader_candidate_pairs"] = len(loaded.pairs)
    return result


def _config_from_args(args: argparse.Namespace, p: int) -> PartitionConfig:
    base = cohort_config(p)
    overrides = {}
    if args.patch_sizes is not None:
        overrides["patch_sizes"] = tuple(int(s) for s in args.patch_sizes)
        probabilities = args.patch_probabilities
        if probabilities is None:
            probabilities = [1.0 / len(args.patch_sizes)] * len(args.patch_sizes)
        if len(probabilities) != len(args.patch_sizes) or abs(sum(probabilities) - 1.0) > 1e-12:
            raise ValueError("--patch-probabilities must match --patch-sizes and sum to one")
        overrides["patch_probabilities"] = tuple(float(x) for x in probabilities)
    if args.coverage is not None:
        overrides["coverage_min"], overrides["coverage_max"] = (float(x) for x in args.coverage)
    if args.max_attempts is not None:
        overrides["max_attempts"] = int(args.max_attempts)
    if args.target_mean_edges is not None:
        overrides["target_mean_edges"] = float(args.target_mean_edges)
    return PartitionConfig(**{**base.__dict__, **overrides})


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True,
                        help="new output directory; the bundle is written to OUT/bundle")
    design = parser.add_argument_group("dataset")
    design.add_argument("--cohort-cell", choices=COHORT_CELLS,
                        help="reproduce a simulation-cohort cell (requires --replicate)")
    design.add_argument("--replicate", type=int, help="cohort replicate index, 1-based")
    design.add_argument("--cohort-master", type=int, default=COHORT_MASTER)
    design.add_argument("--p", type=int, help="number of variables (multiple of 5, >= 10)")
    design.add_argument("--n", type=int, help="sample size")
    design.add_argument("--master-seed", type=int, help="dataset master seed")
    design.add_argument("--snr", type=float, default=1.0, help="target median population SNR")
    design.add_argument("--prefix-of", type=Path,
                        help="derive OUT as the re-standardized first --n observations of this completed output")
    graph = parser.add_argument_group("graph field")
    graph.add_argument("--patch-sizes", type=int, nargs="+")
    graph.add_argument("--patch-probabilities", type=float, nargs="+")
    graph.add_argument("--coverage", type=float, nargs=2, metavar=("MIN", "MAX"))
    graph.add_argument("--max-attempts", type=int)
    graph.add_argument("--target-mean-edges", type=float)
    kl = parser.add_argument_group("K-L design")
    kl.add_argument("--kl-reference", type=Path, help="bundle on the same mesh whose A_kl/eigvals are copied")
    kl.add_argument("--no-kl", action="store_true", help="omit A_kl; the bundle then needs spadag-prepare")
    kl.add_argument("--L", type=int, default=DEFAULT_KL["L"])
    kl.add_argument("--a", type=float, default=DEFAULT_KL["a"])
    kl.add_argument("--b", type=float, default=DEFAULT_KL["b"])
    kl.add_argument("--max-range", type=float, default=DEFAULT_KL["max_range"])
    kl.add_argument("--poly-degree", type=int, default=DEFAULT_KL["poly_degree"])
    kl.add_argument("--rscript")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new output directory")
    if args.prefix_of is not None:
        if args.n is None:
            raise SystemExit("--prefix-of requires --n")
        metadata = derive_prefix(args.prefix_of, args.out, args.n)
        print(json.dumps(dict(status="derived_prefix", n=metadata["n"], p=metadata["p"])), flush=True)
        return
    record = None
    if args.cohort_cell is not None:
        if args.replicate is None:
            raise SystemExit("--cohort-cell requires --replicate")
        p, n = parse_cell(args.cohort_cell)
        master = cohort_master_seed(p, args.replicate, args.cohort_master)
        parent_n = COHORT_PARENT_N[p]
        record = dict(cell=args.cohort_cell, p=p, n=n, replicate=args.replicate, master=master,
                      cohort_master=args.cohort_master, parent_n=parent_n)
        if n != parent_n:
            raise SystemExit(
                f"{args.cohort_cell} is a re-standardized prefix of the n={parent_n} dataset: generate "
                f"P{p}_N{parent_n}_S1 first, then run with --prefix-of <that output> --n {n}")
        if args.p not in (None, p) or args.n not in (None, n) or args.master_seed not in (None, master):
            raise SystemExit("--p/--n/--master-seed conflict with the requested cohort cell")
    else:
        if args.p is None or args.n is None or args.master_seed is None:
            raise SystemExit("provide --cohort-cell/--replicate, or --p, --n and --master-seed")
        p, n, master = args.p, args.n, args.master_seed
    config = _config_from_args(args, p)
    kl = None if (args.no_kl or args.kl_reference is not None) else dict(
        L=args.L, a=args.a, b=args.b, max_range=args.max_range, poly_degree=args.poly_degree)
    metadata = generate_dataset(args.out, n=n, p=p, master_seed=master, config=config, target_snr=args.snr,
                                kl_reference=args.kl_reference, kl=kl, rscript=args.rscript, record=record)
    checks = verify_bundle(args.out / "bundle", args.snr)
    print(json.dumps(dict(status="generated_validated", p=p, n=n, master_seed=master,
                          mean_local_edges=metadata["graph_validation"]["mean_local_edges"],
                          snr_median=metadata["achieved_population_snr_median"],
                          coefficient_multiplier=metadata["coefficient_multiplier"],
                          gamma_sha256=checks["gamma_sha256"], kl=metadata["kl_design"]["source"])), flush=True)


if __name__ == "__main__":
    main()
