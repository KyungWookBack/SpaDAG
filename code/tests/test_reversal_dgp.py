"""Checks for the regional-reversal simulation DGP and its cohort reproduction."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from spadag.acyclic import is_dag, triangle_edge_tensor
from spadag.cpdag import dag_to_cpdag
from spadag.evaluation import evaluate_reversal_recovery
from spadag.sim.generate_reversal import (
    COHORT_MASTER,
    cohort_master_seed,
    derive_prefix,
    generate_dataset,
    parse_cell,
    seeds_from_master,
    verify_bundle,
)
from spadag.mesh.builders import build_regular_mesh
from spadag.sim.reversal_dgp import (
    PartitionConfig,
    cohort_config,
    draw_coefficients,
    draw_partitioned_graph,
    reversal_motifs_for_evaluation,
    smoothness_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_TABLE = REPO_ROOT / "artifacts" / "data" / "reversal_cohort_seeds.json"


def _cohort_dataset(cell: str, replicate: int) -> dict:
    table = json.loads(SEED_TABLE.read_text(encoding="utf-8"))
    assert table["cohort_master"] == COHORT_MASTER
    return next(d for d in table["datasets"] if d["cell"] == cell and d["replicate"] == replicate)


def _npy_sha256(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def test_cohort_seed_rule_matches_seed_table() -> None:
    table = json.loads(SEED_TABLE.read_text(encoding="utf-8"))
    assert len(table["datasets"]) == 150
    for dataset in table["datasets"]:
        master = cohort_master_seed(dataset["p"], dataset["replicate"])
        assert master == dataset["master_seed"]
        graph, coefficient, location, innovation = seeds_from_master(master)
        assert (graph, coefficient, location, innovation) == (
            dataset["graph_seed"], dataset["coefficient_seed"],
            dataset["location_seed"], dataset["innovation_seed"])
    assert parse_cell("P50_N250_S1") == (50, 250)
    with pytest.raises(ValueError):
        parse_cell("P50_N250")


@pytest.mark.parametrize("cell", ["P50_N500_S1", "P25_N500_S1", "P75_N500_S1"])
def test_graph_reproduces_cohort_gamma_hash(cell: str) -> None:
    dataset = _cohort_dataset(cell, 1)
    mesh = build_regular_mesh(5)
    gamma, edge, report = draw_partitioned_graph(
        mesh.points, mesh.triangles, dataset["graph_seed"], cohort_config(dataset["p"]))
    assert _npy_sha256(gamma.astype(np.int64)) == dataset["gamma_sha256"]
    assert report["target_reached"]
    assert report["mean_local_edges"] == pytest.approx(dataset["mean_local_edges"])
    assert len(report["realized_reversal_pairs"]) == dataset["reversal_pairs"] == 2 * (dataset["p"] // 5)


def test_graph_structure_matches_manuscript_rules() -> None:
    mesh = build_regular_mesh(5)
    p = 25
    gamma, edge, report = draw_partitioned_graph(mesh.points, mesh.triangles, 7, cohort_config(p))
    assert gamma.shape == (p, p, 36) and edge.shape == (50, p, p)
    assert np.array_equal(triangle_edge_tensor(gamma, mesh.triangles), edge)
    assert all(is_dag(graph) for graph in edge)
    assert not np.any(edge & edge.transpose(0, 2, 1))
    assert edge.sum(axis=2).max() <= 2
    union = edge.any(axis=0)
    assert not is_dag(union)
    assert int(np.triu(union & union.T, 1).sum()) == 10
    compelled = np.stack([dag_to_cpdag(graph)[0] for graph in edge])
    for motif in report["reversal_motifs"]:
        a, b, c = motif["a"], motif["b"], motif["c"]
        assert compelled[motif["left_triangles"], b, a].all()
        assert compelled[motif["right_triangles"], a, b].all()
        assert not set(motif["left_triangles"]) & set(motif["right_triangles"])
    sizes = [len(patch["vertices"]) for patch in report["accepted_background_patches"]]
    assert set(sizes) <= {2, 3, 4}
    assert all(.08 - 1e-12 <= patch["coverage"] <= .36 + 1e-12
               for patch in report["accepted_background_patches"])


def test_coefficients_share_sign_and_lie_in_uniform_support() -> None:
    mesh = build_regular_mesh(5)
    gamma, edge, _ = draw_partitioned_graph(mesh.points, mesh.triangles, 11, cohort_config(25))
    b = draw_coefficients(mesh.points, mesh.triangles, gamma, edge, 12)
    assert np.all(b[gamma == 0] == 0)
    active = b[gamma != 0]
    assert np.all((np.abs(active) >= .5) & (np.abs(active) <= 1.5))
    for j, k in np.argwhere(gamma.any(axis=2)):
        signs = np.sign(b[j, k, gamma[j, k].astype(bool)])
        assert np.unique(signs).size == 1
    assert np.array_equal(b, draw_coefficients(mesh.points, mesh.triangles, gamma, edge, 12))


@pytest.mark.parametrize("level,sizes", [("rough", {1}), ("smooth", {7, 8, 9})])
def test_smoothness_levels_change_only_patch_geometry(level: str, sizes: set[int]) -> None:
    mesh = build_regular_mesh(5)
    p = 25
    medium = cohort_config(p)
    config = smoothness_config(level, p)
    assert config.target_mean_edges == medium.target_mean_edges == .65 * p
    gamma, edge, report = draw_partitioned_graph(mesh.points, mesh.triangles, 3, config)
    _, _, medium_report = draw_partitioned_graph(mesh.points, mesh.triangles, 3, medium)
    assert report["target_reached"]
    assert report["label_permutation"] == medium_report["label_permutation"]
    assert report["reversal_motifs"] == medium_report["reversal_motifs"]
    assert {len(patch["vertices"]) for patch in report["accepted_background_patches"]} <= sizes
    assert smoothness_config("medium", p).patch_sizes == medium.patch_sizes
    with pytest.raises(ValueError):
        smoothness_config("flat", p)


def test_generator_rejects_unsupported_dimension_and_mesh() -> None:
    mesh = build_regular_mesh(5)
    with pytest.raises(ValueError):
        draw_partitioned_graph(mesh.points, mesh.triangles, 1, PartitionConfig(p=12))
    other = build_regular_mesh(4)
    with pytest.raises(ValueError):
        draw_partitioned_graph(other.points, other.triangles, 1, cohort_config(10))


def test_motif_metadata_feeds_spadag_evaluate() -> None:
    mesh = build_regular_mesh(5)
    gamma, edge, report = draw_partitioned_graph(mesh.points, mesh.triangles, 5, cohort_config(10))
    motifs = reversal_motifs_for_evaluation(report["reversal_motifs"])
    assert len(motifs) == 4
    perfect = evaluate_reversal_recovery(edge.astype(float), motifs)
    assert perfect["motifs_recovered"] == 4 and perfect["both_directions_recovery_rate"] == 1.0
    broken = edge.astype(float)
    first = motifs[0]
    broken[first["left"]["triangles"], first["left"]["child"], first["left"]["parent"]] = 0.0
    assert evaluate_reversal_recovery(broken, motifs)["motifs_recovered"] == 3


def test_generate_dataset_reproduces_cohort_scale_and_prefix(tmp_path: Path) -> None:
    dataset = _cohort_dataset("P25_N500_S1", 1)
    out = tmp_path / "p25"
    metadata = generate_dataset(out, n=500, p=25, master_seed=dataset["master_seed"])
    bundle = out / "bundle"
    assert metadata["coefficient_multiplier"] == pytest.approx(dataset["coefficient_multiplier"], abs=1e-12)
    assert metadata["achieved_population_snr_median"] == pytest.approx(1.0, abs=1e-7)
    assert hashlib.sha256((bundle / "gamma.npy").read_bytes()).hexdigest() == dataset["gamma_sha256"]
    assert not (bundle / "A_kl.npy").exists()
    checks = verify_bundle(bundle)
    assert checks["passed"] and checks["reversal_pairs"] == 10
    assert (bundle / "SHA256SUMS").is_file()
    assert metadata["reversal_motifs"][0]["left"]["triangles"] == metadata["reversal_motif_cores"][0]["left_triangles"]

    prefix = tmp_path / "p25_n200"
    derive_prefix(out, prefix, 200)
    raw = np.load(bundle / "X_raw.npy")[:200]
    X = np.load(prefix / "bundle" / "X.npy")
    np.testing.assert_allclose(X, (raw - raw.mean(0)) / raw.std(0), rtol=0, atol=1e-12)
    for name in ("gamma", "b_coeffs_raw", "edge_triangle"):
        np.testing.assert_array_equal(np.load(prefix / "bundle" / f"{name}.npy"), np.load(bundle / f"{name}.npy"))
    assert json.loads((prefix / "bundle" / "metadata.json").read_text(encoding="utf-8"))["n"] == 200
    assert verify_bundle(prefix / "bundle")["passed"]
    with pytest.raises(FileExistsError):
        generate_dataset(out, n=500, p=25, master_seed=dataset["master_seed"])
