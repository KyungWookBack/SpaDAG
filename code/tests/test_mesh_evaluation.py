"""Location-based recovery evaluation across differing meshes."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from spadag.mesh_evaluation import (
    evaluate_at_locations,
    locate,
    location_recovery,
    overlap_counts,
    relaxed_f1_ceiling,
    sobol_evaluation_points,
)
from spadag.metrics import triangle_edge_recovery
from spadag.mesh.builders import build_regular_mesh
from toy_data_fixture import write_tiny_toy_bundle

CODE_DIR = Path(__file__).resolve().parents[1]


def test_sobol_points_are_reproducible_and_in_unit_square() -> None:
    points = sobol_evaluation_points(6, 20261001)
    assert points.shape == (64, 2)
    assert np.all((points >= 0) & (points < 1))
    np.testing.assert_array_equal(points, sobol_evaluation_points(6, 20261001))
    assert not np.array_equal(points, sobol_evaluation_points(6, 1))


def test_locate_matches_matplotlib_trifinder() -> None:
    mtri = pytest.importorskip("matplotlib.tri")
    mesh = build_regular_mesh(5)
    rng = np.random.default_rng(0)
    query = rng.uniform(0.0, 1.0, size=(2000, 2))
    expected = mtri.Triangulation(*mesh.points.T, triangles=mesh.triangles).get_trifinder()(*query.T)
    np.testing.assert_array_equal(locate(mesh.points, mesh.triangles, query), expected)
    with pytest.raises(ValueError):
        locate(mesh.points, mesh.triangles, np.array([[1.5, 0.5]]))


def test_overlap_and_recovery_use_distinct_location_weights() -> None:
    # Two inference triangles, two truth triangles, one directed edge column.
    fit_ids = np.array([0, 0, 0, 1])
    true_ids = np.array([0, 0, 0, 1])
    overlap = overlap_counts(fit_ids, true_ids, 2, 2)
    np.testing.assert_array_equal(overlap, np.diag([3, 1]))
    truth = np.array([[1], [1]])
    selected = np.array([[True], [False]])
    result = location_recovery(selected, overlap, truth)
    assert result["TP"] == 3 and result["FN"] == 1 and result["FP"] == 0
    assert result["TPR"] == pytest.approx(0.75) and result["FDR"] == pytest.approx(0.0)
    assert result["F1"] == pytest.approx(6 / 7)
    uniform = overlap_counts(np.array([0, 1]), np.array([0, 1]), 2, 2)
    assert location_recovery(selected, uniform, truth)["TPR"] == pytest.approx(0.5)
    empty = location_recovery(np.zeros((2, 1), dtype=bool), overlap, np.zeros((2, 1), dtype=int))
    assert empty["TPR"] is None and empty["FDR"] is None and empty["F1"] is None


def test_relaxed_ceiling_is_exact_optimum() -> None:
    # Inference triangle 0 straddles truth triangles 0 (edge present) and 1 (absent).
    overlap = np.array([[3, 1], [0, 2]])
    truth = np.array([[1], [0]])
    # Selecting fit triangle 0 only: TP 3, FP 1, FN 0 -> F1 = 6/7.
    assert relaxed_f1_ceiling(overlap, truth) == pytest.approx(6 / 7)
    assert relaxed_f1_ceiling(np.eye(2, dtype=int), truth) == pytest.approx(1.0)
    assert relaxed_f1_ceiling(overlap, np.zeros((2, 1), dtype=int)) is None


def test_identical_meshes_give_ceiling_one_and_triangle_metrics_at_vertex_locations() -> None:
    mesh = build_regular_mesh(5)
    p = 3
    rng = np.random.default_rng(1)
    edge_true = (rng.random((50, p, p)) < 0.3).astype(int)
    for t in range(50):
        np.fill_diagonal(edge_true[t], 0)
    edge_prob = np.clip(edge_true + rng.normal(0, 0.3, edge_true.shape), 0, 1)
    for t in range(50):
        np.fill_diagonal(edge_prob[t], 0.0)
    centroids = mesh.points[mesh.triangles].mean(axis=1)
    result = evaluate_at_locations(edge_prob, (mesh.points, mesh.triangles), (mesh.points, mesh.triangles),
                                   edge_true, centroids)
    assert result["representational_f1_ceiling"] == pytest.approx(1.0)
    assert result["n_locations"] == 50 and result["empty_fit_triangles"] == 0
    triangle = triangle_edge_recovery(edge_prob, edge_true, thresh=0.5)
    for key in ("TPR", "FDR", "F1"):
        assert result[key] == pytest.approx(triangle[key.lower()])


def test_coarser_inference_mesh_has_lower_ceiling() -> None:
    truth_mesh = build_regular_mesh(5)
    fit_mesh = build_regular_mesh(2)
    p = 2
    edge_true = np.zeros((50, p, p), dtype=int)
    edge_true[:25, 1, 0] = 1  # the edge exists on half of the truth triangles only
    edge_prob = np.zeros((len(fit_mesh.triangles), p, p))
    points = sobol_evaluation_points(10)
    result = evaluate_at_locations(edge_prob, (fit_mesh.points, fit_mesh.triangles),
                                   (truth_mesh.points, truth_mesh.triangles), edge_true, points)
    assert result["TPR"] == 0.0 and result["FDR"] is None and result["F1"] == 0.0
    assert 0.0 < result["representational_f1_ceiling"] < 1.0
    assert result["fit_triangles"] == 8 and result["truth_triangles"] == 50


def test_evaluate_cli_reports_sobol_and_observed(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    write_tiny_toy_bundle(bundle)
    np.save(bundle / "obs_points.npy", np.array([[0.75, 0.25]] * 9 + [[0.25, 0.75]] * 3))
    edge_true = np.load(bundle / "edge_triangle.npy")
    result = tmp_path / "result.npz"
    np.savez(result, edge_prob=edge_true.astype(float))
    out = tmp_path / "summary.json"
    subprocess.run(
        [sys.executable, "-m", "spadag.evaluation", "--result", str(result), "--data", str(bundle),
         "--truth", str(bundle), "--eval-points", "both", "--sobol-m", "8", "--out", str(out)],
        check=True, cwd=CODE_DIR, capture_output=True, text=True)
    summary = json.loads(out.read_text(encoding="utf-8"))["location_recovery"]
    assert set(summary) == {"sobol", "observed"}
    assert summary["sobol"]["n_locations"] == 256 and summary["observed"]["n_locations"] == 12
    for kind in ("sobol", "observed"):
        assert summary[kind]["F1"] == pytest.approx(1.0)
        assert summary[kind]["representational_f1_ceiling"] == pytest.approx(1.0)
    triangle_run = subprocess.run(
        [sys.executable, "-m", "spadag.evaluation", "--result", str(result), "--data", str(bundle),
         "--truth", str(bundle)], check=True, cwd=CODE_DIR, capture_output=True, text=True)
    assert "triangle_recovery" in triangle_run.stdout and "location_recovery" not in triangle_run.stdout
