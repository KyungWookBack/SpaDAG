"""Rank-deficient (wide) K-L designs in the sampler and in bundle preparation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from spadag.cli import parse_args, run_shared_gp_pg
from spadag.mcmc_data import load_inference_data
from spadag.p1_geometry import p1_integration_weights
from spadag.prepare import parse_args as parse_prepare_args
from spadag.prepare import prepare_inference_bundle
from spadag.shared_gp_pg import build_gp_decomposition
from spadag.spatial_gp import KLDesign
from toy_data_fixture import temporary_toy_bundle, write_tiny_toy_bundle


def _fixed_positive_omega(eta):
    return np.full_like(np.asarray(eta, dtype=float), 0.25)


def _wide_design(H: int, L: int, seed: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(H, L))


def test_sum_to_zero_decomposition_accepts_wide_design() -> None:
    with temporary_toy_bundle(include_truth=False) as bundle:
        A = _wide_design(4, 6)
        np.save(bundle / "A_kl.npy", A)
        data = load_inference_data(bundle)
        assert (data.H, data.L) == (4, 6)
        assert np.linalg.matrix_rank(A) < data.L
        decomposition = build_gp_decomposition(data)
    Q = decomposition.common_basis
    weights, _ = p1_integration_weights(data.mesh_points, data.triangles)
    assert Q.shape == (6, 5)
    np.testing.assert_allclose(Q.T @ Q, np.eye(5), atol=1e-10)
    np.testing.assert_allclose((weights @ A) @ Q, 0.0, atol=1e-10)
    np.testing.assert_allclose(decomposition.integration_weights, weights)


def test_full_rank_design_unchanged_and_zero_free_coefficient_rejected() -> None:
    with temporary_toy_bundle(include_truth=False) as bundle:
        data = load_inference_data(bundle)
        full = build_gp_decomposition(data)
        assert full.common_basis.shape == (2, 1)
        np.save(bundle / "A_kl.npy", np.ones((4, 1)))
        single = load_inference_data(bundle)
        with pytest.raises(ValueError, match="at least one free coefficient"):
            build_gp_decomposition(single)


def test_sampler_runs_with_more_modes_than_vertices() -> None:
    with temporary_toy_bundle(include_truth=False) as bundle:
        np.save(bundle / "A_kl.npy", _wide_design(4, 6))
        data = load_inference_data(bundle)
        args = parse_args([
            "--data", str(bundle), "--storage-profile", "legacy",
            "--iters", "4", "--burn", "1", "--thin", "1",
            "--log-every", "0", "--seed", "3",
        ])
        args.common_gp_var = 1.0
        args.edge_gp_var = 1.0
        result = run_shared_gp_pg(data, args, draw_omega=_fixed_positive_omega)
    assert np.all(np.isfinite(result["edge_prob"]))
    assert result["edge_prob"].shape == (2, 3, 3)
    assert np.all(np.isfinite(result["common_gp_var_full_trace"]))


def _lattice(n_side: int = 12) -> np.ndarray:
    grid = np.linspace(0.0, 10.0, n_side)
    return np.array([[x, y] for x in grid for y in grid])


def test_prepare_accepts_rank_deficient_kl_and_records_rank() -> None:
    X = np.random.default_rng(0).normal(size=(144, 3))
    locations = _lattice()
    H = 9  # n_grid=2
    A_kl = _wide_design(H, 12, seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp) / "bundle"
        prepared = prepare_inference_bundle(X, locations, outdir, mesh_type="regular", n_grid=2, A_kl=A_kl)
        manifest = json.loads((outdir / "preparation_manifest.json").read_text(encoding="utf-8"))
        data = load_inference_data(outdir)
    assert prepared.L == 12 and data.L == 12
    assert manifest["kl"]["rank"] == 9
    assert manifest["kl"]["poly_degree"] is None


def test_prepare_passes_poly_degree_to_kl_generation() -> None:
    X = np.random.default_rng(0).normal(size=(144, 3))
    locations = _lattice()
    H = 9
    fake = KLDesign(A=_wide_design(H, 12, seed=2), eigvals=np.ones(12), L=12, a=0.01, b=1.0,
                    max_range=1.0, energy_kept=0.5)
    with tempfile.TemporaryDirectory() as tmp, patch(
        "spadag.prepare.build_kl_design", return_value=fake
    ) as mocked:
        outdir = Path(tmp) / "bundle"
        prepare_inference_bundle(X, locations, outdir, mesh_type="regular", n_grid=2, L=12, max_range=1.0, poly_degree=10)
        manifest = json.loads((outdir / "preparation_manifest.json").read_text(encoding="utf-8"))
    assert mocked.call_args.kwargs["poly_degree"] == 10
    assert manifest["kl"]["poly_degree"] == 10 and manifest["kl"]["rank"] == 9


def test_prepare_builds_polygon_mesh_in_input_coordinates() -> None:
    rng = np.random.default_rng(4)
    locations = rng.uniform(0.0, 50.0, size=(600, 2)) + np.array([100.0, -20.0])
    X = rng.normal(size=(600, 3))
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp) / "bundle"
        with patch("spadag.prepare.build_kl_design") as mocked:
            def fake_design(points, **kwargs):
                return KLDesign(A=_wide_design(len(points), 66, seed=3), eigvals=np.ones(66), L=66,
                                a=0.01, b=1.0, max_range=1.0, energy_kept=0.5)
            mocked.side_effect = fake_design
            prepared = prepare_inference_bundle(
                X, locations, outdir, mesh_type="polygon", target_tris=40,
                min_obs_per_triangle=5, max_obs_per_triangle=40, L=66, max_range=1.0, poly_degree=10)
        manifest = json.loads((outdir / "preparation_manifest.json").read_text(encoding="utf-8"))
        data = load_inference_data(outdir)
        obs = np.load(outdir / "obs_points.npy")
    assert prepared.mesh_type == "polygon" and manifest["mesh"]["type"] == "polygon"
    assert manifest["mesh"]["domain_source"] == "observation-polygon"
    polygon = manifest["mesh"]["polygon"]
    assert polygon["n_holes"] == 0 and polygon["min_obs_per_triangle"] == 5
    assert data.mesh_points.min(axis=0)[0] >= 99.0 and data.mesh_points.max(axis=0)[1] <= 31.0
    assert data.L == 66 and manifest["kl"]["rank"] == data.H
    counts = np.bincount(np.argmax(data.Phi > 0, axis=1), minlength=data.H)
    assert np.all(data.Phi.sum(axis=1) > 0.999)
    np.testing.assert_array_equal(obs, locations)


def test_prepare_cli_accepts_polygon_and_kl_options() -> None:
    args = parse_prepare_args([
        "--x", "X.npy", "--locations", "obs.npy", "--outdir", "out",
        "--mesh-type", "polygon", "--target-tris", "75", "--no-area-constraint",
        "--min-obs-per-triangle", "6", "--max-obs-per-triangle", "40",
        "--L", "66", "--max-range", "1", "--poly-degree", "10",
    ])
    assert args.mesh_type == "polygon" and args.poly_degree == 10
    assert (args.min_obs_per_triangle, args.max_obs_per_triangle, args.refine_rounds) == (6, 40, 3)
    with pytest.raises(SystemExit):
        parse_prepare_args(["--x", "X.npy", "--locations", "obs.npy", "--outdir", "out",
                            "--min-obs-per-triangle", "50", "--max-obs-per-triangle", "40"])
