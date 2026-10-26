"""Tests for raw observation/location to inference-bundle preparation."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from spadag.mcmc_data import INFERENCE_STEMS, load_inference_data
from spadag.prepare import parse_args, prepare_inference_bundle
from spadag.mesh.builders import build_mesh, build_regular_mesh
from spadag.spatial_gp import KLDesign


def _observations_and_locations() -> tuple[np.ndarray, np.ndarray]:
    X = np.arange(15, dtype=float).reshape(5, 3) / 10.0
    locations = np.asarray(
        [
            [10.0, 100.0],
            [20.0, 100.0],
            [20.0, 140.0],
            [10.0, 140.0],
            [15.0, 120.0],
        ]
    )
    return X, locations


def test_regular_prepare_builds_a_valid_five_array_bundle() -> None:
    X, locations = _observations_and_locations()
    H = 9  # n_grid=2 -> (2 + 1)^2
    A_kl = np.column_stack([np.ones(H), np.arange(H, dtype=float)])

    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp) / "bundle"
        prepared = prepare_inference_bundle(
            X,
            locations,
            outdir,
            mesh_type="regular",
            n_grid=2,
            A_kl=A_kl,
        )
        data = load_inference_data(outdir)
        manifest = json.loads(
            (outdir / "preparation_manifest.json").read_text(encoding="utf-8")
        )

        assert all((outdir / f"{stem}.npy").is_file() for stem in INFERENCE_STEMS)
        assert (data.n, data.p, data.H, data.L) == (5, 3, 9, 2)
        assert len(data.triangles) == 8
        np.testing.assert_allclose(data.Phi.sum(axis=1), 1.0)
        np.testing.assert_allclose(data.mesh_points.min(axis=0), [10.0, 100.0])
        np.testing.assert_allclose(data.mesh_points.max(axis=0), [20.0, 140.0])
        np.testing.assert_array_equal(np.load(outdir / "obs_points.npy"), locations)
        assert prepared.data_fingerprint == data.data_fingerprint
        assert manifest["mesh"]["type"] == "regular"
        assert manifest["mesh"]["domain_source"] == "observation-bounding-box"
        assert manifest["kl"]["source"] == "user-supplied"
        v = 1.0 + np.arange(H) ** 2
        kl_manifest = manifest["kl"]
        assert kl_manifest["pointwise_variance_min"] == pytest.approx(v.min())
        assert kl_manifest["pointwise_variance_median"] == pytest.approx(np.median(v))
        assert kl_manifest["pointwise_variance_max"] == pytest.approx(v.max())
        assert kl_manifest["pointwise_variance_min_max_ratio"] == pytest.approx(
            v.min() / v.max()
        )


def test_prepare_can_generate_kl_and_use_a_custom_mesh() -> None:
    X, locations = _observations_and_locations()
    mesh_points = np.asarray(
        [[10.0, 100.0], [20.0, 100.0], [20.0, 140.0], [10.0, 140.0]]
    )
    mesh_triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    fake_kl = KLDesign(
        A=np.column_stack([np.ones(4), np.arange(4, dtype=float)]),
        eigvals=np.asarray([2.0, 1.0]),
        L=2,
        a=0.01,
        b=1.0,
        max_range=2.0,
        energy_kept=0.9,
    )

    with tempfile.TemporaryDirectory() as tmp, patch(
        "spadag.prepare.build_kl_design", return_value=fake_kl
    ) as mocked_kl:
        outdir = Path(tmp) / "custom"
        prepared = prepare_inference_bundle(
            X,
            locations,
            outdir,
            mesh_points=mesh_points,
            mesh_triangles=mesh_triangles,
            L=2,
        )
        data = load_inference_data(outdir)
        manifest = json.loads(
            (outdir / "preparation_manifest.json").read_text(encoding="utf-8")
        )

    mocked_kl.assert_called_once()
    assert prepared.mesh_type == "custom"
    assert (data.H, len(data.triangles), data.L) == (4, 2, 2)
    assert manifest["mesh"]["type"] == "custom"
    assert manifest["kl"]["source"] == "BayesGPfit"
    assert manifest["kl"]["energy_kept"] == 0.9


def test_prepare_rejects_misaligned_locations_and_existing_bundle() -> None:
    X, locations = _observations_and_locations()
    H = 4
    A_kl = np.column_stack([np.ones(H), np.arange(H, dtype=float)])
    try:
        prepare_inference_bundle(X, locations[:-1], Path("unused"), A_kl=A_kl)
    except ValueError as error:
        assert "align with X rows" in str(error)
    else:
        raise AssertionError("misaligned observation locations should fail")

    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp) / "bundle"
        prepare_inference_bundle(X, locations, outdir, mesh_type="regular", n_grid=1, A_kl=A_kl)
        try:
            prepare_inference_bundle(X, locations, outdir, mesh_type="regular", n_grid=1, A_kl=A_kl)
        except FileExistsError as error:
            assert "refusing to overwrite" in str(error)
        else:
            raise AssertionError("existing prepared files should not be overwritten")


def _unit_square_design(n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261027)
    locations = rng.uniform(0.02, 0.98, size=(n, 2))
    return rng.normal(size=(n, 3)), locations


def test_unit_domain_reproduces_the_simulation_meshes() -> None:
    X, locations = _unit_square_design()
    regular = build_regular_mesh(5)
    cdt = build_mesh(locations, mesh_type="cdt", target_tris=50, seed=0, min_interior=4,
                     n_boundary_per_edge=6, min_angle=10.0, use_area_constraint=True)

    with tempfile.TemporaryDirectory() as tmp:
        for name, expected in (("regular", regular), ("cdt", cdt)):
            outdir = Path(tmp) / name
            A_kl = np.ones((len(expected.points), 1))
            prepare_inference_bundle(X, locations, outdir, mesh_type=name,
                                     domain=(0.0, 1.0, 0.0, 1.0), target_tris=50, A_kl=A_kl)
            manifest = json.loads((outdir / "preparation_manifest.json").read_text(encoding="utf-8"))
            np.testing.assert_array_equal(np.load(outdir / "mesh_points.npy"), expected.points)
            np.testing.assert_array_equal(np.load(outdir / "mesh_triangles.npy"), expected.triangles)
            assert manifest["mesh"]["domain_source"] == "user-rectangle"
            assert manifest["mesh"]["coordinate_min"] == [0.0, 0.0]
            assert manifest["mesh"]["coordinate_max"] == [1.0, 1.0]

        # Without --domain the observation bounding box is meshed instead.
        prepare_inference_bundle(X, locations, Path(tmp) / "bbox", mesh_type="regular",
                                 A_kl=np.ones((36, 1)))
        bbox_points = np.load(Path(tmp) / "bbox" / "mesh_points.npy")
        np.testing.assert_allclose(bbox_points.min(axis=0), locations.min(axis=0))
        np.testing.assert_allclose(bbox_points.max(axis=0), locations.max(axis=0))


def test_domain_rejects_outside_observations_and_non_rectangular_meshes() -> None:
    X, locations = _observations_and_locations()
    A_kl = np.ones((9, 1))
    with pytest.raises(ValueError, match="outside the domain"):
        prepare_inference_bundle(X, locations, Path("unused"), mesh_type="regular", n_grid=2,
                                 domain=(10.0, 20.0, 100.0, 130.0), A_kl=A_kl)
    with pytest.raises(ValueError, match="xmin < xmax"):
        prepare_inference_bundle(X, locations, Path("unused"), mesh_type="regular", n_grid=2,
                                 domain=(20.0, 10.0, 100.0, 140.0), A_kl=A_kl)
    with pytest.raises(ValueError, match="unit-square builders"):
        prepare_inference_bundle(X, locations, Path("unused"), domain=(0.0, 30.0, 0.0, 200.0),
                                 A_kl=A_kl)

    required = ["--x", "X.npy", "--locations", "obs.npy", "--outdir", "bundle"]
    args = parse_args(required + ["--mesh-type", "cdt", "--domain", "0", "1", "0", "1"])
    assert args.domain == [0.0, 1.0, 0.0, 1.0]
    assert parse_args(required).domain is None
    for extra in (["--domain", "0", "1", "0", "1"],
                  ["--mesh-type", "regular", "--domain", "1", "0", "0", "1"]):
        with pytest.raises(SystemExit):
            parse_args(required + extra)


def test_prepare_cli_defaults_to_manuscript_settings_and_accepts_obs_points_alias() -> None:
    args = parse_args(
        [
            "--x",
            "X.npy",
            "--obs-points",
            "locations.npy",
            "--outdir",
            "bundle",
        ]
    )
    assert args.mesh_type == "polygon"
    assert args.n_grid == 5
    assert args.locations == Path("locations.npy")
    assert (args.L, args.max_range, args.poly_degree) == (66, 1.0, 10)
    parameters = inspect.signature(prepare_inference_bundle).parameters
    assert parameters["mesh_type"].default == "polygon"
    assert (parameters["L"].default, parameters["max_range"].default, parameters["poly_degree"].default) == (66, 1.0, 10)
