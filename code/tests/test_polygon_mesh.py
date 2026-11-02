"""Tests for the observation-supported polygon mesh builder."""
from __future__ import annotations

import numpy as np
import pytest

from spadag.mesh.basis import basis_matrix, observation_support
from spadag.mesh.builders import MESH_OPTIONS, build_mesh, build_polygon_mesh
from spadag.mesh.core import Mesh
from spadag.mesh.polygon import (
    _containing_triangle,
    distance_to_boundary,
    estimate_observation_domain,
    median_nn_spacing,
    points_in_domain,
    validate_polygon_mesh,
)


def hex_lattice(width: float, height: float, spacing: float = 1.0) -> np.ndarray:
    """Visium-like lattice: rows offset by half a spacing."""
    dy = spacing * np.sqrt(3.0) / 2.0
    rows = []
    for j in range(int(height / dy) + 1):
        x = np.arange(0.0, width + 1e-9, spacing) + (spacing / 2.0) * (j % 2)
        rows.append(np.c_[x, np.full_like(x, j * dy)])
    return np.vstack(rows)


def l_shaped_tissue() -> np.ndarray:
    """L-shaped lattice with a few missing spots and two isolated spots."""
    P = hex_lattice(30.0, 20.0)
    P = P[~((P[:, 0] > 15.0) & (P[:, 1] > 10.0))]
    rng = np.random.default_rng(7)
    P = np.delete(P, rng.choice(P.shape[0], size=6, replace=False), axis=0)
    return np.vstack([P, [[34.0, 4.0], [-4.0, 16.0]]])


def annulus_tissue() -> np.ndarray:
    P = hex_lattice(20.0, 20.0, spacing=0.5) - 10.0
    r = np.linalg.norm(P, axis=1)
    return P[(r >= 4.0) & (r <= 9.5)]


def test_concave_domain_covers_all_points_and_fills_lattice_gaps() -> None:
    P = l_shaped_tissue()
    mesh, domain = build_polygon_mesh(P, target_tris=50, return_domain=True)

    assert domain.n_holes == 0  # missing single spots are not holes
    assert np.all(points_in_domain(domain, P))
    margin = 1.0 * median_nn_spacing(P)
    assert np.min(distance_to_boundary(domain, P)) >= 0.25 * margin * (1 - 1e-9)

    # The empty quadrant of the L is carved out; the full bounding box is not used.
    assert not points_in_domain(domain, np.array([[24.0, 16.0]]))[0]
    bbox_area = np.prod(P.max(axis=0) - P.min(axis=0))
    assert domain.area < 0.8 * bbox_area

    support = observation_support(mesh, P)
    assert support["n_outside"] == 0
    assert 25 <= mesh.triangles.shape[0] <= 100


def test_meaningful_hole_is_kept_and_not_meshed() -> None:
    P = annulus_tissue()  # empty disk of radius 4 inside
    mesh, domain = build_polygon_mesh(P, target_tris=60, return_domain=True)

    assert domain.n_holes == 1
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    probes = 1.5 * np.c_[np.cos(angles), np.sin(angles)]
    assert not np.any(points_in_domain(domain, np.vstack([[0.0, 0.0], probes])))
    assert np.all(points_in_domain(domain, P))
    _, tri_id = basis_matrix(mesh, P)
    assert np.all(tri_id >= 0)


def test_domain_scale_is_data_driven() -> None:
    P = l_shaped_tissue()
    c, shift = 3.7, np.array([100.0, -50.0])
    d1 = estimate_observation_domain(P)
    d2 = estimate_observation_domain(c * P + shift)
    assert d2.spacing == pytest.approx(c * d1.spacing, rel=1e-12)
    assert d2.area == pytest.approx(c**2 * d1.area, rel=1e-3)

    m1, dom1 = build_polygon_mesh(P, target_tris=50, return_domain=True)
    m2, dom2 = build_polygon_mesh(c * P + shift, target_tris=50, return_domain=True)
    assert dom2.area == pytest.approx(c**2 * dom1.area, rel=1e-2)
    assert abs(m1.triangles.shape[0] - m2.triangles.shape[0]) <= 5


def test_mesh_does_not_depend_on_observation_order() -> None:
    P = l_shaped_tissue()
    perm = np.random.default_rng(1).permutation(P.shape[0])
    m1 = build_polygon_mesh(P, target_tris=40)
    m2 = build_polygon_mesh(P[perm], target_tris=40)
    np.testing.assert_array_equal(m1.points, m2.points)
    np.testing.assert_array_equal(m1.triangles, m2.triangles)


def test_uniform_points_give_a_near_square_domain() -> None:
    P = np.random.default_rng(3).uniform(size=(800, 2))
    mesh, domain = build_polygon_mesh(P, target_tris=50, return_domain=True)
    assert domain.n_holes == 0
    assert 0.9 < domain.area < 1.2
    report = validate_polygon_mesh(mesh, domain, P, target_tris=50)
    assert 0.75 <= report["triangle_count_ratio"] <= 1.5


def test_validation_flags_broken_meshes() -> None:
    P = l_shaped_tissue()
    mesh, domain = build_polygon_mesh(P, target_tris=50, return_domain=True)

    areas = mesh.points[mesh.triangles].mean(axis=1)
    k = int(np.argmax(np.linalg.norm(areas - areas.mean(axis=0), axis=1)))
    holed = Mesh(
        points=mesh.points,
        triangles=np.delete(mesh.triangles, k, axis=0),
        constraints=mesh.constraints,
    )
    report = validate_polygon_mesh(holed, domain, P, raise_on_failure=False)
    assert not report["checks"]["inside_domain"]
    assert not report["checks"]["constraints_represented"]

    flipped_tris = mesh.triangles.copy()
    flipped_tris[0] = flipped_tris[0, ::-1]
    flipped = Mesh(points=mesh.points, triangles=flipped_tris, constraints=mesh.constraints)
    report = validate_polygon_mesh(flipped, domain, P, raise_on_failure=False)
    assert not report["checks"]["ccw"]
    assert not report["checks"]["consistent_connectivity"]
    with pytest.raises(ValueError, match="validation failed"):
        validate_polygon_mesh(flipped, domain, P)

    report = validate_polygon_mesh(mesh, domain, P, target_tris=10, raise_on_failure=False)
    assert not report["checks"]["triangle_count"]


def test_vertex_removal_merges_data_poor_triangles() -> None:
    P = l_shaped_tissue()
    base = build_polygon_mesh(P, target_tris=30)
    pruned, domain = build_polygon_mesh(P, target_tris=30, min_obs_per_triangle=10, return_domain=True)

    before = observation_support(base, P)["obs_per_triangle"]
    after = observation_support(pruned, P)
    assert before.min() == 0 and np.sum(before < 10) > 0
    assert after["obs_per_triangle"].min() >= 10
    assert after["n_outside"] == 0 and after["n_unsupported_nodes"] == 0
    assert np.all(points_in_domain(domain, P))
    assert pruned.triangles.shape[0] < base.triangles.shape[0]


def test_vertex_removal_is_off_by_default_and_a_no_op_when_satisfied() -> None:
    P = annulus_tissue()
    base = build_polygon_mesh(P, target_tris=40)
    same = build_polygon_mesh(P, target_tris=40, min_obs_per_triangle=None)
    np.testing.assert_array_equal(base.triangles, same.triangles)

    assert observation_support(base, P)["obs_per_triangle"].min() >= 10
    pruned, domain = build_polygon_mesh(P, target_tris=40, min_obs_per_triangle=10, return_domain=True)
    np.testing.assert_array_equal(base.points, pruned.points)
    np.testing.assert_array_equal(base.triangles, pruned.triangles)
    assert domain.n_holes == 1


def test_degenerate_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_observation_domain(np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError):
        estimate_observation_domain(np.c_[np.arange(10.0), 2.0 * np.arange(10.0)])


def test_heavy_triangle_splitting_caps_observations_per_triangle() -> None:
    P = l_shaped_tissue()
    base = build_polygon_mesh(P, target_tris=30)
    before = observation_support(base, P)["obs_per_triangle"]
    cap = int(np.quantile(before, 0.75))
    assert before.max() > cap
    split, domain = build_polygon_mesh(P, target_tris=30, max_obs_per_triangle=cap, return_domain=True)
    after = observation_support(split, P)
    # The cap uses the builder's point-location rule (shared-edge points go to
    # the lowest triangle index); observation_support may assign such points
    # differently, so compare with the same rule.
    found = _containing_triangle(split.points, split.triangles, P)
    assert np.bincount(found[found >= 0], minlength=split.triangles.shape[0]).max() <= cap
    assert after["obs_per_triangle"].max() <= cap + 3
    assert after["n_outside"] == 0
    assert split.triangles.shape[0] > base.triangles.shape[0]
    assert np.all(points_in_domain(domain, P))

    def n_obs_vertices(mesh):
        return sum(bool(np.any(np.all(np.isclose(P, q), axis=1))) for q in mesh.points)

    assert n_obs_vertices(split) > n_obs_vertices(base)


def test_heavy_triangle_splitting_is_a_no_op_when_satisfied_and_combines_with_coarsening() -> None:
    P = annulus_tissue()
    base = build_polygon_mesh(P, target_tris=40)
    cap = int(observation_support(base, P)["obs_per_triangle"].max())
    same = build_polygon_mesh(P, target_tris=40, max_obs_per_triangle=cap)
    np.testing.assert_array_equal(base.points, same.points)
    np.testing.assert_array_equal(base.triangles, same.triangles)
    both = build_polygon_mesh(P, target_tris=40, min_obs_per_triangle=10, max_obs_per_triangle=cap - 5)
    found = _containing_triangle(both.points, both.triangles, P)
    counts = np.bincount(found[found >= 0], minlength=both.triangles.shape[0])
    assert counts.min() >= 10 and counts.max() <= cap - 5


def test_build_mesh_dispatches_polygon_in_input_coordinates() -> None:
    assert "polygon" in MESH_OPTIONS
    P = l_shaped_tissue()
    mesh = build_mesh(P, mesh_type="polygon", target_tris=40)
    assert mesh.points.min(axis=0)[0] < 0.0 and mesh.points.max(axis=0)[0] > 30.0
    np.testing.assert_array_equal(mesh.triangles, build_polygon_mesh(P, target_tris=40).triangles)
