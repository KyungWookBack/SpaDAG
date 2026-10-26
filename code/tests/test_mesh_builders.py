"""The unit-square CDT reaches the requested resolution on uniform locations."""
from __future__ import annotations

import numpy as np
import pytest

from spadag.mesh.basis import basis_matrix
from spadag.mesh.builders import _boundary_points_per_edge, build_mesh
from spadag.mesh.core import missing_constraint_edges, triangle_areas


@pytest.mark.parametrize("n_grid", [3, 5, 7])
def test_cdt_matches_regular_resolution(n_grid: int) -> None:
    target = 2 * n_grid**2  # triangles of the regular n_grid x n_grid mesh
    for seed in range(3):
        X = np.random.default_rng(seed).uniform(size=(500, 2))
        mesh = build_mesh(X, mesh_type="cdt", target_tris=target, seed=seed)
        T, H = mesh.triangles.shape[0], mesh.points.shape[0]
        assert target <= T <= 1.05 * target + 1
        assert (n_grid + 1) ** 2 <= H <= (n_grid + 1) ** 2 + 3

        area = triangle_areas(mesh)
        assert np.all(area > 1e-9)  # counter-clockwise and non-degenerate
        assert np.sum(area) == pytest.approx(1.0, rel=1e-9)  # tiles [0,1]^2
        assert np.unique(mesh.triangles).size == H
        assert not missing_constraint_edges(mesh)
        _, tri_id = basis_matrix(mesh, X)
        assert np.all(tri_id >= 0)


def test_regular_mesh_and_explicit_boundary_count_are_unchanged() -> None:
    X = np.random.default_rng(0).uniform(size=(500, 2))
    regular = build_mesh(X, mesh_type="regular", n_grid=5)
    assert (regular.points.shape[0], regular.triangles.shape[0]) == (36, 50)

    four_per_edge = build_mesh(X, mesh_type="cdt", target_tris=50, n_boundary_per_edge=4)
    assert four_per_edge.points.shape[0] == 32  # 12 boundary + 20 interior vertices
    assert _boundary_points_per_edge(50, None) == 6
    assert _boundary_points_per_edge(20, None) == 4
