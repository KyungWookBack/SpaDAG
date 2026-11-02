"""Mesh builders for the three mesh types: ``regular`` and ``cdt`` on the unit
square, and ``polygon`` on a domain estimated from the observations.
"""
from __future__ import annotations

import numpy as np

from spadag.mesh.core import (
    Mesh,
    assert_no_degenerate_triangles,
    build_unit_square_data_adaptive_mesh,
    build_unit_square_regular_triangulation,
)
from spadag.mesh.polygon import build_polygon_data_adaptive_mesh, validate_polygon_mesh


MESH_OPTIONS = ("polygon", "regular", "cdt")


def _boundary_points_per_edge(target_tris: int, n_boundary_per_edge: int | None) -> int:
    """
    Boundary samples per unit-square edge for the ``cdt`` builder.

    The default matches the regular n_grid x n_grid mesh with the same number
    of triangles T = 2 * n_grid^2, i.e. n_grid + 1 samples per edge, so both
    meshes have about the same number of vertices (T = 50: 6 per edge, 36
    vertices).
    """
    if n_boundary_per_edge is not None:
        return int(n_boundary_per_edge)
    return max(2, int(np.round(np.sqrt(max(int(target_tris), 1) / 2.0))) + 1)


def build_data_adaptive_mesh(
    obs_points: np.ndarray,
    target_tris: int = 20,
    seed: int = 0,
    eps: float = 1e-6,
    min_interior: int = 4,
    n_boundary_per_edge: int | None = None,
    min_angle: float = 10.0,
    use_area_constraint: bool = True,
) -> Mesh:
    """
    Build the ``cdt`` mesh: a data-adaptive constrained Delaunay triangulation of [0,1]^2.

    Interior vertices are observation points chosen by farthest-point
    sampling; see `spadag.mesh.core.build_unit_square_data_adaptive_mesh`.

    Parameters
    ----------
    obs_points : np.ndarray
        Observation coordinates of shape (n, 2).
    seed : int, optional
        Seed for the random candidates used only when fewer than 10
        observations lie inside the square.
    eps : float, optional
        Distance from the boundary below which an observation is not used as
        an interior vertex.
    n_boundary_per_edge : int, optional
        Boundary samples per unit-square edge, including endpoints. By default
        round(sqrt(target_tris / 2)) + 1, matching the regular mesh with the
        same triangle count.
    use_area_constraint : bool, optional
        Cap triangle area at twice the mean area for `target_tris`.
    """
    mesh = build_unit_square_data_adaptive_mesh(
        obs_points,
        target_tris=target_tris,
        seed=seed,
        eps=eps,
        min_interior=min_interior,
        n_boundary_per_edge=_boundary_points_per_edge(target_tris, n_boundary_per_edge),
        min_angle=min_angle,
        use_area_constraint=use_area_constraint,
    )
    assert_no_degenerate_triangles(mesh, eps=1e-14)
    return mesh


def build_polygon_mesh(
    obs_points: np.ndarray,
    target_tris: int = 50,
    *,
    hull_scale: float = 4.0,
    margin_scale: float = 1.0,
    boundary_tol: float = 0.5,
    min_hole_area: float | None = None,
    min_interior: int = 4,
    min_angle: float = 10.0,
    use_area_constraint: bool = True,
    max_area_factor: float = 2.0,
    min_obs_per_triangle: int | None = None,
    max_obs_per_triangle: int | None = None,
    refine_rounds: int = 3,
    return_domain: bool = False,
):
    """
    Build and validate a CDT mesh on an observation-supported polygon domain.

    The domain is estimated from the observation coordinates (concave hull
    plus a margin proportional to the median nearest-neighbour spacing), so
    the mesh is in the input coordinates rather than on [0,1]^2. See
    `spadag.mesh.core.build_polygon_data_adaptive_mesh` for the parameters;
    `min_obs_per_triangle` (off by default) deletes vertices around triangles
    holding fewer observations than that, keeping every observation inside;
    `max_obs_per_triangle` (off by default) first splits triangles holding more
    observations than that by adding observation vertices.

    The result is checked with `validate_polygon_mesh` (observations covered,
    no degenerate or clockwise triangles, consistent connectivity, constraint
    segments present, no triangle outside the domain or inside a hole, and a
    triangle count within a factor of two of `target_tris`); a failed check
    raises ValueError.

    Returns
    -------
    Mesh, or (Mesh, PolygonDomain) when `return_domain` is True.
    """
    mesh, domain = build_polygon_data_adaptive_mesh(
        obs_points,
        target_tris=target_tris,
        hull_scale=hull_scale,
        margin_scale=margin_scale,
        boundary_tol=boundary_tol,
        min_hole_area=min_hole_area,
        min_interior=min_interior,
        min_angle=min_angle,
        use_area_constraint=use_area_constraint,
        max_area_factor=max_area_factor,
        min_obs_per_triangle=min_obs_per_triangle,
        max_obs_per_triangle=max_obs_per_triangle,
        refine_rounds=refine_rounds,
        return_domain=True,
    )
    assert_no_degenerate_triangles(mesh, eps=1e-14)
    validate_polygon_mesh(mesh, domain, obs_points, target_tris=target_tris)
    return (mesh, domain) if return_domain else mesh


def build_regular_mesh(n_grid: int = 5) -> Mesh:
    """
    Build a regular right-isosceles triangulation on an n_grid x n_grid grid.

    Produces (n_grid + 1)^2 vertices and 2 * n_grid^2 triangles.
    """
    mesh = build_unit_square_regular_triangulation(n_grid=n_grid)
    assert_no_degenerate_triangles(mesh, eps=1e-14)
    return mesh


def build_mesh(
    obs_points: np.ndarray,
    mesh_type: str = "cdt",
    *,
    target_tris: int = 20,
    seed: int = 0,
    eps: float = 1e-6,
    min_interior: int = 4,
    n_boundary_per_edge: int | None = None,
    min_angle: float = 10.0,
    use_area_constraint: bool = True,
    n_grid: int = 5,
) -> Mesh:
    """
    Build a mesh of the requested type.

    Parameters
    ----------
    obs_points : np.ndarray
        Observation coordinates of shape (n, 2); ``regular`` ignores them.
    mesh_type : str, optional
        One of ``MESH_OPTIONS``.  ``regular`` and ``cdt`` mesh [0,1]^2;
        ``polygon`` meshes a domain estimated from the observations.
    target_tris : int, optional
        Target triangle count for ``cdt`` and ``polygon``.
    seed : int, optional
        Random seed for ``cdt`` interior-vertex selection.
    eps : float, optional
        Tolerance for treating an observation as interior to [0,1]^2.
    min_interior : int, optional
        Minimum number of interior vertices.
    n_boundary_per_edge : int, optional
        ``cdt`` boundary samples per unit-square edge; by default
        round(sqrt(target_tris / 2)) + 1, the regular mesh with the same T.
    min_angle : float, optional
        Minimum angle passed to Triangle's quality refinement.
    use_area_constraint : bool, optional
        Whether to impose a maximum triangle area derived from ``target_tris``.
    n_grid : int, optional
        Cells per axis of the ``regular`` mesh.
    """
    if mesh_type == "cdt":
        return build_data_adaptive_mesh(
            obs_points,
            target_tris=target_tris,
            seed=seed,
            eps=eps,
            min_interior=min_interior,
            n_boundary_per_edge=n_boundary_per_edge,
            min_angle=min_angle,
            use_area_constraint=use_area_constraint,
        )
    if mesh_type == "regular":
        return build_regular_mesh(n_grid=n_grid)
    if mesh_type == "polygon":
        return build_polygon_mesh(
            obs_points,
            target_tris=target_tris,
            min_interior=min_interior,
            min_angle=min_angle,
            use_area_constraint=use_area_constraint,
        )
    raise ValueError(f"Unknown mesh_type: {mesh_type!r}. Choose from {MESH_OPTIONS}.")
