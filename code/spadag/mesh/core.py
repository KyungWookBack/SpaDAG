"""Triangular meshes of the unit square.

Provides the Mesh container, the regular and the data-adaptive constrained
Delaunay triangulations of the unit square, and geometry helpers.  Meshes on
a polygon domain estimated from observation coordinates are in
:mod:`spadag.mesh.polygon`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import triangle as tr
except Exception:
    tr = None


@dataclass(frozen=True)
class Constraints:
    """Constraint segments as pairs of vertex indices (undirected)."""

    segments: np.ndarray  # (m, 2) int


@dataclass(frozen=True)
class Mesh:
    points: np.ndarray  # (Nv, 2) float
    triangles: np.ndarray  # (Nt, 3) int, CCW
    constraints: Optional[Constraints] = None


def orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """2D orientation (signed area * 2): >0 if a,b,c are CCW."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def ensure_ccw(tri: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Return tri with CCW orientation."""
    a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
    if orient2d(a, b, c) > 0:
        return np.asarray(tri, dtype=int)
    return np.array([tri[0], tri[2], tri[1]], dtype=int)


def triangle_areas(mesh: Mesh) -> np.ndarray:
    """Signed triangle areas."""
    pts = np.asarray(mesh.points, dtype=float)
    tris = np.asarray(mesh.triangles, dtype=int)
    a = pts[tris[:, 0]]
    b = pts[tris[:, 1]]
    c = pts[tris[:, 2]]
    return 0.5 * (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )


def assert_no_degenerate_triangles(mesh: Mesh, eps: float = 1e-14) -> None:
    """Raise if mesh contains triangles with near-zero area."""
    area = triangle_areas(mesh)
    bad = np.where(np.abs(area) <= eps)[0]
    if bad.size:
        raise ValueError(
            f"Degenerate triangles found (|area|<=eps). "
            f"First indices: {bad[:10].tolist()}"
        )


def _farthest_points_from_boundary(
    candidates: np.ndarray, boundary_dist: np.ndarray, k: int
) -> np.ndarray:
    """Greedy max-min selection that treats the boundary as already chosen."""
    d = np.asarray(boundary_dist, dtype=float).copy()
    chosen = []
    for _ in range(min(int(k), candidates.shape[0])):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(candidates - candidates[i], axis=1))
    return candidates[chosen]


def unit_square_boundary_pslg(n_per_edge: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return boundary vertices + segments for [0,1]^2.
    n_per_edge includes both endpoints.

    - n_per_edge=2 -> corners only
    - n_per_edge=4 -> {0, 1/3, 2/3, 1} on each edge
    """
    if n_per_edge < 2:
        raise ValueError("n_per_edge must be >= 2")

    t = np.linspace(0.0, 1.0, n_per_edge)

    bottom = np.c_[t, np.zeros_like(t)]
    right = np.c_[np.ones_like(t), t]
    top = np.c_[t[::-1], np.ones_like(t)]
    left = np.c_[np.zeros_like(t), t[::-1]]

    # Stitch without duplicating corners.
    boundary = np.vstack([bottom, right[1:], top[1:], left[1:-1]])
    m = boundary.shape[0]
    segs = np.c_[np.arange(m), np.roll(np.arange(m), -1)].astype(int)
    return boundary.astype(float), segs


def _triangle_zero_base_if_needed(B: dict) -> dict:
    """Shift triangle and segment indices to 0-based if Triangle returned 1-based ones."""
    V = B.get("vertices")
    if V is None:
        return B
    nV = int(np.asarray(V).shape[0])

    def fix(arr_name: str) -> None:
        if arr_name not in B:
            return
        A = np.asarray(B[arr_name])
        if A.size == 0:
            return
        amin, amax = int(A.min()), int(A.max())
        if amin == 1 and amax == nV and not np.any(A == 0):
            B[arr_name] = (A - 1).astype(int)

    fix("triangles")
    fix("segments")
    return B


def triangulate_cdt_pslg(
    vertices: np.ndarray,
    segments: np.ndarray,
    *,
    holes: Optional[np.ndarray] = None,
    min_angle: float = 0.0,
    max_area: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constrained Delaunay triangulation (CDT) of a PSLG with Triangle.

    Segments are always enforced; `min_angle` > 0 adds quality refinement and
    `max_area` caps triangle area, both by inserting Steiner points.
    Returns (vertices, triangles, segments).
    """
    if tr is None:
        raise ImportError(
            "Missing dependency: triangle. Install it with:\n"
            "  pip install triangle\n"
        )

    A = {
        "vertices": np.asarray(vertices, dtype=float),
        "segments": np.asarray(segments, dtype=int),
    }
    if holes is not None:
        A["holes"] = np.asarray(holes, dtype=float)

    if min_angle < 0:
        raise ValueError("min_angle must be >= 0")

    opts = "p"
    if min_angle > 0:
        opts += f"q{float(min_angle):g}"
    if max_area is not None:
        opts += f"a{float(max_area):g}"

    B = tr.triangulate(A, opts)
    B = _triangle_zero_base_if_needed(B)

    pts_out = np.asarray(B["vertices"], dtype=float)
    tris_out = np.asarray(B["triangles"], dtype=int)
    segs_out = np.asarray(B.get("segments", segments), dtype=int)
    return pts_out, tris_out, segs_out


def _vertex_count_for_target_tris(target_tris: int, hull_vertices: int) -> int:
    """Vertex count n for T triangles, from T = 2n - 2 - h with h hull vertices."""
    target_tris = int(target_tris)
    n = int(np.round((target_tris + 2 + hull_vertices) / 2.0))
    return max(n, hull_vertices + 1)


def build_unit_square_data_adaptive_mesh(
    obs_points: np.ndarray,
    *,
    target_tris: int = 20,
    seed: int = 0,
    eps: float = 1e-6,
    min_interior: int = 5,
    n_boundary_per_edge: int = 2,
    min_angle: float = 0.0,
    use_area_constraint: bool = True,
) -> Mesh:
    """
    Build a data-adaptive CDT of Omega=[0,1]^2 with enforced boundary segments.

    Interior vertices are chosen from `obs_points` by farthest-point sampling
    that starts from their distance to the square boundary, which is
    deterministic given the input order. `seed` only affects the random
    candidates used when fewer than 10 observations are interior.
    """
    obs = np.asarray(obs_points, dtype=float)
    if obs.ndim != 2 or obs.shape[1] != 2:
        raise ValueError("obs_points must have shape (n, 2)")

    boundary_pts, boundary_segs = unit_square_boundary_pslg(
        n_per_edge=n_boundary_per_edge
    )
    n_hull = boundary_pts.shape[0]

    mask = (
        (obs[:, 0] > eps)
        & (obs[:, 0] < 1.0 - eps)
        & (obs[:, 1] > eps)
        & (obs[:, 1] < 1.0 - eps)
    )
    candidates = obs[mask]
    if candidates.shape[0] < 10:
        rng = np.random.default_rng(seed)
        candidates = rng.uniform(eps, 1.0 - eps, size=(max(200, obs.shape[0]), 2))

    n_total = _vertex_count_for_target_tris(target_tris, hull_vertices=n_hull)
    n_interior = max(min_interior, n_total - n_hull)
    # Start the max-min selection from the distance to the square boundary;
    # otherwise it favours points next to the boundary, whose slivers make the
    # quality refinement add many Steiner points and overshoot target_tris.
    to_boundary = np.minimum(
        np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
        np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]),
    )
    interior = _farthest_points_from_boundary(candidates, to_boundary, n_interior)

    vertices_in = np.vstack([boundary_pts, interior])

    max_area = None
    if use_area_constraint and target_tris > 0:
        # 1/target_tris is the mean area, so it binds about half the triangles
        # and adds Steiner points; twice the mean only caps oversized triangles.
        max_area = 2.0 / float(target_tris)

    pts_out, tris_out, segs_out = triangulate_cdt_pslg(
        vertices_in,
        boundary_segs,
        holes=None,
        min_angle=min_angle,
        max_area=max_area,
    )

    tris_ccw = np.vstack([ensure_ccw(t, pts_out) for t in tris_out])
    return Mesh(
        points=pts_out,
        triangles=tris_ccw,
        constraints=Constraints(segments=segs_out),
    )


def build_unit_square_regular_triangulation(n_grid: int = 5) -> Mesh:
    """
    Build a regular triangulation of [0,1]^2 using n_grid x n_grid squares,
    each split into 2 right-isosceles triangles by the diagonal.
    """
    if n_grid < 1:
        raise ValueError("n_grid must be >= 1")

    t = np.linspace(0.0, 1.0, n_grid + 1)
    gx, gy = np.meshgrid(t, t)
    points = np.column_stack([gx.ravel(), gy.ravel()])

    def idx(i: int, j: int) -> int:
        return j * (n_grid + 1) + i

    triangles: List[List[int]] = []
    for j in range(n_grid):
        for i in range(n_grid):
            bl = idx(i, j)
            br = idx(i + 1, j)
            tl = idx(i, j + 1)
            tr_ = idx(i + 1, j + 1)

            triangles.append([bl, br, tr_])
            triangles.append([bl, tr_, tl])

    triangles_arr = np.asarray(triangles, dtype=int)
    tris_ccw = np.vstack([ensure_ccw(t, points) for t in triangles_arr])

    boundary_indices: List[int] = []
    for i in range(n_grid + 1):
        boundary_indices.append(idx(i, 0))
    for j in range(1, n_grid + 1):
        boundary_indices.append(idx(n_grid, j))
    for i in range(n_grid - 1, -1, -1):
        boundary_indices.append(idx(i, n_grid))
    for j in range(n_grid - 1, 0, -1):
        boundary_indices.append(idx(0, j))

    m = len(boundary_indices)
    segs = np.array(
        [[boundary_indices[k], boundary_indices[(k + 1) % m]] for k in range(m)],
        dtype=int,
    )

    return Mesh(
        points=points,
        triangles=tris_ccw,
        constraints=Constraints(segments=segs),
    )


def mesh_edges(tris: np.ndarray) -> set[tuple[int, int]]:
    """Return undirected edge set from triangles."""
    edges: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(tris, dtype=int):
        for i, j in ((a, b), (b, c), (c, a)):
            if i > j:
                i, j = j, i
            edges.add((int(i), int(j)))
    return edges


def missing_constraint_edges(mesh: Mesh) -> List[Tuple[int, int]]:
    """Return constraint segments that do not appear as mesh edges."""
    if mesh.constraints is None:
        return []

    E = mesh_edges(mesh.triangles)
    missing: List[Tuple[int, int]] = []
    for i, j in np.asarray(mesh.constraints.segments, dtype=int):
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        if (a, b) not in E:
            missing.append((a, b))
    return missing

