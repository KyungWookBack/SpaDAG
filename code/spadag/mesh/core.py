"""Triangular meshes for the spatial domain.

Provides the Mesh container, regular and data-adaptive constrained Delaunay
triangulations of the unit square, and meshes on a polygon domain estimated
from observation coordinates, together with validation helpers.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


# Observation-supported polygon domains: estimate an irregular domain from the
# observation coordinates, simplify its boundary to the mesh resolution and
# triangulate it.


@dataclass(frozen=True)
class PolygonDomain:
    """
    Polygonal analysis domain described by closed boundary loops.

    Each loop is an (m, 2) array without a repeated closing vertex. Outer loops
    are counter-clockwise and hole loops clockwise, so the domain lies to the
    left of every directed boundary edge and the signed loop areas sum to the
    domain area. `spacing` is the median nearest-neighbour distance of the
    observations that defined the domain.
    """

    loops: Tuple[np.ndarray, ...]
    is_hole: Tuple[bool, ...]
    spacing: float

    @property
    def area(self) -> float:
        return float(sum(_loop_signed_area(loop) for loop in self.loops))

    @property
    def n_holes(self) -> int:
        return int(sum(self.is_hole))

    @property
    def n_boundary_vertices(self) -> int:
        return int(sum(loop.shape[0] for loop in self.loops))

    def boundary_segments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Start and end points of every directed boundary edge."""
        starts = np.vstack(self.loops)
        ends = np.vstack([np.roll(loop, -1, axis=0) for loop in self.loops])
        return starts, ends


def _require_triangle() -> None:
    if tr is None:
        raise ImportError(
            "Missing dependency: triangle. Install it with:\n"
            "  pip install triangle\n"
        )


def _loop_signed_area(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _row_chunks(n_rows: int, n_cols: int, budget: int = 1_000_000):
    """Yield row slices so that each (rows x n_cols) block stays small."""
    step = max(1, budget // max(1, n_cols))
    for start in range(0, n_rows, step):
        yield slice(start, min(n_rows, start + step))


def _points_in_loop(points: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Crossing-number (even-odd) point-in-polygon test against one loop."""
    Q = np.asarray(points, dtype=float)
    a = np.asarray(loop, dtype=float)
    b = np.roll(a, -1, axis=0)
    dy = b[:, 1] - a[:, 1]
    safe_dy = np.where(dy == 0.0, 1.0, dy)  # horizontal edges never straddle
    inside = np.zeros(Q.shape[0], dtype=bool)
    for rows in _row_chunks(Q.shape[0], a.shape[0]):
        x = Q[rows, :1]
        y = Q[rows, 1:2]
        straddle = (a[None, :, 1] > y) != (b[None, :, 1] > y)
        x_cross = a[None, :, 0] + (y - a[None, :, 1]) * (
            (b[None, :, 0] - a[None, :, 0]) / safe_dy[None, :]
        )
        inside[rows] = np.count_nonzero(straddle & (x < x_cross), axis=1) % 2 == 1
    return inside


def points_in_domain(domain: PolygonDomain, points: np.ndarray) -> np.ndarray:
    """Boolean mask of points inside the domain (outer loops minus holes)."""
    Q = np.asarray(points, dtype=float)
    inside = np.zeros(Q.shape[0], dtype=bool)
    for loop in domain.loops:
        inside ^= _points_in_loop(Q, loop)
    return inside


def _nearest_segment(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance from each point to the nearest segment, and that segment."""
    Q = np.asarray(points, dtype=float)
    d = ends - starts
    len2 = np.maximum(np.sum(d * d, axis=1), np.finfo(float).tiny)
    dist = np.empty(Q.shape[0], dtype=float)
    arg = np.empty(Q.shape[0], dtype=int)
    for rows in _row_chunks(Q.shape[0], starts.shape[0]):
        rel = Q[rows, None, :] - starts[None, :, :]
        t = np.clip(np.sum(rel * d[None], axis=2) / len2[None], 0.0, 1.0)
        gap = rel - t[..., None] * d[None]
        D = np.sqrt(np.sum(gap * gap, axis=2))
        arg[rows] = np.argmin(D, axis=1)
        dist[rows] = D[np.arange(D.shape[0]), arg[rows]]
    return dist, arg


def distance_to_boundary(domain: PolygonDomain, points: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to the domain boundary."""
    starts, ends = domain.boundary_segments()
    return _nearest_segment(points, starts, ends)[0]


def _unique_points(points: np.ndarray) -> np.ndarray:
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("obs_points must have shape (n, 2)")
    if not np.all(np.isfinite(P)):
        raise ValueError("obs_points must be finite")
    P = np.unique(P, axis=0)
    if P.shape[0] < 3:
        raise ValueError("need at least three distinct observation points")
    return P


def _unique_edges(tris: np.ndarray) -> np.ndarray:
    E = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    return np.unique(np.sort(E, axis=1), axis=0)


def delaunay_triangulation(points: np.ndarray) -> np.ndarray:
    """Delaunay triangles (indices into `points`) computed by Triangle."""
    _require_triangle()
    B = tr.triangulate({"vertices": np.asarray(points, dtype=float)}, "")
    B = _triangle_zero_base_if_needed(B)
    tris = np.asarray(B.get("triangles", np.zeros((0, 3))), dtype=int)
    if tris.shape[0] == 0:
        raise ValueError("points are collinear; no 2D triangulation exists")
    return tris


def _nn_spacing(points: np.ndarray, tris: np.ndarray) -> float:
    # Each point's nearest neighbour is one of its Delaunay neighbours.
    E = _unique_edges(tris)
    L = np.linalg.norm(points[E[:, 0]] - points[E[:, 1]], axis=1)
    nn = np.full(points.shape[0], np.inf)
    np.minimum.at(nn, E[:, 0], L)
    np.minimum.at(nn, E[:, 1], L)
    return float(np.median(nn))


def median_nn_spacing(points: np.ndarray) -> float:
    """Median nearest-neighbour distance of the distinct points."""
    P = _unique_points(points)
    return _nn_spacing(P, delaunay_triangulation(P))


def _chi_shape(
    points: np.ndarray, tris: np.ndarray, max_edge: float
) -> np.ndarray:
    """
    Characteristic shape (Duckham et al., 2008, Pattern Recognition 41:3224).

    Starting from the Delaunay triangulation, the triangle behind the longest
    boundary edge is removed while that edge is longer than `max_edge`, unless
    its third vertex already lies on the boundary. The kept triangles form a
    simple polygon that contains every point, inside or as a boundary vertex.
    Returns the kept-triangle mask.
    """
    n = points.shape[0]
    edge_tris: Dict[int, List[int]] = {}
    for t, (a, b, c) in enumerate(np.asarray(tris, dtype=int).tolist()):
        for i, j in ((a, b), (b, c), (c, a)):
            edge_tris.setdefault(min(i, j) * n + max(i, j), []).append(t)

    alive = np.ones(tris.shape[0], dtype=bool)
    on_boundary = np.zeros(n, dtype=bool)
    heap: List[Tuple[float, int]] = []

    def push(i: int, j: int) -> None:
        length = float(np.hypot(*(points[i] - points[j])))
        heapq.heappush(heap, (-length, min(i, j) * n + max(i, j)))

    for key, ts in edge_tris.items():
        if len(ts) == 1:
            i, j = divmod(key, n)
            on_boundary[[i, j]] = True
            push(i, j)

    while heap:
        neg_length, key = heapq.heappop(heap)
        if -neg_length <= max_edge:
            break
        live = [t for t in edge_tris[key] if alive[t]]
        if len(live) != 1:
            continue
        i, j = divmod(key, n)
        r = next(int(v) for v in tris[live[0]] if v != i and v != j)
        if on_boundary[r]:
            continue
        alive[live[0]] = False
        on_boundary[r] = True
        push(i, r)
        push(j, r)
    return alive


def _rasterize_triangles(
    points: np.ndarray, tris: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """Grid mask of cell centres covered by the triangles, edges or vertices."""
    step = float(xs[1] - xs[0])
    inside = np.zeros((ys.size, xs.size), dtype=bool)
    for a, b, c in points[tris]:
        lo = np.minimum(np.minimum(a, b), c)
        hi = np.maximum(np.maximum(a, b), c)
        i0 = max(0, int(np.ceil((lo[0] - xs[0]) / step)))
        i1 = min(xs.size, int(np.floor((hi[0] - xs[0]) / step)) + 1)
        j0 = max(0, int(np.ceil((lo[1] - ys[0]) / step)))
        j1 = min(ys.size, int(np.floor((hi[1] - ys[0]) / step)) + 1)
        if i1 <= i0 or j1 <= j0:
            continue
        X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1])
        det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        w1 = ((X - a[0]) * (c[1] - a[1]) - (Y - a[1]) * (c[0] - a[0])) / det
        w2 = ((b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0])) / det
        inside[j0:j1, i0:i1] |= (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)

    # Thin triangles may contain no cell centre; also mark samples along edges.
    E = _unique_edges(tris)
    start = points[E[:, 0]]
    d = points[E[:, 1]] - start
    k = np.ceil(np.linalg.norm(d, axis=1) / (0.5 * step)).astype(int) + 1
    edge = np.repeat(np.arange(E.shape[0]), k)
    t = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(k - 1, k)
    inside[_nearest_cells(start[edge] + t[:, None] * d[edge], xs, ys)] = True
    return inside


def _nearest_cells(
    Q: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """(row, column) indices of the grid cell centre nearest to each point."""
    step = float(xs[1] - xs[0])
    ii = np.clip(np.rint((Q[:, 0] - xs[0]) / step).astype(int), 0, xs.size - 1)
    jj = np.clip(np.rint((Q[:, 1] - ys[0]) / step).astype(int), 0, ys.size - 1)
    return jj, ii


def _drop_loops(
    domain: PolygonDomain,
    min_hole_area: float = 0.0,
    obs_points: Optional[np.ndarray] = None,
) -> PolygonDomain:
    """
    Remove holes with area below `min_hole_area` (filling them) and, when
    `obs_points` is given, outer loops enclosing no observation. Loops nested
    inside a removed loop are removed with it.
    """
    loops = list(domain.loops)
    drop = set()
    for i, (loop, hole) in enumerate(zip(loops, domain.is_hole)):
        small_hole = hole and -_loop_signed_area(loop) < min_hole_area
        empty_island = (
            not hole
            and obs_points is not None
            and not np.any(_points_in_loop(obs_points, loop))
        )
        if small_hole or empty_island:
            drop.add(i)
            drop.update(
                j
                for j in range(len(loops))
                if j != i and _points_in_loop(loops[j][:1], loop)[0]
            )
    keep = [i for i in range(len(loops)) if i not in drop]
    return PolygonDomain(
        loops=tuple(loops[i] for i in keep),
        is_hole=tuple(domain.is_hole[i] for i in keep),
        spacing=domain.spacing,
    )


def estimate_observation_domain(
    obs_points: np.ndarray,
    *,
    hull_scale: float = 4.0,
    margin_scale: float = 1.0,
    min_hole_area: Optional[float] = None,
    grid_scale: float = 0.125,
    max_grid_cells: int = 8_000_000,
) -> PolygonDomain:
    """
    Estimate the spatial domain supported by observation coordinates.

    Only coordinates are used. With h the median nearest-neighbour spacing and
    lambda = hull_scale * h:

    1. Concave hull: the characteristic shape of the Delaunay triangulation
       with edge threshold lambda. It is one simple polygon and contains every
       observation, including isolated spots.
    2. Margin: the domain is the set of locations within `margin_scale * h`
       of the concave hull, so every observation lies strictly inside, about
       `margin_scale * h` from the outer boundary.
    3. Holes: enclosed parts of the hull farther than lambda from every
       observation, grown back to `margin_scale * h` from the nearest
       observation. A hole thus contains an empty disk of radius lambda, which
       a few missing spots of a Visium lattice do not. Holes with area below
       `min_hole_area` (default pi * lambda^2) are filled, and islands that
       enclose no observation are removed.

    Steps 2-3 use distance transforms on a grid of spacing
    `grid_scale * margin_scale * h` and the boundary is traced as their zero
    contour, so the loops follow it at grid resolution; simplify them with
    `simplify_polygon_domain` before meshing.
    """
    import contourpy
    from scipy import ndimage

    if hull_scale <= 0 or margin_scale <= 0 or grid_scale <= 0:
        raise ValueError("hull_scale, margin_scale and grid_scale must be > 0")
    if hull_scale <= margin_scale:
        raise ValueError("hull_scale must exceed margin_scale")

    P = _unique_points(obs_points)
    tris = delaunay_triangulation(P)
    h = _nn_spacing(P, tris)
    if not h > 0:
        raise ValueError("median nearest-neighbour spacing must be positive")
    max_edge = hull_scale * h
    margin = margin_scale * h
    if min_hole_area is None:
        min_hole_area = float(np.pi * max_edge**2)

    alive = _chi_shape(P, tris, max_edge)

    step = grid_scale * margin
    pad = margin + 4.0 * step
    lo = P.min(axis=0) - pad
    shape = np.ceil((P.max(axis=0) + pad - lo) / step).astype(int) + 1
    if int(np.prod(shape)) > max_grid_cells:
        raise ValueError(
            f"distance grid would need {int(np.prod(shape))} cells; "
            "increase grid_scale or max_grid_cells"
        )
    xs = lo[0] + step * np.arange(shape[0])
    ys = lo[1] + step * np.arange(shape[1])
    hull = _rasterize_triangles(P, tris[alive], xs, ys)
    field = margin - ndimage.distance_transform_edt(~hull, sampling=step)

    at_obs = np.zeros_like(hull)
    at_obs[_nearest_cells(P, xs, ys)] = True
    empty = hull & (ndimage.distance_transform_edt(~at_obs, sampling=step) > max_edge)
    label, _ = ndimage.label(empty)
    next_to_outside = ndimage.binary_dilation(~hull, structure=np.ones((3, 3), bool))
    enclosed = empty & ~np.isin(label, np.unique(label[next_to_outside & empty]))
    if enclosed.any():
        to_hole = ndimage.distance_transform_edt(~enclosed, sampling=step)
        field = np.minimum(field, to_hole - (max_edge - margin))

    # Grid distances are step * sqrt(integer), so the margin can equal a grid
    # value exactly; a level offset keeps contours off grid nodes, where
    # separate lines could otherwise touch.
    lines = contourpy.contour_generator(
        x=xs, y=ys, z=field, line_type="Separate"
    ).lines(1e-6 * step)

    loops = []
    for line in lines:
        L = np.asarray(line, dtype=float)
        L = L[np.r_[True, np.any(np.diff(L, axis=0) != 0.0, axis=1)]]
        if L.shape[0] > 1 and np.all(L[0] == L[-1]):
            L = L[:-1]
        if L.shape[0] >= 3 and _loop_signed_area(L) != 0.0:
            loops.append(L)

    # Nesting depth decides outer (even) versus hole (odd) loops.
    is_hole = []
    for i, loop in enumerate(loops):
        depth = sum(
            bool(_points_in_loop(loop[:1], other)[0])
            for j, other in enumerate(loops)
            if j != i
        )
        is_hole.append(depth % 2 == 1)
    loops = [
        L if (_loop_signed_area(L) < 0) == hole else L[::-1].copy()
        for L, hole in zip(loops, is_hole)
    ]
    domain = PolygonDomain(loops=tuple(loops), is_hole=tuple(is_hole), spacing=h)
    return _drop_loops(domain, min_hole_area, obs_points=P)


def _between(i: int, j: int, n: int) -> np.ndarray:
    """Loop indices strictly after i and before j, walking forward."""
    stop = j if j > i else j + n
    return np.arange(i + 1, stop) % n


def _left_offsets(Q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed distance of Q from the line a->b, positive on the left."""
    d = b - a
    return (d[0] * (Q[:, 1] - a[1]) - d[1] * (Q[:, 0] - a[0])) / np.hypot(*d)


def _one_sided_dp(loop: np.ndarray, fill_tol: float, cut_tol: float) -> List[int]:
    """
    Douglas-Peucker on a closed loop with separate outward/inward tolerances.

    The domain is left of the loop, so an original vertex left of a chord means
    the chord adds area (bounded by fill_tol) and one to the right means the
    chord removes area (bounded by cut_tol).
    """
    n = loop.shape[0]
    if n <= 3:
        return list(range(n))
    i0 = int(np.lexsort((loop[:, 1], loop[:, 0]))[0])
    i1 = int(np.argmax(np.linalg.norm(loop - loop[i0], axis=1)))
    keep = {i0, i1}
    stack = [(i0, i1), (i1, i0)]
    while stack:
        i, j = stack.pop()
        idx = _between(i, j, n)
        if idx.size == 0:
            continue
        off = _left_offsets(loop[idx], loop[i], loop[j])
        score = np.maximum(off / fill_tol, -off / cut_tol)
        k = int(np.argmax(score))
        if score[k] > 1.0:
            v = int(idx[k])
            keep.add(v)
            stack.extend([(i, v), (v, j)])
    if len(keep) < 3:
        idx = _between(i0, i1, n)
        if idx.size == 0:
            idx = _between(i1, i0, n)
        off = np.abs(_left_offsets(loop[idx], loop[i0], loop[i1]))
        keep.add(int(idx[int(np.argmax(off))]))
    return sorted(keep)


def _touching_segment_pairs(
    starts: np.ndarray, ends: np.ndarray, owner: np.ndarray, local: np.ndarray
) -> List[Tuple[int, int]]:
    """Pairs of non-adjacent boundary edges that cross or touch."""

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (
            q[..., 1] - p[..., 1]
        ) * (r[..., 0] - p[..., 0])

    a1, b1 = starts[:, None, :], ends[:, None, :]
    a2, b2 = starts[None, :, :], ends[None, :, :]
    lo1, hi1 = np.minimum(a1, b1), np.maximum(a1, b1)
    lo2, hi2 = np.minimum(a2, b2), np.maximum(a2, b2)
    boxes = np.all((lo1 <= hi2) & (lo2 <= hi1), axis=2)
    hit = (
        boxes
        & (orient(a1, b1, a2) * orient(a1, b1, b2) <= 0)
        & (orient(a2, b2, a1) * orient(a2, b2, b1) <= 0)
    )
    size = np.bincount(owner)[owner]
    gap = (local[None, :] - local[:, None]) % size[:, None]
    adjacent = (owner[:, None] == owner[None, :]) & ((gap <= 1) | (gap == size[:, None] - 1))
    i, j = np.nonzero(np.triu(hit & ~adjacent, k=1))
    return list(zip(i.tolist(), j.tolist()))


def simplify_polygon_domain(
    domain: PolygonDomain,
    obs_points: np.ndarray,
    *,
    fill_tol: float,
    cut_tol: float,
    clearance: float,
    max_rounds: int = 10_000,
) -> PolygonDomain:
    """
    Reduce boundary vertices while keeping every observation inside.

    Each loop is simplified by a one-sided Douglas-Peucker recursion: a chord
    that replaces part of the boundary may lie outside it (adding area) by at
    most `fill_tol` but inside it by at most `cut_tol`. Concavities shallower
    than `fill_tol` are filled, while convex parts, where observations are
    close to the boundary, are followed closely. Chords are then refined by
    re-inserting their farthest original vertex until every observation is at
    least `clearance` inside and no two boundary edges touch. Kept vertices
    are original loop vertices, so full refinement recovers `domain`.
    The edge-intersection check is O(S^2) in the number of kept edges.
    """
    if not (fill_tol > 0 and cut_tol > 0 and clearance >= 0):
        raise ValueError("need fill_tol > 0, cut_tol > 0 and clearance >= 0")
    obs = np.asarray(obs_points, dtype=float)
    fine = [np.asarray(loop, dtype=float) for loop in domain.loops]
    kept = [_one_sided_dp(loop, fill_tol, cut_tol) for loop in fine]
    nested = _loop_nesting(fine)

    for _ in range(max_rounds):
        trial = PolygonDomain(
            loops=tuple(loop[k] for loop, k in zip(fine, kept)),
            is_hole=domain.is_hole,
            spacing=domain.spacing,
        )
        starts, ends = trial.boundary_segments()
        owner = np.concatenate([np.full(len(k), li) for li, k in enumerate(kept)])
        local = np.concatenate([np.arange(len(k)) for k in kept])

        def chord(s: int) -> Tuple[int, int, int, np.ndarray]:
            """Loop, original endpoints and skipped original vertices of edge s."""
            li, k = int(owner[s]), int(local[s])
            i, j = kept[li][k], kept[li][(k + 1) % len(kept[li])]
            return li, i, j, _between(i, j, fine[li].shape[0])

        def swept_by(point: np.ndarray, loop: Optional[int] = None) -> List[int]:
            """Chords whose region between chord and original boundary holds point."""
            hits = []
            for s2 in range(starts.shape[0]):
                li, i, j, idx = chord(s2)
                if idx.size and (loop is None or li == loop):
                    if _points_in_loop(point[None], fine[li][np.r_[i, idx, j]])[0]:
                        hits.append(s2)
            return hits

        refine = set()
        dist, nearest = _nearest_segment(obs, starts, ends)
        inside = points_in_domain(trial, obs)
        for q in np.flatnonzero(~inside | (dist < clearance)):
            s = int(nearest[q])
            if chord(s)[3].size:
                refine.add(s)
                continue
            # Nearest edge is original; find the chords that cut this point off.
            cutting = swept_by(obs[q])
            if not cutting:
                raise RuntimeError(
                    "an observation violates the clearance of the unsimplified "
                    "domain; reduce clearance"
                )
            refine.update(cutting)

        # Loop nesting must not change: a bulging chord could swallow a loop.
        for a in range(len(fine)):
            probe = fine[a][kept[a][0]]
            for b in range(len(fine)):
                if b != a and nested[a][b] != _points_in_loop(probe[None], trial.loops[b])[0]:
                    swallowing = swept_by(probe, loop=b)
                    if not swallowing:
                        raise RuntimeError("boundary simplification changed loop nesting")
                    refine.update(swallowing)

        for s, t in _touching_segment_pairs(starts, ends, owner, local):
            refinable = [x for x in (s, t) if chord(x)[3].size]
            if not refinable:
                raise RuntimeError("the unsimplified domain boundary is not simple")
            refine.update(refinable)
        if not refine:
            return trial

        # Choose all insertions before changing `kept` (local indices shift).
        additions: Dict[int, set] = {}
        for s in refine:
            li, i, j, idx = chord(s)
            off = np.abs(_left_offsets(fine[li][idx], fine[li][i], fine[li][j]))
            additions.setdefault(li, set()).add(int(idx[int(np.argmax(off))]))
        for li, new in additions.items():
            kept[li] = sorted(set(kept[li]) | new)
    raise RuntimeError("boundary simplification did not converge")


def _interior_point(loop: np.ndarray) -> np.ndarray:
    """A point strictly inside a simple polygon: its largest CDT triangle's centroid."""
    m = loop.shape[0]
    segs = np.c_[np.arange(m), (np.arange(m) + 1) % m]
    pts, tris, _ = triangulate_cdt_pslg(loop, segs)
    area = np.abs(triangle_areas(Mesh(points=pts, triangles=tris)))
    return pts[tris[int(np.argmax(area))]].mean(axis=0)


def polygon_domain_pslg(
    domain: PolygonDomain,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """PSLG vertices, boundary segments and one seed point inside each hole."""
    vertices, segments, seeds = [], [], []
    offset = 0
    for loop, hole in zip(domain.loops, domain.is_hole):
        m = loop.shape[0]
        idx = np.arange(m)
        vertices.append(loop)
        segments.append(np.c_[offset + idx, offset + (idx + 1) % m])
        offset += m
        if hole:
            seeds.append(_interior_point(loop))
    holes = np.vstack(seeds) if seeds else None
    return np.vstack(vertices), np.vstack(segments).astype(int), holes


def _triangulate_domain(
    domain: PolygonDomain,
    interior: np.ndarray,
    *,
    min_angle: float = 0.0,
    max_area: Optional[float] = None,
) -> Mesh:
    """CDT of the domain loops plus interior vertices (CCW triangles)."""
    boundary_pts, boundary_segs, holes = polygon_domain_pslg(domain)
    pts, tris, segs = triangulate_cdt_pslg(
        np.vstack([boundary_pts, interior]),
        boundary_segs,
        holes=holes,
        min_angle=min_angle,
        max_area=max_area,
    )
    tris_ccw = np.vstack([ensure_ccw(t, pts) for t in tris])
    return Mesh(points=pts, triangles=tris_ccw, constraints=Constraints(segments=segs))


def _loop_nesting(loops: List[np.ndarray]) -> List[List[bool]]:
    """nested[a][b]: whether loop a lies inside loop b."""
    return [
        [b != a and bool(_points_in_loop(loops[a][:1], loops[b])[0]) for b in range(len(loops))]
        for a in range(len(loops))
    ]


def _domain_keeps_observations(
    domain: PolygonDomain,
    obs: np.ndarray,
    clearance: float,
    nested: List[List[bool]],
) -> bool:
    """Observations at least `clearance` inside, simple loops, unchanged nesting."""
    if not np.all(points_in_domain(domain, obs)):
        return False
    if np.min(distance_to_boundary(domain, obs)) < clearance:
        return False
    for loop, hole in zip(domain.loops, domain.is_hole):
        if loop.shape[0] < 3 or (_loop_signed_area(loop) < 0) != hole:
            return False
    starts, ends = domain.boundary_segments()
    owner = np.concatenate([np.full(len(loop), i) for i, loop in enumerate(domain.loops)])
    local = np.concatenate([np.arange(len(loop)) for loop in domain.loops])
    if _touching_segment_pairs(starts, ends, owner, local):
        return False
    return _loop_nesting(list(domain.loops)) == nested


def _split_high_support_triangles(
    domain: PolygonDomain,
    interior: np.ndarray,
    obs: np.ndarray,
    max_obs: int,
    *,
    min_angle: float,
    max_area: Optional[float],
    n_trial: int = 12,
    max_iter: Optional[int] = None,
) -> np.ndarray:
    """
    Add interior vertices until no triangle holds more than `max_obs`
    observations.

    Each pass retriangulates the domain and takes the triangle with the most
    observations above `max_obs`. Up to `n_trial` observations inside it that
    are not already vertices, ordered by distance to its centroid, are tried as
    a new interior vertex; the trial whose mesh has the smallest largest count
    (ties: the largest smallest count) is kept. The loop stops when no triangle
    exceeds the cap, when the heaviest triangle has no observation left to add,
    or after `max_iter` passes (default: number of observations). Observations
    on a shared edge count for the lower triangle index, as in
    `_containing_triangle`. Boundary loops are not changed.
    """
    interior = np.asarray(interior, dtype=float)
    limit = int(max_iter) if max_iter is not None else int(obs.shape[0])

    def counts_for(points_interior):
        mesh = _triangulate_domain(domain, points_interior, min_angle=min_angle, max_area=max_area)
        found = _containing_triangle(mesh.points, mesh.triangles, obs)
        return mesh, found, np.bincount(found[found >= 0], minlength=mesh.triangles.shape[0])

    for _ in range(limit):
        mesh, found, counts = counts_for(interior)
        heavy = np.flatnonzero(counts > max_obs)
        if heavy.size == 0:
            break
        k = int(heavy[np.argmax(counts[heavy])])
        inside = obs[found == k]
        taken = np.array(
            [bool(np.any(np.all(np.isclose(mesh.points, p), axis=1))) for p in inside], dtype=bool
        )
        candidates = inside[~taken]
        if candidates.shape[0] == 0:
            break
        centroid = mesh.points[mesh.triangles[k]].mean(axis=0)
        order = np.argsort(np.sum((candidates - centroid) ** 2, axis=1), kind="stable")
        best = None
        for idx in order[: max(1, int(n_trial))]:
            trial = np.vstack([interior, candidates[idx][None, :]])
            _, _, trial_counts = counts_for(trial)
            score = (int(trial_counts.max()), -int(trial_counts.min()))
            if best is None or score < best[0]:
                best = (score, trial)
        interior = best[1]
    return interior


def _remove_low_support_vertices(
    domain: PolygonDomain,
    interior: np.ndarray,
    obs: np.ndarray,
    min_obs: int,
    *,
    clearance: float,
    min_angle: float,
    max_area: Optional[float],
) -> Tuple[PolygonDomain, np.ndarray]:
    """
    Delete mesh vertices around triangles that hold fewer than `min_obs`
    observations.

    Triangles are visited from the fewest observations upwards. Each vertex of
    the visited triangle that is an input vertex (a boundary-loop vertex or an
    interior vertex, not a Steiner point) is deleted on trial and the domain is
    retriangulated with the final `min_angle` and `max_area`. A boundary vertex
    may be deleted only if every observation stays at least `clearance` inside
    and the loops stay simple with unchanged nesting. The trial that most
    reduces the number of triangles below `min_obs` (ties: that most raises the
    smallest count) is kept; a triangle with no improving trial is skipped until
    another deletion changes the mesh. Greedy, so the threshold is a target: a
    triangle around an isolated observation can stay below it.
    """
    loops = [np.asarray(loop, dtype=float) for loop in domain.loops]
    interior = np.asarray(interior, dtype=float)
    nested = _loop_nesting(loops)

    def build(new_loops, new_interior):
        dom = PolygonDomain(tuple(new_loops), domain.is_hole, domain.spacing)
        mesh = _triangulate_domain(dom, new_interior, min_angle=min_angle, max_area=max_area)
        found = _containing_triangle(mesh.points, mesh.triangles, obs)
        counts = np.bincount(found[found >= 0], minlength=mesh.triangles.shape[0])
        return dom, mesh, counts

    def score(counts):
        return (int(np.sum(counts < min_obs)), -int(counts.min()))

    dom, mesh, counts = build(loops, interior)
    skipped: set = set()
    while True:
        worst = [
            k for k in np.argsort(counts, kind="stable")
            if counts[k] < min_obs and mesh.points[mesh.triangles[k]].tobytes() not in skipped
        ]
        if not worst:
            break
        k = int(worst[0])
        best = None
        for p in mesh.points[mesh.triangles[k]]:
            new_loops, new_interior = loops, interior
            hit = [i for i, loop in enumerate(loops) if np.any(np.all(loop == p, axis=1))]
            if hit:
                li = hit[0]
                row = int(np.flatnonzero(np.all(loops[li] == p, axis=1))[0])
                new_loops = loops[:li] + [np.delete(loops[li], row, axis=0)] + loops[li + 1:]
                trial = PolygonDomain(tuple(new_loops), domain.is_hole, domain.spacing)
                if not _domain_keeps_observations(trial, obs, clearance, nested):
                    continue
            else:
                rows = np.flatnonzero(np.all(interior == p, axis=1))
                if rows.size == 0:
                    continue  # Steiner point added by Triangle
                new_interior = np.delete(interior, rows[0], axis=0)
            candidate = build(new_loops, new_interior)
            if score(candidate[2]) < score(counts) and (
                best is None or score(candidate[2]) < score(best[2][2])
            ):
                best = (new_loops, new_interior, candidate)
        if best is None:
            skipped.add(mesh.points[mesh.triangles[k]].tobytes())
            continue
        loops, interior, (dom, mesh, counts) = best
        skipped.clear()
    return dom, interior


def build_polygon_data_adaptive_mesh(
    obs_points: np.ndarray,
    *,
    target_tris: int = 50,
    hull_scale: float = 4.0,
    margin_scale: float = 1.0,
    boundary_tol: float = 0.5,
    min_hole_area: Optional[float] = None,
    min_interior: int = 4,
    min_angle: float = 10.0,
    use_area_constraint: bool = True,
    max_area_factor: float = 2.0,
    min_obs_per_triangle: Optional[int] = None,
    max_obs_per_triangle: Optional[int] = None,
    refine_rounds: int = 3,
    return_domain: bool = False,
):
    """
    Build a data-adaptive CDT on an observation-supported polygon domain.

    Unlike `build_unit_square_data_adaptive_mesh`, the domain is estimated from
    the observation coordinates instead of being fixed to [0,1]^2, and the mesh
    is in the input coordinates. With h the median nearest-neighbour spacing:

    - Domain: `estimate_observation_domain` (concave hull plus a margin of
      `margin_scale * h`). Unless `min_hole_area` is given, holes with area
      below max(pi*(hull_scale*h)^2, area/target_tris) are filled, since a
      hole smaller than one target triangle cannot be represented.
    - Boundary: `simplify_polygon_domain` with outward tolerance
      `boundary_tol * L`, where L is the side of an equilateral triangle of
      area area/target_tris, inward tolerance half the margin and clearance a
      quarter of the margin.
    - Interior vertices: observations chosen by farthest-point sampling that
      starts from their distance to the boundary. Their number follows
      T = 2V - B - 2*(outer loops) + 2*(holes) for the target T.
    - CDT: `triangulate_cdt_pslg` with the boundary segments, hole seeds,
      `min_angle` and max_area = max_area_factor * area / target_tris using the
      actual polygon area. max_area_factor=1 is the literal area/target_tris
      cap; because it binds every triangle it adds Steiner points and the
      triangle count overshoots the target.
    - Optional coarsening (`min_obs_per_triangle`): vertices of triangles with
      fewer observations are deleted by `_remove_low_support_vertices`, which
      keeps every observation inside and lowers T below the target. Off by
      default. Isolated observations force pockets of empty domain around
      them; this merges such pockets into triangles that reach the dense data.
    - Optional refinement (`max_obs_per_triangle`): triangles holding more
      observations than that are split by `_split_high_support_triangles`,
      which adds the observation nearest to the centroid of the heaviest such
      triangle as an interior vertex and retriangulates, until no triangle
      exceeds the cap. Off by default. When both the cap and
      `min_obs_per_triangle` are set, splitting and coarsening alternate for up
      to `refine_rounds` rounds (stopping early once a round changes nothing),
      because a split can leave a piece below the floor and a merge can
      recreate a heavy triangle. Both bounds are therefore targets, not
      guarantees, and observation counts follow `_containing_triangle` (a point
      on a shared edge or at a vertex counts for the lowest triangle index).

    Returns the Mesh, or (Mesh, PolygonDomain) when `return_domain` is True;
    the domain reflects any boundary vertices deleted by the coarsening.
    """
    obs = np.asarray(obs_points, dtype=float)
    if obs.ndim != 2 or obs.shape[1] != 2:
        raise ValueError("obs_points must have shape (n, 2)")
    target_tris = int(target_tris)
    if target_tris < 1:
        raise ValueError("target_tris must be >= 1")
    if boundary_tol <= 0 or max_area_factor <= 0:
        raise ValueError("boundary_tol and max_area_factor must be > 0")

    fine = estimate_observation_domain(
        obs,
        hull_scale=hull_scale,
        margin_scale=margin_scale,
        min_hole_area=min_hole_area,
    )
    if min_hole_area is None:
        fine = _drop_loops(fine, fine.area / target_tris)

    margin = margin_scale * fine.spacing
    edge = np.sqrt(4.0 * fine.area / (np.sqrt(3.0) * target_tris))
    domain = simplify_polygon_domain(
        fine,
        obs,
        fill_tol=boundary_tol * edge,
        cut_tol=0.5 * margin,
        clearance=0.25 * margin,
    )

    n_boundary = domain.n_boundary_vertices
    n_outer = len(domain.loops) - domain.n_holes
    n_total = int(
        np.round((target_tris + n_boundary + 2 * n_outer - 2 * domain.n_holes) / 2.0)
    )
    n_interior = max(min_interior, n_total - n_boundary)
    candidates = np.unique(obs, axis=0)
    interior = _farthest_points_from_boundary(
        candidates, distance_to_boundary(domain, candidates), n_interior
    )

    max_area = None
    if use_area_constraint:
        max_area = max_area_factor * domain.area / float(target_tris)

    do_split = max_obs_per_triangle is not None and int(max_obs_per_triangle) > 0
    do_merge = min_obs_per_triangle is not None and int(min_obs_per_triangle) > 0
    rounds = int(refine_rounds) if (do_split and do_merge) else 1
    for _ in range(rounds):
        loops_before = [np.array(l, copy=True) for l in domain.loops]
        interior_before = np.array(interior, copy=True)
        if do_split:
            interior = _split_high_support_triangles(
                domain,
                interior,
                obs,
                int(max_obs_per_triangle),
                min_angle=min_angle,
                max_area=max_area,
            )
        if do_merge:
            domain, interior = _remove_low_support_vertices(
                domain,
                interior,
                obs,
                int(min_obs_per_triangle),
                clearance=0.25 * margin,
                min_angle=min_angle,
                max_area=max_area,
            )
        unchanged = (
            len(domain.loops) == len(loops_before)
            and all(np.array_equal(a, b) for a, b in zip(domain.loops, loops_before))
            and np.array_equal(np.asarray(interior), interior_before)
        )
        if unchanged:
            break
    mesh = _triangulate_domain(domain, interior, min_angle=min_angle, max_area=max_area)
    return (mesh, domain) if return_domain else mesh


def _containing_triangle(
    points: np.ndarray, tris: np.ndarray, query: np.ndarray, tol: float = 1e-12
) -> np.ndarray:
    """
    Index of the first closed triangle containing each query point, or -1
    (barycentric test; a point on a shared edge goes to the lower index).
    """
    a = points[tris[:, 0]]
    u = points[tris[:, 1]] - a
    v = points[tris[:, 2]] - a
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    Q = np.asarray(query, dtype=float)
    found = np.full(Q.shape[0], -1, dtype=int)
    for rows in _row_chunks(Q.shape[0], tris.shape[0]):
        rx = Q[rows, :1] - a[None, :, 0]
        ry = Q[rows, 1:2] - a[None, :, 1]
        w1 = (rx * v[None, :, 1] - ry * v[None, :, 0]) / det[None]
        w2 = (u[None, :, 0] * ry - u[None, :, 1] * rx) / det[None]
        inside = (w1 >= -tol) & (w2 >= -tol) & (w1 + w2 <= 1 + tol)
        found[rows] = np.where(inside.any(axis=1), inside.argmax(axis=1), -1)
    return found


def triangle_min_angles(mesh: Mesh) -> np.ndarray:
    """Smallest interior angle (degrees) of each triangle."""
    p = np.asarray(mesh.points, dtype=float)[np.asarray(mesh.triangles, dtype=int)]
    angles = []
    for k in range(3):
        u = p[:, (k + 1) % 3] - p[:, k]
        v = p[:, (k + 2) % 3] - p[:, k]
        cos = np.sum(u * v, axis=1) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        )
        angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return np.min(angles, axis=0)


def validate_polygon_mesh(
    mesh: Mesh,
    domain: PolygonDomain,
    obs_points: Optional[np.ndarray] = None,
    *,
    target_tris: Optional[int] = None,
    count_range: Tuple[float, float] = (0.5, 2.0),
    rtol: float = 1e-9,
    raise_on_failure: bool = True,
) -> Dict[str, object]:
    """
    Check that `mesh` is a valid triangulation of `domain`.

    report["checks"] holds one boolean per check:
    - valid_indices: finite points, indices in range, three distinct vertices
      per triangle and every vertex used by some triangle;
    - ccw: every triangle has positive signed area;
    - nondegenerate: every triangle area exceeds rtol * domain area;
    - consistent_connectivity: each directed edge occurs once, so an interior
      edge is shared by exactly two oppositely oriented triangles, and the
      boundary edges form one simple cycle per domain loop;
    - constraints_represented: every constraint segment is a mesh edge, the
      mesh boundary edges are exactly the constraint segments and they lie on
      the domain boundary;
    - inside_domain: every triangle centroid is inside the domain (so not in a
      hole) and the mesh area equals the domain area;
    - obs_covered: every observation lies in a closed triangle (if obs_points);
    - triangle_count: count_range[0] <= T / target_tris <= count_range[1]
      (if target_tris).
    """
    pts = np.asarray(mesh.points, dtype=float)
    tris = np.asarray(mesh.triangles, dtype=int)
    n = pts.shape[0]
    checks: Dict[str, bool] = {}

    indexable = (
        tris.ndim == 2 and tris.shape[1] == 3 and tris.size > 0
        and tris.min() >= 0 and tris.max() < n
    )
    if not indexable:
        raise ValueError("mesh triangles are empty or index outside mesh.points")
    distinct = np.all(
        (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    )
    checks["valid_indices"] = bool(
        np.all(np.isfinite(pts)) and distinct and np.unique(tris).size == n
    )

    area = triangle_areas(mesh)
    domain_area = domain.area
    checks["ccw"] = bool(np.all(area > 0))
    checks["nondegenerate"] = bool(np.min(np.abs(area)) > rtol * abs(domain_area))

    directed = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    _, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    boundary = directed[counts[inverse.ravel()] == 1]
    out_deg = np.bincount(boundary[:, 0], minlength=n)
    in_deg = np.bincount(boundary[:, 1], minlength=n)
    n_cycles = -1
    if np.all(out_deg == in_deg) and np.all(out_deg <= 1):
        nxt = dict(zip(boundary[:, 0].tolist(), boundary[:, 1].tolist()))
        seen: set = set()
        n_cycles = 0
        for v in nxt:
            if v in seen:
                continue
            n_cycles += 1
            while v not in seen:
                seen.add(v)
                v = nxt[v]
    checks["consistent_connectivity"] = bool(
        np.unique(directed[:, 0] * n + directed[:, 1]).size == directed.shape[0]
        and n_cycles == len(domain.loops)
    )

    scale = np.sqrt(abs(domain_area))
    segs = (
        np.zeros((0, 2), dtype=int)
        if mesh.constraints is None
        else np.asarray(mesh.constraints.segments, dtype=int)
    )
    seg_set = set(map(tuple, np.sort(segs, axis=1).tolist()))
    boundary_set = set(map(tuple, np.sort(boundary, axis=1).tolist()))
    probes = np.vstack([pts[boundary[:, 0]], 0.5 * (pts[boundary[:, 0]] + pts[boundary[:, 1]])])
    checks["constraints_represented"] = bool(
        not missing_constraint_edges(mesh)
        and seg_set == boundary_set
        and np.all(distance_to_boundary(domain, probes) <= rtol * scale)
    )

    centroids = pts[tris].mean(axis=1)
    mesh_area = float(np.sum(area))
    checks["inside_domain"] = bool(
        np.all(points_in_domain(domain, centroids))
        and abs(mesh_area - domain_area) <= rtol * abs(domain_area)
    )

    if obs_points is not None:
        checks["obs_covered"] = bool(
            np.all(_containing_triangle(pts, tris, np.asarray(obs_points, dtype=float)) >= 0)
        )
    ratio = None
    if target_tris is not None:
        ratio = tris.shape[0] / float(target_tris)
        checks["triangle_count"] = bool(count_range[0] <= ratio <= count_range[1])

    report: Dict[str, object] = {
        "checks": checks,
        "n_vertices": int(n),
        "n_triangles": int(tris.shape[0]),
        "n_boundary_vertices": int(np.unique(boundary).size),
        "n_holes": domain.n_holes,
        "domain_area": float(domain_area),
        "mesh_area": mesh_area,
        "min_area": float(np.min(np.abs(area))),
        "median_area": float(np.median(np.abs(area))),
        "max_area": float(np.max(np.abs(area))),
        "min_angle_deg": float(np.min(triangle_min_angles(mesh))),
        "triangle_count_ratio": ratio,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed and raise_on_failure:
        raise ValueError(f"polygon mesh validation failed: {failed}")
    return report
