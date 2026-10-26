"""P1 finite-element basis construction on triangular meshes."""
from __future__ import annotations

import matplotlib.tri as mtri
import numpy as np
from scipy import sparse

from spadag.mesh.core import Mesh


def basis_matrix_p1(
    mesh: Mesh,
    query_points: np.ndarray,
    *,
    clip: bool = True,
    tol: float = 1e-12,
) -> sparse.csr_matrix:
    """
    Evaluate the P1 nodal basis at query points.

    Returns the sparse matrix Phi of shape (n_query, n_vertices) whose rows
    hold the barycentric weights of each point in its triangle (at most three
    nonzeros). Raises ValueError if a point lies outside the mesh or a
    triangle is degenerate.

    Parameters
    ----------
    clip : bool, optional
        Clip barycentric weights to [-tol, 1 + tol] to remove round-off
        violations near triangle edges.
    """
    pts = np.asarray(mesh.points, float)
    tris = np.asarray(mesh.triangles, int)
    Q = np.asarray(query_points, float)

    tri_obj = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles=tris)
    trifinder = tri_obj.get_trifinder()

    tri_idx = trifinder(Q[:, 0], Q[:, 1]).astype(int)
    if np.any(tri_idx < 0):
        outside = np.flatnonzero(tri_idx < 0)
        bad = outside[:10]
        raise ValueError(
            f"{outside.size} query points are outside the triangulation "
            f"(first indices: {bad.tolist()})."
        )

    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]

    A11 = v1[:, 0] - v0[:, 0]
    A21 = v1[:, 1] - v0[:, 1]
    A12 = v2[:, 0] - v0[:, 0]
    A22 = v2[:, 1] - v0[:, 1]
    det = A11 * A22 - A12 * A21

    if np.any(np.abs(det) <= 1e-14):
        bad = np.where(np.abs(det) <= 1e-14)[0][:10]
        raise ValueError(
            f"Degenerate triangle(s) found (det ~ 0), indices: {bad.tolist()}."
        )

    inv11 = A22 / det
    inv12 = -A12 / det
    inv21 = -A21 / det
    inv22 = A11 / det

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    # Group queries by triangle for vectorization.
    order = np.argsort(tri_idx)
    tri_sorted = tri_idx[order]
    Q_sorted = Q[order]
    unique_tris, start_idx = np.unique(tri_sorted, return_index=True)

    start_idx = list(start_idx) + [len(Q_sorted)]
    for u, s, e in zip(unique_tris, start_idx[:-1], start_idx[1:]):
        qq = Q_sorted[s:e]
        t = int(u)

        qx = qq[:, 0] - v0[t, 0]
        qy = qq[:, 1] - v0[t, 1]

        w1 = inv11[t] * qx + inv12[t] * qy
        w2 = inv21[t] * qx + inv22[t] * qy
        w0 = 1.0 - w1 - w2

        if clip:
            w0 = np.clip(w0, -tol, 1.0 + tol)
            w1 = np.clip(w1, -tol, 1.0 + tol)
            w2 = np.clip(w2, -tol, 1.0 + tol)

        orig_rows = order[s:e]
        tri_verts = tris[t]

        rows.extend(np.repeat(orig_rows, 3).tolist())
        cols.extend(np.tile(tri_verts, (e - s, 1)).ravel().tolist())
        data.extend(np.column_stack([w0, w1, w2]).ravel().tolist())

    Phi = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(Q.shape[0], pts.shape[0]),
    ).tocsr()
    return Phi


def basis_matrix(
    mesh: Mesh,
    query_points: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Evaluate the P1 basis, allowing query points outside the mesh.

    Returns
    -------
    Phi : sparse.csr_matrix
        Sparse matrix of shape (n_query, n_vertices). Rows corresponding to
        outside points are all zeros.
    tri_id : np.ndarray
        Triangle index for each query point, or -1 if outside the mesh.
    """
    pts = np.asarray(mesh.points, dtype=float)
    tris = np.asarray(mesh.triangles, dtype=int)
    q = np.asarray(query_points, dtype=float)

    tri_obj = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles=tris)
    finder = tri_obj.get_trifinder()
    tri_id = finder(q[:, 0], q[:, 1]).astype(int)

    n_q = q.shape[0]
    n_nodes = pts.shape[0]
    valid = np.where(tri_id >= 0)[0]
    if valid.size == 0:
        return sparse.csr_matrix((n_q, n_nodes), dtype=float), tri_id

    if valid.size == n_q:
        Phi = basis_matrix_p1(mesh, q, clip=True, tol=1e-12)
        return Phi, tri_id

    Phi_valid = basis_matrix_p1(mesh, q[valid], clip=True, tol=1e-12)
    row_counts = np.diff(Phi_valid.indptr)
    rows = np.repeat(valid, row_counts)
    cols = Phi_valid.indices
    data = Phi_valid.data
    Phi = sparse.coo_matrix((data, (rows, cols)), shape=(n_q, n_nodes)).tocsr()
    return Phi, tri_id


def observation_support(
    mesh: Mesh,
    query_points: np.ndarray,
    *,
    tol: float = 1e-12,
) -> dict:
    """
    Summarize how observations support the P1 basis of a mesh.

    Returns
    -------
    dict
        ``triangle_id`` (containing triangle or -1), ``obs_per_triangle``,
        ``node_weight`` (column sums of Phi), and the counts ``n_outside``
        (points outside the mesh), ``n_empty_triangles`` (triangles containing
        no point) and ``n_unsupported_nodes`` (nodes with total P1 weight
        <= tol, i.e. no observation informs them).
    """
    Phi, tri_id = basis_matrix(mesh, query_points)
    n_tri = np.asarray(mesh.triangles).shape[0]
    obs_per_triangle = np.bincount(tri_id[tri_id >= 0], minlength=n_tri)
    node_weight = np.asarray(Phi.sum(axis=0)).ravel()
    return {
        "triangle_id": tri_id,
        "obs_per_triangle": obs_per_triangle,
        "node_weight": node_weight,
        "n_outside": int(np.sum(tri_id < 0)),
        "n_empty_triangles": int(np.sum(obs_per_triangle == 0)),
        "n_unsupported_nodes": int(np.sum(node_weight <= tol)),
    }
