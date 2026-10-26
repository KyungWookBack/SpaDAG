"""Location-based graph recovery when the inference mesh differs from the truth mesh.

Triangle-level recovery (``spadag.metrics.triangle_edge_recovery``) compares
posterior triangle graphs with the truth on the same triangulation.  When the
inference mesh is built from the observed design (for example a
constrained Delaunay triangulation) its triangles do not correspond to the
generating triangles, so recovery is evaluated at a finite set of spatial
locations instead:

* ``sobol``: a scrambled Sobol design on the unit square (``2**m`` points,
  default ``m = 14`` and seed ``20261001``), weighting the whole domain
  approximately uniformly;
* ``observed``: the realized observation locations, weighting the sampled
  design.

At each location the fitted graph of the inference triangle containing it is
compared with the true graph of the generating triangle containing it, and
true/false positives and false negatives are pooled over location--directed-edge
pairs.  Observed locations are training locations, so the observed summary is
not a hold-out performance measure.

The *representational ceiling* is the largest F1 attainable on the same
locations by any triangle-constant graph on the inference mesh given the true
field, ignoring the DAG and nodal-OR constraints.  It separates what the mesh
can represent from what the sampler recovered.  With identical meshes the
ceiling is one.
"""
from __future__ import annotations

import numpy as np

SOBOL_DEFAULT_M = 14
SOBOL_DEFAULT_SEED = 20261001
EVALUATION_KINDS = ("sobol", "observed")


def sobol_evaluation_points(m: int = SOBOL_DEFAULT_M, seed: int = SOBOL_DEFAULT_SEED) -> np.ndarray:
    """Scrambled two-dimensional Sobol points on ``[0, 1]^2`` (``2**m`` rows)."""
    from scipy.stats import qmc

    return qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(int(m))


def locate(points: np.ndarray, triangles: np.ndarray, query: np.ndarray, *, tolerance: float = 1e-12) -> np.ndarray:
    """Index of the triangle containing each query point.

    Uses barycentric coordinates on every triangle; a point on a shared edge
    is assigned to the lowest-index containing triangle.  Raises ``ValueError``
    when a point lies outside the mesh.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    query = np.asarray(query, dtype=float)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("query must have shape (m, 2)")
    a, b, c = (points[triangles[:, i]] for i in range(3))
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    if np.any(np.abs(det) < 1e-300):
        raise ValueError("degenerate triangle in mesh")
    dx = query[:, None, 0] - a[None, :, 0]
    dy = query[:, None, 1] - a[None, :, 1]
    l1 = ((c[None, :, 1] - a[None, :, 1]) * dx - (c[None, :, 0] - a[None, :, 0]) * dy) / det[None, :]
    l2 = (-(b[None, :, 1] - a[None, :, 1]) * dx + (b[None, :, 0] - a[None, :, 0]) * dy) / det[None, :]
    l0 = 1.0 - l1 - l2
    inside = (l0 >= -tolerance) & (l1 >= -tolerance) & (l2 >= -tolerance)
    found = inside.any(axis=1)
    if not np.all(found):
        raise ValueError(f"{int((~found).sum())} evaluation point(s) outside the mesh")
    return inside.argmax(axis=1).astype(int)


def overlap_counts(fit_ids: np.ndarray, true_ids: np.ndarray, n_fit: int, n_true: int) -> np.ndarray:
    """``(n_fit, n_true)`` matrix counting locations in each triangle pair."""
    fit_ids = np.asarray(fit_ids, dtype=int)
    true_ids = np.asarray(true_ids, dtype=int)
    if fit_ids.shape != true_ids.shape:
        raise ValueError("fit_ids and true_ids must have the same length")
    return np.bincount(fit_ids * n_true + true_ids, minlength=n_fit * n_true).reshape(n_fit, n_true)


def location_recovery(selected: np.ndarray, overlap: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    """TP/FP/FN and TPR/FDR/F1 pooled over locations and directed edges.

    ``selected`` is ``(T_fit, M)`` boolean, ``truth`` is ``(T_true, M)``
    binary, and ``overlap`` is the location-count matrix.  Undefined rates
    (no positives, or no selections for FDR) are reported as ``None``.
    """
    selected = np.asarray(selected, dtype=bool)
    truth = np.asarray(truth)
    overlap = np.asarray(overlap)
    if selected.shape[0] != overlap.shape[0] or truth.shape[0] != overlap.shape[1]:
        raise ValueError("selected/truth triangle counts do not match overlap")
    if selected.shape[1] != truth.shape[1]:
        raise ValueError("selected and truth must index the same directed edges")
    positive = overlap @ truth.astype(np.int64)
    tp = int((selected * positive).sum())
    predicted = int((selected * overlap.sum(axis=1)[:, None]).sum())
    actual = int(positive.sum())
    return dict(TP=tp, FP=predicted - tp, FN=actual - tp,
                TPR=tp / actual if actual else None,
                FDR=(predicted - tp) / predicted if predicted else None,
                F1=2 * tp / (predicted + actual) if predicted + actual else None)


def relaxed_f1_ceiling(overlap: np.ndarray, truth: np.ndarray) -> float | None:
    """Exact F1 optimum over triangle-constant graphs on the finite location set.

    Every inference-triangle/edge decision carries true-positive mass ``a`` and
    selected mass ``w``; sorting by ``a / w`` and checking all prefixes
    maximizes ``2 sum(a) / (truth_mass + sum(w))``.  DAG and nodal-OR
    constraints are not imposed.  Returns ``None`` when the truth is empty on
    the locations.
    """
    overlap = np.asarray(overlap, dtype=float)
    truth = np.asarray(truth)
    a = (overlap @ truth.astype(float)).ravel()
    w = np.repeat(overlap.sum(axis=1), truth.shape[1])
    valid = w > 0
    a, w = a[valid], w[valid]
    total = float(a.sum())
    if not total:
        return None
    order = np.argsort(-(a / w), kind="stable")
    values = 2 * np.cumsum(a[order]) / (total + np.cumsum(w[order]))
    return float(values.max())


def _flatten_pairs(tensor: np.ndarray) -> np.ndarray:
    """``(T, p, p)`` -> ``(T, p(p-1))`` over off-diagonal ordered pairs."""
    tensor = np.asarray(tensor)
    p = tensor.shape[1]
    mask = ~np.eye(p, dtype=bool)
    return tensor[:, mask]


def evaluate_at_locations(
    edge_prob: np.ndarray,
    fit_mesh: tuple[np.ndarray, np.ndarray],
    truth_mesh: tuple[np.ndarray, np.ndarray],
    edge_true: np.ndarray,
    points: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Location-based recovery of ``edge_prob >= threshold`` against ``edge_true``.

    ``fit_mesh`` and ``truth_mesh`` are ``(points, triangles)`` tuples;
    ``edge_prob`` has shape ``(T_fit, p, p)`` and ``edge_true`` ``(T_true, p, p)``.
    """
    edge_prob = np.asarray(edge_prob, dtype=float)
    edge_true = np.asarray(edge_true)
    if edge_prob.ndim != 3 or edge_prob.shape[1] != edge_prob.shape[2]:
        raise ValueError("edge_prob must have shape (T_fit, p, p)")
    if edge_true.shape[1:] != edge_prob.shape[1:]:
        raise ValueError("edge_true and edge_prob must share the variable dimension")
    fit_points, fit_triangles = (np.asarray(x) for x in fit_mesh)
    true_points, true_triangles = (np.asarray(x) for x in truth_mesh)
    if len(fit_triangles) != edge_prob.shape[0] or len(true_triangles) != edge_true.shape[0]:
        raise ValueError("mesh triangle counts do not match the graph arrays")
    fit_ids = locate(fit_points, fit_triangles, points)
    true_ids = locate(true_points, true_triangles, points)
    overlap = overlap_counts(fit_ids, true_ids, len(fit_triangles), len(true_triangles))
    truth = _flatten_pairs(edge_true).astype(np.int64)
    selected = _flatten_pairs(edge_prob) >= threshold
    result = location_recovery(selected, overlap, truth)
    result.update(
        n_locations=int(len(points)),
        threshold=float(threshold),
        representational_f1_ceiling=relaxed_f1_ceiling(overlap, truth),
        fit_triangles=int(len(fit_triangles)),
        truth_triangles=int(len(true_triangles)),
        empty_fit_triangles=int((overlap.sum(axis=1) == 0).sum()),
    )
    return result
