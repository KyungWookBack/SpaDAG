"""P1 nodal integration weights and the design of an area-centred baseline surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space


@dataclass(frozen=True)
class BaselineGeometry:
    """Fixed design objects for an area-centered P1 baseline surface."""

    integration_weights: np.ndarray
    spatial_basis: np.ndarray
    design: np.ndarray
    crossproduct: np.ndarray


def p1_integration_weights(
    mesh_points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return P1 nodal integration weights normalized to sum to one, and the mesh area.

    The weight of node h is one third of the area of the triangles containing
    h, so ``w @ a`` is the domain average of the P1 surface with nodal values a.
    """
    points = np.asarray(mesh_points, dtype=float)
    cells = np.asarray(triangles, dtype=int)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("mesh_points must have shape (H, 2)")
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("triangles must have shape (T, 3)")
    if points.shape[0] == 0 or cells.shape[0] == 0:
        raise ValueError("mesh must contain points and triangles")
    if np.any(~np.isfinite(points)):
        raise ValueError("mesh_points must be finite")
    if np.any(cells < 0) or np.any(cells >= points.shape[0]):
        raise ValueError("triangle index is outside mesh_points")

    triangle_points = points[cells]
    first = triangle_points[:, 1] - triangle_points[:, 0]
    second = triangle_points[:, 2] - triangle_points[:, 0]
    signed_double_area = (
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    )
    areas = 0.5 * np.abs(signed_double_area)
    area_tolerance = (
        np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(points)))) ** 2
        * 32.0
    )
    if np.any(~np.isfinite(areas)) or np.any(areas <= area_tolerance):
        raise ValueError("all mesh triangles must have positive finite area")

    weights = np.zeros(points.shape[0], dtype=float)
    for local_vertex in range(3):
        np.add.at(weights, cells[:, local_vertex], areas / 3.0)
    mesh_area = float(np.sum(areas))
    if np.any(weights <= 0.0) or not np.isfinite(mesh_area):
        raise ValueError("every mesh node must have positive integration weight")
    if not np.isclose(
        float(np.sum(weights)),
        mesh_area,
        rtol=1e-12,
        atol=1e-14 * max(mesh_area, 1.0),
    ):
        raise RuntimeError("P1 integration weights do not sum to mesh area")
    return weights / mesh_area, mesh_area


def build_baseline_geometry(
    Phi: np.ndarray,
    mesh_points: np.ndarray,
    triangles: np.ndarray,
) -> BaselineGeometry:
    """Build the baseline design ``[1, Phi Q_w]``.

    ``Q_w`` is an orthonormal basis of the null space of the integration
    weights w, so the spatial part ``a = Q_w c`` satisfies ``w' a = 0`` and the
    intercept is the domain average of the baseline.
    """
    basis = np.asarray(Phi, dtype=float)
    if basis.ndim != 2:
        raise ValueError("Phi must be a two-dimensional P1 design")
    n, H = basis.shape
    weights, _ = p1_integration_weights(mesh_points, triangles)
    if weights.shape != (H,):
        raise ValueError("Phi and mesh_points must have the same node count")

    spatial_basis = null_space(weights[None, :])
    expected_shape = (H, H - 1)
    if spatial_basis.shape != expected_shape:
        raise RuntimeError(
            "baseline null-space basis has shape "
            f"{spatial_basis.shape}, expected {expected_shape}"
        )
    constraint_error = float(np.max(np.abs(weights @ spatial_basis)))
    orthogonality_error = float(
        np.max(
            np.abs(
                spatial_basis.T @ spatial_basis
                - np.eye(spatial_basis.shape[1])
            )
        )
    )
    if constraint_error > 1e-10 or orthogonality_error > 1e-10:
        raise RuntimeError("failed to construct the baseline null-space basis")

    design = np.column_stack([np.ones(n, dtype=float), basis @ spatial_basis])
    return BaselineGeometry(
        integration_weights=weights,
        spatial_basis=spatial_basis,
        design=design,
        crossproduct=design.T @ design,
    )
