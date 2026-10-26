"""Linear structural equation model on a triangulated domain.

Provides simulation, per-column standardization and calibration of the
population signal-to-noise ratio (SNR).  Coefficients are P1 surfaces ``b[j, k]``; within
each triangle the local graph ``edge_triangle[t]`` is a DAG and observations
follow ``X_j = sum_k beta_jk(s) X_k + e_j`` in a topological order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SNR_RELATIVE_TOLERANCE = 1e-3


MAX_COVARIANCE_CONDITION = 1e6


@dataclass(frozen=True)
class PopulationSNR:
    values: np.ndarray
    slot_indices: np.ndarray
    median: float
    covariance_condition_max: float
    covariance_min_eigenvalue: float


def normalized_triangle_quadrature(order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return barycentric points and weights that average over a triangle.

    The rule is a collapsed (Duffy) tensor product of ``order``-point
    Gauss-Legendre rules; the weights sum to one.
    """
    if order <= 0:
        raise ValueError("quadrature order must be positive")
    nodes, raw_weights = np.polynomial.legendre.leggauss(order)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * raw_weights
    barycentric: list[tuple[float, float, float]] = []
    weights: list[float] = []
    for u, weight_u in zip(unit_nodes, unit_weights):
        for v, weight_v in zip(unit_nodes, unit_weights):
            barycentric.append(
                ((1.0 - u) * (1.0 - v), u, (1.0 - u) * v)
            )
            weights.append(2.0 * weight_u * weight_v * (1.0 - u))
    barycentric_array = np.asarray(barycentric, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if not np.isclose(weight_array.sum(), 1.0, atol=1e-13):
        raise RuntimeError("triangle quadrature weights do not sum to one")
    return barycentric_array, weight_array


def _quadrature_coefficients(
    b_coeffs: np.ndarray,
    triangles: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    barycentric, weights = normalized_triangle_quadrature(order)
    nodal = b_coeffs[:, :, triangles].transpose(2, 3, 0, 1)
    coefficient = np.einsum("tvjk,qv->tqjk", nodal, barycentric)
    return coefficient, weights


def population_child_triangle_snr(
    b_coeffs: np.ndarray,
    triangles: np.ndarray,
    edge_triangle: np.ndarray,
    sigma_noise: np.ndarray,
    *,
    order: int = 5,
    compute_covariance_diagnostics: bool = True,
) -> PopulationSNR:
    """Population SNR of every (triangle, child) pair whose child has a parent there.

    The SNR is the triangle average of Var(sum_k beta_jk X_k) / sigma_j^2 under
    the local linear SEM, computed by quadrature.
    """
    b_coeffs = np.asarray(b_coeffs, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    edge_triangle = np.asarray(edge_triangle, dtype=bool)
    sigma_noise = np.asarray(sigma_noise, dtype=float)
    p = b_coeffs.shape[0]
    if b_coeffs.shape[:2] != (p, p) or sigma_noise.shape != (p,):
        raise ValueError("coefficient and noise shapes do not agree")
    if np.any(sigma_noise <= 0.0):
        raise ValueError("noise standard deviations must be positive")

    coefficient, weights = _quadrature_coefficients(b_coeffs, triangles, order)
    identity = np.eye(p)
    inverse = np.linalg.inv(identity[None, None] - coefficient)
    noise_variance = sigma_noise * sigma_noise
    covariance = (inverse * noise_variance[None, None, None, :]) @ np.swapaxes(
        inverse, -1, -2
    )
    signal = np.sum((coefficient @ covariance) * coefficient, axis=-1)
    integrated = np.einsum("q,tqj->tj", weights, signal) / noise_variance
    slot_mask = edge_triangle.any(axis=2)
    slot_indices = np.argwhere(slot_mask)
    values = integrated[slot_mask]
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError("population SNR is empty or non-finite")

    if compute_covariance_diagnostics:
        covariance_flat = covariance.reshape((-1, p, p))
        eigenvalues = np.linalg.eigvalsh(covariance_flat)
        minimum_eigenvalue = float(eigenvalues[:, 0].min())
        condition_max = float(
            np.max(eigenvalues[:, -1] / eigenvalues[:, 0])
        )
    else:
        minimum_eigenvalue = float("nan")
        condition_max = float("nan")
    return PopulationSNR(
        values=values,
        slot_indices=slot_indices,
        median=float(np.median(values)),
        covariance_condition_max=condition_max,
        covariance_min_eigenvalue=minimum_eigenvalue,
    )


def calibrate_global_scale(
    base_coefficients: np.ndarray,
    triangles: np.ndarray,
    edge_triangle: np.ndarray,
    sigma_noise: np.ndarray,
    target_snr: float,
    *,
    order: int = 5,
) -> tuple[float, PopulationSNR]:
    """Find by bisection the common coefficient multiplier whose SNR median equals ``target_snr``."""
    if target_snr <= 0.0:
        raise ValueError("target SNR must be positive")

    upper = 1.0
    upper_summary = population_child_triangle_snr(
        upper * base_coefficients,
        triangles,
        edge_triangle,
        sigma_noise,
        order=order,
        compute_covariance_diagnostics=False,
    )
    while upper_summary.median < target_snr:
        upper *= 2.0
        if upper > 128.0:
            raise RuntimeError("could not bracket target population SNR")
        upper_summary = population_child_triangle_snr(
            upper * base_coefficients,
            triangles,
            edge_triangle,
            sigma_noise,
            order=order,
            compute_covariance_diagnostics=False,
        )

    lower = 0.0
    summary = upper_summary
    for _ in range(48):
        scale = 0.5 * (lower + upper)
        summary = population_child_triangle_snr(
            scale * base_coefficients,
            triangles,
            edge_triangle,
            sigma_noise,
            order=order,
            compute_covariance_diagnostics=False,
        )
        if summary.median < target_snr:
            lower = scale
        else:
            upper = scale
    scale = 0.5 * (lower + upper)
    summary = population_child_triangle_snr(
        scale * base_coefficients,
        triangles,
        edge_triangle,
        sigma_noise,
        order=order,
        compute_covariance_diagnostics=True,
    )
    tolerance = target_snr * SNR_RELATIVE_TOLERANCE
    if abs(summary.median - target_snr) > tolerance:
        raise RuntimeError(
            f"target SNR gate failed: {summary.median} versus {target_snr}"
        )
    if summary.covariance_min_eigenvalue <= 0.0:
        raise RuntimeError("population covariance is not positive definite")
    if summary.covariance_condition_max > MAX_COVARIANCE_CONDITION:
        raise RuntimeError("population covariance condition-number gate failed")
    return scale, summary


def topological_order(adj: np.ndarray) -> np.ndarray:
    """Return one topological order for adj[j,k]=1 meaning k -> j."""
    adj = np.asarray(adj, dtype=int)
    indegree = adj.sum(axis=1).astype(int)
    queue = [int(v) for v in np.flatnonzero(indegree == 0)]
    order: list[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in np.flatnonzero(adj[:, node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    if len(order) != adj.shape[0]:
        raise RuntimeError("triangle graph is cyclic")
    return np.asarray(order, dtype=np.int64)


def simulate_sem(
    Phi: np.ndarray,
    triangle_ids: np.ndarray,
    edge_triangle: np.ndarray,
    b_coeffs: np.ndarray,
    sigma_noise: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta_obs = np.einsum("ih,jkh->ijk", Phi, b_coeffs)
    orders = np.stack([topological_order(graph) for graph in edge_triangle])
    observation_orders = orders[triangle_ids]
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, sigma_noise, size=(len(Phi), len(sigma_noise)))
    observations = np.zeros_like(innovations)
    for index, order in enumerate(observation_orders):
        for child in order:
            observations[index, child] = (
                beta_obs[index, child] @ observations[index]
                + innovations[index, child]
            )
    return observations, innovations, beta_obs, orders


def standardize_sem(
    X_raw: np.ndarray,
    innovations_raw: np.ndarray,
    b_raw: np.ndarray,
    Phi: np.ndarray,
    integration_weights: np.ndarray,
    sigma_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    """Standardize each column and rewrite the SEM in the standardized scale.

    Coefficients scale by sd_k / sd_j, and centring introduces a spatially
    varying intercept split into its domain average ``a0`` and a P1 surface
    ``a_spatial`` with zero area-weighted mean.
    """
    means = X_raw.mean(axis=0)
    standard_deviations = X_raw.std(axis=0)
    if np.any(~np.isfinite(standard_deviations)) or np.any(
        standard_deviations <= 1e-12
    ):
        raise RuntimeError("cannot standardize a degenerate observation column")
    X = (X_raw - means) / standard_deviations
    scale_ratio = (
        standard_deviations[None, :, None]
        / standard_deviations[:, None, None]
    )
    b_coeffs = b_raw * scale_ratio
    beta_obs = np.einsum("ih,jkh->ijk", Phi, b_coeffs)
    alpha_node = (
        np.einsum("jkh,k->jh", b_raw, means) - means[:, None]
    ) / standard_deviations[:, None]
    a0 = alpha_node @ integration_weights
    a_spatial = alpha_node - a0[:, None]
    alpha_obs = Phi @ a_spatial.T + a0[None, :]
    innovations = innovations_raw / standard_deviations
    sigma_noise = sigma_raw / standard_deviations
    return {
        "X": X,
        "b_coeffs": b_coeffs,
        "beta_obs": beta_obs,
        "a0": a0,
        "a_spatial": a_spatial,
        "alpha_obs": alpha_obs,
        "innovations": innovations,
        "sigma_noise": sigma_noise,
        "column_mean": means,
        "column_sd": standard_deviations,
    }


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    labels = ("min", "q10", "q25", "median", "q75", "q90", "max")
    probabilities = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    return {
        label: float(value)
        for label, value in zip(labels, np.quantile(values, probabilities))
    }
