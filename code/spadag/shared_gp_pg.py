"""Pólya-Gamma update of the spatial inclusion fields.

The inclusion model is

    logit(q_eh) = mu + A_h' theta0 + A_h' theta_e,

under the constraints

    w' A theta0 = 0,       sum_e theta_e = 0,

where ``w`` contains normalized P1 finite-element integration weights.  Given
the Pólya-Gamma variables, all coefficients are drawn jointly from their
Gaussian conditional: the edge coefficients are integrated out of the common
block, and the density of their sum at zero adds a Schur-complement term.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve, null_space, solve_triangular

from spadag.mcmc_data import InferenceData
from spadag.mcmc_steps import SamplerState
from spadag.numerics import sigmoid
from spadag.p1_geometry import p1_integration_weights


@dataclass(frozen=True)
class SharedGPDecomposition:
    """Fixed geometry of the constrained inclusion fields."""

    integration_weights: np.ndarray
    common_basis: np.ndarray
    mesh_area: float


@dataclass
class SharedGPState:
    """Common inclusion field and the current mean inclusion probability."""

    mu: float
    theta0: np.ndarray
    f0: np.ndarray
    q_bar: np.ndarray
    decomposition: SharedGPDecomposition


@dataclass
class SharedGPPGStats:
    """Diagnostics from one blocked Pólya-Gamma update."""

    omega_mean: float
    omega_min: float
    omega_max: float
    common_precision_condition: float
    eta_min: float
    eta_max: float
    common_constraint_error: float = float("nan")
    edge_constraint_error: float = float("nan")
    edge_gp_var_raw: float = float("nan")
    backend: str = "bayeslogit_r"


def build_gp_decomposition(data: InferenceData) -> SharedGPDecomposition:
    """Construct the integration weights and the common-field null-space basis."""
    A = np.asarray(data.A_kl, dtype=float)
    if A.shape != (data.H, data.L) or data.L < 1 or not np.all(np.isfinite(A)):
        raise ValueError(f"A_kl must be finite with shape ({data.H}, {data.L}), L > 0")
    # A_kl may have more columns than rows (L = 66 modes on 36 vertices).
    # The Gaussian priors keep every precision positive definite, so all L
    # coordinates are kept; only the constraint w' A theta0 = 0 is imposed.
    weights, mesh_area = p1_integration_weights(
        data.mesh_points, data.triangles
    )
    constraint = A.T @ weights
    tolerance = np.finfo(float).eps * max(
        1.0, float(np.linalg.norm(A, ord=2))
    ) * 64.0
    if float(np.linalg.norm(constraint)) <= tolerance:
        common_basis = np.eye(data.L)
    else:
        common_basis = null_space(constraint[None, :])
    if common_basis.shape[0] != data.L:
        raise RuntimeError("common null-space basis has an invalid shape")
    if common_basis.shape[1] == 0:
        raise ValueError("the common GP must retain at least one free coefficient")
    orthogonality_error = float(np.max(np.abs(
        common_basis.T @ common_basis
        - np.eye(common_basis.shape[1])
    ))) if common_basis.shape[1] else 0.0
    constraint_error = float(np.max(np.abs(
        constraint @ common_basis
    ))) if common_basis.shape[1] else 0.0
    if orthogonality_error > 1e-10 or constraint_error > 1e-10:
        raise RuntimeError("failed to construct the common-GP null-space basis")
    return SharedGPDecomposition(
        integration_weights=weights,
        common_basis=common_basis,
        mesh_area=mesh_area,
    )


def constrained_edge_raw_variance(
    edge_gp_var: float,
    n_pairs: int,
) -> float:
    """Rescale raw variance so constrained edge marginals retain edge_gp_var."""
    if not np.isfinite(edge_gp_var) or edge_gp_var <= 0.0:
        raise ValueError("edge_gp_var must be finite and positive")
    if n_pairs < 2:
        raise ValueError("sum-to-zero decomposition requires at least two pairs")
    return float(edge_gp_var * n_pairs / (n_pairs - 1))


def edge_gp_variance_posterior_parameters(
    data: InferenceData,
    st: SamplerState,
    *,
    prior_shape: float,
    prior_scale: float,
) -> tuple[float, float]:
    """Return shape-scale parameters for the conditional law of ``tau^2``.

    The inverse-Gamma convention is

        p(x) proportional to x^(-a-1) exp(-b/x).

    ``tau^2`` is the marginal variance of each edge coefficient after
    imposing ``sum_e theta_e = 0``, so the raw variance on the
    ``(E - 1) * L`` dimensional contrast space is ``tau^2 * E / (E - 1)``.
    """
    if not np.isfinite(prior_shape) or prior_shape <= 0.0:
        raise ValueError("prior_shape must be finite and positive")
    if not np.isfinite(prior_scale) or prior_scale <= 0.0:
        raise ValueError("prior_scale must be finite and positive")

    pair_theta = np.asarray(
        [st.theta[j, k] for j, k in data.pairs], dtype=float
    )
    if pair_theta.size == 0:
        pair_theta = np.empty((0, data.L), dtype=float)
    if pair_theta.shape != (len(data.pairs), data.L):
        raise ValueError("edge coefficient array has an invalid shape")
    if np.any(~np.isfinite(pair_theta)):
        raise ValueError("edge coefficients must be finite")

    n_pairs = len(data.pairs)
    if n_pairs < 2:
        raise ValueError("sum-to-zero decomposition requires at least two pairs")
    sum_squares = float(np.sum(pair_theta * pair_theta))
    constraint_error = float(np.max(np.abs(np.sum(pair_theta, axis=0))))
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(pair_theta), initial=0.0)))
    if constraint_error > tolerance:
        raise ValueError("edge coefficients do not satisfy the sum-to-zero constraint")
    free_dimension = (n_pairs - 1) * data.L
    quadratic_scale = (n_pairs - 1) / n_pairs * sum_squares

    posterior_shape = prior_shape + 0.5 * free_dimension
    posterior_scale = prior_scale + 0.5 * quadratic_scale
    return float(posterior_shape), float(posterior_scale)


def draw_edge_gp_variance(
    data: InferenceData,
    st: SamplerState,
    rng: np.random.Generator,
    *,
    prior_shape: float,
    prior_scale: float,
) -> float:
    """Draw ``tau^2`` from its inverse-Gamma full conditional."""
    posterior_shape, posterior_scale = (
        edge_gp_variance_posterior_parameters(
            data,
            st,
            prior_shape=prior_shape,
            prior_scale=prior_scale,
        )
    )
    precision = float(rng.gamma(
        shape=posterior_shape,
        scale=1.0 / posterior_scale,
    ))
    edge_gp_var = 1.0 / precision
    if not np.isfinite(edge_gp_var) or edge_gp_var <= 0.0:
        raise RuntimeError("inverse-Gamma edge variance draw is invalid")
    return float(edge_gp_var)


def common_gp_variance_posterior_parameters(
    shared: SharedGPState,
    *,
    prior_shape: float,
    prior_scale: float,
) -> tuple[float, float]:
    """Return shape-scale parameters for the conditional law of ``sigma0^2``.

    The inverse-Gamma convention matches
    ``edge_gp_variance_posterior_parameters``:

        p(x) proportional to x^(-a-1) exp(-b/x).

    With ``theta0 = Q0 xi0`` and ``xi0 ~ N(0, sigma0^2 I_d0)`` the common-GP
    coefficients contribute a single Gaussian quadratic form, so

        sigma0^2 | xi0 ~ IG(a0 + d0 / 2, b0 + ||xi0||^2 / 2).

    ``d0 = Q0.shape[1]`` is ``L - 1`` unless ``w' A = 0``.  ``xi0`` is
    recovered as ``Q0' theta0``; since ``Q0`` has orthonormal columns,
    ``||xi0|| == ||theta0||`` exactly when ``theta0`` lies in its column
    space, which the check below enforces.
    """
    if not np.isfinite(prior_shape) or prior_shape <= 0.0:
        raise ValueError("prior_shape must be finite and positive")
    if not np.isfinite(prior_scale) or prior_scale <= 0.0:
        raise ValueError("prior_scale must be finite and positive")

    theta0 = np.asarray(shared.theta0, dtype=float)
    if theta0.ndim != 1:
        raise ValueError("theta0 must be one-dimensional")
    if np.any(~np.isfinite(theta0)):
        raise ValueError("theta0 must be finite")

    common_basis = np.asarray(shared.decomposition.common_basis, dtype=float)
    if common_basis.ndim != 2 or common_basis.shape[0] != theta0.size:
        raise ValueError("common basis rows must match theta0")
    free_dimension = common_basis.shape[1]
    if free_dimension == 0:
        raise ValueError("the common GP has no free coefficient direction")

    xi0 = common_basis.T @ theta0
    sum_squares = float(xi0 @ xi0)
    theta0_squares = float(theta0 @ theta0)
    tolerance = 1e-10 * max(1.0, theta0_squares)
    if abs(theta0_squares - sum_squares) > tolerance:
        raise ValueError(
            "theta0 does not lie in the common-GP basis column space"
        )

    posterior_shape = prior_shape + 0.5 * free_dimension
    posterior_scale = prior_scale + 0.5 * sum_squares
    return float(posterior_shape), float(posterior_scale)


def draw_common_gp_variance(
    shared: SharedGPState,
    rng: np.random.Generator,
    *,
    prior_shape: float,
    prior_scale: float,
) -> float:
    """Draw ``sigma0^2`` from its inverse-Gamma full conditional."""
    posterior_shape, posterior_scale = (
        common_gp_variance_posterior_parameters(
            shared,
            prior_shape=prior_shape,
            prior_scale=prior_scale,
        )
    )
    precision = float(rng.gamma(
        shape=posterior_shape,
        scale=1.0 / posterior_scale,
    ))
    common_gp_var = 1.0 / precision
    if not np.isfinite(common_gp_var) or common_gp_var <= 0.0:
        raise RuntimeError("inverse-Gamma common variance draw is invalid")
    return float(common_gp_var)


def shared_gp_constraint_errors(
    data: InferenceData,
    st: SamplerState,
    shared: SharedGPState,
) -> tuple[float, float]:
    """Return weighted-common and edge-coefficient constraint errors."""
    common_error = float(abs(shared.decomposition.integration_weights @ shared.f0))
    if data.pairs:
        theta_sum = np.sum(
            [st.theta[j, k] for j, k in data.pairs], axis=0
        )
        edge_error = float(np.max(np.abs(theta_sum)))
    else:
        edge_error = 0.0
    return common_error, edge_error


def _validate_prior_scales(
    *,
    common_gp_var: float,
    edge_gp_var: float,
    mu_prior_sd: float,
) -> None:
    if not np.isfinite(common_gp_var) or common_gp_var <= 0.0:
        raise ValueError("common_gp_var must be finite and positive")
    if not np.isfinite(edge_gp_var) or edge_gp_var <= 0.0:
        raise ValueError("edge_gp_var must be finite and positive")
    if not np.isfinite(mu_prior_sd) or mu_prior_sd <= 0.0:
        raise ValueError("mu_prior_sd must be finite and positive")


def _draw_from_precision_factor(
    factor: tuple[np.ndarray, bool],
    rhs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``N(Q^-1 rhs, Q^-1)`` from a Cholesky factorization of ``Q``."""
    chol, low = factor
    mean = cho_solve((chol, low), rhs, check_finite=False)
    lower = np.tril(chol) if low else np.tril(chol.T)
    noise = solve_triangular(
        lower,
        rng.standard_normal(rhs.size),
        lower=True,
        trans="T",
        check_finite=False,
    )
    return mean + noise


def refresh_shared_gp_probabilities(
    data: InferenceData,
    st: SamplerState,
    shared: SharedGPState,
) -> tuple[float, float]:
    """Recompute ``q = logit^-1(eta)`` for every candidate pair from the current fields."""
    shared.f0 = data.A_kl @ shared.theta0
    q_sum = np.zeros(data.H, dtype=float)
    eta_min = float("inf")
    eta_max = float("-inf")
    for j, k in data.pairs:
        eta = shared.mu + shared.f0 + data.A_kl @ st.theta[j, k]
        q = sigmoid(eta)
        st.m[j, k] = q
        # The indicator and reversal updates read q from st.pi.
        st.pi[j, k] = q
        q_sum += q
        eta_min = min(eta_min, float(np.min(eta)))
        eta_max = max(eta_max, float(np.max(eta)))
    shared.q_bar = q_sum / max(len(data.pairs), 1)
    if not data.pairs:
        eta_min = float("nan")
        eta_max = float("nan")
    return eta_min, eta_max


def init_shared_gp_state(
    data: InferenceData,
    st: SamplerState,
    *,
    mu_init: float,
    decomposition: SharedGPDecomposition,
    theta0_init: np.ndarray | None = None,
) -> SharedGPState:
    """Initialize the inclusion fields and project them onto the constraints."""
    if not np.isfinite(mu_init):
        raise ValueError("mu_init must be finite")
    if theta0_init is None:
        theta0 = np.zeros(data.L, dtype=float)
    else:
        theta0 = np.asarray(theta0_init, dtype=float).copy()
        if theta0.shape != (data.L,):
            raise ValueError(f"theta0_init must have shape ({data.L},)")
        if np.any(~np.isfinite(theta0)):
            raise ValueError("theta0_init must be finite")
    common_basis = decomposition.common_basis
    theta0 = common_basis @ (common_basis.T @ theta0)
    pair_theta = np.asarray([st.theta[j, k] for j, k in data.pairs])
    if pair_theta.size:
        pair_theta -= pair_theta.mean(axis=0, keepdims=True)
        for (j, k), theta in zip(data.pairs, pair_theta):
            st.theta[j, k] = theta
    shared = SharedGPState(
        mu=float(mu_init),
        theta0=theta0,
        f0=np.zeros(data.H, dtype=float),
        q_bar=np.zeros(data.H, dtype=float),
        decomposition=decomposition,
    )
    refresh_shared_gp_probabilities(data, st, shared)
    common_error, edge_error = shared_gp_constraint_errors(data, st, shared)
    if max(common_error, edge_error) > 1e-10:
        raise RuntimeError("constrained shared-GP initialization failed")
    return shared


def _split_by_pair(
    values: np.ndarray,
    *,
    n_pairs: int,
    H: int,
) -> list[np.ndarray]:
    if values.size != n_pairs * H:
        raise ValueError(
            f"expected {n_pairs * H} Pólya-Gamma draws, got {values.size}"
        )
    return [
        values[position * H:(position + 1) * H]
        for position in range(n_pairs)
    ]


def draw_constrained_coefficients(
    data: InferenceData,
    st: SamplerState,
    shared: SharedGPState,
    omega_by_pair: list[np.ndarray],
    rng: np.random.Generator,
    *,
    common_gp_var: float,
    edge_gp_var: float,
    mu_prior_mean: float,
    mu_prior_sd: float,
) -> tuple[float, float]:
    """Draw the joint Gaussian coefficient block under both constraints.

    ``edge_gp_var`` is the target marginal variance after imposing
    ``sum_e theta_e = 0``.  The independent Gaussian representation therefore
    uses raw variance ``edge_gp_var * E / (E - 1)`` before conditioning.
    """
    _validate_prior_scales(
        common_gp_var=common_gp_var,
        edge_gp_var=edge_gp_var,
        mu_prior_sd=mu_prior_sd,
    )
    if len(omega_by_pair) != len(data.pairs):
        raise ValueError("omega_by_pair must align with data.pairs")
    if not np.isfinite(mu_prior_mean):
        raise ValueError("mu_prior_mean must be finite")
    geometry = shared.decomposition
    n_pairs = len(data.pairs)
    edge_gp_var_raw = constrained_edge_raw_variance(
        edge_gp_var, n_pairs
    )
    A = data.A_kl
    Q0 = geometry.common_basis
    A0 = A @ Q0
    X0 = np.column_stack([np.ones(data.H), A0])
    common_dim = X0.shape[1]
    common_precision = np.zeros((common_dim, common_dim), dtype=float)
    common_precision[0, 0] = 1.0 / (mu_prior_sd * mu_prior_sd)
    if common_dim > 1:
        common_precision[1:, 1:] = (
            np.eye(common_dim - 1) / common_gp_var
        )
    common_rhs = np.zeros(common_dim, dtype=float)
    common_rhs[0] = mu_prior_mean / (mu_prior_sd * mu_prior_sd)

    edge_factors: list[tuple[np.ndarray, bool]] = []
    edge_covariances: list[np.ndarray] = []
    edge_a: list[np.ndarray] = []
    edge_B: list[np.ndarray] = []
    covariance_sum = np.zeros((data.L, data.L), dtype=float)
    mean_numerator_sum = np.zeros(data.L, dtype=float)
    mean_design_sum = np.zeros((data.L, common_dim), dtype=float)

    for (j, k), omega in zip(data.pairs, omega_by_pair):
        omega = np.asarray(omega, dtype=float)
        if omega.shape != (data.H,):
            raise ValueError(f"each omega vector must have shape ({data.H},)")
        if np.any(~np.isfinite(omega)) or np.any(omega <= 0.0):
            raise ValueError("Pólya-Gamma draws must be finite and positive")

        kappa = st.gamma[j, k].astype(float) - 0.5
        weighted_A = omega[:, None] * A
        weighted_X0 = omega[:, None] * X0
        edge_precision = (
            A.T @ weighted_A
            + np.eye(data.L) / edge_gp_var_raw
        )
        edge_factor = cho_factor(
            edge_precision, lower=True, check_finite=False
        )
        covariance = cho_solve(
            edge_factor, np.eye(data.L), check_finite=False
        )
        a_e = A.T @ kappa
        B_e = X0.T @ weighted_A
        covariance_a = covariance @ a_e
        covariance_Bt = covariance @ B_e.T

        common_precision += (
            X0.T @ weighted_X0
            - B_e @ covariance_Bt
        )
        common_rhs += X0.T @ kappa - B_e @ covariance_a
        covariance_sum += covariance
        mean_numerator_sum += covariance_a
        mean_design_sum += covariance_Bt
        edge_factors.append(edge_factor)
        edge_covariances.append(covariance)
        edge_a.append(a_e)
        edge_B.append(B_e)

    covariance_sum_factor = cho_factor(
        covariance_sum, lower=True, check_finite=False
    )
    solved_mean_design = cho_solve(
        covariance_sum_factor,
        mean_design_sum,
        check_finite=False,
    )
    solved_mean_numerator = cho_solve(
        covariance_sum_factor,
        mean_numerator_sum,
        check_finite=False,
    )
    common_precision += mean_design_sum.T @ solved_mean_design
    common_rhs += mean_design_sum.T @ solved_mean_numerator
    common_precision = 0.5 * (common_precision + common_precision.T)

    common_factor = cho_factor(
        common_precision, lower=True, check_finite=False
    )
    z = _draw_from_precision_factor(common_factor, common_rhs, rng)
    shared.mu = float(z[0])
    shared.theta0 = Q0 @ z[1:]

    unconstrained_draws: list[np.ndarray] = []
    for edge_factor, a_e, B_e in zip(
        edge_factors, edge_a, edge_B
    ):
        conditional_rhs = a_e - B_e.T @ z
        unconstrained_draws.append(_draw_from_precision_factor(
            edge_factor, conditional_rhs, rng
        ))

    draw_sum = np.sum(unconstrained_draws, axis=0)
    correction_weight = cho_solve(
        covariance_sum_factor, draw_sum, check_finite=False
    )
    for (j, k), draw, covariance in zip(
        data.pairs, unconstrained_draws, edge_covariances
    ):
        st.theta[j, k] = draw - covariance @ correction_weight

    return float(np.linalg.cond(common_precision)), edge_gp_var_raw


def update_shared_gp_pg(
    data: InferenceData,
    st: SamplerState,
    shared: SharedGPState,
    rng: np.random.Generator,
    *,
    draw_omega: Callable[[np.ndarray], np.ndarray],
    common_gp_var: float,
    edge_gp_var: float,
    mu_prior_mean: float,
    mu_prior_sd: float,
) -> SharedGPPGStats:
    """Draw the Pólya-Gamma variables, then all inclusion-field coefficients jointly."""
    _validate_prior_scales(
        common_gp_var=common_gp_var,
        edge_gp_var=edge_gp_var,
        mu_prior_sd=mu_prior_sd,
    )
    pairs = list(data.pairs)
    common_surface = shared.mu + data.A_kl @ shared.theta0
    eta_by_pair = [
        common_surface + data.A_kl @ st.theta[j, k]
        for j, k in pairs
    ]
    eta_all = np.concatenate(eta_by_pair) if eta_by_pair else np.empty(0)
    omega_all = np.asarray(draw_omega(eta_all), dtype=float).reshape(-1)
    omega_by_pair = _split_by_pair(
        omega_all, n_pairs=len(pairs), H=data.H
    )
    condition, edge_gp_var_raw = draw_constrained_coefficients(
        data,
        st,
        shared,
        omega_by_pair,
        rng,
        common_gp_var=common_gp_var,
        edge_gp_var=edge_gp_var,
        mu_prior_mean=mu_prior_mean,
        mu_prior_sd=mu_prior_sd,
    )
    eta_min, eta_max = refresh_shared_gp_probabilities(data, st, shared)
    common_error, edge_error = shared_gp_constraint_errors(data, st, shared)

    return SharedGPPGStats(
        omega_mean=float(np.mean(omega_all)) if omega_all.size else float("nan"),
        omega_min=float(np.min(omega_all)) if omega_all.size else float("nan"),
        omega_max=float(np.max(omega_all)) if omega_all.size else float("nan"),
        common_precision_condition=condition,
        eta_min=eta_min,
        eta_max=eta_max,
        common_constraint_error=common_error,
        edge_constraint_error=edge_error,
        edge_gp_var_raw=edge_gp_var_raw,
    )
