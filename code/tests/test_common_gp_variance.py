"""Conditional-law checks for the ``sigma0^2`` update."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import invgamma, norm

from spadag.cli import parse_args, run_shared_gp_pg
from spadag.mcmc_data import load_inference_data
from spadag.shared_gp_pg import (
    SharedGPState,
    build_gp_decomposition,
    common_gp_variance_posterior_parameters,
    draw_common_gp_variance,
)
from tests.toy_data_fixture import temporary_toy_bundle


def _fixed_positive_omega(eta):
    return np.full(np.asarray(eta).shape, 0.25, dtype=float)


def _shared_state(data, decomposition, xi0):
    """Return a state whose theta0 is Q0 xi0 for the given decomposition."""
    common_basis = decomposition.common_basis
    theta0 = common_basis @ np.asarray(xi0, dtype=float)
    return SharedGPState(
        mu=0.0,
        theta0=theta0,
        f0=data.A_kl @ theta0,
        q_bar=np.zeros(data.H, dtype=float),
        decomposition=decomposition,
    )


def test_posterior_parameters_use_runtime_d0_and_xi0_norm() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
    decomposition = build_gp_decomposition(data)
    d0 = decomposition.common_basis.shape[1]
    assert d0 == data.L - 1

    xi0 = np.linspace(0.5, 1.5, d0)
    shared = _shared_state(data, decomposition, xi0)
    shape, scale = common_gp_variance_posterior_parameters(
        shared, prior_shape=2.0, prior_scale=1.0
    )
    assert shape == pytest.approx(2.0 + 0.5 * d0)
    assert scale == pytest.approx(1.0 + 0.5 * float(xi0 @ xi0))


def test_orthonormal_basis_equates_xi0_and_theta0_norms() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
    decomposition = build_gp_decomposition(data)
    common_basis = decomposition.common_basis
    assert common_basis.T @ common_basis == pytest.approx(
        np.eye(common_basis.shape[1]), abs=1e-12
    )

    xi0 = np.full(common_basis.shape[1], 1.75)
    shared = _shared_state(data, decomposition, xi0)
    theta0 = shared.theta0
    assert float(theta0 @ theta0) == pytest.approx(float(xi0 @ xi0), rel=1e-12)

    _, scale = common_gp_variance_posterior_parameters(
        shared, prior_shape=1.0, prior_scale=1.0
    )
    assert scale == pytest.approx(1.0 + 0.5 * float(theta0 @ theta0))


def test_theta0_outside_the_basis_column_space_is_rejected() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
    decomposition = build_gp_decomposition(data)
    common_basis = decomposition.common_basis
    shared = _shared_state(
        data, decomposition, np.ones(common_basis.shape[1])
    )
    # The constraint direction A' w is orthogonal to every basis column, so
    # adding it takes theta0 out of the constrained space.
    constraint = data.A_kl.T @ decomposition.integration_weights
    shared.theta0 = shared.theta0 + constraint
    with pytest.raises(ValueError, match="column space"):
        common_gp_variance_posterior_parameters(
            shared, prior_shape=2.0, prior_scale=1.0
        )


@pytest.mark.parametrize(
    "prior_shape,prior_scale",
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0), (np.nan, 1.0)],
)
def test_invalid_priors_are_rejected(prior_shape, prior_scale) -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
    decomposition = build_gp_decomposition(data)
    shared = _shared_state(
        data, decomposition, np.ones(decomposition.common_basis.shape[1])
    )
    with pytest.raises(ValueError):
        common_gp_variance_posterior_parameters(
            shared, prior_shape=prior_shape, prior_scale=prior_scale
        )


def test_draw_matches_the_inverse_gamma_full_conditional() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
    decomposition = build_gp_decomposition(data)
    d0 = decomposition.common_basis.shape[1]
    shared = _shared_state(data, decomposition, np.full(d0, 1.25))
    shape, scale = common_gp_variance_posterior_parameters(
        shared, prior_shape=3.0, prior_scale=2.0
    )

    drawn = draw_common_gp_variance(
        shared,
        np.random.default_rng(11),
        prior_shape=3.0,
        prior_scale=2.0,
    )
    expected = 1.0 / np.random.default_rng(11).gamma(
        shape=shape, scale=1.0 / scale
    )
    assert drawn == expected
    assert drawn > 0.0 and np.isfinite(drawn)


def test_conditional_log_density_matches_normal_times_hyperprior() -> None:
    """Check the derivation independently, including the normalizer's d0/2."""
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
    geometry = build_gp_decomposition(data)
    xi0 = np.linspace(-1.2, 0.8, geometry.common_basis.shape[1])
    shared = _shared_state(data, geometry, xi0)
    prior_shape, prior_scale = 2.3, 0.7
    shape, scale = common_gp_variance_posterior_parameters(
        shared, prior_shape=prior_shape, prior_scale=prior_scale
    )
    variances = np.geomspace(0.03, 40.0, 41)
    log_joint = invgamma.logpdf(variances, a=prior_shape, scale=prior_scale)
    log_joint += norm.logpdf(
        xi0[None, :], scale=np.sqrt(variances[:, None])
    ).sum(axis=1)
    log_conditional = invgamma.logpdf(variances, a=shape, scale=scale)
    # Densities are equal as functions of sigma0^2 up to one constant.
    assert log_joint - log_joint[0] == pytest.approx(
        log_conditional - log_conditional[0], abs=1e-11
    )


def test_rotating_the_orthonormal_common_basis_preserves_the_conditional() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
    geometry = build_gp_decomposition(data)
    d0 = geometry.common_basis.shape[1]
    shared = _shared_state(data, geometry, np.linspace(0.2, 1.2, d0))
    rotation, _ = np.linalg.qr(np.random.default_rng(8).normal(size=(d0, d0)))
    rotated = replace(
        shared,
        decomposition=replace(
            geometry, common_basis=geometry.common_basis @ rotation
        ),
    )
    expected = common_gp_variance_posterior_parameters(
        shared, prior_shape=2.0, prior_scale=1.0
    )
    assert common_gp_variance_posterior_parameters(
        rotated, prior_shape=2.0, prior_scale=1.0
    ) == pytest.approx(expected)


def test_already_centered_design_keeps_L_common_coordinates() -> None:
    """A zero centering row removes no direction: d0 must not be hard-coded."""
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
    weights = build_gp_decomposition(data).integration_weights
    centered_A = data.A_kl - np.outer(np.ones(data.H), weights @ data.A_kl)
    assert np.linalg.matrix_rank(centered_A) == data.L
    data = replace(data, A_kl=centered_A)
    geometry = build_gp_decomposition(data)
    assert geometry.common_basis.shape == (data.L, data.L)
    shared = _shared_state(data, geometry, np.ones(data.L))
    shape, scale = common_gp_variance_posterior_parameters(
        shared, prior_shape=2.0, prior_scale=1.0
    )
    assert shape == pytest.approx(2.0 + data.L / 2)
    assert scale == pytest.approx(1.0 + data.L / 2)


def test_nodal_field_norm_is_not_a_common_variance_sufficient_statistic() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
    geometry = build_gp_decomposition(data)
    shared = _shared_state(
        data, geometry, np.ones(geometry.common_basis.shape[1])
    )
    scaled_data = replace(data, A_kl=7.0 * data.A_kl)
    scaled_geometry = build_gp_decomposition(scaled_data)
    scaled = replace(
        shared, f0=scaled_data.A_kl @ shared.theta0,
        decomposition=scaled_geometry,
    )
    assert np.linalg.norm(scaled.f0) == pytest.approx(
        7.0 * np.linalg.norm(shared.f0)
    )
    assert common_gp_variance_posterior_parameters(
        scaled, prior_shape=2.0, prior_scale=1.0
    ) == pytest.approx(common_gp_variance_posterior_parameters(
        shared, prior_shape=2.0, prior_scale=1.0
    ))


def _short_run(bundle, data):
    args = parse_args([
        "--data", str(bundle), "--storage-profile", "legacy",
        "--iters", "12",
        "--burn", "4",
        "--thin", "1",
        "--log-every", "0",
        "--seed", "5",
    ])
    args.common_gp_var = 1.0
    args.edge_gp_var = 1.0
    return run_shared_gp_pg(data, args, draw_omega=_fixed_positive_omega)


def test_learned_sigma0_varies_and_stays_finite() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        learned = _short_run(bundle, data)

    full_trace = learned["common_gp_var_full_trace"]
    retained = learned["common_gp_var_trace"]
    assert full_trace.shape == (12,)
    assert retained.shape == (8,)
    assert np.all(np.isfinite(full_trace)) and np.all(full_trace > 0.0)
    assert np.unique(full_trace).size == full_trace.size
    assert learned["common_gp_var"] == pytest.approx(float(np.mean(retained)))
    assert learned["common_gp_var_init"] == 1.0
    assert learned["a_sigma0"] == 2.0 and learned["b_sigma0"] == 1.0
    low, high = learned["common_gp_var_ci"]
    assert 0.0 < low <= high
