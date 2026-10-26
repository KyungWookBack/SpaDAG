"""Checks for the area-centered baseline-surface Gibbs block."""

from __future__ import annotations

import numpy as np

from spadag.mcmc_data import load_inference_data
from spadag.mcmc_steps import SamplerConfig, init_state, update_alpha

from tests.toy_data_fixture import temporary_toy_bundle


def test_loaded_baseline_geometry_is_centered_and_orthonormal() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)

    assert data.baseline_weights is not None
    assert data.baseline_basis is not None
    assert data.U.shape == (data.n, data.H)
    assert data.UtU.shape == (data.H, data.H)
    assert data.baseline_basis.shape == (data.H, data.H - 1)
    np.testing.assert_allclose(np.sum(data.baseline_weights), 1.0)
    np.testing.assert_allclose(
        data.baseline_weights @ data.baseline_basis,
        0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        data.baseline_basis.T @ data.baseline_basis,
        np.eye(data.H - 1),
        atol=1e-14,
    )


def test_area_centering_preserves_a_p1_surface() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)

    a0 = np.asarray([0.7, -0.2])
    a_spatial = np.asarray([[0.2, 0.8, -0.4, 0.5], [-1.0, 0.3, 0.9, -0.1]])
    mean_level = a_spatial @ data.baseline_weights
    centered_a0 = a0 + mean_level
    centered_spatial = a_spatial - mean_level[:, None]

    before = a0[None, :] + data.Phi @ a_spatial.T
    after = centered_a0[None, :] + data.Phi @ centered_spatial.T
    np.testing.assert_allclose(after, before, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        centered_spatial @ data.baseline_weights,
        0.0,
        atol=1e-14,
    )


def test_baseline_update_enforces_constraint_and_uses_h_minus_one_df() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
    config = SamplerConfig()
    state = init_state(
        data,
        config,
        np.random.default_rng(913),
        gamma_init_prob=0.0,
    )

    class RecordingRng:
        def __init__(self) -> None:
            self.gamma_shapes = []

        @staticmethod
        def standard_normal(size):
            return np.zeros(size)

        def gamma(self, *, shape, scale):
            self.gamma_shapes.append(shape)
            return shape * scale

    rng = RecordingRng()
    update_alpha(data, state, config, rng)

    np.testing.assert_allclose(
        state.a_spatial @ data.baseline_weights,
        0.0,
        atol=1e-13,
    )
    assert rng.gamma_shapes == [config.a_alpha + 0.5 * (data.H - 1)] * data.p
    assert np.all(np.isfinite(state.a0))
    assert np.all(np.isfinite(state.a_spatial))
    assert np.all(state.tau_alpha > 0.0)
