"""Regression checks for the cached update_b implementation."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from spadag.mcmc_data import load_inference_data
from spadag.mcmc_steps import (
    SamplerConfig,
    _alpha_at_obs,
    _parent_pred,
    _sample_gaussian_precision,
    init_state,
    spike_slab_variances,
    update_b,
)
from tests.toy_data_fixture import temporary_toy_bundle


def _update_b_reference(data, st, cfg, rng) -> None:
    """Uncached update used as a transition-equivalence test oracle."""
    for j in range(data.p):
        alpha_j = _alpha_at_obs(data, st, j)
        s2 = st.sigma2[j]
        for k in np.where(data.parents[j] == 1)[0]:
            r = (
                data.X[:, j]
                - alpha_j
                - _parent_pred(data, st, j, exclude_k=k)
            )
            v = spike_slab_variances(
                st.gamma[j, k], st.sigma_b2, cfg
            )
            precision = data.ZtZ[k] / s2 + np.diag(1.0 / v)
            rhs = (data.Zk[k].T @ r) / s2
            st.b[j, k] = _sample_gaussian_precision(
                precision, rhs, rng
            )


def _nontrivial_state(data, cfg, seed: int):
    rng = np.random.default_rng(seed)
    state = init_state(data, cfg, rng)
    state.a0 = rng.normal(0.0, 0.2, size=data.p)
    state.a_spatial = rng.normal(0.0, 0.1, size=(data.p, data.H))
    state.sigma2 = rng.lognormal(0.0, 0.2, size=data.p)
    return state


def _assert_same_transition(data, cfg) -> None:
    initial = _nontrivial_state(data, cfg, seed=731)
    reference = deepcopy(initial)
    cached = deepcopy(initial)
    reference_rng = np.random.default_rng(991)
    cached_rng = np.random.default_rng(991)

    _update_b_reference(data, reference, cfg, reference_rng)
    update_b(data, cached, cfg, cached_rng)

    np.testing.assert_allclose(
        cached.b,
        reference.b,
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.array_equal(
        cached_rng.bit_generator.state["state"]["state"],
        reference_rng.bit_generator.state["state"]["state"],
    )


def test_cached_update_matches_reference() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        _assert_same_transition(data, SamplerConfig(v0=1e-3))


if __name__ == "__main__":
    test_cached_update_matches_reference()
    print("cached update_b transition checks passed")
