"""Focused checks for the spike-and-slab prior."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from spadag.mcmc_steps import (
    SamplerConfig,
    spike_slab_variances,
    update_gamma,
    update_sigma_b2,
)


class _ThresholdRng:
    def __init__(self, value: float):
        self.value = value

    def random(self, size=None):
        return np.full(size, self.value) if size is not None else self.value


class _CaptureGammaRng:
    def __init__(self):
        self.shape = None
        self.scale = None

    def gamma(self, *, shape, scale):
        self.shape = float(shape)
        self.scale = float(scale)
        return 2.0


def _one_coordinate_state():
    data = SimpleNamespace(
        H=1, pairs=[(1, 0)], triangles=np.empty((0, 3), dtype=int)
    )
    state = SimpleNamespace(
        gamma=np.zeros((2, 2, 1), dtype=int),
        b=np.zeros((2, 2, 1)),
        pi=np.full((2, 2, 1), 0.5),
        sigma_b2=2.0,
        _node_tris=None,
    )
    return data, state


def test_prior_variance_mapping() -> None:
    gamma = np.array([0, 1, 0])
    cfg = SamplerConfig(v0=0.03)
    np.testing.assert_allclose(
        spike_slab_variances(gamma, 2.0, cfg), [0.03, 2.0, 0.03]
    )


def test_conditional_gamma_uses_spike_variance() -> None:
    cfg, spike_var = SamplerConfig(v0=0.03), 0.03
    data, state = _one_coordinate_state()
    expected = np.sqrt(spike_var) / (
        np.sqrt(spike_var) + np.sqrt(state.sigma_b2)
    )
    update_gamma(data, state, cfg, _ThresholdRng(expected - 1e-6))
    assert state.gamma[1, 0, 0] == 1
    update_gamma(data, state, cfg, _ThresholdRng(expected + 1e-6))
    assert state.gamma[1, 0, 0] == 0


def test_sigma_b2_conditionals() -> None:
    data = SimpleNamespace(H=3, pairs=[(1, 0)])
    state = SimpleNamespace(
        gamma=np.zeros((2, 2, 3), dtype=int),
        b=np.zeros((2, 2, 3)),
        sigma_b2=1.0,
    )
    state.gamma[1, 0] = [0, 1, 1]
    state.b[1, 0] = [0.2, 0.5, -0.7]

    rng = _CaptureGammaRng()
    update_sigma_b2(data, state, SamplerConfig(), rng)
    assert rng.shape == 2.0
    expected_scale = 1.0 + 0.5 * (0.5**2 + 0.7**2)
    np.testing.assert_allclose(rng.scale, 1.0 / expected_scale)


def test_invalid_hyperparameter_contracts() -> None:
    for cfg, message in (
        (SamplerConfig(v0=0.0), "v0"),
        (SamplerConfig(a_b=0.0, b_b=1.0), "a_b"),
        (SamplerConfig(sigma_b2_init=0.0), "sigma_b2_init"),
    ):
        try:
            cfg.validate()
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected validation failure for {cfg}")


if __name__ == "__main__":
    test_prior_variance_mapping()
    test_conditional_gamma_uses_spike_variance()
    test_sigma_b2_conditionals()
    test_invalid_hyperparameter_contracts()
    print("spike-and-slab checks passed")
