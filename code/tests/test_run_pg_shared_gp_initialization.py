"""Checks for truth-independent main-sampler initializations."""
from __future__ import annotations

import numpy as np

from spadag.mcmc_data import load_inference_data
from spadag.mcmc_steps import SamplerConfig, init_state
from spadag.cli import _apply_initialization
from tests.toy_data_fixture import temporary_toy_bundle


def test_prior_random_and_empty_initializations() -> None:
    with temporary_toy_bundle(include_truth=False) as bundle:
        data = load_inference_data(bundle)
        cfg = SamplerConfig()

        prior_random = init_state(
            data,
            cfg,
            np.random.default_rng(0),
            gamma_init_prob=0.5,
        )
        _apply_initialization(prior_random, "prior-random")
        assert np.any(prior_random.gamma)

        empty = init_state(
            data,
            cfg,
            np.random.default_rng(0),
            gamma_init_prob=0.5,
        )
        _apply_initialization(empty, "empty")
        assert not np.any(empty.gamma)
        assert not np.any(empty.b)


def test_true_initialization_is_not_a_main_sampler_option() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        state = init_state(
            data,
            SamplerConfig(),
            np.random.default_rng(0),
        )

    try:
        _apply_initialization(state, "true")
    except ValueError as exc:
        assert str(exc) == "unknown initialization: true"
    else:
        raise AssertionError("true initialization was unexpectedly accepted")


def test_initial_state_uses_configured_slab_variance() -> None:
    with temporary_toy_bundle(include_truth=False) as bundle:
        data = load_inference_data(bundle)
        cfg = SamplerConfig(a_b=3.0, b_b=2.0, sigma_b2_init=7.5)
        first = init_state(
            data,
            cfg,
            np.random.default_rng(91),
            gamma_init_prob=0.5,
        )
        second = init_state(
            data,
            cfg,
            np.random.default_rng(91),
            gamma_init_prob=0.5,
        )

    np.testing.assert_array_equal(first.gamma, second.gamma)
    np.testing.assert_array_equal(first.b, second.b)
    assert first.sigma_b2 == 7.5
    assert second.sigma_b2 == 7.5
    np.testing.assert_array_equal(first.m, second.m)
    np.testing.assert_array_equal(first.pi, second.pi)
