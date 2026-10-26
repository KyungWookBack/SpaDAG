"""Contract checks for the production binary BayesLogit bridge."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from spadag.polya_gamma import BayesLogitRPGSampler, bayeslogit_available


_R_AVAILABLE = bayeslogit_available()


def test_missing_bayeslogit_has_clear_error() -> None:
    with patch("spadag.polya_gamma._resolve_rscript", return_value="Rscript"), patch(
        "spadag.polya_gamma.bayeslogit_available", return_value=False
    ):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "BayesLogit is required"):
            BayesLogitRPGSampler(seed=1)


@unittest.skipUnless(_R_AVAILABLE, "Rscript and BayesLogit are required")
def test_binary_draw_preserves_shape_and_is_finite_positive() -> None:
    tilt = np.linspace(-3.0, 3.0, 12).reshape(3, 4)
    with BayesLogitRPGSampler(seed=72931) as sampler:
        draw = sampler.draw(tilt)
    assert draw.shape == tilt.shape
    assert np.all(np.isfinite(draw))
    assert np.all(draw > 0.0)


@unittest.skipUnless(_R_AVAILABLE, "Rscript and BayesLogit are required")
def test_binary_rng_state_round_trip() -> None:
    with BayesLogitRPGSampler(seed=72932) as sampler:
        sampler.draw(np.linspace(-2.0, 2.0, 19))
        state = sampler.get_rng_state()
        first = sampler.draw(np.linspace(-5.0, 5.0, 101))
        sampler.set_rng_state(state)
        second = sampler.draw(np.linspace(-5.0, 5.0, 101))
    np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    test_missing_bayeslogit_has_clear_error()
    test_binary_draw_preserves_shape_and_is_finite_positive()
    test_binary_rng_state_round_trip()
    print("binary BayesLogit bridge checks passed")
