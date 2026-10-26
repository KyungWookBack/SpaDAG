"""Unit checks for the K-L design helpers and public defaults."""
from __future__ import annotations

import inspect

import numpy as np
from numpy.testing import assert_allclose

from spadag.spatial_gp import build_kl_design, kl_pointwise_variance


def test_kl_pointwise_variance_helper() -> None:
    A = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert_allclose(kl_pointwise_variance(A), [5.0, 25.0])


def test_build_kl_design_default_max_range_is_two() -> None:
    assert inspect.signature(build_kl_design).parameters["max_range"].default == 2.0
