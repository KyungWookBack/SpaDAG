"""Checks for the default TPR/FDR/F1 recovery schema."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spadag.metrics import triangle_edge_recovery


_EXPECTED_KEYS = {"tpr", "fdr", "f1", "tp", "fp", "fn"}


def test_triangle_edge_recovery() -> None:
    edge_prob = np.zeros((1, 2, 2), dtype=float)
    edge_true = np.zeros((1, 2, 2), dtype=int)
    edge_prob[0, 0, 1] = 0.9
    edge_prob[0, 1, 0] = 0.8
    edge_true[0, 0, 1] = 1

    summary = triangle_edge_recovery(edge_prob, edge_true)

    assert set(summary) == _EXPECTED_KEYS
    assert summary["tpr"] == 1.0
    assert summary["fdr"] == 0.5
    assert np.isclose(summary["f1"], 2.0 / 3.0)


def test_fdr_is_zero_when_nothing_is_selected() -> None:
    summary = triangle_edge_recovery(
        np.zeros((1, 2, 2), dtype=float),
        np.zeros((1, 2, 2), dtype=int),
    )
    assert summary["tpr"] == 0.0
    assert summary["fdr"] == 0.0
    assert summary["f1"] == 0.0


def test_triangle_mask_restricts_pooled_units() -> None:
    edge_prob = np.zeros((2, 2, 2), dtype=float)
    edge_true = np.zeros((2, 2, 2), dtype=int)
    edge_prob[0, 0, 1] = 0.9
    edge_true[0, 0, 1] = 1
    edge_true[1, 1, 0] = 1  # missed on the second triangle

    pooled = triangle_edge_recovery(edge_prob, edge_true)
    masked = triangle_edge_recovery(
        edge_prob, edge_true, triangle_mask=np.array([True, False])
    )

    assert (pooled["tp"], pooled["fn"]) == (1, 1)
    assert set(masked) == _EXPECTED_KEYS
    assert (masked["tp"], masked["fp"], masked["fn"]) == (1, 0, 0)
    assert masked["f1"] == 1.0
    for bad in (np.array([1, 0]), np.array([True]), np.array([False, False])):
        with np.testing.assert_raises(ValueError):
            triangle_edge_recovery(edge_prob, edge_true, triangle_mask=bad)


def test_recovery_inputs_are_validated_before_thresholding() -> None:
    valid_prob = np.zeros((1, 2, 2), dtype=float)
    valid_true = np.zeros((1, 2, 2), dtype=int)
    cases = []

    bad_prob = np.zeros((2, 2), dtype=float)
    cases.append(
        (bad_prob, np.zeros((2, 2), dtype=int), 0.5, "3-dimensional")
    )
    cases.append((bad_prob, valid_true, 0.5, "same shape"))

    bad_prob = valid_prob.copy()
    bad_prob[0, 0, 1] = np.nan
    cases.append((bad_prob, valid_true, 0.5, "only finite"))

    bad_prob = valid_prob.copy()
    bad_prob[0, 0, 1] = 1.1
    cases.append((bad_prob, valid_true, 0.5, "lie in \\[0, 1\\]"))

    bad_true = valid_true.copy()
    bad_true[0, 0, 1] = 2
    cases.append((valid_prob, bad_true, 0.5, "must be binary"))

    bad_prob = valid_prob.copy()
    bad_prob[0, 0, 0] = 0.1
    cases.append((bad_prob, valid_true, 0.5, "self-edge diagonal"))
    cases.append((valid_prob, valid_true, np.nan, "finite value"))

    for edge_prob, edge_true, threshold, message in cases:
        with np.testing.assert_raises_regex(ValueError, message):
            triangle_edge_recovery(edge_prob, edge_true, threshold)


if __name__ == "__main__":
    test_triangle_edge_recovery()
    test_fdr_is_zero_when_nothing_is_selected()
    test_triangle_mask_restricts_pooled_units()
    test_recovery_inputs_are_validated_before_thresholding()
    print("recovery metric checks passed")
