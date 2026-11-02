"""Checks for simulation-only evaluation helpers."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spadag.evaluation import evaluate_reversal_recovery, evaluate_triangle_recovery


def test_evaluate_triangle_recovery() -> None:
    edge_true = np.zeros((1, 2, 2), dtype=int)
    edge_true[0, 0, 1] = 1
    edge_prob = np.zeros((1, 2, 2), dtype=float)
    edge_prob[0, 0, 1] = 0.9
    edge_prob[0, 1, 0] = 0.2

    result = {"edge_prob": edge_prob}

    summary = evaluate_triangle_recovery(result, edge_true)

    assert set(summary) == {"triangle_recovery"}
    assert summary["triangle_recovery"]["f1"] == 1.0

    strict = evaluate_triangle_recovery(result, edge_true, threshold=0.95)
    assert strict["triangle_recovery"]["tp"] == 0
    assert strict["triangle_recovery"]["fn"] == 1


def test_reversal_recovery_requires_both_correct_regional_directions() -> None:
    edge_prob = np.zeros((3, 4, 4), dtype=float)
    edge_prob[0, 2, 1] = 0.9
    edge_prob[2, 1, 2] = 0.8
    motifs = [
        {
            "name": "R1",
            "left": {"parent": 1, "child": 2, "triangles": [0]},
            "right": {"parent": 2, "child": 1, "triangles": [2]},
        }
    ]

    recovered = evaluate_reversal_recovery(edge_prob, motifs)
    assert recovered["motifs"]["R1"] == {
        "left_direction_recovered": True,
        "right_direction_recovered": True,
        "both_directions_recovered": True,
    }
    assert recovered["both_directions_recovery_rate"] == 1.0

    edge_prob[2, 2, 1] = 0.7
    failed = evaluate_reversal_recovery(edge_prob, motifs)
    assert failed["motifs"]["R1"]["right_direction_recovered"] is False
    assert failed["both_directions_recovery_rate"] == 0.0


if __name__ == "__main__":
    test_evaluate_triangle_recovery()
    test_reversal_recovery_requires_both_correct_regional_directions()
    print("triangle recovery evaluation checks passed")
