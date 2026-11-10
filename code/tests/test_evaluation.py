"""Checks for simulation-only evaluation helpers."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spadag.evaluation import (
    evaluate_reversal_recovery,
    evaluate_triangle_recovery,
    observed_triangle_mask,
)


def _p1_rows(points, mesh_points, triangles):
    """Barycentric P1 rows for points located in the first containing triangle."""
    rows = np.zeros((len(points), len(mesh_points)))
    for i, s in enumerate(points):
        for tri in triangles:
            a, b, c = mesh_points[tri]
            weights = np.linalg.solve(np.column_stack([b - a, c - a]), s - a)
            bary = np.array([1.0 - weights.sum(), weights[0], weights[1]])
            if np.all(bary >= -1e-12):
                rows[i, tri] = np.clip(bary, 0.0, None)
                break
    return rows


def test_observed_triangle_mask_uses_closed_triangles() -> None:
    # Unit square split into four triangles around the centre vertex 4.
    mesh_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], float)
    triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    interior = [0.5, 0.2]          # strictly inside triangle 0
    shared_edge = [0.75, 0.75]     # on the edge 2-4 shared by triangles 1 and 2
    Phi = _p1_rows(np.array([interior, shared_edge]), mesh_points, triangles)

    mask = observed_triangle_mask(Phi, triangles)
    np.testing.assert_array_equal(mask, [True, True, True, False])

    vertex_only = _p1_rows(np.array([[0.5, 0.5]]), mesh_points, triangles)
    assert observed_triangle_mask(vertex_only, triangles).all()


def test_evaluate_triangle_recovery_excludes_unobserved_triangles() -> None:
    edge_true = np.zeros((2, 2, 2), dtype=int)
    edge_true[0, 0, 1] = 1
    edge_true[1, 0, 1] = 1
    edge_prob = np.zeros((2, 2, 2), dtype=float)
    edge_prob[0, 0, 1] = 0.9

    summary = evaluate_triangle_recovery(
        {"edge_prob": edge_prob}, edge_true, triangle_mask=np.array([True, False])
    )

    assert summary["triangle_recovery"]["f1"] == 1.0
    assert summary["evaluated_triangles"] == 1
    assert summary["excluded_triangles"] == [1]


def test_evaluate_triangle_recovery_uses_threshold() -> None:
    edge_true = np.zeros((1, 2, 2), dtype=int)
    edge_true[0, 0, 1] = 1
    edge_prob = np.zeros((1, 2, 2), dtype=float)
    edge_prob[0, 0, 1] = 0.6

    summary = evaluate_triangle_recovery(
        {"edge_prob": edge_prob}, edge_true, threshold=0.7
    )
    assert summary["triangle_recovery"]["tp"] == 0


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
    test_observed_triangle_mask_uses_closed_triangles()
    test_evaluate_triangle_recovery_excludes_unobserved_triangles()
    test_evaluate_triangle_recovery_uses_threshold()
    test_reversal_recovery_requires_both_correct_regional_directions()
    print("triangle recovery evaluation checks passed")
