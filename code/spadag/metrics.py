"""Triangle-level graph recovery versus known simulation truth."""
from __future__ import annotations

from typing import Dict

import numpy as np


def _recovery_summary(tp: int, fp: int, fn: int) -> Dict[str, float]:
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0
    return {
        "tpr": tpr,
        "fdr": fdr,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def triangle_edge_recovery(
    edge_prob: np.ndarray,
    edge_true: np.ndarray,
    thresh: float = 0.5,
    *,
    triangle_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    """TPR, FDR, and F1 for the triangle-level posterior graph.

    ``edge_prob[K, j, k]`` must be the posterior mean of the triangle-edge
    indicator formed within each MCMC draw.  Applying the triangle OR only
    after thresholding marginal node-level PIPs is a different estimand and
    must not be used for triangle-level recovery.

    ``triangle_mask`` (boolean, length K) restricts the pooled triangle--edge
    units to the selected triangles; ``None`` pools over every triangle.
    When no unit is selected, FDR is reported as 0 (the false discovery
    proportion is undefined in that case).
    """
    probability = np.asarray(edge_prob, dtype=float)
    truth = np.asarray(edge_true)
    if probability.shape != truth.shape:
        raise ValueError(
            "edge_prob and edge_true must have the same shape, got "
            f"{probability.shape} and {truth.shape}"
        )
    if probability.ndim != 3:
        raise ValueError(
            "edge_prob and edge_true must be 3-dimensional arrays with "
            f"shape (K, p, p), got {probability.shape}"
        )
    K, p, p_second = probability.shape
    if K == 0 or p < 2 or p_second != p:
        raise ValueError(
            "edge_prob and edge_true must have shape (K, p, p) with "
            f"K > 0 and p >= 2, got {probability.shape}"
        )
    if not np.all(np.isfinite(probability)):
        raise ValueError("edge_prob must contain only finite values")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("edge_prob values must lie in [0, 1]")
    if np.any((truth != 0) & (truth != 1)):
        raise ValueError("edge_true must be binary")
    if np.any(np.diagonal(probability, axis1=1, axis2=2) != 0.0):
        raise ValueError("edge_prob self-edge diagonal entries must be zero")
    if np.any(np.diagonal(truth, axis1=1, axis2=2) != 0):
        raise ValueError("edge_true self-edge diagonal entries must be zero")
    try:
        threshold = float(thresh)
    except (TypeError, ValueError) as exc:
        raise ValueError("thresh must be a finite value in [0, 1]") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("thresh must be a finite value in [0, 1]")

    if triangle_mask is not None:
        keep = np.asarray(triangle_mask)
        if keep.dtype != bool or keep.shape != (K,):
            raise ValueError(f"triangle_mask must be a boolean array of shape ({K},)")
        if not np.any(keep):
            raise ValueError("triangle_mask must select at least one triangle")
        probability = probability[keep]
        truth = truth[keep]

    et = truth > 0
    eh = probability >= threshold
    tp = int(np.sum(eh & et))
    fp = int(np.sum(eh & ~et))
    fn = int(np.sum(~eh & et))
    return _recovery_summary(tp, fp, fn)
