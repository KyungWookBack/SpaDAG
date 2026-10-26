"""Self-contained `.npy` bundle used by sampler integration tests."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import numpy as np


def write_tiny_toy_bundle(path: Path, *, include_truth: bool = True) -> None:
    """Write a deterministic common-GP bundle, optionally without truth."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    n, p, H = 12, 3, 4
    mesh_points = np.asarray([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    A_kl = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.5],
        [0.0, -0.5],
    ])
    Phi = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.5, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5, 0.0],
        [0.2, 0.3, 0.5, 0.0],
        [0.4, 0.0, 0.2, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ])
    X = np.asarray([
        [-1.2, 0.3, 1.1],
        [-0.8, -0.5, 0.7],
        [-0.3, 1.0, -0.6],
        [0.1, -1.1, 0.4],
        [0.5, 0.8, -1.0],
        [0.9, -0.2, 0.2],
        [1.3, 0.5, 0.9],
        [-1.0, -0.8, -0.3],
        [0.7, 1.2, 0.1],
        [-0.5, 0.1, -1.2],
        [0.2, -0.7, 1.3],
        [1.1, 0.9, -0.4],
    ])

    gamma = np.zeros((p, p, H), dtype=int)
    gamma[1, 0, :2] = 1
    gamma[2, 1, 2:] = 1
    b_coeffs = np.zeros((p, p, H), dtype=float)
    b_coeffs[1, 0] = [0.45, 0.30, 0.0, 0.0]
    b_coeffs[2, 1] = [0.0, 0.0, -0.35, -0.25]
    beta_obs = np.einsum("nh,jkh->njk", Phi, b_coeffs)

    edge_triangle = np.zeros((len(triangles), p, p), dtype=int)
    for triangle_index, vertices in enumerate(triangles):
        edge_triangle[triangle_index] = (
            gamma[:, :, vertices].max(axis=2) > 0
        ).astype(int)

    q_node = np.full((p, p, H), 0.2, dtype=float)
    for node in range(p):
        q_node[node, node] = 0.0
    q_node[gamma == 1] = 0.8

    arrays = {
        "X": X,
        "Phi_obs": Phi,
        "mesh_points": mesh_points,
        "mesh_triangles": triangles,
        "A_kl": A_kl,
    }
    truth_arrays = {
        "q_node": q_node,
        "gamma": gamma,
        "b_coeffs": b_coeffs,
        "beta_obs": beta_obs,
        "edge_triangle": edge_triangle,
        "sigma_noise": np.asarray([0.8, 1.0, 1.2]),
        "hyper_sigma_b": np.asarray([1.0]),
        "hyper_mu": np.asarray([-1.0]),
        "hyper_tau2": np.asarray([1.0]),
    }
    if include_truth:
        arrays.update(truth_arrays)
    for stem, array in arrays.items():
        np.save(path / f"{stem}.npy", array)


@contextmanager
def temporary_toy_bundle(*, include_truth: bool = True) -> Iterator[Path]:
    """Yield a temporary bundle path and remove it after the test."""
    with TemporaryDirectory(prefix="spadag_test_bundle_") as directory:
        path = Path(directory)
        write_tiny_toy_bundle(path, include_truth=include_truth)
        yield path
