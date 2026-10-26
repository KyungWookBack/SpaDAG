"""The bundled example dataset is intact and consistent with its metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from spadag.acyclic import is_dag
from spadag.mcmc_data import load_inference_data

EXAMPLE = Path(__file__).resolve().parents[2] / "artifacts" / "data" / "example_reversal_p50_n500"


def test_example_checksums_match() -> None:
    lines = (EXAMPLE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    listed = dict(reversed(line.split("  ", 1)) for line in lines)
    assert set(listed) == {path.name for path in EXAMPLE.iterdir()} - {"SHA256SUMS"}
    for name, digest in listed.items():
        assert hashlib.sha256((EXAMPLE / name).read_bytes()).hexdigest() == digest, name


def test_example_loads_with_documented_dimensions() -> None:
    metadata = json.loads((EXAMPLE / "metadata.json").read_text(encoding="utf-8"))
    data = load_inference_data(EXAMPLE)
    dims = metadata["dimensions"]
    assert (data.n, data.p, data.H, len(data.triangles), data.L) == (
        dims["n"], dims["p"], dims["H"], dims["T"], dims["L"]
    )
    np.testing.assert_allclose(data.Phi.sum(axis=1), 1.0)
    locations = np.load(EXAMPLE / "obs_points.npy")
    assert locations.shape == (data.n, 2)
    assert np.all((locations >= 0.0) & (locations <= 1.0))


def test_example_truth_is_acyclic_in_every_triangle() -> None:
    edge_triangle = np.load(EXAMPLE / "edge_triangle.npy")
    assert edge_triangle.shape == (50, 50, 50)
    assert np.all(np.diagonal(edge_triangle, axis1=1, axis2=2) == 0)
    assert all(is_dag(graph) for graph in edge_triangle)
    assert edge_triangle.sum() > 0
