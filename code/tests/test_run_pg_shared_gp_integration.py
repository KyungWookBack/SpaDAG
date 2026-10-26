"""Self-contained integration checks for the main shared-GP runner."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spadag.compact_storage import load_stream
from spadag.mcmc_data import load_inference_data
from spadag.cli import parse_args, run_shared_gp_pg
from tests.toy_data_fixture import temporary_toy_bundle


def _fixed_positive_omega(eta):
    return np.full(np.asarray(eta).shape, 0.25, dtype=float)


def _manifest(trace_dir: Path) -> dict:
    return json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))


def _stream(trace_dir: Path, stream: str) -> dict:
    return load_stream(trace_dir, _manifest(trace_dir)["head"], stream)


def _assert_probabilities(result, data) -> None:
    p, H, T = data.p, data.H, len(data.triangles)
    assert result["gamma_prob"].shape == (p, p, H)
    assert result["b_mean"].shape == (p, p, H)
    assert result["edge_prob"].shape == (T, p, p)
    for key in ("gamma_prob", "edge_prob"):
        assert np.all((result[key] >= 0.0) & (result[key] <= 1.0)), key
    assert np.all(np.isfinite(result["b_mean"]))


def test_short_runner_integration(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    with temporary_toy_bundle(include_truth=False) as bundle:
        data = load_inference_data(bundle)
        args = parse_args([
            "--data", str(bundle), "--trace-dir", str(trace),
            "--iters", "3",
            "--burn", "1",
            "--thin", "1",
            "--scalar-thin", "1",
            "--field-thin", "2",
            "--gamma-trace-thin", "1",
            "--log-every", "0",
            "--seed", "17",
        ])
        args.common_gp_var = 1.0
        args.edge_gp_var = 1.0
        result = run_shared_gp_pg(
            data, args, draw_omega=_fixed_positive_omega
        )

    for removed in (
        "node_tpr",
        "node_fdr",
        "node_f1",
        "triangle_tpr",
        "triangle_fdr",
        "triangle_f1",
        "edge_tpr",
        "edge_fdr",
        "edge_f1",
        "q_recovery",
    ):
        assert removed not in result
    assert result["kept"] == 2
    manifest = _manifest(trace)
    E = len(data.pairs)
    field = _stream(trace, "field")
    assert field["eta"].shape == (1, E, data.H)
    assert np.all(np.isfinite(field["eta"]))
    _assert_probabilities(result, data)
    packed = _stream(trace, "gamma")["packed"]
    assert packed.shape[0] == 2
    unpacked = np.unpackbits(
        packed,
        axis=1,
        count=E * data.H,
        bitorder=manifest["gamma_bitorder"],
    ).reshape(2, E, data.H)
    expected_gamma_prob = np.stack([
        result["gamma_prob"][j, k] for j, k in data.pairs
    ])
    assert np.allclose(np.mean(unpacked, axis=0), expected_gamma_prob)
    assert manifest["integrity"]["failures"] == 0
    assert manifest["integrity"]["baseline_max"] < 1e-10
    scalar = _stream(trace, "scalar")
    edge_var = scalar["values"][scalar["postburn"], manifest["scalar_names"].index("edge_var")]
    assert edge_var.shape == (2,)
    assert np.all(np.isfinite(edge_var))
    assert np.all(edge_var > 0.0)


def test_short_constrained_runner_integration(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        args = parse_args([
            "--data", str(bundle), "--trace-dir", str(trace),
            "--iters", "3",
            "--burn", "1",
            "--thin", "1",
            "--field-thin", "2",
            "--latent-thin", "1",
            "--log-every", "0",
            "--seed", "23",
        ])
        args.common_gp_var = 1.0
        args.edge_gp_var = 1.0
        result = run_shared_gp_pg(
            data, args, draw_omega=_fixed_positive_omega
        )

    assert result["kept"] == 2
    with np.load(trace / "geometry.npz") as geometry:
        common_basis = geometry["common_basis"]
        integration_weights = geometry["integration_weights"]
        A = geometry["A"]
    assert common_basis.shape == (data.L, data.L - 1)
    assert np.isclose(np.sum(integration_weights), 1.0)
    integrity = _manifest(trace)["integrity"]
    assert integrity["common_max"] < 1e-10
    assert integrity["edge_max"] < 1e-10
    xi0 = _stream(trace, "latent")["xi0"]
    assert xi0.shape == (2, data.L - 1)
    weighted_f0 = (xi0 @ common_basis.T @ A.T) @ integration_weights
    assert np.max(np.abs(weighted_f0)) < 1e-10
    _assert_probabilities(result, data)


def test_diagnostic_storage_preserves_trajectory(tmp_path: Path) -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        common = [
            "--data", str(bundle),
            "--iters", "8",
            "--burn", "2",
            "--thin", "2",
            "--scalar-thin", "1",
            "--field-thin", "3",
            "--log-every", "0",
            "--seed", "91",
        ]
        args_off = parse_args(common + [
            "--trace-dir", str(tmp_path / "off"),
            "--gamma-trace-thin", "0",
        ])
        args_on = parse_args(common + [
            "--trace-dir", str(tmp_path / "on"),
            "--gamma-trace-thin", "2",
        ])
        for args in (args_off, args_on):
            args.common_gp_var = 1.0
            args.edge_gp_var = 1.0
        result_off = run_shared_gp_pg(
            data, args_off, draw_omega=_fixed_positive_omega
        )
        result_on = run_shared_gp_pg(
            data, args_on, draw_omega=_fixed_positive_omega
        )

    assert _stream(args_off.trace_dir, "gamma") == {}
    assert len(_stream(args_on.trace_dir, "gamma")["iteration"]) == 3
    keys = [
        "gamma_prob",
        "edge_prob",
        "b_mean",
        "b_candidate_sd",
    ]
    for key in keys:
        assert np.array_equal(result_off[key], result_on[key]), key
    for stream in ("scalar", "field"):
        off = _stream(args_off.trace_dir, stream)
        on = _stream(args_on.trace_dir, stream)
        assert off.keys() == on.keys(), stream
        for key in off:
            assert np.array_equal(off[key], on[key]), f"{stream}/{key}"


def test_adding_or_changing_truth_does_not_affect_trajectory(tmp_path: Path) -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        argv = [
            "--data", str(bundle),
            "--iters", "3",
            "--burn", "1",
            "--thin", "1",
            "--scalar-thin", "1",
            "--common-gp-var", "1",
            "--edge-gp-var", "1",
            "--log-every", "0",
            "--seed", "41",
        ]
        first = run_shared_gp_pg(
            data,
            parse_args(argv + ["--trace-dir", str(tmp_path / "first")]),
            draw_omega=_fixed_positive_omega,
        )
        np.save(bundle / "gamma.npy", np.ones((3, 3, 4), dtype=int))
        np.save(bundle / "b_coeffs.npy", np.full((3, 3, 4), 99.0))
        np.save(bundle / "q_node.npy", np.full((3, 3, 4), 0.99))
        changed_truth = load_inference_data(bundle)
        second = run_shared_gp_pg(
            changed_truth,
            parse_args(argv + ["--trace-dir", str(tmp_path / "second")]),
            draw_omega=_fixed_positive_omega,
        )

    for key in (
        "gamma_prob",
        "edge_prob",
        "b_mean",
    ):
        assert np.array_equal(first[key], second[key]), key
    for stream in ("scalar", "gamma"):
        a = _stream(tmp_path / "first", stream)
        b = _stream(tmp_path / "second", stream)
        assert a.keys() == b.keys(), stream
        for key in a:
            assert np.array_equal(a[key], b[key]), f"{stream}/{key}"


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        root = Path(directory)
        test_short_runner_integration(root / "short")
        test_short_constrained_runner_integration(root / "constrained")
        test_diagnostic_storage_preserves_trajectory(root / "storage")
        test_adding_or_changing_truth_does_not_affect_trajectory(root / "truth")
    print("shared-GP runner integration checks passed")
