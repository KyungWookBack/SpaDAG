"""Self-contained integration checks for the main shared-GP runner."""
from __future__ import annotations

import numpy as np

from spadag.mcmc_data import load_inference_data
from spadag.cli import parse_args, run_shared_gp_pg
from tests.toy_data_fixture import temporary_toy_bundle


def _fixed_positive_omega(eta):
    return np.full(np.asarray(eta).shape, 0.25, dtype=float)


def test_short_runner_integration() -> None:
    with temporary_toy_bundle(include_truth=False) as bundle:
        data = load_inference_data(bundle)
        args = parse_args([
            "--data", str(bundle), "--storage-profile", "legacy",
            "--iters", "3",
            "--burn", "1",
            "--thin", "1",
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
    assert result["field_kept"] == 1
    assert result["q_mean"].shape == (data.p, data.p, data.H)
    assert result["eta_mean"].shape == (data.p, data.p, data.H)
    assert result["theta0_trace"].shape == (2, data.L)
    assert result["gamma_trace_packed"].shape[0] == 2
    unpacked = np.unpackbits(
        result["gamma_trace_packed"],
        axis=1,
        count=int(result["gamma_trace_n_bits"]),
        bitorder=str(result["gamma_trace_bitorder"]),
    ).reshape(2, len(data.pairs), data.H)
    expected_gamma_prob = np.stack([
        result["gamma_prob"][j, k] for j, k in data.pairs
    ])
    assert np.allclose(np.mean(unpacked, axis=0), expected_gamma_prob)
    assert np.all(np.isfinite(result["q_mean"]))
    assert np.all((result["q_mean"] >= 0.0) & (result["q_mean"] <= 1.0))
    assert np.isfinite(result["common_precision_condition_max"])
    assert np.isclose(np.sum(result["baseline_integration_weights"]), 1.0)
    assert result["baseline_constraint_error"] < 1e-10
    assert result["edge_gp_var_trace"].shape == (2,)
    assert np.all(np.isfinite(result["edge_gp_var_trace"]))
    assert np.all(result["edge_gp_var_trace"] > 0.0)
    assert np.isclose(
        result["edge_gp_var"],
        np.mean(result["edge_gp_var_trace"]),
    )


def test_short_constrained_runner_integration() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        args = parse_args([
            "--data", str(bundle), "--storage-profile", "legacy",
            "--iters", "3",
            "--burn", "1",
            "--thin", "1",
            "--field-thin", "2",
            "--log-every", "0",
            "--seed", "23",
        ])
        args.common_gp_var = 1.0
        args.edge_gp_var = 1.0
        result = run_shared_gp_pg(
            data, args, draw_omega=_fixed_positive_omega
        )

    assert result["kept"] == 2
    assert result["common_null_dimension"] == data.L - 1
    assert np.isclose(np.sum(result["integration_weights"]), 1.0)
    assert np.isclose(
        result["edge_gp_var_raw"],
        result["edge_gp_var"] * len(data.pairs) / (len(data.pairs) - 1),
    )
    assert np.allclose(
        result["edge_gp_var_raw_trace"],
        result["edge_gp_var_trace"]
        * len(data.pairs) / (len(data.pairs) - 1),
    )
    assert result["common_constraint_error_max"] < 1e-10
    assert result["edge_constraint_error_max"] < 1e-10
    weighted_f0 = result["f0_trace"] @ result["integration_weights"]
    assert np.max(np.abs(weighted_f0)) < 1e-10
    assert np.max(np.abs(result["fbar_trace"])) < 1e-10
    assert np.all(np.isfinite(result["q_mean"]))
    assert np.all((result["q_mean"] >= 0.0) & (result["q_mean"] <= 1.0))


def test_diagnostic_storage_preserves_trajectory() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        common = [
            "--data", str(bundle), "--storage-profile", "legacy",
            "--iters", "8",
            "--burn", "2",
            "--thin", "2",
            "--field-thin", "3",
            "--log-every", "0",
            "--seed", "91",
        ]
        args_off = parse_args(common)
        args_on = parse_args(common + [
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

    keys = [
        "mu_full_trace",
        "active_gamma_full_trace",
        "sigma_b2_trace",
        "theta0_trace",
        "gamma_prob",
        "edge_prob",
        "q_mean",
        "eta_mean",
    ]
    for key in keys:
        assert np.array_equal(result_off[key], result_on[key]), key


def test_adding_or_changing_truth_does_not_affect_trajectory() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        argv = [
            "--data", str(bundle), "--storage-profile", "legacy",
            "--iters", "3",
            "--burn", "1",
            "--thin", "1",
            "--common-gp-var", "1",
            "--edge-gp-var", "1",
            "--log-every", "0",
            "--seed", "41",
        ]
        first = run_shared_gp_pg(
            data, parse_args(argv), draw_omega=_fixed_positive_omega
        )
        np.save(bundle / "gamma.npy", np.ones((3, 3, 4), dtype=int))
        np.save(bundle / "b_coeffs.npy", np.full((3, 3, 4), 99.0))
        np.save(bundle / "q_node.npy", np.full((3, 3, 4), 0.99))
        changed_truth = load_inference_data(bundle)
        second = run_shared_gp_pg(
            changed_truth,
            parse_args(argv),
            draw_omega=_fixed_positive_omega,
        )

    for key in (
        "active_gamma_full_trace",
        "sigma_b2_trace",
        "mu_full_trace",
        "edge_gp_var_full_trace",
        "gamma_prob",
        "b_mean",
        "q_mean",
    ):
        assert np.array_equal(first[key], second[key]), key


if __name__ == "__main__":
    test_short_runner_integration()
    test_short_constrained_runner_integration()
    test_diagnostic_storage_preserves_trajectory()
    test_adding_or_changing_truth_does_not_affect_trajectory()
    print("shared-GP runner integration checks passed")
