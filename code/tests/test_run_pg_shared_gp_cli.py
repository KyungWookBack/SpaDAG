"""Argument-parser checks for the main shared-GP runner."""
from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from spadag.cli import parse_args


_DATA = ["--data", "bundle", "--trace-dir", "trace"]


def _assert_parse_error(argv) -> None:
    try:
        with redirect_stderr(StringIO()):
            parse_args(argv)
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError(f"expected parse error for {argv}")


def test_required_data_and_defaults() -> None:
    _assert_parse_error([])
    _assert_parse_error(["--data", "bundle"])
    args = parse_args(_DATA)
    assert args.v0 == 1e-3
    assert args.common_gp_var == 1.0
    assert tuple(args.common_gp_var_prior) == (2.0, 1.0)
    assert args.a_sigma0 == 2.0
    assert args.b_sigma0 == 1.0
    assert args.edge_gp_var == 1.0
    assert args.a_tau == 3.0
    assert args.b_tau == 2.0
    assert args.mu_init == 0.0
    assert args.mu_prior_mean == 0.0
    assert args.mu_prior_sd == 2.0
    assert args.data == Path("bundle")
    assert args.trace_dir == Path("trace")
    assert args.a_b == 1.0
    assert args.b_b == 1.0
    assert args.sigma_b2_init == 1.0
    assert args.initialization == "empty"
    assert args.thin == 5
    assert args.field_thin == 0
    assert args.iters == 50000 and args.burn == 25000
    assert args.checkpoint_every == 0 and args.milestone_every == 0
    assert args.gamma_trace_thin == 10


def test_optional_settings_are_parsed() -> None:
    args = parse_args(_DATA + [
        "--common-gp-var", "0.4",
        "--common-gp-var-prior", "5", "4",
        "--edge-gp-var", "1.7",
        "--a-tau", "4",
        "--b-tau", "3",
        "--mu-init", "-2",
        "--mu-prior-mean", "-1",
        "--mu-prior-sd", "3",
        "--initialization", "prior-random",
        "--v0", "0.002",
        "--a-b", "4",
        "--b-b", "3",
        "--sigma-b2-init", "20",
        "--thin", "2",
        "--field-thin", "50",
        "--gamma-trace-thin", "20",
    ])
    assert args.common_gp_var == 0.4
    assert args.common_gp_var_prior == [5.0, 4.0]
    assert args.a_sigma0 == 5.0
    assert args.b_sigma0 == 4.0
    assert args.edge_gp_var == 1.7
    assert args.a_tau == 4.0
    assert args.b_tau == 3.0
    assert args.mu_init == -2.0
    assert args.mu_prior_mean == -1.0
    assert args.mu_prior_sd == 3.0
    assert args.initialization == "prior-random"
    assert args.v0 == 0.002
    assert args.a_b == 4.0
    assert args.b_b == 3.0
    assert args.sigma_b2_init == 20.0
    assert args.thin == 2
    assert args.field_thin == 50
    assert args.gamma_trace_thin == 20


def test_prior_random_name_and_removed_default() -> None:
    args = parse_args(_DATA + ["--initialization", "prior-random"])
    assert args.initialization == "prior-random"
    _assert_parse_error(_DATA + ["--initialization", "default"])
    _assert_parse_error(_DATA + ["--initialization", "true"])


def test_invalid_values_are_rejected() -> None:
    _assert_parse_error(_DATA + ["--common-gp-var", "0"])
    _assert_parse_error(_DATA + ["--edge-gp-var", "-1"])
    _assert_parse_error(_DATA + ["--a-tau", "0"])
    _assert_parse_error(_DATA + ["--common-gp-var-prior", "0", "1"])
    _assert_parse_error(_DATA + ["--common-gp-var-prior", "2", "-1"])
    _assert_parse_error(_DATA + ["--common-gp-var-prior", "2", "nan"])
    _assert_parse_error(_DATA + ["--common-gp-var-prior", "2"])
    _assert_parse_error(_DATA + ["--b-tau", "nan"])
    _assert_parse_error(_DATA + ["--a-b", "0"])
    _assert_parse_error(_DATA + ["--a-b", "nan"])
    _assert_parse_error(_DATA + ["--b-b", "nan"])
    _assert_parse_error(_DATA + ["--a-b", "4", "--b-b", "0"])
    _assert_parse_error(_DATA + ["--v0", "0"])
    _assert_parse_error(_DATA + ["--sigma-b2-init", "nan"])
    _assert_parse_error(_DATA + ["--mu-prior-sd", "0"])
    _assert_parse_error(_DATA + ["--mu-init", "nan"])
    _assert_parse_error(_DATA + ["--iters", "20", "--burn", "20"])
    _assert_parse_error(_DATA + ["--field-thin", "-1"])
    _assert_parse_error(_DATA + ["--gamma-trace-thin", "-1"])


def test_removed_options_are_rejected() -> None:
    for option in (
        ["--gp-decomposition", "sum-to-zero"],
        ["--fixed-common-gp-var"],
        ["--learn-common-gp-var"],
        ["--edge-gp-var-mode", "ig"],
        ["--spike-mode", "decoupled"],
        ["--kappa0", "0.001"],
        ["--diagnostic-windows", "3"],
        ["--storage-profile", "legacy"],
        ["--storage-profile", "grid"],
        ["--continue-from", "checkpoint.pkl"],
    ):
        _assert_parse_error(_DATA + option)


if __name__ == "__main__":
    test_required_data_and_defaults()
    test_optional_settings_are_parsed()
    test_prior_random_name_and_removed_default()
    test_invalid_values_are_rejected()
    test_removed_options_are_rejected()
    print("shared-GP runner CLI checks passed")
