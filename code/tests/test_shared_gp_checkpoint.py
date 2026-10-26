"""Exact-continuation checks for the shared-GP runner."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import pickle
import shutil
import unittest

import numpy as np

from spadag.mcmc_data import load_inference_data
from spadag.polya_gamma import BayesLogitRPGSampler, bayeslogit_available
from spadag.cli import (
    CheckpointStop,
    _CHECKPOINT_VERSION,
    _read_checkpoint,
    parse_args,
    run_shared_gp_pg,
)
from tests.toy_data_fixture import temporary_toy_bundle


_R_AVAILABLE = bayeslogit_available()


def _args(
    data_path: Path,
    checkpoint: Path | None = None,
    *,
    resume: bool = False,
    continue_from: bool = False,
    iters: int = 8,
    burn: int = 2,
):
    argv = [
        "--data", str(data_path), "--storage-profile", "legacy",
        "--iters", str(iters),
        "--burn", str(burn),
        "--thin", "1",
        "--field-thin", "2",
        "--gamma-trace-thin", "1",
        "--log-every", "0",
        "--seed", "73002",
        "--initialization", "empty",
        "--common-gp-var", "1",
        "--edge-gp-var", "1",
        "--a-tau", "3",
        "--b-tau", "2",
    ]
    if checkpoint is not None:
        argv.extend(["--checkpoint", str(checkpoint), "--checkpoint-every", "2"])
    if resume:
        argv.extend(["--resume", str(checkpoint)])
    if continue_from:
        argv.extend(["--continue-from", str(checkpoint)])
    return parse_args(argv)


def _run(data, args):
    with BayesLogitRPGSampler(seed=args.seed) as sampler:
        return run_shared_gp_pg(
            data,
            args,
            draw_omega=sampler.draw,
            get_omega_state=sampler.get_rng_state,
            set_omega_state=sampler.set_rng_state,
        )


class _MockPGSampler:
    """Deterministic stateful test double for checkpoint protocol coverage."""

    def __init__(self) -> None:
        self.calls = 0

    def draw(self, eta):
        self.calls += 1
        return np.full(np.asarray(eta).shape, 0.25, dtype=float)

    def get_rng_state(self):
        return np.asarray([self.calls], dtype=np.int64)

    def set_rng_state(self, state):
        self.calls = int(np.asarray(state).ravel()[0])


def _run_mock(data, args):
    sampler = _MockPGSampler()
    return run_shared_gp_pg(
        data,
        args,
        draw_omega=sampler.draw,
        get_omega_state=sampler.get_rng_state,
        set_omega_state=sampler.set_rng_state,
    )


def test_portable_exact_continuation_and_content_rejection() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        uninterrupted = _run_mock(data, _args(bundle))
        with TemporaryDirectory(prefix="shared_gp_portable_checkpoint_") as directory:
            checkpoint = Path(directory) / "chain.pkl"
            partial_args = _args(bundle, checkpoint)
            partial_args.stop_after = 4
            with unittest.TestCase().assertRaises(CheckpointStop):
                _run_mock(data, partial_args)

            moved_bundle = Path(directory) / "moved_bundle"
            shutil.copytree(bundle, moved_bundle)
            moved_data = load_inference_data(moved_bundle)
            resumed = _run_mock(
                moved_data, _args(moved_bundle, checkpoint, resume=True)
            )
            resumed_from_final = _run_mock(
                moved_data, _args(moved_bundle, checkpoint, resume=True)
            )

            changed_X = np.load(moved_bundle / "X.npy", allow_pickle=False)
            changed_X[0, 0] += 0.01
            np.save(moved_bundle / "X.npy", changed_X)
            changed_data = load_inference_data(moved_bundle)
            with unittest.TestCase().assertRaisesRegex(
                ValueError, "settings do not match"
            ):
                _run_mock(
                    changed_data,
                    _args(moved_bundle, checkpoint, resume=True),
                )

    for key in (
        "active_gamma_full_trace",
        "mu_full_trace",
        "edge_gp_var_full_trace",
        "sigma_b2_trace",
        "theta0_trace",
        "gamma_trace_packed",
        "gamma_prob",
        "edge_prob",
        "b_mean",
        "q_mean",
        "eta_mean",
    ):
        assert np.array_equal(uninterrupted[key], resumed[key]), key
        assert np.array_equal(uninterrupted[key], resumed_from_final[key]), key


@unittest.skipUnless(
    _R_AVAILABLE,
    "Rscript and BayesLogit are required",
)
def test_exact_continuation() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        uninterrupted = _run(data, _args(bundle))

        with TemporaryDirectory(prefix="shared_gp_checkpoint_") as directory:
            checkpoint = Path(directory) / "chain.pkl"
            partial_args = _args(bundle, checkpoint)
            partial_args.stop_after = 4
            try:
                _run(data, partial_args)
            except CheckpointStop:
                pass
            else:
                raise AssertionError(
                    "partial run did not stop at its checkpoint"
                )
            assert checkpoint.exists()
            moved_bundle = Path(directory) / "moved_bundle"
            shutil.copytree(bundle, moved_bundle)
            moved_data = load_inference_data(moved_bundle)
            resumed = _run(
                moved_data, _args(moved_bundle, checkpoint, resume=True)
            )
            resumed_from_final = _run(
                data, _args(bundle, checkpoint, resume=True)
            )

    exact_keys = (
        "active_gamma_full_trace",
        "mu_full_trace",
        "edge_gp_var_full_trace",
        "sigma_b2_trace",
        "theta0_trace",
        "gamma_trace_packed",
        "gamma_prob",
        "edge_prob",
        "b_mean",
        "q_mean",
        "eta_mean",
    )
    for key in exact_keys:
        assert np.array_equal(uninterrupted[key], resumed[key]), key
        assert np.array_equal(uninterrupted[key], resumed_from_final[key]), key


@unittest.skipUnless(
    _R_AVAILABLE,
    "Rscript and BayesLogit are required",
)
def test_state_only_extension() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        uninterrupted = _run(
            data, _args(bundle, iters=12, burn=2)
        )
        with TemporaryDirectory(prefix="shared_gp_extension_") as directory:
            checkpoint = Path(directory) / "base.pkl"
            base_args = _args(bundle, checkpoint)
            _run(data, base_args)
            extension = _run(
                data,
                _args(
                    bundle,
                    checkpoint,
                    continue_from=True,
                    iters=4,
                    burn=0,
                ),
            )
    assert int(extension["continued_from_iteration"]) == 8
    full_trace_keys = (
        "active_gamma_full_trace",
        "mu_full_trace",
        "edge_gp_var_full_trace",
    )
    retained_trace_keys = (
        "sigma_b2_trace",
        "theta0_trace",
        "q_bar_trace",
        "gamma_trace_packed",
    )
    for key in full_trace_keys:
        assert np.array_equal(uninterrupted[key][-4:], extension[key]), key
    for key in retained_trace_keys:
        assert np.array_equal(uninterrupted[key][-4:], extension[key]), key


def test_mock_state_only_extension() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        uninterrupted = _run_mock(data, _args(bundle, iters=12, burn=2))
        with TemporaryDirectory(prefix="shared_gp_mock_extension_") as directory:
            checkpoint = Path(directory) / "base.pkl"
            _run_mock(data, _args(bundle, checkpoint))
            extension = _run_mock(
                data,
                _args(
                    bundle,
                    checkpoint,
                    continue_from=True,
                    iters=4,
                    burn=0,
                ),
            )
    assert int(extension["continued_from_iteration"]) == 8
    for key in (
        "active_gamma_full_trace",
        "mu_full_trace",
        "edge_gp_var_full_trace",
        "sigma_b2_trace",
        "theta0_trace",
        "q_bar_trace",
        "gamma_trace_packed",
    ):
        assert np.array_equal(uninterrupted[key][-4:], extension[key]), key


def test_mock_sigma0_checkpoint_round_trip() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        uninterrupted = _run_mock(data, _args(bundle))
        with TemporaryDirectory(prefix="shared_gp_sigma0_") as directory:
            checkpoint = Path(directory) / "chain.pkl"
            partial_args = _args(bundle, checkpoint)
            partial_args.stop_after = 4
            with unittest.TestCase().assertRaises(CheckpointStop):
                _run_mock(data, partial_args)

            payload = _read_checkpoint(checkpoint)
            assert "common_gp_var" in payload
            assert payload["common_gp_var"] > 0.0
            accumulators = payload["accumulators"]
            assert len(accumulators["common_gp_var_full_trace"]) == 4

            resumed = _run_mock(
                data, _args(bundle, checkpoint, resume=True)
            )

    assert np.unique(uninterrupted["common_gp_var_full_trace"]).size == 8
    for key in (
        "common_gp_var_full_trace",
        "common_gp_var_trace",
        "active_gamma_full_trace",
        "mu_full_trace",
        "edge_gp_var_full_trace",
        "theta0_trace",
        "gamma_prob",
    ):
        assert np.array_equal(uninterrupted[key], resumed[key]), key


def test_corrupted_and_incomplete_checkpoints_are_rejected() -> None:
    with TemporaryDirectory(prefix="shared_gp_invalid_checkpoint_") as directory:
        path = Path(directory) / "checkpoint.pkl"
        path.write_bytes(b"not a pickle")
        with unittest.TestCase().assertRaisesRegex(ValueError, "corrupted checkpoint"):
            _read_checkpoint(path)

        with path.open("wb") as stream:
            pickle.dump({"version": _CHECKPOINT_VERSION}, stream)
        with unittest.TestCase().assertRaisesRegex(ValueError, "missing keys"):
            _read_checkpoint(path)

        with path.open("wb") as stream:
            pickle.dump({"version": _CHECKPOINT_VERSION - 2}, stream)
        with unittest.TestCase().assertRaisesRegex(
            ValueError, f"expected {_CHECKPOINT_VERSION}"
        ):
            _read_checkpoint(path)


if __name__ == "__main__":
    test_portable_exact_continuation_and_content_rejection()
    test_mock_state_only_extension()
    test_corrupted_and_incomplete_checkpoints_are_rejected()
    test_exact_continuation()
    test_state_only_extension()
    print("shared-GP checkpoint checks passed")
