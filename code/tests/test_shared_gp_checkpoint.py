"""Exact-continuation checks for the shared-GP runner."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil
import unittest

import numpy as np

from spadag.mcmc_data import load_inference_data
from spadag.compact_runner import read_checkpoint
from spadag.compact_storage import load_stream
from spadag.cli import (
    CheckpointStop,
    parse_args,
    run_shared_gp_pg,
)
from tests.toy_data_fixture import temporary_toy_bundle


_STREAMS = ("scalar", "gamma", "field", "component")


def _args(
    data_path: Path,
    trace_dir: Path,
    checkpoint: Path | None = None,
    *,
    resume: bool = False,
    iters: int = 8,
    burn: int = 2,
):
    argv = [
        "--data", str(data_path), "--trace-dir", str(trace_dir),
        "--iters", str(iters),
        "--burn", str(burn),
        "--thin", "1",
        "--scalar-thin", "1",
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
    return parse_args(argv)


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


def _manifest(trace_dir: Path) -> dict:
    return json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))


def _stream(trace_dir: Path, stream: str) -> dict:
    return load_stream(trace_dir, _manifest(trace_dir)["head"], stream)


def _assert_same_chain(expected: Path, actual: Path) -> None:
    with np.load(expected / "summary.npz") as a, np.load(actual / "summary.npz") as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            if key != "time":
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    for stream in _STREAMS:
        a, b = _stream(expected, stream), _stream(actual, stream)
        assert a.keys() == b.keys(), stream
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"{stream}/{key}")


def test_portable_exact_continuation_and_content_rejection() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        with TemporaryDirectory(prefix="shared_gp_portable_checkpoint_") as directory:
            root = Path(directory)
            uninterrupted = root / "uninterrupted"
            _run_mock(data, _args(bundle, uninterrupted))
            checkpoint = root / "chain.pkl"
            trace = root / "trace"
            partial_args = _args(bundle, trace, checkpoint)
            partial_args.stop_after = 4
            with unittest.TestCase().assertRaises(CheckpointStop):
                _run_mock(data, partial_args)

            moved_bundle = root / "moved_bundle"
            shutil.copytree(bundle, moved_bundle)
            moved_data = load_inference_data(moved_bundle)
            _run_mock(
                moved_data, _args(moved_bundle, trace, checkpoint, resume=True)
            )
            _assert_same_chain(uninterrupted, trace)
            _run_mock(
                moved_data, _args(moved_bundle, trace, checkpoint, resume=True)
            )
            _assert_same_chain(uninterrupted, trace)

            changed_X = np.load(moved_bundle / "X.npy", allow_pickle=False)
            changed_X[0, 0] += 0.01
            np.save(moved_bundle / "X.npy", changed_X)
            changed_data = load_inference_data(moved_bundle)
            with unittest.TestCase().assertRaisesRegex(
                ValueError, "settings do not match"
            ):
                _run_mock(
                    changed_data,
                    _args(moved_bundle, trace, checkpoint, resume=True),
                )


def test_mock_sigma0_checkpoint_round_trip() -> None:
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        with TemporaryDirectory(prefix="shared_gp_sigma0_") as directory:
            root = Path(directory)
            uninterrupted = root / "uninterrupted"
            _run_mock(data, _args(bundle, uninterrupted))
            checkpoint = root / "chain.pkl"
            trace = root / "trace"
            partial_args = _args(bundle, trace, checkpoint)
            partial_args.stop_after = 4
            with unittest.TestCase().assertRaises(CheckpointStop):
                _run_mock(data, partial_args)

            payload = read_checkpoint(checkpoint)
            assert "common_gp_var" in payload
            assert payload["common_gp_var"] > 0.0
            assert len(_stream(trace, "scalar")["iteration"]) == 4

            _run_mock(data, _args(bundle, trace, checkpoint, resume=True))
            names = _manifest(uninterrupted)["scalar_names"]
            expected = _stream(uninterrupted, "scalar")
            resumed = _stream(trace, "scalar")

    common = expected["values"][:, names.index("common_var")]
    assert common.shape == (8,)
    assert np.unique(common).size == 8
    for key in ("iteration", "values", "postburn"):
        np.testing.assert_array_equal(expected[key], resumed[key], err_msg=key)


if __name__ == "__main__":
    test_portable_exact_continuation_and_content_rejection()
    test_mock_sigma0_checkpoint_round_trip()
    print("shared-GP checkpoint checks passed")
