"""Observer invariance, online estimands and crash/continuation protocol tests."""
import json
import shutil

import numpy as np
import pytest

from spadag.cli import CheckpointStop, parse_args, run_shared_gp_pg
from spadag.compact_runner import read_checkpoint
from spadag.compact_storage import (ChunkWriter, Moments, chunk_links,
                                   load_stream, triangle_coefficients)
from spadag.mcmc_data import load_inference_data
from spadag.polya_gamma import BayesLogitRPGSampler, bayeslogit_available
from tests.test_shared_gp_checkpoint import _MockPGSampler
from tests.toy_data_fixture import temporary_toy_bundle


def args_for(bundle, root, *, iters=37, extra=()):
    argv = ["--data", str(bundle), "--iters", str(iters), "--burn", "3", "--thin", "2",
            "--field-thin", "0", "--gamma-trace-thin", "3", "--common-gp-var", "1", "--edge-gp-var", "1",
            "--log-every", "0", "--seed", "902", "--checkpoint", str(root / "checkpoint.pkl"),
            "--checkpoint-every", "7", "--trace-dir", str(root / "trace"), "--scalar-thin", "2",
            "--block-iterations", "10", "--chunk-iterations", "5"]
    return parse_args(argv + list(extra))


def run(data, args, real=False):
    if real:
        with BayesLogitRPGSampler(seed=args.seed) as sampler:
            return run_shared_gp_pg(data, args, draw_omega=sampler.draw,
                get_omega_state=sampler.get_rng_state, set_omega_state=sampler.set_rng_state)
    sampler = _MockPGSampler()
    return run_shared_gp_pg(data, args, draw_omega=sampler.draw,
                get_omega_state=sampler.get_rng_state, set_omega_state=sampler.set_rng_state)


def same_state(a, b):
    for name in ("b", "gamma", "theta", "sigma2", "sigma_b2", "a0", "a_spatial", "tau_alpha", "pi", "m"):
        assert np.array_equal(getattr(a["sampler_state"], name), getattr(b["sampler_state"], name)), name
    assert a["numpy_rng_state"] == b["numpy_rng_state"]
    assert np.array_equal(a["omega_rng_state"], b["omega_rng_state"])
    assert np.array_equal(a["shared_gp_state"].theta0, b["shared_gp_state"].theta0)
    assert a["common_gp_var"] == b["common_gp_var"]


def test_observer_and_online_estimands(tmp_path):
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        plain_args = args_for(bundle, tmp_path / "plain")
        plain = run(data, plain_args)
        grid_args = args_for(bundle, tmp_path / "grid", extra=["--b-trace-thin", "2", "--latent-thin", "2"])
        compact = run(data, grid_args)
        new = read_checkpoint(grid_args.checkpoint)
        same_state(read_checkpoint(plain_args.checkpoint), new)
        assert np.array_equal(plain["edge_prob"], compact["edge_prob"])
        assert np.array_equal(plain["b_mean"], compact["b_mean"])
        draws = load_stream(grid_args.trace_dir, new["head"], "b")["values"]
        for key, raw in (("b", draws), ("signed", np.stack([triangle_coefficients(x, data.triangles)[0] for x in draws])),
                         ("rms", np.stack([triangle_coefficients(x, data.triangles)[1] for x in draws]))):
            np.testing.assert_allclose(compact[f"{key}_candidate_mean"], raw.mean(axis=0), atol=1e-14)
            np.testing.assert_allclose(compact[f"{key}_candidate_sd"], raw.std(axis=0, ddof=1), atol=1e-14)
        assert "q_ci" not in compact and "eta_ci" not in compact
        assert not any("draws" in k for k in new["accumulators"])
        scalar = load_stream(grid_args.trace_dir, new["head"], "scalar")
        np.testing.assert_array_equal(scalar["iteration"], np.arange(1, 38, 2))
        p = data.p
        logL = -.5 * (data.n * np.log(2*np.pi*scalar["values"][:, 6:6+p])
                       + scalar["values"][:, 6+p:6+2*p] / scalar["values"][:, 6:6+p]).sum(axis=1)
        np.testing.assert_allclose(logL, scalar["values"][:, 5], atol=1e-12)
        blocks = load_stream(grid_args.trace_dir, new["head"], "blocks")
        np.testing.assert_array_equal(blocks["start"], [4, 14, 24])
        np.testing.assert_array_equal(blocks["iteration"], [13, 23, 33])
        np.testing.assert_array_equal(blocks["n"], [5, 5, 5])
        np.testing.assert_array_equal(blocks["count"].sum(axis=0) + compact["partial_block_count"], compact["edge_count"])


@pytest.mark.parametrize("real", [False, pytest.param(True, marks=pytest.mark.skipif(not bayeslogit_available(), reason="R BayesLogit unavailable"))])
def test_resume_and_global_extension(tmp_path, real):
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        full_args = args_for(bundle, tmp_path / "full")
        full = run(data, full_args, real)
        interrupted = args_for(bundle, tmp_path / "resume", extra=["--stop-after", "19"])
        with pytest.raises(CheckpointStop):
            run(data, interrupted, real)
        resumed_args = args_for(bundle, tmp_path / "resume", extra=["--resume", str(interrupted.checkpoint)])
        resumed = run(data, resumed_args, real)
        # A completed short run is extended to a global target, retaining burn,
        # summary phase, partial fixed block, history and RNG.
        prefix = args_for(bundle, tmp_path / "extend", iters=16)
        run(data, prefix, real)
        extension = args_for(bundle, tmp_path / "extend", extra=["--resume", str(prefix.checkpoint)])
        extended = run(data, extension, real)
        full_cp = read_checkpoint(full_args.checkpoint)
        for result, args in ((resumed, resumed_args), (extended, extension)):
            same_state(full_cp, read_checkpoint(args.checkpoint))
            for key in ("edge_count", "gamma_count", "b_candidate_mean", "b_candidate_sd", "signed_candidate_mean", "rms_candidate_mean", "partial_block_count"):
                np.testing.assert_array_equal(full[key], result[key])
            cp = read_checkpoint(args.checkpoint)
            for stream in ("scalar", "gamma", "blocks"):
                a = load_stream(full_args.trace_dir, full_cp["head"], stream)
                b = load_stream(args.trace_dir, cp["head"], stream)
                assert a.keys() == b.keys()
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
        # Running resume on the final checkpoint is idempotent.
        head_before = read_checkpoint(extension.checkpoint)["head"]
        run(data, extension, real)
        assert read_checkpoint(extension.checkpoint)["head"] == head_before


def test_orphan_recovery_corruption_and_portability(tmp_path, monkeypatch):
    import spadag.compact_runner as runner
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        root = tmp_path / "original"
        args = args_for(bundle, root, extra=["--stop-after", "14"])
        with pytest.raises(CheckpointStop):
            run(data, args)
        old = read_checkpoint(args.checkpoint)
        real_writer = runner.write_checkpoint
        monkeypatch.setattr(runner, "write_checkpoint", lambda *a, **kw: (_ for _ in ()).throw(OSError("injected crash after chunk flush")))
        with pytest.raises(OSError, match="injected crash"):
            run(data, args_for(bundle, root, extra=["--resume", str(args.checkpoint)]))
        assert read_checkpoint(args.checkpoint)["head"] == old["head"]
        monkeypatch.setattr(runner, "write_checkpoint", real_writer)
        moved = tmp_path / "moved"
        shutil.copytree(root, moved)
        resume = args_for(bundle, moved, extra=["--resume", str(moved / "checkpoint.pkl")])
        with pytest.warns(RuntimeWarning, match="uncommitted"):
            result = run(data, resume)
        baseline = run(data, args_for(bundle, tmp_path / "baseline"))
        np.testing.assert_array_equal(result["edge_count"], baseline["edge_count"])
        cp = read_checkpoint(resume.checkpoint)
        (resume.trace_dir / cp["head"]["file"]).write_bytes(b"corruption")
        with pytest.raises(ValueError, match="missing or corrupt"):
            run(data, resume)


def test_bounded_buffers_and_single_draw_moments(tmp_path):
    writer = ChunkWriter(tmp_path, max_bytes=4096)
    for i in range(200):
        writer.add("x", i, values=np.ones(100))
    writer.flush()
    assert writer.peak_buffer_bytes <= 4096
    assert writer.buffer_bytes == 0
    assert len(chunk_links(tmp_path, writer.head)) > 1
    assert len(load_stream(tmp_path, writer.head, "x")["iteration"]) == 200
    moments = Moments((2,))
    moments.add(np.ones(2))
    assert np.isnan(moments.sd()).all()


def test_constant_checkpoint_size(tmp_path):
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        # One attempt per sweep: component records waiting in the chunk buffer
        # make the size fluctuate by a few KB at larger --m-rev, without growth.
        small = args_for(bundle, tmp_path / "size", iters=20, extra=["--milestone-every", "20", "--m-rev", "1"])
        run(data, small)
        size_before = small.checkpoint.stat().st_size
        longer = args_for(bundle, tmp_path / "size", iters=100,
                          extra=["--resume", str(small.checkpoint), "--milestone-every", "20", "--m-rev", "1"])
        run(data, longer)
        assert abs(longer.checkpoint.stat().st_size - size_before) < 1024
        assert len(list((small.checkpoint.parent / "milestones").glob("*.pkl"))) == 5


def test_writer_lock_and_resume_cli(tmp_path):
    from spadag.compact_storage import writer_lock
    with writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already has a writer"):
            with writer_lock(tmp_path):
                pass
    args = parse_args(["--data", "bundle", "--trace-dir", "run/trace", "--resume", "run/checkpoint.pkl", "--checkpoint-every", "5000"])
    assert args.checkpoint == args.resume


def test_targets_precision_and_grid_cli(tmp_path):
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        target = tmp_path / "targets.json"
        target.write_text(json.dumps({"reason": "test", "projections": [[1, 0, 0]], "edges": [[1, 0]],
                                    "orientations": [[1, 0, 0]], "residuals": [{"child": 1, "observations": [0, 1]}]}))
        args = args_for(bundle, tmp_path / "selected", extra=["--diagnostic-targets", str(target),
                        "--trace-dtype", "float32", "--b-trace-thin", "2", "--latent-thin", "2", "--field-thin", "2"])
        run(data, args)
        cp = read_checkpoint(args.checkpoint)
        assert cp["sampler_state"].b.dtype == np.float64
        assert load_stream(args.trace_dir, cp["head"], "b")["values"].dtype == np.float32
        assert load_stream(args.trace_dir, cp["head"], "b")["values"].shape[1] == 1
        assert len(load_stream(args.trace_dir, cp["head"], "orientation")["iteration"]) == 34
        assert len(load_stream(args.trace_dir, cp["head"], "projection")["iteration"]) == 4
        float64 = args_for(bundle, tmp_path / "selected64", extra=["--diagnostic-targets", str(target),
                           "--b-trace-thin", "2", "--latent-thin", "2", "--field-thin", "2"])
        run(data, float64)
        cp64 = read_checkpoint(float64.checkpoint)
        same_state(cp, cp64)
        for stream, key in (("b", "values"), ("latent", "theta"), ("field", "eta")):
            a = load_stream(args.trace_dir, cp["head"], stream)[key]
            b = load_stream(float64.trace_dir, cp64["head"], stream)[key]
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        latent = load_stream(float64.trace_dir, cp64["head"], "latent")
        field = load_stream(float64.trace_dir, cp64["head"], "field")
        basis = cp64["shared_gp_state"].decomposition.common_basis
        recovered = (latent["theta"] @ data.A_kl.T
                     + (latent["xi0"] @ basis.T @ data.A_kl.T)[:, None, :]
                     + latent["mu"][:, None, None])
        np.testing.assert_allclose(recovered, field["eta"], atol=1e-12)
        default = parse_args(["--data", str(bundle), "--trace-dir", "fresh"])
        assert (default.thin, default.scalar_thin, default.gamma_trace_thin, default.field_thin) == (5, 5, 10, 0)
