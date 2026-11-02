import json

import numpy as np
import pytest

pytest.importorskip("arviz")

from spadag.diagnostics import diagnose, scalar_diagnostic
from spadag.mcmc_data import load_inference_data
from tests.test_compact_storage import args_for, run
from tests.toy_data_fixture import temporary_toy_bundle


def test_gate_rejects_shift_constant_and_missing():
    rng = np.random.default_rng(1)
    iid = rng.normal(size=(4, 2000))
    assert scalar_diagnostic(iid)["status"] == "PASS"
    iid[0] += 3
    assert scalar_diagnostic(iid)["status"] == "FAIL"
    assert scalar_diagnostic(np.ones((4, 2000)))["status"] == "UNAVAILABLE"
    assert scalar_diagnostic(np.ones((4, 2)))["status"] == "UNAVAILABLE"
    assert scalar_diagnostic(np.full((4, 200), np.nan))["status"] == "UNAVAILABLE"


def test_report_chain_identity_noise_and_recovery(tmp_path):
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        target = tmp_path / "target.json"
        target.write_text(json.dumps({"projections": [[1, 0, 0]]}))
        directories = []
        for seed in (91, 92):
            args = args_for(bundle, tmp_path / str(seed), iters=45,
                            extra=["--seed", str(seed), "--diagnostic-targets", str(target), "--projection-thin", "2"])
            run(data, args)
            directories.append(args.trace_dir)
        truth = np.load(bundle / "edge_triangle.npy")
        report = diagnose(directories, truth=truth, child=1, top=2)
        assert report["status"] == "FAIL"
        assert report["noise_parameters"] == data.p
        assert len(report["scalars"]) == 6 + 4 * data.p + 1
        assert report["projection_status"] == "AVAILABLE"
        assert len(report["directions_and_noise"]) == 2
        assert report["chains"][0]["seed"] == 91
        assert set(report["chains"][0]["triangle_recovery"]) >= {"tpr", "fdr", "f1"}
        assert report["graph_and_coefficient_review_required"]
        json.dumps(report, allow_nan=False)
        with pytest.raises(ValueError, match="distinct"):
            diagnose([directories[0], directories[0]])
        recent = diagnose(directories, top=0, recent_iterations=10)
        assert recent["scalar_aligned_draws_per_chain"] == 5


def test_load_truth_accepts_directory_or_file(tmp_path):
    from spadag.diagnostics import load_truth

    truth = np.zeros((2, 3, 3), dtype=int)
    truth[0, 1, 0] = 1
    np.save(tmp_path / "edge_triangle.npy", truth)
    motifs = [{"name": "R1", "nodes": {"a": 0, "b": 1, "c": 2},
               "left": {"parent": 0, "child": 1, "triangles": [0]},
               "right": {"parent": 1, "child": 0, "triangles": [1]}}]
    (tmp_path / "metadata.json").write_text(json.dumps({"reversal_motifs": motifs}), encoding="utf-8")

    loaded, loaded_motifs = load_truth(tmp_path, None)
    np.testing.assert_array_equal(loaded, truth)
    assert loaded_motifs == motifs

    loaded, loaded_motifs = load_truth(tmp_path / "edge_triangle.npy", None)
    np.testing.assert_array_equal(loaded, truth)
    assert loaded_motifs is None

    explicit = tmp_path / "motifs.json"
    explicit.write_text(json.dumps([]), encoding="utf-8")
    assert load_truth(tmp_path, explicit) == (pytest.approx(truth), [])
    assert load_truth(None, None) == (None, None)
