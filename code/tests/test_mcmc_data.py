"""Unit tests for inference and simulation-truth bundle loading."""
from __future__ import annotations

from inspect import signature
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from spadag.mcmc_data import (
    load_inference_data,
    load_simulation_truth,
    load_triangle_edge_truth,
)
from tests.toy_data_fixture import temporary_toy_bundle


class InferenceDataTests(unittest.TestCase):
    def test_loaders_only_accept_bundle_path(self) -> None:
        self.assertEqual(
            tuple(signature(load_inference_data).parameters), ("npy_dir",)
        )
        self.assertEqual(
            tuple(signature(load_simulation_truth).parameters), ("npy_dir", "data")
        )
        self.assertEqual(
            tuple(signature(load_triangle_edge_truth).parameters),
            ("npy_dir", "data"),
        )

    def test_truth_free_bundle_loads_for_inference(self) -> None:
        with temporary_toy_bundle(include_truth=False) as bundle:
            data = load_inference_data(bundle)
            with self.assertRaisesRegex(
                FileNotFoundError, "q_node.npy"
            ):
                load_simulation_truth(bundle)

        self.assertEqual((data.n, data.p, data.H, data.L), (12, 3, 4, 2))
        self.assertEqual(data.U.shape, (data.n, data.H))
        self.assertEqual(data.UtU.shape, (data.H, data.H))
        self.assertEqual(data.baseline_weights.shape, (data.H,))
        self.assertEqual(data.baseline_basis.shape, (data.H, data.H - 1))
        self.assertFalse(hasattr(data, "gamma"))
        self.assertFalse(hasattr(data, "gamma_true"))

    def test_loader_always_uses_all_nonself_directed_pairs(self) -> None:
        with temporary_toy_bundle() as bundle:
            data = load_inference_data(bundle)

        expected_parents = np.ones((data.p, data.p), dtype=int)
        np.fill_diagonal(expected_parents, 0)
        np.testing.assert_array_equal(data.parents, expected_parents)
        self.assertEqual(len(data.pairs), data.p * (data.p - 1))
        self.assertEqual(
            set(data.pairs),
            {
                (j, k)
                for j in range(data.p)
                for k in range(data.p)
                if j != k
            },
        )

    def test_simulation_truth_is_loaded_separately(self) -> None:
        with temporary_toy_bundle() as bundle:
            data = load_inference_data(bundle)
            truth = load_simulation_truth(bundle, data)

        self.assertEqual(truth.gamma.shape, (data.p, data.p, data.H))
        self.assertEqual(
            truth.edge_triangle.shape,
            (len(data.triangles), data.p, data.p),
        )
        self.assertEqual(truth.sigma_noise.shape, (data.p,))

    def test_prior_specific_truth_scales_are_optional(self) -> None:
        with temporary_toy_bundle() as bundle:
            data = load_inference_data(bundle)
            (bundle / "hyper_sigma_b.npy").unlink()
            (bundle / "hyper_tau2.npy").unlink()
            truth = load_simulation_truth(bundle, data)

        self.assertIsNone(truth.sigma_b)
        self.assertIsNone(truth.tau2)
        self.assertEqual(truth.mu, -1.0)

    def test_triangle_truth_loader_requires_only_edge_triangle(self) -> None:
        unused_truth = (
            "q_node",
            "gamma",
            "b_coeffs",
            "beta_obs",
            "sigma_noise",
            "hyper_sigma_b",
            "hyper_mu",
            "hyper_tau2",
        )
        with temporary_toy_bundle() as bundle:
            data = load_inference_data(bundle)
            expected = np.load(bundle / "edge_triangle.npy", allow_pickle=False)
            for stem in unused_truth:
                (bundle / f"{stem}.npy").unlink()

            loaded = load_triangle_edge_truth(bundle, data)

        np.testing.assert_array_equal(loaded, expected)

    def test_triangle_truth_shape_and_binary_values_are_validated(self) -> None:
        with temporary_toy_bundle() as bundle:
            data = load_inference_data(bundle)
            np.save(bundle / "edge_triangle.npy", np.zeros((1, 2, 2), dtype=int))
            with self.assertRaisesRegex(ValueError, "must have shape"):
                load_triangle_edge_truth(bundle, data)

        with temporary_toy_bundle() as bundle:
            data = load_inference_data(bundle)
            edge = np.load(bundle / "edge_triangle.npy", allow_pickle=False)
            edge[0, 0, 1] = 2
            np.save(bundle / "edge_triangle.npy", edge)
            with self.assertRaisesRegex(ValueError, "must be binary"):
                load_triangle_edge_truth(bundle, data)

    def test_each_required_inference_stem_has_explicit_error(self) -> None:
        stems = ("X", "Phi_obs", "mesh_points", "mesh_triangles", "A_kl")
        for stem in stems:
            with self.subTest(stem=stem), temporary_toy_bundle() as bundle:
                (bundle / f"{stem}.npy").unlink()
                with self.assertRaisesRegex(
                    FileNotFoundError, rf"schema-v1.*{stem}\.npy"
                ):
                    load_inference_data(bundle)

    def test_inference_shape_finite_and_geometry_validation(self) -> None:
        cases = (
            ("X", np.zeros(3), "X.npy must be 2-dimensional"),
            ("Phi_obs", np.full((12, 4), np.nan), "only finite"),
            ("Phi_obs", np.full((12, 4), 0.5), "rows must sum to one"),
            ("mesh_points", np.zeros((3, 2)), "must have shape"),
            ("mesh_triangles", np.asarray([[0, 1, 9]]), "indices must be"),
            ("mesh_triangles", np.asarray([[0, 1, 1]]), "repeated-vertex"),
            ("mesh_triangles", np.asarray([[0, 1, 2.5]]), "integer values"),
            ("A_kl", np.zeros((3, 2)), "must have shape"),
        )
        for stem, replacement, message in cases:
            with self.subTest(stem=stem, message=message), temporary_toy_bundle() as bundle:
                np.save(bundle / f"{stem}.npy", replacement)
                with self.assertRaisesRegex(ValueError, message):
                    load_inference_data(bundle)

    def test_single_variable_inference_bundle_is_rejected(self) -> None:
        with temporary_toy_bundle() as bundle:
            X = np.load(bundle / "X.npy", allow_pickle=False)[:, :1]
            np.save(bundle / "X.npy", X)
            with self.assertRaisesRegex(ValueError, "at least two variables"):
                load_inference_data(bundle)

    def test_degenerate_triangle_is_rejected(self) -> None:
        with temporary_toy_bundle() as bundle:
            np.save(
                bundle / "mesh_points.npy",
                np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
            )
            with self.assertRaisesRegex(ValueError, "degenerate triangle"):
                load_inference_data(bundle)

    def test_truth_dimensions_are_checked_against_inference_data(self) -> None:
        with temporary_toy_bundle() as bundle:
            data = load_inference_data(bundle)
            np.save(bundle / "gamma.npy", np.zeros((2, 2, data.H), dtype=int))
            with self.assertRaisesRegex(ValueError, "gamma.npy must have shape"):
                load_simulation_truth(bundle, data)

    def test_truth_q_allows_only_self_edge_nan(self) -> None:
        with temporary_toy_bundle() as bundle:
            data = load_inference_data(bundle)
            q = np.load(bundle / "q_node.npy", allow_pickle=False)
            for node in range(data.p):
                q[node, node] = np.nan
            np.save(bundle / "q_node.npy", q)
            load_simulation_truth(bundle, data)
            q[0, 1, 0] = np.nan
            np.save(bundle / "q_node.npy", q)
            with self.assertRaisesRegex(ValueError, "candidate-edge values must be finite"):
                load_simulation_truth(bundle, data)

    def test_unrelated_npy_is_not_opened(self) -> None:
        with temporary_toy_bundle() as bundle:
            extra = bundle / "large_unrelated.npy"
            extra.write_bytes(b"not even a valid npy file")
            original_load = np.load
            opened: list[Path] = []

            def recording_load(path, *args, **kwargs):
                opened.append(Path(path))
                return original_load(path, *args, **kwargs)

            with patch("spadag.mcmc_data.np.load", side_effect=recording_load):
                load_inference_data(bundle)
            self.assertNotIn(extra, opened)

    def test_fingerprint_depends_on_values_not_directory(self) -> None:
        with temporary_toy_bundle() as first, temporary_toy_bundle() as second:
            first_data = load_inference_data(first)
            second_data = load_inference_data(second)
            self.assertEqual(first_data.data_fingerprint, second_data.data_fingerprint)
            X = np.load(second / "X.npy", allow_pickle=False)
            X[0, 0] += 0.01
            np.save(second / "X.npy", X)
            changed_data = load_inference_data(second)
            self.assertNotEqual(
                first_data.data_fingerprint, changed_data.data_fingerprint
            )


if __name__ == "__main__":
    unittest.main()
