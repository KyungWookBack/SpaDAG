"""Graph recovery of a saved inference result against simulation truth.

Inference never reads ground truth; this module loads it separately and
compares triangle graphs on the shared mesh, or edge probabilities at spatial
locations when the inference mesh differs from the truth mesh.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np

from .mcmc_data import (
    load_inference_data,
    load_triangle_edge_truth,
)
from .mesh_evaluation import (
    EVALUATION_KINDS,
    SOBOL_DEFAULT_M,
    SOBOL_DEFAULT_SEED,
    evaluate_at_locations,
    sobol_evaluation_points,
)
from .metrics import triangle_edge_recovery

EVAL_POINT_CHOICES = ("triangle",) + EVALUATION_KINDS + ("both",)


def evaluate_reversal_recovery(
    edge_prob: np.ndarray,
    reversal_motifs: list[dict[str, object]],
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Evaluate strict recovery of both regional directions of each motif."""
    edge_prob = np.asarray(edge_prob, dtype=float)
    if edge_prob.ndim != 3 or edge_prob.shape[1] != edge_prob.shape[2]:
        raise ValueError("edge_prob must have shape (T, p, p)")
    if not np.all(np.isfinite(edge_prob)) or np.any(
        (edge_prob < 0.0) | (edge_prob > 1.0)
    ):
        raise ValueError("edge_prob values must be finite and lie in [0, 1]")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    selected = edge_prob >= threshold
    rows: dict[str, dict[str, bool]] = {}
    for motif in reversal_motifs:
        name = str(motif["name"])
        left = motif["left"]
        right = motif["right"]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError(f"malformed reversal metadata for {name}")

        def recovered(side: Mapping[str, object]) -> bool:
            parent = int(side["parent"])
            child = int(side["child"])
            triangles = np.asarray(side["triangles"], dtype=int)
            if triangles.size == 0 or np.any(triangles < 0) or np.any(
                triangles >= edge_prob.shape[0]
            ):
                raise ValueError(f"invalid triangle set for reversal motif {name}")
            correct = selected[triangles, child, parent]
            wrong = selected[triangles, parent, child]
            return bool(np.all(correct) and not np.any(wrong))

        left_ok = recovered(left)
        right_ok = recovered(right)
        rows[name] = {
            "left_direction_recovered": left_ok,
            "right_direction_recovered": right_ok,
            "both_directions_recovered": left_ok and right_ok,
        }
    joint = [row["both_directions_recovered"] for row in rows.values()]
    return {
        "motifs": rows,
        "both_directions_recovery_rate": float(np.mean(joint)) if joint else 0.0,
        "motifs_recovered": int(np.sum(joint)),
        "motifs_total": len(joint),
    }


def evaluate_triangle_recovery(
    result: Mapping[str, object],
    edge_true: np.ndarray,
    *,
    reversal_motifs: list[dict[str, object]] | None = None,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Evaluate the triangle-level graph target against simulation truth.

    Edges with posterior probability at least ``threshold`` are selected.
    """
    required = ("edge_prob",)
    missing = [name for name in required if name not in result]
    if missing:
        raise KeyError(f"result is missing required arrays: {', '.join(missing)}")

    edge_prob = np.asarray(result["edge_prob"], dtype=float)
    summary: dict[str, object] = {
        "triangle_recovery": triangle_edge_recovery(
            edge_prob, edge_true, thresh=threshold
        ),
    }
    if reversal_motifs:
        summary["reversal_recovery"] = evaluate_reversal_recovery(
            edge_prob,
            reversal_motifs,
            threshold=threshold,
        )
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute triangle-graph recovery for a simulation run."
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--truth",
        type=Path,
        required=True,
        help="directory containing edge_triangle.npy",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--eval-points",
        choices=EVAL_POINT_CHOICES,
        default="triangle",
        help=(
            "triangle: compare triangle graphs on the shared mesh (default); "
            "sobol/observed/both: compare at spatial locations, which also allows "
            "an inference mesh that differs from the truth mesh"
        ),
    )
    parser.add_argument(
        "--truth-mesh",
        type=Path,
        default=None,
        help="directory with the truth mesh_points.npy/mesh_triangles.npy (default: --truth)",
    )
    parser.add_argument("--sobol-m", type=int, default=SOBOL_DEFAULT_M, help="2**m Sobol points")
    parser.add_argument("--sobol-seed", type=int, default=SOBOL_DEFAULT_SEED)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="posterior probability at or above which an edge is selected (default 0.5)")
    return parser.parse_args(argv)


def _load_truth_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = np.load(path / "mesh_points.npy", allow_pickle=False)
    triangles = np.load(path / "mesh_triangles.npy", allow_pickle=False)
    return np.asarray(points, dtype=float), np.asarray(triangles, dtype=int)


def evaluate_location_recovery(
    result: Mapping[str, object],
    data,
    *,
    truth_dir: Path,
    data_dir: Path | None = None,
    truth_mesh_dir: Path | None = None,
    kinds: tuple[str, ...] = EVALUATION_KINDS,
    sobol_m: int = SOBOL_DEFAULT_M,
    sobol_seed: int = SOBOL_DEFAULT_SEED,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Location-based recovery on Sobol and/or observed locations.

    ``data`` is the loaded inference bundle (its mesh is the inference mesh);
    ``data_dir`` must contain ``obs_points.npy`` when ``"observed"`` is
    requested.  The truth mesh and ``edge_triangle.npy`` come from
    ``truth_mesh_dir``/``truth_dir``.
    """
    edge_prob = np.asarray(result["edge_prob"], dtype=float)
    truth_mesh_dir = truth_dir if truth_mesh_dir is None else truth_mesh_dir
    truth_mesh = _load_truth_mesh(truth_mesh_dir)
    edge_true = np.load(truth_dir / "edge_triangle.npy", allow_pickle=False)
    if edge_true.shape != (len(truth_mesh[1]), data.p, data.p):
        raise ValueError(
            f"edge_triangle.npy must have shape {(len(truth_mesh[1]), data.p, data.p)}, got {edge_true.shape}"
        )
    fit_mesh = (np.asarray(data.mesh_points, dtype=float), np.asarray(data.triangles, dtype=int))
    summary: dict[str, object] = {}
    for kind in kinds:
        if kind == "sobol":
            points = sobol_evaluation_points(sobol_m, sobol_seed)
            extra = {"sobol_m": int(sobol_m), "sobol_seed": int(sobol_seed)}
        elif kind == "observed":
            if data_dir is None or not (Path(data_dir) / "obs_points.npy").is_file():
                raise FileNotFoundError("observed evaluation needs obs_points.npy in the inference bundle")
            points = np.asarray(np.load(Path(data_dir) / "obs_points.npy", allow_pickle=False), dtype=float)
            if points.shape != (data.n, 2):
                raise ValueError(f"obs_points.npy must have shape ({data.n}, 2), got {points.shape}")
            extra = {"note": "observed locations are training locations, not a hold-out set"}
        else:
            raise ValueError(f"unknown evaluation kind {kind!r}")
        entry = evaluate_at_locations(edge_prob, fit_mesh, truth_mesh, edge_true, points, threshold=threshold)
        entry.update(extra)
        summary[kind] = entry
    return summary


def main() -> None:
    args = parse_args()
    data = load_inference_data(args.data)
    with np.load(args.result, allow_pickle=False) as result:
        result = {key: result[key] for key in result.files}
    summary: dict[str, object] = {}
    if args.eval_points == "triangle":
        edge_true = load_triangle_edge_truth(args.truth, data)
        metadata_path = args.truth / "metadata.json"
        reversal_motifs = None
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            reversal_motifs = metadata.get("reversal_motifs")
        summary = evaluate_triangle_recovery(
            result,
            edge_true,
            reversal_motifs=reversal_motifs,
            threshold=args.threshold,
        )
    else:
        kinds = EVALUATION_KINDS if args.eval_points == "both" else (args.eval_points,)
        summary["location_recovery"] = evaluate_location_recovery(
            result,
            data,
            truth_dir=args.truth,
            data_dir=args.data,
            truth_mesh_dir=args.truth_mesh,
            kinds=kinds,
            sobol_m=args.sobol_m,
            sobol_seed=args.sobol_seed,
            threshold=args.threshold,
        )

    rendered = json.dumps(
        summary,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
