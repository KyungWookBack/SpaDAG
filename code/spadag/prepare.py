"""Build a SpaDAG inference bundle from observations and their locations.

The sampler reads a fixed five-array bundle (``X``, ``Phi_obs``,
``mesh_points``, ``mesh_triangles``, ``A_kl``), so the mesh and the K-L design
are fixed before sampling.  This module constructs the bundle from ``X`` and
two-dimensional locations; a custom mesh or K-L design can be supplied instead.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np

from spadag.mcmc_data import INFERENCE_SCHEMA_VERSION, INFERENCE_STEMS, load_inference_data
from spadag.mesh.builders import MESH_OPTIONS, build_mesh, build_polygon_mesh
from spadag.mesh.core import Mesh, assert_no_degenerate_triangles
from spadag.spatial_gp import build_kl_design, kl_pointwise_variance


_MANAGED_FILENAMES = tuple(f"{stem}.npy" for stem in INFERENCE_STEMS) + (
    "obs_points.npy",
    "preparation_manifest.json",
)


@dataclass(frozen=True)
class PreparedBundle:
    """Summary of a validated bundle written by :func:`prepare_inference_bundle`."""

    path: Path
    n: int
    p: int
    H: int
    T: int
    L: int
    data_fingerprint: str
    mesh_type: str


def _finite_float_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array, got {array.ndim}D")
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must have a real numeric dtype, got {array.dtype}")
    converted = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _integer_triangles(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(
            f"mesh_triangles must have shape (T, 3), got {array.shape}"
        )
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError("mesh_triangles must contain finite integer values")
    converted = array.astype(np.int64)
    if not np.array_equal(array, converted):
        raise ValueError("mesh_triangles must contain integer values")
    return converted


def _rectangular_domain(
    obs_points: np.ndarray,
    domain: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the rectangle mapped to the unit square: the observation
    bounding box, or ``domain = (xmin, xmax, ymin, ymax)`` when given."""
    if domain is None:
        coordinate_min = np.min(obs_points, axis=0)
        coordinate_max = np.max(obs_points, axis=0)
        if np.any(coordinate_max - coordinate_min <= 0.0):
            raise ValueError(
                "obs_points must span a two-dimensional domain; each coordinate "
                "column must have positive range"
            )
        return coordinate_min, coordinate_max

    bounds = np.asarray(domain, dtype=float)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("domain must be four finite values (xmin, xmax, ymin, ymax)")
    coordinate_min = bounds[[0, 2]]
    coordinate_max = bounds[[1, 3]]
    if np.any(coordinate_max <= coordinate_min):
        raise ValueError("domain must satisfy xmin < xmax and ymin < ymax")
    outside = np.flatnonzero(
        np.any((obs_points < coordinate_min) | (obs_points > coordinate_max), axis=1)
    )
    if outside.size:
        raise ValueError(
            f"{outside.size} observation locations are outside the domain "
            f"(first indices: {outside[:10].tolist()})"
        )
    return coordinate_min, coordinate_max


def _mesh_from_observation_domain(
    obs_points: np.ndarray,
    *,
    mesh_type: str,
    n_grid: int,
    target_tris: int,
    seed: int,
    min_interior: int,
    n_boundary_per_edge: int | None,
    min_angle: float,
    use_area_constraint: bool,
    domain: Sequence[float] | None = None,
) -> tuple[Mesh, np.ndarray, np.ndarray]:
    """Build in the unit square and map the mesh back to input coordinates."""
    coordinate_min, coordinate_max = _rectangular_domain(obs_points, domain)
    coordinate_span = coordinate_max - coordinate_min

    unit_points = (obs_points - coordinate_min) / coordinate_span
    unit_mesh = build_mesh(
        unit_points,
        mesh_type=mesh_type,
        n_grid=n_grid,
        target_tris=target_tris,
        seed=seed,
        min_interior=min_interior,
        n_boundary_per_edge=n_boundary_per_edge,
        min_angle=min_angle,
        use_area_constraint=use_area_constraint,
    )
    physical_points = coordinate_min + unit_mesh.points * coordinate_span
    mesh = Mesh(
        points=np.asarray(physical_points, dtype=float),
        triangles=np.asarray(unit_mesh.triangles, dtype=np.int64),
        constraints=unit_mesh.constraints,
    )
    return mesh, coordinate_min, coordinate_max


def _custom_mesh(
    mesh_points: np.ndarray,
    mesh_triangles: np.ndarray,
) -> Mesh:
    points = _finite_float_matrix(mesh_points, "mesh_points")
    if points.shape[1:] != (2,) or len(points) < 3:
        raise ValueError(f"mesh_points must have shape (H, 2), H >= 3, got {points.shape}")
    triangles = _integer_triangles(mesh_triangles)
    if len(triangles) == 0:
        raise ValueError("mesh_triangles must contain at least one triangle")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError(
            f"mesh_triangles indices must be in [0, {len(points)})"
        )
    if np.any(np.apply_along_axis(lambda row: len(set(row)) < 3, 1, triangles)):
        raise ValueError("mesh_triangles contains a repeated-vertex triangle")
    mesh = Mesh(points=points, triangles=triangles, constraints=None)
    assert_no_degenerate_triangles(mesh)
    return mesh


def _validate_kl_design(A_kl: np.ndarray, H: int) -> np.ndarray:
    """Validate the K-L design shape; ``L > H`` (rank deficiency) is allowed.

    Every coefficient keeps a proper Gaussian prior, so a wide design such as
    66 modes on a 36-vertex mesh is valid.  The rank goes into the manifest.
    """
    design = _finite_float_matrix(A_kl, "A_kl")
    if design.shape[0] != H or design.shape[1] == 0:
        raise ValueError(f"A_kl must have shape ({H}, L) with L > 0, got {design.shape}")
    return design


def _kl_rank(design: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(design))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_inference_bundle(
    X: np.ndarray,
    obs_points: np.ndarray,
    outdir: Path,
    *,
    mesh_type: str = "polygon",
    domain: Sequence[float] | None = None,
    n_grid: int = 5,
    target_tris: int = 20,
    seed: int = 0,
    min_interior: int = 4,
    n_boundary_per_edge: int | None = None,
    min_angle: float = 10.0,
    use_area_constraint: bool = True,
    mesh_points: np.ndarray | None = None,
    mesh_triangles: np.ndarray | None = None,
    A_kl: np.ndarray | None = None,
    L: int = 66,
    a: float = 0.01,
    b: float = 1.0,
    max_range: float = 1.0,
    poly_degree: int | None = 10,
    rscript: str | None = None,
    overwrite: bool = False,
    min_obs_per_triangle: int | None = None,
    max_obs_per_triangle: int | None = None,
    refine_rounds: int = 3,
    hull_scale: float = 4.0,
    margin_scale: float = 1.0,
) -> PreparedBundle:
    """Create and validate a schema-v1 inference bundle.

    ``X`` and ``obs_points`` are the only required scientific inputs.  By
    default (``mesh_type="polygon"``) an observation-supported polygon domain
    (concave hull with holes) is estimated in the input coordinates and a
    constrained Delaunay mesh is built on it, optionally coarsened so that
    every triangle holds at least ``min_obs_per_triangle`` observations and
    refined so that none holds more than ``max_obs_per_triangle``.  Passing
    both ``mesh_points`` and ``mesh_triangles`` uses a custom triangulation.
    ``mesh_type="regular"`` or ``"cdt"`` builds a
    unit-square mesh over the observation bounding box, or over the rectangle
    ``domain = (xmin, xmax, ymin, ymax)`` when given; ``domain=(0, 1, 0, 1)``
    reproduces the simulation studies' meshes.  Passing ``A_kl`` skips
    BayesGPfit K-L generation; otherwise
    ``L``, ``a``, ``b``, ``max_range`` and ``poly_degree`` parameterize the
    generated design (default: the manuscript's 66-mode, range-1, degree-10
    expansion), which may have more modes than mesh vertices.
    """
    observations = _finite_float_matrix(X, "X")
    locations = _finite_float_matrix(obs_points, "obs_points")
    n, p = observations.shape
    if n == 0 or p < 2:
        raise ValueError(f"X must have shape (n, p) with n > 0 and p > 1, got {observations.shape}")
    if locations.shape != (n, 2):
        raise ValueError(
            f"obs_points must have shape ({n}, 2) to align with X rows, got {locations.shape}"
        )
    if mesh_type not in MESH_OPTIONS:
        raise ValueError(f"unknown mesh_type {mesh_type!r}; choose from {MESH_OPTIONS}")
    if (mesh_points is None) != (mesh_triangles is None):
        raise ValueError("mesh_points and mesh_triangles must be supplied together")
    if domain is not None and (mesh_points is not None or mesh_type == "polygon"):
        raise ValueError(
            "domain applies only to the unit-square builders (regular, cdt)"
        )
    if A_kl is None:
        if L < 1:
            raise ValueError("L must be positive")
        if not all(np.isfinite(value) for value in (a, b, max_range)):
            raise ValueError("a, b, and max_range must be finite")
        if a <= 0.0 or b <= 0.0 or max_range <= 0.0:
            raise ValueError("a, b, and max_range must be positive")

    outdir = Path(outdir)
    conflicts = [outdir / name for name in _MANAGED_FILENAMES if (outdir / name).exists()]
    if conflicts and not overwrite:
        names = ", ".join(path.name for path in conflicts)
        raise FileExistsError(
            f"refusing to overwrite existing prepared-bundle files in {outdir}: {names}"
        )

    polygon_info: dict[str, object] | None = None
    if mesh_points is None and mesh_type == "polygon":
        mesh, domain = build_polygon_mesh(
            locations,
            target_tris=target_tris,
            hull_scale=hull_scale,
            margin_scale=margin_scale,
            min_interior=min_interior,
            min_angle=min_angle,
            use_area_constraint=use_area_constraint,
            min_obs_per_triangle=min_obs_per_triangle,
            max_obs_per_triangle=max_obs_per_triangle,
            refine_rounds=refine_rounds,
            return_domain=True,
        )
        mesh = Mesh(
            points=np.asarray(mesh.points, dtype=float),
            triangles=np.asarray(mesh.triangles, dtype=np.int64),
            constraints=mesh.constraints,
        )
        coordinate_min = np.min(mesh.points, axis=0)
        coordinate_max = np.max(mesh.points, axis=0)
        selected_mesh_type = "polygon"
        domain_source = "observation-polygon"
        polygon_info = {
            "area": float(domain.area),
            "n_loops": int(len(domain.loops)),
            "n_holes": int(domain.n_holes),
            "spacing": float(domain.spacing),
            "hull_scale": float(hull_scale),
            "margin_scale": float(margin_scale),
            "min_obs_per_triangle": None if min_obs_per_triangle is None else int(min_obs_per_triangle),
            "max_obs_per_triangle": None if max_obs_per_triangle is None else int(max_obs_per_triangle),
            "refine_rounds": int(refine_rounds),
        }
    elif mesh_points is None:
        mesh, coordinate_min, coordinate_max = _mesh_from_observation_domain(
            locations,
            mesh_type=mesh_type,
            n_grid=n_grid,
            target_tris=target_tris,
            seed=seed,
            min_interior=min_interior,
            n_boundary_per_edge=n_boundary_per_edge,
            min_angle=min_angle,
            use_area_constraint=use_area_constraint,
            domain=domain,
        )
        selected_mesh_type = mesh_type
        domain_source = "observation-bounding-box" if domain is None else "user-rectangle"
    else:
        mesh = _custom_mesh(mesh_points, mesh_triangles)
        coordinate_min = np.min(mesh.points, axis=0)
        coordinate_max = np.max(mesh.points, axis=0)
        selected_mesh_type = "custom"
        domain_source = "user-supplied-mesh"

    try:
        from spadag.mesh.basis import basis_matrix
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise ImportError(
                "spadag-prepare requires Matplotlib for P1 basis construction; "
                "install SpaDAG with the 'prepare' extra"
            ) from exc
        raise

    Phi_sparse, triangle_id = basis_matrix(mesh, locations)
    outside = np.flatnonzero(triangle_id < 0)
    if outside.size:
        raise ValueError(
            f"{outside.size} observation locations are outside the mesh "
            f"(first indices: {outside[:10].tolist()})"
        )
    Phi = np.asarray(Phi_sparse.toarray(), dtype=float)

    if A_kl is None:
        kl = build_kl_design(
            mesh.points,
            L=L,
            a=a,
            b=b,
            max_range=max_range,
            poly_degree=poly_degree,
            rscript=rscript,
        )
        design = _validate_kl_design(kl.A, len(mesh.points))
        kl_source = "BayesGPfit"
        energy_kept: float | None = float(kl.energy_kept)
    else:
        design = _validate_kl_design(A_kl, len(mesh.points))
        kl_source = "user-supplied"
        energy_kept = None
    kl_rank = _kl_rank(design)

    pointwise_var = kl_pointwise_variance(design)
    pv_max = float(pointwise_var.max())
    pv_ratio = float(pointwise_var.min() / pv_max) if pv_max > 0.0 else 0.0

    outdir.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "X": observations,
        "Phi_obs": Phi,
        "mesh_points": np.asarray(mesh.points, dtype=float),
        "mesh_triangles": np.asarray(mesh.triangles, dtype=np.int64),
        "A_kl": design,
        "obs_points": locations,
    }
    with tempfile.TemporaryDirectory(prefix="spadag-prepare-", dir=outdir.parent) as tmp:
        stage = Path(tmp)
        for stem, array in arrays.items():
            np.save(stage / f"{stem}.npy", array)
        validated = load_inference_data(stage)
        manifest = {
            "created_by": "spadag-prepare",
            "inference_schema_version": INFERENCE_SCHEMA_VERSION,
            "dimensions": {
                "n": validated.n,
                "p": validated.p,
                "H": validated.H,
                "T": int(len(validated.triangles)),
                "L": validated.L,
            },
            "mesh": {
                "type": selected_mesh_type,
                "domain_source": domain_source,
                "coordinate_min": coordinate_min.tolist(),
                "coordinate_max": coordinate_max.tolist(),
                "n_grid": int(n_grid) if selected_mesh_type == "regular" else None,
                "target_tris": (
                    int(target_tris)
                    if selected_mesh_type == "cdt"
                    else None
                ),
                "seed": int(seed) if selected_mesh_type not in ("regular", "custom", "polygon") else None,
                "polygon": polygon_info,
            },
            "kl": {
                "source": kl_source,
                "L": validated.L,
                "rank": kl_rank,
                "a": float(a) if kl_source == "BayesGPfit" else None,
                "b": float(b) if kl_source == "BayesGPfit" else None,
                "max_range": float(max_range) if kl_source == "BayesGPfit" else None,
                "poly_degree": (
                    (None if poly_degree is None else int(poly_degree))
                    if kl_source == "BayesGPfit" else None
                ),
                "energy_kept": energy_kept,
                "pointwise_variance_min": float(pointwise_var.min()),
                "pointwise_variance_median": float(np.median(pointwise_var)),
                "pointwise_variance_max": pv_max,
                "pointwise_variance_min_max_ratio": pv_ratio,
            },
            "data_fingerprint": validated.data_fingerprint,
            "sha256": {
                f"{stem}.npy": _sha256(stage / f"{stem}.npy")
                for stem in (*INFERENCE_STEMS, "obs_points")
            },
        }
        (stage / "preparation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        outdir.mkdir(parents=True, exist_ok=True)
        for filename in _MANAGED_FILENAMES:
            os.replace(stage / filename, outdir / filename)

    return PreparedBundle(
        path=outdir,
        n=validated.n,
        p=validated.p,
        H=validated.H,
        T=int(len(validated.triangles)),
        L=validated.L,
        data_fingerprint=validated.data_fingerprint,
        mesh_type=selected_mesh_type,
    )


def _load_npy(path: Path, label: str) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load {label} array {path}: {exc}") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a validated SpaDAG inference bundle from X and spatial locations."
        )
    )
    parser.add_argument("--x", type=Path, required=True, help="X.npy with shape (n, p)")
    parser.add_argument(
        "--locations",
        "--obs-points",
        dest="locations",
        type=Path,
        required=True,
        help="observation coordinates .npy with shape (n, 2)",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mesh-type", choices=MESH_OPTIONS, default="polygon",
                        help="polygon (default) meshes the observation-supported domain in input coordinates; "
                             "regular/cdt build unit-square meshes over the bounding box "
                             "or --domain")
    parser.add_argument(
        "--domain", type=float, nargs=4, default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="regular/cdt: rectangle mapped to the unit square instead of the "
        "observation bounding box; 0 1 0 1 reproduces the simulation meshes",
    )
    parser.add_argument("--n-grid", type=int, default=5)
    parser.add_argument("--target-tris", type=int, default=20)
    parser.add_argument("--mesh-seed", type=int, default=0)
    parser.add_argument("--min-interior", type=int, default=4)
    parser.add_argument(
        "--n-boundary-per-edge",
        type=int,
        default=None,
        help="boundary samples per unit-square edge for cdt "
        "(default: round(sqrt(target_tris / 2)) + 1, the regular mesh with the same T)",
    )
    parser.add_argument("--min-angle", type=float, default=10.0)
    parser.add_argument("--no-area-constraint", action="store_true")
    parser.add_argument("--mesh-points", type=Path)
    parser.add_argument("--mesh-triangles", type=Path)
    parser.add_argument(
        "--a-kl",
        type=Path,
        help="optional existing A_kl.npy; skips BayesGPfit generation",
    )
    parser.add_argument(
        "--min-obs-per-triangle", type=int, default=None,
        help="polygon mesh: remove vertices around triangles holding fewer observations (default off)",
    )
    parser.add_argument(
        "--max-obs-per-triangle", type=int, default=None,
        help="polygon mesh: split triangles holding more observations by adding observation vertices (default off)",
    )
    parser.add_argument("--refine-rounds", type=int, default=3,
                        help="polygon mesh: alternating coarsen/split rounds when both bounds are set")
    parser.add_argument("--hull-scale", type=float, default=4.0,
                        help="polygon mesh: concave-hull scale in median nearest-neighbour spacings")
    parser.add_argument("--margin-scale", type=float, default=1.0,
                        help="polygon mesh: boundary margin in median nearest-neighbour spacings")
    parser.add_argument("--L", type=int, default=66)
    parser.add_argument("--a", type=float, default=0.01)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--max-range", type=float, default=1.0)
    parser.add_argument(
        "--poly-degree", type=int, default=10,
        help="Hermite polynomial degree of the BayesGPfit expansion (default 10, the full 66-mode "
        "expansion used by the manuscript); L may exceed the number of mesh vertices",
    )
    parser.add_argument("--rscript")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if (args.mesh_points is None) != (args.mesh_triangles is None):
        parser.error("--mesh-points and --mesh-triangles must be supplied together")
    if args.domain is not None:
        if args.mesh_points is not None or args.mesh_type == "polygon":
            parser.error("--domain requires --mesh-type regular or cdt")
        xmin, xmax, ymin, ymax = args.domain
        if not (xmin < xmax and ymin < ymax):
            parser.error("--domain must satisfy XMIN < XMAX and YMIN < YMAX")
    if args.n_grid < 1:
        parser.error("--n-grid must be at least 1")
    if args.target_tris < 1:
        parser.error("--target-tris must be positive")
    if args.L < 1:
        parser.error("--L must be positive")
    if args.poly_degree is not None and args.poly_degree < 0:
        parser.error("--poly-degree cannot be negative")
    for name in ("min_obs_per_triangle", "max_obs_per_triangle"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.refine_rounds < 1:
        parser.error("--refine-rounds must be positive")
    if (args.min_obs_per_triangle is not None and args.max_obs_per_triangle is not None
            and args.min_obs_per_triangle > args.max_obs_per_triangle):
        parser.error("--min-obs-per-triangle cannot exceed --max-obs-per-triangle")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    mesh_points = (
        _load_npy(args.mesh_points, "mesh_points")
        if args.mesh_points is not None
        else None
    )
    mesh_triangles = (
        _load_npy(args.mesh_triangles, "mesh_triangles")
        if args.mesh_triangles is not None
        else None
    )
    A_kl = _load_npy(args.a_kl, "A_kl") if args.a_kl is not None else None
    prepared = prepare_inference_bundle(
        _load_npy(args.x, "X"),
        _load_npy(args.locations, "obs_points"),
        args.outdir,
        mesh_type=args.mesh_type,
        domain=args.domain,
        n_grid=args.n_grid,
        target_tris=args.target_tris,
        seed=args.mesh_seed,
        min_interior=args.min_interior,
        n_boundary_per_edge=args.n_boundary_per_edge,
        min_angle=args.min_angle,
        use_area_constraint=not args.no_area_constraint,
        mesh_points=mesh_points,
        mesh_triangles=mesh_triangles,
        A_kl=A_kl,
        L=args.L,
        a=args.a,
        b=args.b,
        max_range=args.max_range,
        poly_degree=args.poly_degree,
        rscript=args.rscript,
        overwrite=args.overwrite,
        min_obs_per_triangle=args.min_obs_per_triangle,
        max_obs_per_triangle=args.max_obs_per_triangle,
        refine_rounds=args.refine_rounds,
        hull_scale=args.hull_scale,
        margin_scale=args.margin_scale,
    )
    summary = asdict(prepared)
    summary["path"] = str(prepared.path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
