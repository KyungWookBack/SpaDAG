"""Load SpaDAG inference inputs, optional simulation truth, and caches."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np

from spadag.p1_geometry import build_baseline_geometry


INFERENCE_SCHEMA_VERSION = 1
INFERENCE_STEMS = (
    "X", "Phi_obs", "mesh_points", "mesh_triangles", "A_kl",
)
TRUTH_STEMS = (
    "q_node", "gamma", "b_coeffs", "beta_obs", "edge_triangle",
    "sigma_noise", "hyper_mu",
)
OPTIONAL_TRUTH_STEMS = ("hyper_sigma_b", "hyper_tau2")


@dataclass
class InferenceData:
    """Observed and fixed inputs used by the SpaDAG sampler."""

    n: int
    p: int
    H: int
    L: int
    X: np.ndarray
    Phi: np.ndarray
    mesh_points: np.ndarray
    triangles: np.ndarray
    A_kl: np.ndarray
    # Candidate pair (j, k) means the directed edge k -> j.
    pairs: List[Tuple[int, int]]
    parents: np.ndarray
    # Cached designs used repeatedly during each sweep.
    Zk: np.ndarray
    ZtZ: np.ndarray
    U: np.ndarray
    UtU: np.ndarray
    baseline_weights: np.ndarray | None = None
    baseline_basis: np.ndarray | None = None
    data_fingerprint: str = ""


@dataclass
class SimulationTruth:
    """Ground truth available only for simulation evaluation."""

    q: np.ndarray
    gamma: np.ndarray
    b: np.ndarray
    beta: np.ndarray
    edge_triangle: np.ndarray
    sigma_noise: np.ndarray
    # sigma_b and tau2 are None unless the truth was drawn from the prior.
    sigma_b: float | None
    mu: float
    tau2: float | None


def _load_array(npy_dir: Path, stem: str) -> np.ndarray:
    path = npy_dir / f"{stem}.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"required SpaDAG schema-v{INFERENCE_SCHEMA_VERSION} array "
            f"is missing: {path}"
        )
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load bundle array {path}: {exc}") from exc


def _float_array(
    array: np.ndarray,
    stem: str,
    ndim: int,
    *,
    allow_nan: bool = False,
) -> np.ndarray:
    if array.ndim != ndim:
        raise ValueError(f"{stem}.npy must be {ndim}-dimensional, got {array.ndim}")
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{stem}.npy must have a real numeric dtype, got {array.dtype}")
    converted = array.astype(float)
    valid = ~np.isinf(converted) if allow_nan else np.isfinite(converted)
    if not np.all(valid):
        raise ValueError(f"{stem}.npy must contain only finite values")
    return converted


def _integer_array(array: np.ndarray, stem: str, ndim: int) -> np.ndarray:
    if array.ndim != ndim:
        raise ValueError(f"{stem}.npy must be {ndim}-dimensional, got {array.ndim}")
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{stem}.npy must contain finite integer values")
    converted = array.astype(np.int64)
    if not np.array_equal(array, converted):
        raise ValueError(f"{stem}.npy must contain integer values")
    return converted


def _validate_triangles(
    mesh_points: np.ndarray,
    triangles: np.ndarray,
) -> None:
    H = len(mesh_points)
    if triangles.shape[1:] != (3,):
        raise ValueError(
            f"mesh_triangles.npy must have shape (T, 3), got {triangles.shape}"
        )
    if len(triangles) == 0:
        raise ValueError("mesh_triangles.npy must contain at least one triangle")
    if np.any(triangles < 0) or np.any(triangles >= H):
        raise ValueError(
            f"mesh_triangles.npy indices must be in [0, {H}), got range "
            f"[{int(triangles.min())}, {int(triangles.max())}]"
        )
    if np.any(np.apply_along_axis(lambda row: len(set(row)) < 3, 1, triangles)):
        raise ValueError("mesh_triangles.npy contains a repeated-vertex triangle")
    vertices = mesh_points[triangles]
    twice_area = (
        (vertices[:, 1, 0] - vertices[:, 0, 0])
        * (vertices[:, 2, 1] - vertices[:, 0, 1])
        - (vertices[:, 1, 1] - vertices[:, 0, 1])
        * (vertices[:, 2, 0] - vertices[:, 0, 0])
    )
    scale = max(1.0, float(np.max(np.abs(mesh_points))))
    if np.any(np.abs(twice_area) <= np.finfo(float).eps * scale * scale * 32):
        raise ValueError("mesh_triangles.npy contains a degenerate triangle")


def _data_fingerprint(arrays: tuple[tuple[str, np.ndarray], ...]) -> str:
    """Hash validated inference values independently of their directory path."""
    digest = hashlib.sha256(f"SpaDAG-schema-v{INFERENCE_SCHEMA_VERSION}".encode())
    for stem, array in arrays:
        canonical = np.ascontiguousarray(array)
        digest.update(stem.encode())
        digest.update(canonical.dtype.str.encode())
        digest.update(repr(canonical.shape).encode())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def load_inference_data(npy_dir: Path) -> InferenceData:
    """Load and validate the arrays needed for inference and build the design caches.

    Every ordered pair (j, k), j != k, is a candidate edge k -> j.
    """
    npy_dir = Path(npy_dir)
    X = _float_array(_load_array(npy_dir, "X"), "X", 2)
    Phi = _float_array(_load_array(npy_dir, "Phi_obs"), "Phi_obs", 2)
    mesh_points = _float_array(
        _load_array(npy_dir, "mesh_points"), "mesh_points", 2
    )
    triangles = _integer_array(
        _load_array(npy_dir, "mesh_triangles"), "mesh_triangles", 2
    )
    A_kl = _float_array(_load_array(npy_dir, "A_kl"), "A_kl", 2)

    n, p = X.shape
    H = Phi.shape[1]
    L = A_kl.shape[1]
    if n == 0:
        raise ValueError(f"X.npy must have at least one row, got {X.shape}")
    if p < 2:
        raise ValueError(
            f"X.npy must contain at least two variables (p >= 2), got {X.shape}"
        )
    if Phi.shape != (n, H) or H == 0:
        raise ValueError(
            f"Phi_obs.npy must have shape ({n}, H) with H > 0, got {Phi.shape}"
        )
    if mesh_points.shape != (H, 2):
        raise ValueError(
            f"mesh_points.npy must have shape ({H}, 2), got {mesh_points.shape}"
        )
    if A_kl.shape[0] != H or L == 0:
        raise ValueError(
            f"A_kl.npy must have shape ({H}, L) with L > 0, got {A_kl.shape}"
        )
    if np.any(Phi < -1e-12):
        raise ValueError("Phi_obs.npy P1 basis rows cannot contain negative weights")
    if not np.allclose(Phi.sum(axis=1), 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("Phi_obs.npy P1 basis rows must sum to one")
    _validate_triangles(mesh_points, triangles)

    parents = np.ones((p, p), dtype=int)
    np.fill_diagonal(parents, 0)
    pairs = [
        (j, k)
        for j in range(p)
        for k in np.where(parents[j] == 1)[0]
    ]

    # Per-parent design: Zk[k][:, h] = Phi[:, h] * X[:, k].
    Zk = np.empty((p, n, H), dtype=float)
    for k in range(p):
        Zk[k] = Phi * X[:, k][:, None]
    ZtZ = np.einsum("pnh,png->phg", Zk, Zk)

    baseline = build_baseline_geometry(Phi, mesh_points, triangles)
    U = baseline.design
    UtU = baseline.crossproduct

    return InferenceData(
        n=n,
        p=p,
        H=H,
        L=L,
        X=X,
        Phi=Phi,
        mesh_points=mesh_points,
        triangles=triangles,
        A_kl=A_kl,
        pairs=pairs,
        parents=parents,
        Zk=Zk,
        ZtZ=ZtZ,
        U=U,
        UtU=UtU,
        baseline_weights=baseline.integration_weights,
        baseline_basis=baseline.spatial_basis,
        data_fingerprint=_data_fingerprint((
            ("X", X),
            ("Phi_obs", Phi),
            ("mesh_points", mesh_points),
            ("mesh_triangles", triangles),
            ("A_kl", A_kl),
        )),
    )


def load_triangle_edge_truth(
    npy_dir: Path,
    data: InferenceData,
) -> np.ndarray:
    """Load only the triangle-level graph truth required for recovery."""
    npy_dir = Path(npy_dir)
    edge_triangle = _integer_array(
        _load_array(npy_dir, "edge_triangle"), "edge_triangle", 3
    )
    expected = (len(data.triangles), data.p, data.p)
    if edge_triangle.shape != expected:
        raise ValueError(
            f"edge_triangle.npy must have shape {expected}, "
            f"got {edge_triangle.shape}"
        )
    if np.any((edge_triangle != 0) & (edge_triangle != 1)):
        raise ValueError("edge_triangle.npy must be binary")
    return edge_triangle


def load_simulation_truth(
    npy_dir: Path,
    data: InferenceData | None = None,
) -> SimulationTruth:
    """Load simulation truth, including prior scales when the DGP defines them."""
    npy_dir = Path(npy_dir)
    q = _float_array(
        _load_array(npy_dir, "q_node"), "q_node", 3, allow_nan=True
    )
    gamma = _integer_array(_load_array(npy_dir, "gamma"), "gamma", 3)
    b = _float_array(_load_array(npy_dir, "b_coeffs"), "b_coeffs", 3)
    beta = _float_array(_load_array(npy_dir, "beta_obs"), "beta_obs", 3)
    edge_triangle = _integer_array(
        _load_array(npy_dir, "edge_triangle"), "edge_triangle", 3
    )
    sigma_noise = _float_array(
        _load_array(npy_dir, "sigma_noise"), "sigma_noise", 1
    )

    def scalar(stem: str) -> float:
        value = _float_array(_load_array(npy_dir, stem), stem, 1)
        if value.shape != (1,):
            raise ValueError(f"{stem}.npy must have shape (1,), got {value.shape}")
        return float(value[0])

    def optional_scalar(stem: str) -> float | None:
        if not (npy_dir / f"{stem}.npy").is_file():
            return None
        return scalar(stem)

    truth = SimulationTruth(
        q=q,
        gamma=gamma,
        b=b,
        beta=beta,
        edge_triangle=edge_triangle,
        sigma_noise=sigma_noise,
        sigma_b=optional_scalar("hyper_sigma_b"),
        mu=scalar("hyper_mu"),
        tau2=optional_scalar("hyper_tau2"),
    )
    if q.shape[0] != q.shape[1] or q.shape[2] == 0:
        raise ValueError(f"q_node.npy must have shape (p, p, H), got {q.shape}")
    p, _, H = q.shape
    expected_node = (p, p, H)
    for stem, array in (("gamma", gamma), ("b_coeffs", b)):
        if array.shape != expected_node:
            raise ValueError(
                f"{stem}.npy must have shape {expected_node}, got {array.shape}"
            )
    if beta.shape[1:] != (p, p):
        raise ValueError(
            f"beta_obs.npy must have trailing shape ({p}, {p}), got {beta.shape}"
        )
    if edge_triangle.shape[1:] != (p, p):
        raise ValueError(
            "edge_triangle.npy must have trailing shape "
            f"({p}, {p}), got {edge_triangle.shape}"
        )
    if sigma_noise.shape != (p,):
        raise ValueError(
            f"sigma_noise.npy must have shape ({p},), got {sigma_noise.shape}"
        )
    if data is not None:
        expected_node = (data.p, data.p, data.H)
        expected_beta = (data.n, data.p, data.p)
        expected_triangle = (len(data.triangles), data.p, data.p)
        for stem, array in (("q_node", q), ("gamma", gamma), ("b_coeffs", b)):
            if array.shape != expected_node:
                raise ValueError(
                    f"{stem}.npy must have shape {expected_node}, got {array.shape}"
                )
        if beta.shape != expected_beta:
            raise ValueError(
                f"beta_obs.npy must have shape {expected_beta}, got {beta.shape}"
            )
        if edge_triangle.shape != expected_triangle:
            raise ValueError(
                "edge_triangle.npy must have shape "
                f"{expected_triangle}, got {edge_triangle.shape}"
            )
        if sigma_noise.shape != (data.p,):
            raise ValueError(
                f"sigma_noise.npy must have shape ({data.p},), got {sigma_noise.shape}"
            )
    if np.any((gamma != 0) & (gamma != 1)):
        raise ValueError("gamma.npy must be binary")
    if np.any((edge_triangle != 0) & (edge_triangle != 1)):
        raise ValueError("edge_triangle.npy must be binary")
    if data is not None:
        candidate_q = np.stack([q[j, k] for j, k in data.pairs])
    else:
        candidate_q = q[~np.eye(q.shape[0], dtype=bool)]
    if not np.all(np.isfinite(candidate_q)):
        raise ValueError("q_node.npy candidate-edge values must be finite")
    if np.any((candidate_q < 0.0) | (candidate_q > 1.0)):
        raise ValueError("q_node.npy candidate-edge values must lie in [0, 1]")
    optional_scales = (truth.sigma_b, truth.tau2)
    if np.any(sigma_noise <= 0.0) or any(
        scale is not None and scale <= 0.0 for scale in optional_scales
    ):
        raise ValueError(
            "simulation noise and provided variance scales must be positive"
        )
    return truth
