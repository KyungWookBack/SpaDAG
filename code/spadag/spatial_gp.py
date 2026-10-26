"""Karhunen-Loeve (K-L) design for the inclusion fields.

The logit inclusion field of each directed pair (j, k) is approximated at the
mesh nodes by a truncated K-L expansion

    g_{jkh} = sum_{l=1}^L theta_{jkl} sqrt(lambda_l) phi_l(s_h) = A_h^T theta_{jk},
    A[h, l] = sqrt(lambda_l) * phi_l(s_h),   theta_{jkl} ~ N(0, tau^2),

of the modified squared-exponential (MSE) kernel of Kang, Reich & Staicu
(2018), Biometrika 105(1):165-184,

    kappa(v, v') = tau^2 exp{ -a(||v||^2 + ||v'||^2) - b ||v - v'||^2 },  a, b > 0,

whose eigenpairs are available in closed form.  They are computed by the R
package BayesGPfit through the bridge script ``r_bridges/kl_bayesgpfit.R``.
The K-L design A is shared by all directed pairs; the sampler reads it from the
prepared bundle and never evaluates the kernel.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np

from spadag.polya_gamma import _resolve_rscript

_HERE = Path(__file__).resolve().parent
_BRIDGE = _HERE / "r_bridges" / "kl_bayesgpfit.R"


@dataclass(frozen=True)
class KLDesign:
    """Finite-dimensional K-L representation evaluated at mesh nodes."""

    A: np.ndarray            # (H, L) design matrix, A[h, l] = sqrt(lambda_l) phi_l(s_h)
    eigvals: np.ndarray      # (L,) retained eigenvalues, descending
    L: int                   # truncation order
    a: float                 # MSE kernel decay-rate parameter
    b: float                 # MSE kernel smoothness parameter
    max_range: float         # GP.std.grids coordinate standardization range
    energy_kept: float       # fraction of finite-mode energy through poly_degree retained by L


def kl_pointwise_variance(A: np.ndarray) -> np.ndarray:
    """Prior variance ``v_h = sum_l A_hl^2`` of the K-L field at each node when ``theta ~ N(0, I_L)``."""
    design = np.asarray(A, dtype=float)
    return np.sum(design ** 2, axis=1)


def _min_poly_degree(L: int, d: int) -> int:
    """Smallest total-degree m such that C(m+d, d) >= L (number of 2D modes)."""
    m = 0
    while comb(m + d, d) < L:
        m += 1
    return m


def _run_bayesgpfit(
    points: np.ndarray, a: float, b: float, poly_degree: int, max_range: float,
    rscript: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Call the BayesGPfit R bridge; return (phi (H, M), lambda (M,))."""
    exe = rscript or _resolve_rscript()
    if not _BRIDGE.exists():
        raise FileNotFoundError(f"R bridge not found: {_BRIDGE}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pts_csv, phi_csv, lam_csv = td / "pts.csv", td / "phi.csv", td / "lam.csv"
        np.savetxt(pts_csv, np.asarray(points, dtype=float), delimiter=",")
        cmd = [exe, str(_BRIDGE), str(pts_csv), repr(float(a)), repr(float(b)),
               str(int(poly_degree)), repr(float(max_range)), str(phi_csv), str(lam_csv)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"BayesGPfit bridge failed (rc={proc.returncode}).\n"
                f"cmd: {' '.join(cmd)}\nstderr:\n{proc.stderr}"
            )
        phi = np.loadtxt(phi_csv, delimiter=",", ndmin=2)
        lam = np.loadtxt(lam_csv, delimiter=",", ndmin=1)
    return phi, lam


def build_kl_design(
    mesh_points: np.ndarray,
    *,
    L: int = 15,
    a: float = 0.01,
    b: float = 1.0,
    max_range: float = 2.0,
    poly_degree: int | None = None,
    rscript: str | None = None,
) -> KLDesign:
    """Build the K-L design matrix A (H x L) from the MSE kernel via BayesGPfit.

    Parameters
    ----------
    mesh_points : (H, d) array
        Mesh node coordinates s_h (standardized internally by GP.std.grids).
    L : int
        Number of retained K-L terms (truncation order). Clamped to <= M, the
        number of closed-form modes for the chosen poly_degree.
    a, b : float
        MSE kernel parameters. `a` = variance decay rate (a=0 recovers the SE
        kernel), `b` = smoothness. Both feed BayesGPfit's GP.eigen.* routines.
    max_range : float
        Coordinate standardization span passed to GP.std.grids.
    poly_degree : int, optional
        Largest total Hermite degree m. If None, the smallest m giving at least
        L modes is used, raised by one if it gives exactly L modes so that
        ``energy_kept`` includes a discarded mode.
    rscript : str, optional
        Path to the Rscript executable (defaults to PATH lookup).

    Notes
    -----
    ``energy_kept`` is the fraction among the finite set of modes computed
    through ``poly_degree`` for truncation diagnostics, not the fraction of
    the full kernel trace.
    """
    pts = np.asarray(mesh_points, dtype=float)
    H, d = pts.shape

    if poly_degree is None:
        poly_degree = _min_poly_degree(L, d)
        if comb(poly_degree + d, d) == L:
            poly_degree += 1

    phi, lam = _run_bayesgpfit(pts, a, b, poly_degree, max_range, rscript)
    if phi.shape[0] != H:
        raise RuntimeError(f"bridge returned phi with {phi.shape[0]} rows, expected {H}")

    order = np.argsort(lam)[::-1]
    lam_sorted = np.clip(lam[order], 0.0, None)
    phi_sorted = phi[:, order]

    L_eff = int(min(L, lam_sorted.size))
    lam_keep = lam_sorted[:L_eff]
    phi_keep = phi_sorted[:, :L_eff]

    A = phi_keep * np.sqrt(lam_keep)[None, :]
    total = float(lam_sorted.sum())
    energy = float(lam_keep.sum() / total) if total > 0 else 1.0

    return KLDesign(
        A=A,
        eigvals=lam_keep,
        L=L_eff,
        a=float(a),
        b=float(b),
        max_range=float(max_range),
        energy_kept=energy,
    )
