"""Sampler state and the conjugate Gibbs blocks of a SpaDAG sweep.

The Pólya-Gamma inclusion block is in ``shared_gp_pg.py`` and the sweep
schedule in ``cli.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from spadag.acyclic import creates_cycle, node_triangles, triangle_edge_tensor
from spadag.mcmc_data import InferenceData


@dataclass
class SamplerConfig:
    v0: float = 1e-3              # spike (inactive) variance
    a_b: float = 1.0              # slab IG shape
    b_b: float = 1.0              # slab IG scale
    sigma_b2_init: float = 1.0    # initial slab variance
    a_sig: float = 0.01           # noise IG shape
    b_sig: float = 0.01           # noise IG scale
    a_alpha: float = 1.0          # tau_alpha Gamma shape
    b_alpha: float = 1.0          # tau_alpha Gamma rate
    sigma_a0_2: float = 100.0     # intercept prior variance

    def validate(self) -> None:
        if not np.isfinite(self.v0) or self.v0 <= 0.0:
            raise ValueError("v0 must be finite and positive")
        if not np.isfinite(self.a_b) or self.a_b <= 0.0:
            raise ValueError("a_b must be finite and positive")
        if not np.isfinite(self.b_b) or self.b_b <= 0.0:
            raise ValueError("b_b must be finite and positive")
        if not np.isfinite(self.sigma_b2_init) or self.sigma_b2_init <= 0.0:
            raise ValueError("sigma_b2_init must be finite and positive")


def spike_slab_variances(gamma: np.ndarray, sigma_b2: float,
                         cfg: SamplerConfig) -> np.ndarray:
    """Return the prior variance of each coefficient: slab if active, else spike."""
    return np.where(np.asarray(gamma) == 1, sigma_b2, cfg.v0)


@dataclass
class SamplerState:
    gamma: np.ndarray             # (p, p, H) int
    b: np.ndarray                 # (p, p, H)
    pi: np.ndarray                # (p, p, H)
    sigma_b2: float
    a0: np.ndarray                # (p,)
    a_spatial: np.ndarray         # (p, H)
    tau_alpha: np.ndarray         # (p,)
    sigma2: np.ndarray            # (p,) noise variances
    theta: np.ndarray             # (p, p, L) K-L coefficients
    m: np.ndarray                 # (p, p, H) working inclusion-prob mean
    # node -> incident triangles, used by the local-DAG checks
    _node_tris: list = field(default=None, repr=False)


def init_state(
    data: InferenceData,
    cfg: SamplerConfig,
    rng: np.random.Generator,
    gamma_init_prob: float | None = None,
) -> SamplerState:
    """Initialize the sampler state.

    Indicators are Bernoulli(``gamma_init_prob``, default 0.5) subject to the
    local-DAG constraint, and coefficients are drawn from their prior.
    Inclusion probabilities start at 0.5 until the inclusion fields are set.
    """
    cfg.validate()
    if gamma_init_prob is not None and (
        not np.isfinite(gamma_init_prob)
        or gamma_init_prob < 0.0
        or gamma_init_prob > 1.0
    ):
        raise ValueError("gamma_init_prob must lie in [0, 1]")
    p, H, L = data.p, data.H, data.L
    gamma = np.zeros((p, p, H), dtype=int)
    b = np.zeros((p, p, H), dtype=float)
    pi = np.full((p, p, H), 0.5, dtype=float)
    m = np.zeros((p, p, H), dtype=float)
    theta = np.zeros((p, p, L), dtype=float)
    node_tris = node_triangles(data.triangles, H)
    edge = np.zeros((data.triangles.shape[0], p, p), dtype=np.int8)
    sigma_b2_init = float(cfg.sigma_b2_init)
    initial_probability = 0.5 if gamma_init_prob is None else gamma_init_prob
    for (j, k) in data.pairs:
        m[j, k] = 0.5
        pi[j, k] = 0.5
        for hi in range(H):
            if rng.random() >= initial_probability:
                continue
            if any(creates_cycle(edge[K], j, k) for K in node_tris[hi]):
                continue
            gamma[j, k, hi] = 1
            for K in node_tris[hi]:
                edge[K, j, k] = 1
        variances = spike_slab_variances(gamma[j, k], sigma_b2_init, cfg)
        b[j, k] = rng.normal(0.0, np.sqrt(variances), size=H)
    st = SamplerState(
        gamma=gamma, b=b, pi=pi,
        sigma_b2=sigma_b2_init,
        a0=np.zeros(p), a_spatial=np.zeros((p, H)),
        tau_alpha=np.ones(p), sigma2=data.X.var(axis=0).astype(float),
        theta=theta, m=m,
    )
    st._node_tris = node_tris
    return st


def _alpha_at_obs(data: InferenceData, st: SamplerState, j: int) -> np.ndarray:
    return st.a0[j] + data.Phi @ st.a_spatial[j]


def _parent_pred(data: InferenceData, st: SamplerState, j: int, exclude_k: int | None = None) -> np.ndarray:
    """Sum_{k in pa(j), k != exclude_k} Zk[k] @ b[j,k]."""
    pred = np.zeros(data.n)
    for k in np.where(data.parents[j] == 1)[0]:
        if k == exclude_k:
            continue
        pred += data.Zk[k] @ st.b[j, k]
    return pred


def _sample_gaussian_precision(Lambda: np.ndarray, rhs: np.ndarray,
                               rng: np.random.Generator) -> np.ndarray:
    """Draw x ~ N(Lambda^{-1} rhs, Lambda^{-1}) via Cholesky of the precision."""
    # Inputs come from validated sampler state; finiteness checks would
    # dominate the cost of these small per-pair solves.
    c, low = cho_factor(Lambda, lower=True, check_finite=False)
    mean = cho_solve((c, low), rhs, check_finite=False)
    L = c if low else c.T                       # lower Cholesky factor of Lambda
    z = rng.standard_normal(Lambda.shape[0])
    # Cov = Lambda^{-1} = L^{-T} L^{-1}; sample = mean + L^{-T} z
    u = solve_triangular(
        L, z, lower=True, trans="T", check_finite=False
    )
    return mean + u


def update_gamma(data: InferenceData, st: SamplerState, cfg: SamplerConfig,
                 rng: np.random.Generator) -> None:
    """Draw each nodal indicator; a switch-on that would create a cycle in an
    incident triangle graph is rejected."""
    log2pi = np.log(2.0 * np.pi)
    sb2 = st.sigma_b2
    spike_var = cfg.v0
    if getattr(st, "_node_tris", None) is None:
        st._node_tris = node_triangles(data.triangles, data.H)
    edge = triangle_edge_tensor(st.gamma, data.triangles)
    for (j, k) in data.pairs:
        bvec = st.b[j, k]
        pivec = np.clip(st.pi[j, k], 1e-12, 1 - 1e-12)
        logN1 = -0.5 * (log2pi + np.log(sb2)) - bvec**2 / (2.0 * sb2)
        logN0 = -0.5 * (log2pi + np.log(spike_var)) - bvec**2 / (2.0 * spike_var)
        logq1 = np.log(pivec) + logN1
        logq0 = np.log(1.0 - pivec) + logN0
        prob1 = 1.0 / (1.0 + np.exp(logq0 - logq1))
        draw = (rng.random(data.H) < prob1).astype(int)
        for hi, new_val in enumerate(draw):
            old_val = st.gamma[j, k, hi]
            if new_val == old_val:
                continue

            tris_hi = st._node_tris[hi]
            if new_val == 1:
                if any(creates_cycle(edge[K], j, k) for K in tris_hi):
                    continue
                st.gamma[j, k, hi] = 1
                for K in tris_hi:
                    edge[K, j, k] = 1
            else:
                st.gamma[j, k, hi] = 0
                for K in tris_hi:
                    verts = data.triangles[K]
                    edge[K, j, k] = int(np.any(st.gamma[j, k, verts] == 1))


def update_b(data: InferenceData, st: SamplerState, cfg: SamplerConfig,
             rng: np.random.Generator) -> None:
    """Draw the coefficient surface of each candidate parent, one pair at a time."""
    for j in range(data.p):
        alpha_j = _alpha_at_obs(data, st, j)
        s2 = st.sigma2[j]
        parents = np.where(data.parents[j] == 1)[0]
        if parents.size == 0:
            continue

        # Keep the running total of parent predictions so each leave-one-out
        # residual costs O(n H) rather than O(n H p).
        pair_pred = [data.Zk[k] @ st.b[j, k] for k in parents]
        total_pred = np.zeros(data.n)
        for prediction in pair_pred:
            total_pred += prediction

        base = data.X[:, j] - alpha_j
        for index, k in enumerate(parents):
            old_prediction = pair_pred[index]
            r = base - total_pred + old_prediction
            v = spike_slab_variances(st.gamma[j, k], st.sigma_b2, cfg)
            Lambda = data.ZtZ[k] / s2 + np.diag(1.0 / v)
            rhs = (data.Zk[k].T @ r) / s2
            st.b[j, k] = _sample_gaussian_precision(Lambda, rhs, rng)
            new_prediction = data.Zk[k] @ st.b[j, k]
            total_pred += new_prediction - old_prediction
            pair_pred[index] = new_prediction


def update_sigma_b2(data: InferenceData, st: SamplerState, cfg: SamplerConfig,
                    rng: np.random.Generator) -> None:
    """Draw the slab variance from its inverse-Gamma conditional given active coefficients."""
    weighted_ssq = 0.0
    n_coeff = 0
    for (j, k) in data.pairs:
        active = st.gamma[j, k] == 1
        weighted_ssq += float(np.sum(st.b[j, k, active] ** 2))
        n_coeff += int(np.sum(active))
    shape = cfg.a_b + 0.5 * n_coeff
    scale = cfg.b_b + 0.5 * weighted_ssq
    st.sigma_b2 = float(1.0 / rng.gamma(shape=shape, scale=1.0 / scale))


def update_alpha(data: InferenceData, st: SamplerState, cfg: SamplerConfig,
                 rng: np.random.Generator) -> None:
    """Draw each baseline (intercept and centered surface) and its precision."""
    if data.baseline_basis is None:
        raise ValueError("InferenceData is missing the centered-baseline basis")
    baseline_basis = np.asarray(data.baseline_basis, dtype=float)
    if baseline_basis.shape != (data.H, data.H - 1):
        raise ValueError("centered-baseline basis has an invalid shape")
    reduced_dimension = data.H - 1
    prior_prec = np.empty(data.H)
    prior_prec[0] = 1.0 / cfg.sigma_a0_2
    for j in range(data.p):
        r = data.X[:, j] - _parent_pred(data, st, j)
        prior_prec[1:] = st.tau_alpha[j]
        s2 = st.sigma2[j]
        Lambda = data.UtU / s2 + np.diag(prior_prec)
        rhs = (data.U.T @ r) / s2
        c = _sample_gaussian_precision(Lambda, rhs, rng)
        st.a0[j] = c[0]
        st.a_spatial[j] = baseline_basis @ c[1:]
        shape = cfg.a_alpha + 0.5 * reduced_dimension
        rate = cfg.b_alpha + 0.5 * float(np.sum(st.a_spatial[j] ** 2))
        st.tau_alpha[j] = float(rng.gamma(shape=shape, scale=1.0 / rate))


def update_sigma2(data: InferenceData, st: SamplerState, cfg: SamplerConfig,
                  rng: np.random.Generator) -> None:
    """Draw each noise variance from its inverse-Gamma conditional."""
    log_likelihood = 0.0
    rss = np.empty(data.p, dtype=float)
    for j in range(data.p):
        e = data.X[:, j] - _alpha_at_obs(data, st, j) - _parent_pred(data, st, j)
        shape = cfg.a_sig + 0.5 * data.n
        scale = cfg.b_sig + 0.5 * float(np.sum(e**2))
        st.sigma2[j] = float(1.0 / rng.gamma(shape=shape, scale=1.0 / scale))
        variance = st.sigma2[j]
        rss[j] = float(e @ e)
        log_likelihood += -0.5 * (
            data.n * np.log(2.0 * np.pi * variance)
            + float(e @ e) / variance
        )
    # Cached for the trace writers, which would otherwise recompute residuals.
    st._log_likelihood = float(log_likelihood)
    st._rss = rss
