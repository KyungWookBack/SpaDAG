"""The ``spadag`` command: run one SpaDAG chain on an inference bundle.

Each sweep updates, in order, the nodal indicators, the regression
coefficients, the slab variance, the inclusion fields (``shared_gp_pg.py``),
the baselines and noise variances, the two inclusion-field variances, and
finally attempts ``m_rev`` component reversals.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict

import numpy as np

from spadag.acyclic import triangle_edge_tensor
from spadag.mcmc_data import InferenceData, load_inference_data
from spadag.mcmc_steps import (
    SamplerConfig,
    init_state,
    update_alpha,
    update_b,
    update_gamma,
    update_sigma2,
    update_sigma_b2,
)
from spadag.polya_gamma import BayesLogitRPGSampler
from spadag.shared_gp_pg import (
    SharedGPState,
    build_gp_decomposition,
    constrained_edge_raw_variance,
    draw_common_gp_variance,
    draw_edge_gp_variance,
    init_shared_gp_state,
    update_shared_gp_pg,
)

DEFAULT_V0 = 1e-3
# sigma0^2 ~ IG(2, 1) has mean one with the smallest integer shape that keeps
# the mean finite, so the d0 / 2 data term dominates the conditional shape.
DEFAULT_A_SIGMA0 = 2.0
DEFAULT_B_SIGMA0 = 1.0
DEFAULT_M_REV = 50
INITIALIZATIONS = ("prior-random", "empty")
_CHECKPOINT_VERSION = 6
_CHECKPOINT_REQUIRED_KEYS = {
    "version", "signature", "completed_iterations", "elapsed_seconds",
    "numpy_rng_state", "omega_rng_state", "sampler_state", "sampler_config",
    "shared_gp_state", "edge_gp_var", "common_gp_var", "accumulators",
}


def _apply_initialization(st, kind: str) -> None:
    """Apply the requested structural initialization."""
    if kind == "prior-random":
        return
    if kind == "empty":
        st.gamma.fill(0)
        st.b.fill(0.0)
        return
    raise ValueError(f"unknown initialization: {kind}")


class CheckpointStop(RuntimeError):
    """Signal an intentional stop immediately after a checkpoint."""


def _resolved_m_rev(args: argparse.Namespace) -> int:
    """Use DEFAULT_M_REV unless the caller explicitly selects a count."""
    return DEFAULT_M_REV if args.m_rev is None else args.m_rev


def _checkpoint_signature(data: InferenceData, args: argparse.Namespace) -> dict:
    """Return settings that must agree for an exact continuation."""
    names = (
        "iters", "burn", "thin", "field_thin", "gamma_trace_thin", "seed",
        "initialization", "mu_init", "mu_prior_mean", "mu_prior_sd",
        "common_gp_var", "a_sigma0", "b_sigma0", "edge_gp_var", "a_tau",
        "b_tau", "v0", "a_b", "b_b", "sigma_b2_init",
    )
    settings = {name: getattr(args, name) for name in names}
    settings["m_rev"] = _resolved_m_rev(args)
    return {
        "data_fingerprint": data.data_fingerprint,
        "dimensions": (data.n, data.p, data.H, data.L, len(data.pairs)),
        "settings": settings,
    }


def _continuation_signature(signature: dict) -> dict:
    """Select model settings that must agree for a state-only continuation."""
    run_control = {"iters", "burn", "thin", "field_thin", "gamma_trace_thin"}
    return {
        "data_fingerprint": signature["data_fingerprint"],
        "dimensions": signature["dimensions"],
        "settings": {
            name: value
            for name, value in signature["settings"].items()
            if name not in run_control
        },
    }


def _write_checkpoint(path: Path, payload: dict) -> None:
    """Atomically replace one local checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _read_checkpoint(path: Path) -> dict:
    """Read a trusted local checkpoint and validate its top-level envelope."""
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except (EOFError, pickle.UnpicklingError, AttributeError, ImportError, IndexError) as exc:
        raise ValueError(f"invalid or corrupted checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint {path}: expected a dictionary payload")
    version = payload.get("version")
    if version != _CHECKPOINT_VERSION:
        raise ValueError(
            "unsupported checkpoint version: "
            f"{version!r}; expected {_CHECKPOINT_VERSION}"
        )
    missing = sorted(_CHECKPOINT_REQUIRED_KEYS - payload.keys())
    if missing:
        raise ValueError(
            f"incomplete checkpoint {path}: missing keys {', '.join(missing)}"
        )
    if not isinstance(payload["signature"], dict):
        raise ValueError(f"invalid checkpoint {path}: signature must be a dictionary")
    if not isinstance(payload["accumulators"], dict):
        raise ValueError(f"invalid checkpoint {path}: accumulators must be a dictionary")
    return payload


def _ess_1d(values: np.ndarray) -> float:
    """Geyer initial-positive-sequence ESS for one retained scalar trace."""
    x = np.asarray(values, dtype=float)
    n = x.size
    centered = x - np.mean(x)
    if n < 4 or float(centered @ centered) <= 1e-14:
        return float("nan")
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(
        spectrum * np.conjugate(spectrum), n=fft_size
    )[:n]
    autocovariance /= n
    if autocovariance[0] <= 0.0:
        return float("nan")
    autocorrelation = autocovariance / autocovariance[0]
    pair_sums = autocorrelation[1:-1:2] + autocorrelation[2::2]
    nonpositive = np.flatnonzero(pair_sums <= 0.0)
    positive = pair_sums[:nonpositive[0]] if nonpositive.size else pair_sums
    tau = max(1.0, 1.0 + 2.0 * float(np.sum(positive)))
    return float(min(n, n / tau))


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return float("nan")
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _eta_tensor(
    data: InferenceData,
    st,
    shared: SharedGPState,
) -> np.ndarray:
    eta = np.zeros((data.p, data.p, data.H), dtype=float)
    for j, k in data.pairs:
        eta[j, k] = (
            shared.mu + shared.f0 + data.A_kl @ st.theta[j, k]
        )
    return eta


def _mean_edge_deviation(
    data: InferenceData,
    st,
) -> np.ndarray:
    if not data.pairs:
        return np.zeros(data.H, dtype=float)
    total = np.zeros(data.H, dtype=float)
    for j, k in data.pairs:
        total += data.A_kl @ st.theta[j, k]
    return total / len(data.pairs)


def _candidate_gamma_vector(data: InferenceData, st) -> np.ndarray:
    """Return candidate-pair gamma values in the fixed ``data.pairs`` order."""
    if not data.pairs:
        return np.empty(0, dtype=np.uint8)
    return np.concatenate([
        np.asarray(st.gamma[j, k], dtype=np.uint8)
        for j, k in data.pairs
    ])


def sweep_shared_gp_pg(
    data: InferenceData,
    st,
    shared: SharedGPState,
    cfg: SamplerConfig,
    rng: np.random.Generator,
    draw_omega,
    *,
    common_gp_var: float,
    edge_gp_var: float,
    a_tau: float,
    b_tau: float,
    a_sigma0: float,
    b_sigma0: float,
    mu_prior_mean: float,
    mu_prior_sd: float,
    m_rev: int = 0,
    component_mesh=None,
):
    """Run one sweep; return the PG statistics and the new tau^2 and sigma0^2."""
    update_gamma(data, st, cfg, rng)
    update_b(data, st, cfg, rng)
    update_sigma_b2(data, st, cfg, rng)
    pg_stats = update_shared_gp_pg(
        data,
        st,
        shared,
        rng,
        draw_omega=draw_omega,
        common_gp_var=common_gp_var,
        edge_gp_var=edge_gp_var,
        mu_prior_mean=mu_prior_mean,
        mu_prior_sd=mu_prior_sd,
    )
    update_alpha(data, st, cfg, rng)
    update_sigma2(data, st, cfg, rng)
    edge_gp_var = draw_edge_gp_variance(
        data, st, rng, prior_shape=a_tau, prior_scale=b_tau,
    )
    common_gp_var = draw_common_gp_variance(
        shared, rng, prior_shape=a_sigma0, prior_scale=b_sigma0,
    )
    if m_rev:
        from spadag.component_reversal import MeshComponents, attempt_component_reversal
        from spadag.mcmc_steps import _alpha_at_obs, _parent_pred
        mesh = component_mesh if component_mesh is not None else MeshComponents(data)
        st._component_events = [
            attempt_component_reversal(data, st, cfg, rng, mesh) for _ in range(m_rev)
        ]
        st._component_accepted = getattr(st, "_component_accepted", 0) + sum(
            event["accepted"] for event in st._component_events)
        # The reversal redraws coefficients, so refresh the cached residuals
        # and likelihood; no random numbers are consumed.
        st._rss = np.array([
            np.sum((data.X[:, j] - _alpha_at_obs(data, st, j)
                    - _parent_pred(data, st, j))**2) for j in range(data.p)
        ])
        st._log_likelihood = float(-0.5 * np.sum(
            data.n * np.log(2 * np.pi * st.sigma2) + st._rss / st.sigma2))
    return pg_stats, edge_gp_var, common_gp_var


def run_shared_gp_pg(
    data: InferenceData,
    args: argparse.Namespace,
    *,
    draw_omega,
    get_omega_state=None,
    set_omega_state=None,
) -> Dict:
    """Run one chain and return posterior summaries and diagnostic traces."""
    # Copy so that resolving defaults does not mutate the caller's namespace.
    args = argparse.Namespace(**vars(args))
    args.m_rev = _resolved_m_rev(args)
    rng = np.random.default_rng(args.seed)
    decomposition = build_gp_decomposition(data)
    edge_gp_var = float(args.edge_gp_var)
    common_gp_var = float(args.common_gp_var)
    cfg = SamplerConfig(
        v0=args.v0,
        a_b=args.a_b,
        b_b=args.b_b,
        sigma_b2_init=args.sigma_b2_init,
    )
    st = init_state(
        data, cfg, rng, gamma_init_prob=0.5
    )
    _apply_initialization(st, args.initialization)
    shared = init_shared_gp_state(
        data,
        st,
        mu_init=args.mu_init,
        decomposition=decomposition,
    )

    if getattr(args, "storage_profile", "legacy") == "grid":
        from spadag.compact_runner import run_compact
        from spadag.compact_storage import writer_lock
        with writer_lock(args.trace_dir):
            return run_compact(
                data, args, st=st, cfg=cfg, shared=shared, rng=rng,
                sweep=sweep_shared_gp_pg, signature=_checkpoint_signature(data, args),
                continuation_signature=_continuation_signature, read_legacy=_read_checkpoint,
                stop_exception=CheckpointStop, draw_omega=draw_omega,
                get_omega_state=get_omega_state, set_omega_state=set_omega_state,
            )

    p, H = data.p, data.H
    gamma_sum = np.zeros((p, p, H), dtype=float)
    edge_sum = np.zeros((data.triangles.shape[0], p, p), dtype=float)
    b_sum = np.zeros((p, p, H), dtype=float)
    q_sum = np.zeros((p, p, H), dtype=float)
    eta_sum = np.zeros((p, p, H), dtype=float)

    sigma_b2_draws = []
    sigma2_draws = []
    q_field_draws = []
    eta_field_draws = []
    mu_full_trace = []
    active_gamma_full_trace = []
    omega_mean = []
    common_condition = []
    eta_min = []
    eta_max = []
    common_constraint_error = []
    edge_constraint_error = []
    edge_gp_var_full_trace = []
    edge_gp_var_draws = []
    edge_gp_var_raw_draws = []
    common_gp_var_full_trace = []
    common_gp_var_draws = []

    theta0_draws = []
    f0_draws = []
    fbar_draws = []
    q_bar_draws = []
    mu_star_draws = []
    f0_star_draws = []

    gamma_trace_packed = []
    gamma_trace_n_bits = len(data.pairs) * H

    kept = 0
    field_kept = 0
    initial_active_gamma = int(st.gamma.sum())

    iteration_start = 0
    elapsed_before = 0.0
    continued_from_iteration = 0
    if args.resume is not None:
        payload = _read_checkpoint(args.resume)
        expected_signature = _checkpoint_signature(data, args)
        if payload["signature"] != expected_signature:
            raise ValueError("checkpoint settings do not match this run")
        if set_omega_state is None:
            raise ValueError("resume requires a Pólya-Gamma RNG state setter")
        rng.bit_generator.state = payload["numpy_rng_state"]
        set_omega_state(payload["omega_rng_state"])
        st = payload["sampler_state"]
        cfg = payload["sampler_config"]
        shared = payload["shared_gp_state"]
        edge_gp_var = payload["edge_gp_var"]
        common_gp_var = float(payload["common_gp_var"])
        iteration_start = int(payload["completed_iterations"])
        elapsed_before = float(payload["elapsed_seconds"])
        restored = payload["accumulators"]
        gamma_sum = restored["gamma_sum"]
        edge_sum = restored["edge_sum"]
        b_sum = restored["b_sum"]
        q_sum = restored["q_sum"]
        eta_sum = restored["eta_sum"]
        sigma_b2_draws = restored["sigma_b2_draws"]
        sigma2_draws = restored["sigma2_draws"]
        q_field_draws = restored["q_field_draws"]
        eta_field_draws = restored["eta_field_draws"]
        mu_full_trace = restored["mu_full_trace"]
        active_gamma_full_trace = restored["active_gamma_full_trace"]
        omega_mean = restored["omega_mean"]
        common_condition = restored["common_condition"]
        eta_min = restored["eta_min"]
        eta_max = restored["eta_max"]
        common_constraint_error = restored["common_constraint_error"]
        edge_constraint_error = restored["edge_constraint_error"]
        edge_gp_var_full_trace = restored["edge_gp_var_full_trace"]
        edge_gp_var_draws = restored["edge_gp_var_draws"]
        edge_gp_var_raw_draws = restored["edge_gp_var_raw_draws"]
        common_gp_var_full_trace = restored["common_gp_var_full_trace"]
        common_gp_var_draws = restored["common_gp_var_draws"]
        theta0_draws = restored["theta0_draws"]
        f0_draws = restored["f0_draws"]
        fbar_draws = restored["fbar_draws"]
        q_bar_draws = restored["q_bar_draws"]
        mu_star_draws = restored["mu_star_draws"]
        f0_star_draws = restored["f0_star_draws"]
        gamma_trace_packed = restored["gamma_trace_packed"]
        kept = restored["kept"]
        field_kept = restored["field_kept"]
        initial_active_gamma = restored["initial_active_gamma"]
        continued_from_iteration = int(
            payload.get("continued_from_iteration", 0)
        )
        if iteration_start > args.iters:
            raise ValueError("checkpoint is beyond --iters")
    elif args.continue_from is not None:
        payload = _read_checkpoint(args.continue_from)
        source = _continuation_signature(payload["signature"])
        target = _continuation_signature(_checkpoint_signature(data, args))
        if source != target:
            raise ValueError("checkpoint model settings do not match continuation")
        if set_omega_state is None:
            raise ValueError("continuation requires a Pólya-Gamma RNG state setter")
        rng.bit_generator.state = payload["numpy_rng_state"]
        set_omega_state(payload["omega_rng_state"])
        st = payload["sampler_state"]
        cfg = payload["sampler_config"]
        shared = payload["shared_gp_state"]
        edge_gp_var = payload["edge_gp_var"]
        common_gp_var = float(payload["common_gp_var"])
        continued_from_iteration = int(
            payload.get("continued_from_iteration", 0)
        ) + int(payload["completed_iterations"])
        initial_active_gamma = int(st.gamma.sum())

    from spadag.component_reversal import MeshComponents
    component_mesh = MeshComponents(data) if args.m_rev else None
    t0 = time.perf_counter()
    for iteration in range(iteration_start, args.iters):
        pg_stats, edge_gp_var, common_gp_var = sweep_shared_gp_pg(
            data,
            st,
            shared,
            cfg,
            rng,
            draw_omega,
            common_gp_var=common_gp_var,
            edge_gp_var=edge_gp_var,
            a_tau=args.a_tau,
            b_tau=args.b_tau,
            a_sigma0=args.a_sigma0,
            b_sigma0=args.b_sigma0,
            mu_prior_mean=args.mu_prior_mean,
            mu_prior_sd=args.mu_prior_sd,
            m_rev=args.m_rev, component_mesh=component_mesh,
        )
        mu_full_trace.append(shared.mu)
        active_gamma_full_trace.append(int(st.gamma.sum()))
        omega_mean.append(pg_stats.omega_mean)
        common_condition.append(pg_stats.common_precision_condition)
        eta_min.append(pg_stats.eta_min)
        eta_max.append(pg_stats.eta_max)
        common_constraint_error.append(pg_stats.common_constraint_error)
        edge_constraint_error.append(pg_stats.edge_constraint_error)
        edge_gp_var_full_trace.append(edge_gp_var)
        common_gp_var_full_trace.append(common_gp_var)

        if iteration >= args.burn and (iteration - args.burn) % args.thin == 0:
            current_eta = _eta_tensor(data, st, shared)
            fbar = _mean_edge_deviation(data, st)
            combined_common = shared.f0 + fbar
            centering_constant = float(
                decomposition.integration_weights @ combined_common
            )
            # The triangle-edge PIP averages the OR over the triangle's
            # vertices within each draw, not the vertex PIPs afterwards.
            current_edge = triangle_edge_tensor(st.gamma, data.triangles)

            gamma_sum += st.gamma
            edge_sum += current_edge
            b_sum += st.b
            q_sum += st.m
            eta_sum += current_eta
            edge_gp_var_draws.append(edge_gp_var)
            edge_gp_var_raw_draws.append(
                constrained_edge_raw_variance(edge_gp_var, len(data.pairs))
            )
            common_gp_var_draws.append(common_gp_var)
            sigma_b2_draws.append(st.sigma_b2)
            sigma2_draws.append(st.sigma2.copy())
            theta0_draws.append(shared.theta0.copy())
            f0_draws.append(shared.f0.copy())
            fbar_draws.append(fbar)
            q_bar_draws.append(shared.q_bar.copy())
            mu_star_draws.append(shared.mu + centering_constant)
            f0_star_draws.append(combined_common - centering_constant)
            kept += 1

        if (
            args.gamma_trace_thin > 0
            and iteration >= args.burn
            and (iteration - args.burn) % args.gamma_trace_thin == 0
        ):
            gamma_trace_packed.append(np.packbits(
                _candidate_gamma_vector(data, st),
                bitorder="little",
            ))

        if (
            iteration >= args.burn
            and (iteration - args.burn) % args.field_thin == 0
        ):
            q_field_draws.append(st.m.copy())
            eta_field_draws.append(_eta_tensor(data, st, shared))
            field_kept += 1

        completed = iteration + 1
        if args.log_every > 0 and (
            completed % args.log_every == 0 or completed == args.iters
        ):
            elapsed_now = elapsed_before + time.perf_counter() - t0
            rate = completed / elapsed_now if elapsed_now > 0 else float("nan")
            remaining = (
                (args.iters - completed) / rate if rate > 0 else float("nan")
            )
            print(
                f"[progress] iter={completed}/{args.iters} "
                f"({100.0 * completed / args.iters:.1f}%) "
                f"elapsed={elapsed_now:.1f}s eta={remaining:.1f}s "
                f"kept={kept} active_gamma={int(st.gamma.sum())} "
                f"sigma_b2={st.sigma_b2:.4f} mu={shared.mu:.4f} "
                f"tau2={edge_gp_var:.4f} "
                f"sigma0_2={common_gp_var:.4f} "
                f"mean_q={float(np.mean(shared.q_bar)):.4f} "
                f"common_cond={pg_stats.common_precision_condition:.2e}",
                flush=True,
            )

        checkpoint_due = (
            args.checkpoint is not None
            and (
                completed == args.iters
                or (
                    args.checkpoint_every > 0
                    and completed % args.checkpoint_every == 0
                )
                or completed == args.stop_after
            )
        )
        if checkpoint_due:
            if get_omega_state is None:
                raise ValueError(
                    "checkpointing requires a Pólya-Gamma RNG state getter"
                )
            elapsed_checkpoint = elapsed_before + time.perf_counter() - t0
            accumulators = {
                "gamma_sum": gamma_sum,
                "edge_sum": edge_sum,
                "b_sum": b_sum,
                "q_sum": q_sum,
                "eta_sum": eta_sum,
                "sigma_b2_draws": sigma_b2_draws,
                "sigma2_draws": sigma2_draws,
                "q_field_draws": q_field_draws,
                "eta_field_draws": eta_field_draws,
                "mu_full_trace": mu_full_trace,
                "active_gamma_full_trace": active_gamma_full_trace,
                "omega_mean": omega_mean,
                "common_condition": common_condition,
                "eta_min": eta_min,
                "eta_max": eta_max,
                "common_constraint_error": common_constraint_error,
                "edge_constraint_error": edge_constraint_error,
                "edge_gp_var_full_trace": edge_gp_var_full_trace,
                "edge_gp_var_draws": edge_gp_var_draws,
                "edge_gp_var_raw_draws": edge_gp_var_raw_draws,
                "common_gp_var_full_trace": common_gp_var_full_trace,
                "common_gp_var_draws": common_gp_var_draws,
                "theta0_draws": theta0_draws,
                "f0_draws": f0_draws,
                "fbar_draws": fbar_draws,
                "q_bar_draws": q_bar_draws,
                "mu_star_draws": mu_star_draws,
                "f0_star_draws": f0_star_draws,
                "gamma_trace_packed": gamma_trace_packed,
                "kept": kept,
                "field_kept": field_kept,
                "initial_active_gamma": initial_active_gamma,
            }
            _write_checkpoint(args.checkpoint, {
                "version": _CHECKPOINT_VERSION,
                "signature": _checkpoint_signature(data, args),
                "completed_iterations": completed,
                "continued_from_iteration": continued_from_iteration,
                "elapsed_seconds": elapsed_checkpoint,
                "numpy_rng_state": rng.bit_generator.state,
                "omega_rng_state": get_omega_state(),
                "sampler_state": st,
                "sampler_config": cfg,
                "shared_gp_state": shared,
                "edge_gp_var": edge_gp_var,
                "common_gp_var": common_gp_var,
                "accumulators": accumulators,
            })
            print(
                f"[checkpoint] iter={completed} path={args.checkpoint}",
                flush=True,
            )
        if completed == args.stop_after:
            raise CheckpointStop(
                f"stopped after checkpoint at iteration {completed}"
            )

    elapsed = elapsed_before + time.perf_counter() - t0
    if kept == 0:
        raise RuntimeError("No posterior draws kept; decrease burn or increase iters.")
    if field_kept == 0:
        raise RuntimeError("No field draws kept; decrease field_thin.")

    gamma_prob = gamma_sum / kept
    edge_prob = edge_sum / kept
    b_mean = b_sum / kept
    q_mean = q_sum / kept
    eta_mean = eta_sum / kept

    sigma_b2_trace = np.asarray(sigma_b2_draws)
    sigma2_trace = np.asarray(sigma2_draws)
    edge_gp_var_trace = np.asarray(edge_gp_var_draws)
    edge_gp_var_raw_trace = np.asarray(edge_gp_var_raw_draws)
    edge_gp_var_full_trace_array = np.asarray(edge_gp_var_full_trace)
    common_gp_var_trace = np.asarray(common_gp_var_draws)
    common_gp_var_full_trace_array = np.asarray(common_gp_var_full_trace)
    q_field_trace = np.asarray(q_field_draws)
    eta_field_trace = np.asarray(eta_field_draws)
    theta0_trace = np.asarray(theta0_draws)
    f0_trace = np.asarray(f0_draws)
    fbar_trace = np.asarray(fbar_draws)
    q_bar_trace = np.asarray(q_bar_draws)
    mu_star_trace = np.asarray(mu_star_draws)
    f0_star_trace = np.asarray(f0_star_draws)
    mu_full_trace_array = np.asarray(mu_full_trace)
    mu_trace = mu_full_trace_array[args.burn::args.thin]
    active_gamma_full_trace_array = np.asarray(
        active_gamma_full_trace, dtype=int
    )
    active_gamma_trace = active_gamma_full_trace_array[
        args.burn::args.thin
    ]
    gamma_trace_packed_array = np.asarray(
        gamma_trace_packed, dtype=np.uint8
    )
    if args.gamma_trace_thin > 0:
        expected_bytes = (gamma_trace_n_bits + 7) // 8
        if gamma_trace_packed_array.shape != (
            len(gamma_trace_packed), expected_bytes
        ):
            raise RuntimeError("packed gamma trace has an unexpected shape")
    else:
        gamma_trace_packed_array = np.empty((0, 0), dtype=np.uint8)

    q_ci_low, q_ci_high = np.quantile(
        q_field_trace, [0.025, 0.975], axis=0
    )
    eta_ci_low, eta_ci_high = np.quantile(
        eta_field_trace, [0.025, 0.975], axis=0
    )
    sigma_b2_ci = np.quantile(sigma_b2_trace, [0.025, 0.975])
    edge_gp_var_ci = np.quantile(
        edge_gp_var_trace, [0.025, 0.975]
    )
    common_gp_var_ci = np.quantile(
        common_gp_var_trace, [0.025, 0.975]
    )
    sigma_j_trace = np.sqrt(sigma2_trace)
    sigma_j_ci_low, sigma_j_ci_high = np.quantile(
        sigma_j_trace, [0.025, 0.975], axis=0
    )

    mean_f0_trace = np.mean(f0_trace, axis=1)
    mean_fbar_trace = np.mean(fbar_trace, axis=1)
    pointwise_f0_fbar_corr = np.asarray([
        _safe_correlation(f0_trace[:, h], fbar_trace[:, h])
        for h in range(data.H)
    ])
    mean_q_trace = np.mean(q_bar_trace, axis=1)

    common_constraint_error_array = np.asarray(
        common_constraint_error, dtype=float
    )
    edge_constraint_error_array = np.asarray(
        edge_constraint_error, dtype=float
    )
    if data.baseline_weights is None:
        raise ValueError("InferenceData is missing baseline integration weights")
    baseline_constraint_error = float(
        np.max(np.abs(st.a_spatial @ data.baseline_weights))
    )

    return {
        "sigma_b2": float(np.mean(sigma_b2_trace)),
        "sigma_b2_ci": sigma_b2_ci,
        "sigma_j_mean": np.mean(sigma_j_trace, axis=0),
        "sigma_j_ci_low": sigma_j_ci_low,
        "sigma_j_ci_high": sigma_j_ci_high,
        "omega_mean": float(np.mean(omega_mean[args.burn:])),
        "common_precision_condition_max": float(np.max(common_condition)),
        "common_precision_condition_mean": float(np.mean(
            common_condition[args.burn:]
        )),
        "eta_min": float(np.min(eta_min)),
        "eta_max": float(np.max(eta_max)),
        "integration_weights": decomposition.integration_weights,
        "mesh_area": decomposition.mesh_area,
        "baseline_integration_weights": data.baseline_weights,
        "baseline_constraint_error": baseline_constraint_error,
        "common_null_dimension": decomposition.common_basis.shape[1],
        "common_constraint_error": common_constraint_error_array,
        "edge_constraint_error": edge_constraint_error_array,
        "common_constraint_error_max": float(np.nanmax(common_constraint_error_array)),
        "edge_constraint_error_max": float(np.nanmax(edge_constraint_error_array)),
        "time": elapsed,
        "kept": kept,
        "field_kept": field_kept,
        "field_thin": args.field_thin,
        "gamma_trace_packed": gamma_trace_packed_array,
        "gamma_trace_thin": args.gamma_trace_thin,
        "gamma_trace_n_bits": gamma_trace_n_bits,
        "gamma_trace_shape": np.asarray(
            [len(data.pairs), H], dtype=np.int64
        ),
        "gamma_trace_pairs": np.asarray(data.pairs, dtype=np.int64),
        "gamma_trace_bitorder": "little",
        "reversal_moves": int(getattr(st, "_component_accepted", 0)),
        "m_rev": args.m_rev,
        "v0": cfg.v0,
        "a_b": cfg.a_b,
        "b_b": cfg.b_b,
        "sigma_b2_init": cfg.sigma_b2_init,
        "initialization": args.initialization,
        "continued_from_iteration": continued_from_iteration,
        "common_gp_var": float(np.mean(common_gp_var_trace)),
        "common_gp_var_init": args.common_gp_var,
        "a_sigma0": args.a_sigma0,
        "b_sigma0": args.b_sigma0,
        "edge_gp_var": float(np.mean(edge_gp_var_trace)),
        "edge_gp_var_raw": float(np.mean(edge_gp_var_raw_trace)),
        "edge_gp_var_init": args.edge_gp_var,
        "a_tau": args.a_tau,
        "b_tau": args.b_tau,
        "mu_init": args.mu_init,
        "mu_prior_mean": args.mu_prior_mean,
        "mu_prior_sd": args.mu_prior_sd,
        "initial_active_gamma": initial_active_gamma,
        "active_gamma_mean": float(gamma_prob.sum()),
        "active_gamma_full_trace": active_gamma_full_trace_array,
        "active_gamma_trace": active_gamma_trace,
        "mu_full_trace": mu_full_trace_array,
        "mu_trace": mu_trace,
        "mu_mean": float(np.mean(mu_trace)),
        "mu_sd": float(np.std(mu_trace, ddof=1)),
        "theta0_trace": theta0_trace,
        "f0_trace": f0_trace,
        "fbar_trace": fbar_trace,
        "q_bar_trace": q_bar_trace,
        "mu_star_trace": mu_star_trace,
        "f0_star_trace": f0_star_trace,
        "mu_mean_f0_corr": _safe_correlation(mu_trace, mean_f0_trace),
        "mean_f0_mean_fbar_corr": _safe_correlation(
            mean_f0_trace, mean_fbar_trace
        ),
        "pointwise_f0_fbar_corr": pointwise_f0_fbar_corr,
        "gamma_prob": gamma_prob,
        "edge_prob": edge_prob,
        "b_mean": b_mean,
        "q_mean": q_mean,
        "eta_mean": eta_mean,
        "q_ci_low": q_ci_low,
        "q_ci_high": q_ci_high,
        "eta_ci_low": eta_ci_low,
        "eta_ci_high": eta_ci_high,
        "sigma_b2_trace": sigma_b2_trace,
        "edge_gp_var_full_trace": edge_gp_var_full_trace_array,
        "edge_gp_var_trace": edge_gp_var_trace,
        "edge_gp_var_raw_trace": edge_gp_var_raw_trace,
        "edge_gp_var_ci": edge_gp_var_ci,
        "common_gp_var_full_trace": common_gp_var_full_trace_array,
        "common_gp_var_trace": common_gp_var_trace,
        "common_gp_var_ci": common_gp_var_ci,
        "ess_sigma_b2": _ess_1d(sigma_b2_trace),
        "ess_edge_gp_var": _ess_1d(edge_gp_var_trace),
        "ess_common_gp_var": _ess_1d(common_gp_var_trace),
        "ess_active_gamma": _ess_1d(active_gamma_trace),
        "ess_mu": _ess_1d(mu_trace),
        "ess_mean_q": _ess_1d(mean_q_trace),
    }


def print_result(result: Dict) -> None:
    if "storage_format" in result:
        print(f"[grid summary] retained={result['kept']} time={result['time']:.2f}s")
        print("Triangle PIP and coefficient moments saved; multi-chain diagnostics are separate.")
        return
    print(f"{'metric':<32}{'value':>26}")
    print("-" * 58)
    rows = [
        ("sigma_b^2 mean", result["sigma_b2"], ".4f"),
        ("tau^2 retained mean", result["edge_gp_var"], ".4f"),
        ("sigma0^2 retained mean", result["common_gp_var"], ".4f"),
        ("mu posterior mean", result["mu_mean"], ".4f"),
        ("mu posterior sd", result["mu_sd"], ".4f"),
        ("mean active gamma", result["active_gamma_mean"], ".1f"),
        ("ESS sigma_b^2", result["ess_sigma_b2"], ".1f"),
        ("ESS tau^2", result["ess_edge_gp_var"], ".1f"),
        ("ESS sigma0^2", result["ess_common_gp_var"], ".1f"),
        ("ESS active gamma", result["ess_active_gamma"], ".1f"),
        ("ESS mu", result["ess_mu"], ".1f"),
        ("ESS mean q", result["ess_mean_q"], ".1f"),
        (
            "max common precision cond",
            result["common_precision_condition_max"],
            ".2e",
        ),
        ("PG omega mean", result["omega_mean"], ".4f"),
        ("time (s)", result["time"], ".1f"),
    ]
    for label, value, spec in rows:
        print(f"{label:<32}{format(value, spec):>26}")
    for label, key in (("sigma_b^2", "sigma_b2_ci"), ("tau^2", "edge_gp_var_ci"),
                       ("sigma0^2", "common_gp_var_ci")):
        interval = f"[{result[key][0]:.3f}, {result[key][1]:.3f}]"
        print(f"{label + ' 95% CrI':<32}{interval:>26}")
    print(f"{'max common constraint err':<32}{result['common_constraint_error_max']:>26.3e}")
    print(f"{'max edge constraint err':<32}{result['edge_constraint_error_max']:>26.3e}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one SpaDAG chain on a prepared inference bundle."
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="path to a SpaDAG inference bundle")
    parser.add_argument("--iters", type=int, default=50000,
                        help="total sweeps per chain including burn-in (default 50,000)")
    parser.add_argument("--burn", type=int, default=25000,
                        help="burn-in sweeps (default 25,000)")
    parser.add_argument("--thin", type=int, default=5,
                        help="retention interval for posterior summaries (default 5)")
    parser.add_argument("--storage-profile", choices=("legacy", "grid"), default="grid",
                        help="grid (default) writes bounded chunk storage to --trace-dir; "
                             "legacy keeps every retained draw in memory, for short runs")
    parser.add_argument("--trace-dir", type=Path, help="grid trace directory; transfer with the checkpoint")
    parser.add_argument("--scalar-thin", type=int, default=5)
    parser.add_argument("--block-iterations", type=int, default=5000)
    parser.add_argument("--chunk-iterations", type=int, default=500)
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--milestone-every", type=int, default=None,
                        help="grid only: keep immutable checkpoint snapshots at this interval "
                             "(default 5000 when --checkpoint is given, else 0)")
    parser.add_argument("--diagnostic-targets", type=Path)
    parser.add_argument("--projection-thin", type=int, default=10)
    parser.add_argument("--b-trace-thin", type=int, default=0)
    parser.add_argument("--latent-thin", type=int, default=0)
    parser.add_argument("--trace-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--field-thin", type=int, default=None,
                        help="legacy q/eta draw interval; grid eta stream interval (grid default 0, off)")
    parser.add_argument("--gamma-trace-thin", type=int, default=None,
                        help="interval for packed gamma states (grid default 10, legacy 0); 0 disables")
    parser.add_argument("--log-every", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initialization", choices=INITIALIZATIONS, default="empty",
                        help="empty graph (default), or Bernoulli(0.5) indicators restricted to local DAGs")
    parser.add_argument("--m-rev", type=int, default=None,
                        help=f"component-reversal attempts per sweep (default {DEFAULT_M_REV}); 0 disables")
    parser.add_argument("--mu-init", type=float, default=0.0)
    parser.add_argument("--mu-prior-mean", type=float, default=0.0)
    parser.add_argument("--mu-prior-sd", type=float, default=2.0)
    parser.add_argument("--common-gp-var", type=float, default=1.0,
                        help="initial value of the common-field variance sigma0^2")
    parser.add_argument("--common-gp-var-prior", type=float, nargs=2,
                        default=(DEFAULT_A_SIGMA0, DEFAULT_B_SIGMA0), metavar=("A0", "B0"),
                        help=f"inverse-Gamma shape and scale of sigma0^2 (default "
                             f"{DEFAULT_A_SIGMA0:g} {DEFAULT_B_SIGMA0:g})")
    parser.add_argument("--edge-gp-var", type=float, default=None,
                        help="initial value of the edge-field variance tau^2 (default: prior mean)")
    parser.add_argument("--a-tau", type=float, default=3.0, help="inverse-Gamma shape of tau^2")
    parser.add_argument("--b-tau", type=float, default=2.0, help="inverse-Gamma scale of tau^2")
    parser.add_argument("--v0", type=float, default=DEFAULT_V0, help="spike (inactive) coefficient variance")
    parser.add_argument("--a-b", type=float, default=1.0, help="inverse-Gamma shape of the slab variance")
    parser.add_argument("--b-b", type=float, default=1.0, help="inverse-Gamma scale of the slab variance")
    parser.add_argument("--sigma-b2-init", type=float, default=1.0, help="initial slab variance")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="atomically replaced local pickle checkpoint; "
                             "never load a checkpoint from an untrusted source")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="checkpoint interval in sweeps (default 250 with --checkpoint); 0 disables")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume exactly from a checkpoint of the same run")
    parser.add_argument("--continue-from", type=Path, default=None,
                        help="start a new saved segment from a completed checkpoint, "
                             "keeping the sampler and Python/R RNG states")
    parser.add_argument("--stop-after", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    grid = args.storage_profile == "grid"
    if args.gamma_trace_thin is None:
        args.gamma_trace_thin = 10 if grid else 0
    if args.milestone_every is None:
        args.milestone_every = 5000 if (grid and args.checkpoint is not None) else 0
    if args.checkpoint_every is None:
        args.checkpoint_every = 250 if (args.checkpoint is not None or args.resume is not None) else 0
    if args.field_thin is None:
        args.field_thin = 0 if grid else args.thin
    args.a_sigma0, args.b_sigma0 = (float(value) for value in args.common_gp_var_prior)
    if args.edge_gp_var is None:
        args.edge_gp_var = args.b_tau / (args.a_tau - 1.0) if args.a_tau > 1.0 else 1.0

    if grid and args.trace_dir is None:
        parser.error("--storage-profile grid (the default) requires --trace-dir; "
                     "use --storage-profile legacy for a short in-memory run")
    if not grid and (args.trace_dir or args.diagnostic_targets or args.b_trace_thin
                     or args.latent_thin or args.milestone_every):
        parser.error("trace options require --storage-profile grid")
    for name in ("scalar_thin", "block_iterations", "chunk_iterations", "chunk_mb",
                 "projection_thin", "thin"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("b_trace_thin", "latent_thin", "milestone_every", "gamma_trace_thin",
                 "checkpoint_every"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if args.field_thin < 0 or (not grid and args.field_thin == 0):
        parser.error("--field-thin must be positive (grid also accepts 0 to disable)")
    if args.iters <= 0 or args.burn < 0 or args.burn >= args.iters:
        parser.error("require --iters > --burn >= 0")
    if args.m_rev is not None and args.m_rev < 0:
        parser.error("--m-rev cannot be negative")
    for name in ("mu_init", "mu_prior_mean"):
        if not np.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    for name in ("mu_prior_sd", "common_gp_var", "edge_gp_var", "a_tau", "b_tau",
                 "a_sigma0", "b_sigma0", "v0", "a_b", "b_b", "sigma_b2_init"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")

    if args.resume is not None and args.checkpoint is None:
        args.checkpoint = args.resume
    if args.checkpoint_every > 0 and args.checkpoint is None:
        parser.error("--checkpoint-every requires --checkpoint")
    if args.milestone_every and args.checkpoint is None:
        parser.error("--milestone-every requires --checkpoint")
    if args.resume is not None and args.continue_from is not None:
        parser.error("--resume and --continue-from are mutually exclusive")
    if args.stop_after is not None:
        if args.checkpoint is None:
            parser.error("--stop-after requires --checkpoint")
        if args.stop_after <= 0 or args.stop_after >= args.iters:
            parser.error("require 0 < --stop-after < --iters")
    return args


def _save_result(path: Path, result: Dict, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value
        for key, value in result.items()
        if isinstance(
            value,
            (np.ndarray, float, int, str, np.floating, np.integer),
        )
    }
    arrays.update(
        seed=args.seed,
        iters=args.iters,
        burn=args.burn,
        thin=args.thin,
    )
    np.savez(path, **arrays)


def main() -> None:
    args = parse_args()
    data = load_inference_data(args.data)
    args.m_rev = _resolved_m_rev(args)
    print(
        f"[data] n={data.n} p={data.p} H={data.H} L={data.L} "
        f"pairs={len(data.pairs)}"
    )
    print(
        f"[spadag] iters={args.iters} burn={args.burn} thin={args.thin} "
        f"m_rev={args.m_rev} initialization={args.initialization} seed={args.seed} | "
        f"v0={args.v0} slab_IG=({args.a_b}, {args.b_b}) "
        f"tau2_IG=({args.a_tau}, {args.b_tau}) "
        f"sigma0_2_IG=({args.a_sigma0}, {args.b_sigma0}) "
        f"mu_prior=N({args.mu_prior_mean}, {args.mu_prior_sd}^2)\n"
    )
    with BayesLogitRPGSampler(seed=args.seed) as sampler:
        result = run_shared_gp_pg(
            data,
            args,
            draw_omega=sampler.draw,
            get_omega_state=sampler.get_rng_state,
            set_omega_state=sampler.set_rng_state,
        )
    print_result(result)

    if args.out is not None:
        _save_result(args.out, result, args)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
