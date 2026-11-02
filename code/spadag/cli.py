"""The ``spadag`` command: run one SpaDAG chain on an inference bundle.

Each sweep updates, in order, the nodal indicators, the regression
coefficients, the slab variance, the inclusion fields (``shared_gp_pg.py``),
the baselines and noise variances, the two inclusion-field variances, and
finally attempts ``m_rev`` component reversals.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from spadag.compact_runner import run_compact
from spadag.compact_storage import writer_lock
from spadag.mcmc_data import InferenceData, load_inference_data
from spadag.component_reversal import MeshComponents, attempt_component_reversal
from spadag.mcmc_steps import (
    SamplerConfig,
    init_state,
    refresh_residual_cache,
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
        mesh = component_mesh if component_mesh is not None else MeshComponents(data)
        st.component_events = [
            attempt_component_reversal(data, st, cfg, rng, mesh) for _ in range(m_rev)
        ]
        st.component_accepted += sum(event["accepted"] for event in st.component_events)
        # The reversal redraws coefficients, so the cached residuals and
        # likelihood are refreshed; no random numbers are consumed.
        refresh_residual_cache(data, st)
    return pg_stats, edge_gp_var, common_gp_var


def run_shared_gp_pg(
    data: InferenceData,
    args: argparse.Namespace,
    *,
    draw_omega,
    get_omega_state=None,
    set_omega_state=None,
) -> Dict:
    """Run one chain, writing its traces to ``args.trace_dir``, and return the summaries."""
    # Copy so that resolving defaults does not mutate the caller's namespace.
    args = argparse.Namespace(**vars(args))
    args.m_rev = _resolved_m_rev(args)
    rng = np.random.default_rng(args.seed)
    decomposition = build_gp_decomposition(data)
    cfg = SamplerConfig(
        v0=args.v0,
        a_b=args.a_b,
        b_b=args.b_b,
        sigma_b2_init=args.sigma_b2_init,
    )
    st = init_state(data, cfg, rng, gamma_init_prob=0.5)
    _apply_initialization(st, args.initialization)
    shared = init_shared_gp_state(
        data,
        st,
        mu_init=args.mu_init,
        decomposition=decomposition,
    )
    with writer_lock(args.trace_dir):
        return run_compact(
            data, args, st=st, cfg=cfg, shared=shared, rng=rng,
            sweep=sweep_shared_gp_pg, signature=_checkpoint_signature(data, args),
            stop_exception=CheckpointStop, draw_omega=draw_omega,
            get_omega_state=get_omega_state, set_omega_state=set_omega_state,
        )


def print_result(result: Dict) -> None:
    print(f"[summary] retained={result['kept']} reversal_moves={result['reversal_moves']} "
          f"time={result['time']:.2f}s")
    print("Triangle PIPs and coefficient moments are in summary.npz; "
          "use spadag-diagnose for multi-chain diagnostics.")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one SpaDAG chain on a prepared inference bundle."
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="path to a SpaDAG inference bundle")
    parser.add_argument("--trace-dir", type=Path, required=True,
                        help="new directory for the chain's traces and summary; "
                             "keep it with the checkpoint")
    parser.add_argument("--iters", type=int, default=50000,
                        help="total sweeps per chain including burn-in (default 50,000)")
    parser.add_argument("--burn", type=int, default=25000,
                        help="burn-in sweeps (default 25,000)")
    parser.add_argument("--thin", type=int, default=5,
                        help="retention interval for posterior summaries (default 5)")
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
    parser.add_argument("--scalar-thin", type=int, default=5)
    parser.add_argument("--gamma-trace-thin", type=int, default=10,
                        help="interval for packed indicator states (default 10); 0 disables")
    parser.add_argument("--field-thin", type=int, default=0,
                        help="interval for the eta field stream; 0 (default) disables")
    parser.add_argument("--projection-thin", type=int, default=10)
    parser.add_argument("--b-trace-thin", type=int, default=0)
    parser.add_argument("--latent-thin", type=int, default=0)
    parser.add_argument("--diagnostic-targets", type=Path)
    parser.add_argument("--trace-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--block-iterations", type=int, default=5000)
    parser.add_argument("--chunk-iterations", type=int, default=500)
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=1500)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="atomically replaced local pickle checkpoint; "
                             "never load a checkpoint from an untrusted source")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="checkpoint interval in sweeps (default 250 with --checkpoint); 0 disables")
    parser.add_argument("--milestone-every", type=int, default=None,
                        help="keep immutable checkpoint snapshots at this interval "
                             "(default 5000 with --checkpoint)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume from a checkpoint of the same run; a larger --iters extends it")
    parser.add_argument("--stop-after", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=None,
                        help="optional .npz copy of the returned summaries")
    args = parser.parse_args(argv)

    if args.resume is not None and args.checkpoint is None:
        args.checkpoint = args.resume
    if args.checkpoint_every is None:
        args.checkpoint_every = 250 if args.checkpoint is not None else 0
    if args.milestone_every is None:
        args.milestone_every = 5000 if args.checkpoint is not None else 0
    args.a_sigma0, args.b_sigma0 = (float(value) for value in args.common_gp_var_prior)
    if args.edge_gp_var is None:
        args.edge_gp_var = args.b_tau / (args.a_tau - 1.0) if args.a_tau > 1.0 else 1.0

    for name in ("thin", "scalar_thin", "projection_thin", "block_iterations",
                 "chunk_iterations", "chunk_mb"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("gamma_trace_thin", "field_thin", "b_trace_thin", "latent_thin",
                 "checkpoint_every", "milestone_every"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
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
    if args.checkpoint_every > 0 and args.checkpoint is None:
        parser.error("--checkpoint-every requires --checkpoint")
    if args.milestone_every and args.checkpoint is None:
        parser.error("--milestone-every requires --checkpoint")
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
