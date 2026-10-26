"""Grid storage for a SpaDAG chain.

The sweep is the same as in the in-memory (legacy) profile; only what is
retained differs.  Retained draws are written to append-only chunk files, so the
checkpoint holds no draw history.  Iteration counts are one-based completed
global sweeps.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import time
import warnings

import numpy as np
import scipy

from .acyclic import all_triangle_dags, triangle_edge_tensor
from .compact_storage import (ChunkWriter, Moments, atomic_json, atomic_npz,
                              chunk_links, digest, triangle_coefficients)
from .mcmc_steps import _alpha_at_obs, _parent_pred


FORMAT = "spadag-compact-1"


def read_checkpoint(path):
    with Path(path).open("rb") as f:
        result = pickle.load(f)
    if not isinstance(result, dict) or result.get("format") != FORMAT:
        raise ValueError("expected a grid checkpoint; use --continue-from for a legacy-profile checkpoint")
    return result


def write_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_targets(path, data):
    targets = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    if set(targets) - {"reason", "projections", "edges", "residuals", "orientations"}:
        raise ValueError("unknown diagnostic target keys")
    pairs = set(data.pairs)
    for key in ("projections", "orientations"):
        for j, k, t in targets.get(key, []):
            if (j, k) not in pairs or not 0 <= t < len(data.triangles):
                raise ValueError(f"invalid {key} target (zero-based child,parent,triangle)")
    for pair in targets.get("edges", []):
        if tuple(pair) not in pairs:
            raise ValueError("invalid selected edge")
    for item in targets.get("residuals", []):
        ids = np.asarray(item["observations"], dtype=int)
        if not 0 <= item["child"] < data.p or ids.size == 0 or np.any((ids < 0) | (ids >= data.n)):
            raise ValueError("invalid residual child/observation target")
    return targets


def scalar_names(data, shared):
    return (["mu", "sigma_b2", "edge_var", "common_var", "active_gamma", "working_loglik"]
            + [f"{key}[{j}]" for key in ("sigma2", "rss", "a0", "tau_alpha") for j in range(data.p)]
            + [f"xi0[{l}]" for l in range(shared.decomposition.common_basis.shape[1])])


def run_compact(data, args, *, st, cfg, shared, rng, sweep, signature,
                continuation_signature, read_legacy, stop_exception,
                draw_omega, get_omega_state, set_omega_state):
    directory = Path(args.trace_dir)
    targets = load_targets(args.diagnostic_targets, data)
    observer_names = ("scalar_thin", "block_iterations", "projection_thin", "b_trace_thin",
                      "latent_thin", "trace_dtype")
    signature = dict(signature)
    signature["settings"] = {k: v for k, v in signature["settings"].items() if k != "iters"}
    signature["observer"] = {k: getattr(args, k) for k in observer_names}
    signature["targets"] = targets
    signature["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "code_sha256": {
                                path.name: digest(path) for path in [Path(sweep.__code__.co_filename),
                                *[Path(__file__).with_name(name) for name in
                                  ("compact_runner.py", "compact_storage.py", "mcmc_steps.py", "shared_gp_pg.py", "component_reversal.py")]]}}
    pair_j, pair_k = np.asarray(data.pairs, dtype=int).T
    E, T = len(data.pairs), len(data.triangles)
    acc = {
        "gamma_count": np.zeros((E, data.H), dtype=np.int64),
        "edge_count": np.zeros((E, T), dtype=np.int64),
        "b": Moments((E, data.H)), "signed": Moments((E, T)), "rms": Moments((E, T)),
        "block_count": np.zeros((E, T), dtype=np.int64), "block_n": 0, "block_id": 0,
        "kept": 0, "origin": 0, "initial_active_gamma": int(st.gamma.sum()),
        "integrity": {"checks": 0, "dag_checks": 0, "failures": 0, "first_failure_iteration": None,
                      "common_max": 0., "edge_max": 0., "baseline_max": 0.},
        "resources": {"checkpoint_seconds": 0., "checkpoint_bytes_max": 0,
                      "sampled_process_tree_rss_max": None, "buffer_bytes_max": 0},
        "rows": {},
    }
    start, elapsed_before, head = 0, 0., None
    edge_var, common_var = float(args.edge_gp_var), float(args.common_gp_var)
    payload = None
    if args.resume:
        payload = read_checkpoint(args.resume)
        if payload["signature"] != signature:
            raise ValueError("compact checkpoint settings do not match this run")
        start = int(payload["completed_iterations"])
        if args.iters < start:
            raise ValueError("--iters cannot precede the resumed global iteration")
        if not directory.is_dir():
            raise ValueError("missing trace directory for compact resume")
        head = payload["head"]
        links = chunk_links(directory, head)
        committed = {link["file"] for link in links}
        orphans = {p.name for p in directory.glob("chunk-*.npz")} - committed
        if orphans:
            warnings.warn(f"ignoring {len(orphans)} uncommitted chunks; files preserved", RuntimeWarning)
        acc = payload["accumulators"]
        elapsed_before = payload["elapsed_seconds"]
        if digest(directory / "geometry.npz") != payload["geometry_sha256"]:
            raise ValueError("missing or corrupt geometry")
    else:
        if directory.exists() and any(p.name != ".writer.lock" for p in directory.iterdir()):
            raise ValueError("new compact run requires an empty trace directory")
        if args.checkpoint and Path(args.checkpoint).exists():
            raise ValueError("new compact checkpoint path already exists; choose a fresh run path")
        if args.continue_from:
            # A legacy-profile checkpoint continues state only; its retained
            # draws are not merged into the new summaries.
            payload = read_legacy(args.continue_from)
            if continuation_signature(payload["signature"]) != continuation_signature(signature):
                raise ValueError("legacy continuation model settings do not match")
            if args.burn != 0:
                raise ValueError("legacy-to-compact continuation requires --burn 0")
            start = int(payload["completed_iterations"]) + int(payload.get("continued_from_iteration", 0))
            acc["origin"] = start
            if args.iters <= start:
                raise ValueError("compact --iters is a global target, including legacy iterations")
    if payload is not None:
        if set_omega_state is None:
            raise ValueError("continuation requires a Pólya-Gamma RNG state setter")
        rng.bit_generator.state = payload["numpy_rng_state"]
        set_omega_state(payload["omega_rng_state"])
        st, cfg, shared = payload["sampler_state"], payload["sampler_config"], payload["shared_gp_state"]
        edge_var = payload["edge_gp_var"]
        common_var = payload["common_gp_var"]
        if args.continue_from:
            acc["initial_active_gamma"] = int(st.gamma.sum())
    if args.checkpoint and get_omega_state is None:
        raise ValueError("checkpoint requires a Pólya-Gamma RNG state getter")
    directory.mkdir(parents=True, exist_ok=True)
    geometry_path = directory / "geometry.npz"
    if not args.resume:
        atomic_npz(geometry_path, pairs=np.asarray(data.pairs), triangles=data.triangles,
                   common_basis=shared.decomposition.common_basis, A=data.A_kl,
                   integration_weights=shared.decomposition.integration_weights)
    geometry_sha = digest(geometry_path)
    writer = ChunkWriter(directory, head=head, max_bytes=args.chunk_mb * 1024**2)
    names = scalar_names(data, shared)
    from .component_reversal import MeshComponents
    component_mesh = MeshComponents(data) if args.m_rev else None
    clock = time.perf_counter()
    dtype = np.dtype(args.trace_dtype)
    pair_lookup = {pair: i for i, pair in enumerate(data.pairs)}
    selected = [pair_lookup[tuple(pair)] for pair in targets.get("edges", [])]
    try:
        import psutil
    except ImportError:
        psutil = None

    def add(stream, iteration, **values):
        writer.add(stream, iteration, **values)
        acc["rows"][stream] = acc["rows"].get(stream, 0) + 1

    def resources():
        acc["resources"]["buffer_bytes_max"] = max(acc["resources"]["buffer_bytes_max"], writer.peak_buffer_bytes)
        try:
            if psutil is None:
                return
            process = psutil.Process()
            rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)])
            old = acc["resources"]["sampled_process_tree_rss_max"] or 0
            acc["resources"]["sampled_process_tree_rss_max"] = max(old, rss)
        except (OSError, psutil.Error):
            pass  # Missing resource observation is null, never zero.

    def save(completed):
        before = time.perf_counter()
        writer.flush()
        resources()
        elapsed = elapsed_before + time.perf_counter() - clock
        if args.checkpoint:
            write_checkpoint(args.checkpoint, {
                "format": FORMAT, "signature": signature, "completed_iterations": completed,
                "elapsed_seconds": elapsed, "numpy_rng_state": rng.bit_generator.state,
                "omega_rng_state": get_omega_state(), "sampler_state": st, "sampler_config": cfg,
                "shared_gp_state": shared, "edge_gp_var": edge_var, "common_gp_var": common_var,
                "accumulators": acc, "head": writer.head, "geometry_sha256": geometry_sha,
            })
            acc["resources"]["checkpoint_bytes_max"] = max(acc["resources"]["checkpoint_bytes_max"], Path(args.checkpoint).stat().st_size)
            if args.milestone_every and completed % args.milestone_every == 0:
                milestone = Path(args.checkpoint).parent / "milestones" / f"iteration_{completed:09d}.pkl"
                milestone.parent.mkdir(parents=True, exist_ok=True)
                if not milestone.exists():
                    tmp = milestone.with_suffix(".tmp")
                    shutil.copyfile(args.checkpoint, tmp)
                    tmp.replace(milestone)
        acc["resources"]["checkpoint_seconds"] += time.perf_counter() - before
        atomic_json(directory / "manifest.json", {
            "format": FORMAT, "signature": signature, "head": writer.head,
            "completed_iterations": completed, "elapsed_seconds": elapsed,
            "geometry_sha256": geometry_sha, "scalar_names": names,
            "data_reference": str(Path(args.data).resolve()), "n": data.n, "p": data.p, "H": data.H,
            "seed": args.seed, "initialization": args.initialization, "origin": acc["origin"],
            "rows": acc["rows"], "summary_thin": args.thin, "scalar_thin": args.scalar_thin,
            "gamma_thin": args.gamma_trace_thin, "gamma_bitorder": "little", "burn": args.burn,
            "threshold": .5, "empty_selection_fdr": 0, "targets": targets,
            "fields": "eta only; q = sigmoid(eta)" if args.field_thin else "not stored",
            "coefficient_summary": "unconditional; posterior SD, no credible intervals",
            "integrity": acc["integrity"], "resources": acc["resources"],
        })

    def due(i, interval, origin):
        return interval > 0 and i >= origin and (i - origin) % interval == 0

    origin = acc["origin"] + args.burn
    for i in range(start, args.iters):
        pg, edge_var, common_var = sweep(
            data, st, shared, cfg, rng, draw_omega,
            common_gp_var=common_var, edge_gp_var=edge_var,
            a_tau=args.a_tau, b_tau=args.b_tau, a_sigma0=args.a_sigma0, b_sigma0=args.b_sigma0,
            mu_prior_mean=args.mu_prior_mean, mu_prior_sd=args.mu_prior_sd,
            m_rev=args.m_rev, component_mesh=component_mesh,
        )
        completed = i + 1
        if args.m_rev:
            events = st._component_events
            add("component", completed,
                accepted=np.array([e["accepted"] for e in events], dtype=np.uint8),
                size=np.array([e["size"] for e in events], dtype=np.int64),
                reason=np.array([e["reason"] for e in events], dtype="U24"),
                child=np.array([e.get("child", -1) for e in events]),
                parent=np.array([e.get("parent", -1) for e in events]),
                log_ratio=np.array([e.get("log_ratio", np.nan) for e in events]),
                delta_likelihood=np.array([e.get("delta_likelihood", np.nan) for e in events]),
                delta_prior=np.array([e.get("delta_prior", np.nan) for e in events]))
        integrity = acc["integrity"]
        integrity["checks"] += 1
        errors = {"common_max": abs(pg.common_constraint_error),
                  "edge_max": abs(pg.edge_constraint_error),
                  "baseline_max": float(np.max(np.abs(st.a_spatial @ data.baseline_weights)))}
        for key, value in errors.items():
            integrity[key] = max(integrity[key], value)
        finite = all(np.all(np.isfinite(x)) for x in (st.b, st.theta, st.sigma2, st.a0,
                     st.a_spatial, st.tau_alpha, st.pi, st._rss, st._log_likelihood,
                     shared.theta0, shared.mu, st.sigma_b2, edge_var, common_var))
        finite = finite and all(np.all(np.asarray(x) > 0) for x in (st.sigma2, st.tau_alpha, st.sigma_b2, edge_var, common_var))
        bad = not finite or any(v > 1e-8 or not np.isfinite(v) for v in errors.values())
        if due(i, args.thin, origin):
            gamma = st.gamma[pair_j, pair_k]
            edge = gamma[:, data.triangles].max(axis=-1)
            b = st.b[pair_j, pair_k]
            m, a = triangle_coefficients(b, data.triangles)
            block_id = (i - origin) // args.block_iterations
            if block_id != acc["block_id"] and acc["block_n"]:
                add("blocks", origin + (acc["block_id"] + 1) * args.block_iterations,
                    start=origin + acc["block_id"] * args.block_iterations + 1,
                    count=acc["block_count"], n=acc["block_n"])
                acc["block_count"].fill(0)
                acc["block_n"] = 0
            acc["block_id"] = block_id
            acc["gamma_count"] += gamma
            acc["edge_count"] += edge
            acc["block_count"] += edge
            acc["block_n"] += 1
            acc["kept"] += 1
            acc["b"].add(b)
            acc["signed"].add(m)
            acc["rms"].add(a)
            integrity["dag_checks"] += 1
            bad = bad or not all_triangle_dags(triangle_edge_tensor(st.gamma, data.triangles))
        if bad:
            integrity["failures"] += 1
            if integrity["first_failure_iteration"] is None:
                integrity["first_failure_iteration"] = completed
            save(completed)
            raise FloatingPointError(f"sampler integrity failure at iteration {completed}")
        xi0 = shared.decomposition.common_basis.T @ shared.theta0
        if due(i, args.scalar_thin, acc["origin"]):
            values = np.concatenate(([shared.mu, st.sigma_b2, edge_var, common_var,
                                      st.gamma.sum(), st._log_likelihood], st.sigma2,
                                     st._rss, st.a0, st.tau_alpha, xi0))
            add("scalar", completed, values=values, postburn=i >= origin)
        if due(i, args.gamma_trace_thin, origin):
            add("gamma", completed, packed=np.packbits(st.gamma[pair_j, pair_k].astype(np.uint8).ravel(), bitorder="little"))
        if due(i, args.projection_thin, origin) and targets.get("projections"):
            values = []
            for j, k, t in targets["projections"]:
                m, a = triangle_coefficients(st.b[j, k][None, :], data.triangles[t:t+1])
                values.extend((float(m[0, 0]), float(a[0, 0])))
            add("projection", completed, values=values)
        if targets.get("orientations") and i >= origin:
            states = [int(st.gamma[j, k, data.triangles[t]].any())
                      + 2 * int(st.gamma[k, j, data.triangles[t]].any()) for j, k, t in targets["orientations"]]
            add("orientation", completed, states=np.asarray(states, dtype=np.uint8))
        if due(i, args.scalar_thin, acc["origin"]) and targets.get("residuals"):
            values = []
            for target in targets["residuals"]:
                j, ids = target["child"], target["observations"]
                residual = data.X[:, j] - _alpha_at_obs(data, st, j) - _parent_pred(data, st, j)
                values.append(np.mean(residual[ids]**2))
            add("residual", completed, mse=values)
        if due(i, args.b_trace_thin, origin):
            b = st.b[pair_j, pair_k]
            add("b", completed, values=(b[selected] if selected else b).astype(dtype))
        if due(i, args.latent_thin, origin):
            theta = st.theta[pair_j, pair_k]
            add("latent", completed, theta=(theta[selected] if selected else theta).astype(dtype),
                xi0=xi0, mu=shared.mu)
        if due(i, args.field_thin, origin):
            eta = shared.mu + shared.f0 + st.theta[pair_j, pair_k] @ data.A_kl.T
            add("field", completed, eta=(eta[selected] if selected else eta).astype(dtype))
        if completed % args.chunk_iterations == 0:
            writer.flush()
            resources()
        if args.log_every and completed % args.log_every == 0:
            print(f"[compact] iteration={completed} kept={acc['kept']} logL={st._log_likelihood:.3f}", flush=True)
        stopping = args.stop_after is not None and completed == args.stop_after
        if (stopping or completed == args.iters or (args.checkpoint_every and completed % args.checkpoint_every == 0)
                or (args.milestone_every and completed % args.milestone_every == 0)):
            save(completed)
        if stopping:
            raise stop_exception(f"stopped after checkpoint at {completed}")
    if start == args.iters:
        save(start)
    if not acc["kept"]:
        raise ValueError("no posterior summary draws")
    p, H = data.p, data.H
    edge_prob = np.zeros((T, p, p))
    edge_prob[:, pair_j, pair_k] = acc["edge_count"].T / acc["kept"]
    gamma_prob, b_mean = np.zeros((p, p, H)), np.zeros((p, p, H))
    gamma_prob[pair_j, pair_k] = acc["gamma_count"] / acc["kept"]
    b_mean[pair_j, pair_k] = acc["b"].mean
    result = {"storage_format": FORMAT, "kept": acc["kept"], "edge_prob": edge_prob,
              "gamma_prob": gamma_prob, "b_mean": b_mean, "edge_count": acc["edge_count"],
              "gamma_count": acc["gamma_count"], "pairs": np.asarray(data.pairs), "triangles": data.triangles,
              "partial_block_count": acc["block_count"], "partial_block_n": acc["block_n"],
              "partial_block_start": origin + acc["block_id"] * args.block_iterations + 1,
              "completed_iterations": args.iters, "seed": args.seed, "burn": args.burn,
              "thin": args.thin, "initialization": args.initialization,
              "m_rev": args.m_rev, "reversal_moves": int(getattr(st, "_component_accepted", 0)),
              "data_fingerprint": data.data_fingerprint, "time": elapsed_before + time.perf_counter() - clock}
    for name in ("b", "signed", "rms"):
        result[f"{name}_candidate_mean"] = acc[name].mean
        result[f"{name}_candidate_sd"] = acc[name].sd()
    atomic_npz(directory / "summary.npz", **result)
    return result
