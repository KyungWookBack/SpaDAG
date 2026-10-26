"""Multi-chain convergence diagnostics for saved SpaDAG runs.

Rank-normalized split R-hat and bulk/tail ESS are computed with ArviZ.  Passing
the scalar checks is necessary but does not establish convergence of the
graph; between-chain PIP differences and direction traces are reported for
review.  Recovery against truth is computed in :mod:`spadag.evaluation`.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .compact_storage import atomic_json, digest, load_stream
from .cpdag import dag_to_cpdag
from .evaluation import evaluate_reversal_recovery
from .metrics import triangle_edge_recovery


def scalar_diagnostic(values):
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError("Install spadag[diagnostics] to run multi-chain diagnostics") from exc
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 8 or not np.all(np.isfinite(x)):
        return {"status": "UNAVAILABLE", "reason": "need >=2 chains, >=8 finite draws per chain"}
    if np.any(np.ptp(x, axis=1) == 0):
        return {"status": "UNAVAILABLE", "reason": "at least one chain is constant"}
    rhat = float(az.rhat(x, method="rank"))
    bulk, tail = float(az.ess(x, method="bulk")), float(az.ess(x, method="tail"))
    mcse = float(az.mcse(x, method="mean"))
    valid = np.all(np.isfinite([rhat, bulk, tail, mcse]))
    return {"status": "PASS" if valid and rhat < 1.01 and min(bulk, tail) >= 400 else "FAIL",
            "rhat": rhat, "ess_bulk": bulk, "ess_tail": tail, "mcse_mean": mcse}


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def compare_pip(a, b, mask):
    difference = np.abs(a - b)[mask]
    return {"mean_abs_difference": float(np.mean(difference)),
            "max_abs_difference": float(np.max(difference)),
            "threshold_disagreements": int(np.count_nonzero(((a >= .5) != (b >= .5))[mask]))}


def read_chain(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if digest(directory / "geometry.npz") != manifest["geometry_sha256"]:
        raise ValueError("corrupt geometry")
    with np.load(directory / "summary.npz", allow_pickle=False) as z:
        summary = {k: z[k] for k in z.files}
    if int(summary["completed_iterations"]) != manifest["completed_iterations"]:
        raise ValueError("summary is stale relative to manifest; finish/resume the run first")
    with np.load(directory / "geometry.npz", allow_pickle=False) as z:
        geometry = {k: z[k] for k in z.files}
    scalar = load_stream(directory, manifest["head"], "scalar")
    gamma = load_stream(directory, manifest["head"], "gamma")
    projection = load_stream(directory, manifest["head"], "projection")
    blocks = load_stream(directory, manifest["head"], "blocks")
    if len(scalar.get("iteration", [])) != manifest["rows"].get("scalar", 0):
        raise ValueError("scalar row count disagrees with manifest")
    for stream in (scalar, gamma, projection):
        if "iteration" in stream and np.any(np.diff(stream["iteration"]) <= 0):
            raise ValueError("duplicate/nonmonotone stream iterations")
    return {"directory": directory, "manifest": manifest, "summary": summary,
            "geometry": geometry, "scalar": scalar, "gamma": gamma,
            "projection": projection, "blocks": blocks}


def _align(chains, stream, *, postburn=False):
    iterations = []
    for c in chains:
        x = c[stream]
        if "iteration" not in x:
            return np.empty(0, dtype=int), []
        mask = x["postburn"] if postburn else np.ones(len(x["iteration"]), dtype=bool)
        iterations.append(x["iteration"][mask])
    common = iterations[0]
    for x in iterations[1:]:
        common = np.intersect1d(common, x)
    indices = [np.searchsorted(c[stream]["iteration"], common) for c in chains]
    return common, indices


def _direction_trace(chain, j, k, triangle):
    pairs = {tuple(pair): i for i, pair in enumerate(chain["geometry"]["pairs"])}
    gamma = chain["gamma"]
    count = len(gamma.get("iteration", []))
    if not count:
        return np.empty(0, dtype=np.uint8)
    H = chain["manifest"]["H"]
    vertices = chain["geometry"]["triangles"][triangle]

    def active(pair):
        if pair not in pairs:
            return np.zeros(count, dtype=bool)
        bits = pairs[pair] * H + vertices
        return np.any(((gamma["packed"][:, bits // 8] >> (bits % 8)) & 1) != 0, axis=1)

    return active((j, k)).astype(np.uint8) + 2 * active((k, j)).astype(np.uint8)


def _parent_modes(chain, j, triangle):
    """Modal parent set and modal undirected neighbourhood of child j over the stored gamma draws."""
    p = chain["manifest"]["p"]
    states = np.stack([_direction_trace(chain, j, k, triangle) for k in range(p) if k != j], axis=1)
    neighbours = [k for k in range(p) if k != j]
    result = {}
    for label, mask in (("parent_set", states == 1), ("skeleton_neighbourhood", states > 0)):
        rows, counts = np.unique(mask, axis=0, return_counts=True)
        index = int(np.argmax(counts))
        result[label] = [k for k, present in zip(neighbours, rows[index]) if present]
        result[label + "_fraction"] = float(counts[index] / len(states))
    return result


def direction_noise_report(chain, target):
    j, k, t = target
    state = _direction_trace(chain, j, k, t)
    if not len(state):
        return {"status": "UNAVAILABLE", "reason": "gamma stream disabled"}
    gamma, scalar = chain["gamma"], chain["scalar"]
    common, gi, si = np.intersect1d(gamma["iteration"], scalar["iteration"], return_indices=True)
    columns = chain["manifest"]["scalar_names"]
    variables = [f"sigma2[{j}]", f"rss[{j}]", "working_loglik"]
    result = {"aligned_draws": len(common), "gamma_thin": chain["manifest"]["gamma_thin"],
              "observed_transitions": int(np.count_nonzero(np.diff(state))),
              "state_meanings": {"0": "absent", "1": "parent->child", "2": "child->parent", "3": "invalid two-cycle"},
              "modal_sets": _parent_modes(chain, j, t), "conditional": {}}
    for s in (0, 1, 2, 3):
        ids = si[state[gi] == s]
        result["conditional"][str(s)] = {"count": len(ids), **{
            name: {"mean": float(np.mean(scalar["values"][ids, columns.index(name)])),
                   "q05_q50_q95": np.quantile(scalar["values"][ids, columns.index(name)], [.05, .5, .95]).tolist()}
            for name in variables if len(ids)}}
    return result


def diagnose(directories, *, truth=None, motifs=None, top=20, child=None, recent_iterations=None):
    import arviz as az
    chains = [read_chain(path) for path in directories]
    if len(chains) < 2 or len({str(c["directory"].resolve()) for c in chains}) != len(chains):
        raise ValueError("provide at least two distinct chain directories")
    seeds = [c["manifest"]["seed"] for c in chains]
    if len(set(seeds)) != len(seeds):
        raise ValueError("chains must have distinct RNG seeds")
    first = chains[0]
    ignored = {"seed", "initialization", "burn", "thin", "field_thin", "gamma_trace_thin"}
    model = lambda c: {k: v for k, v in c["manifest"]["signature"]["settings"].items() if k not in ignored}
    for c in chains[1:]:
        if c["manifest"]["signature"]["data_fingerprint"] != first["manifest"]["signature"]["data_fingerprint"] or model(c) != model(first):
            raise ValueError("chains must fit the same data and model")
        for key in ("pairs", "triangles", "common_basis"):
            if not np.array_equal(c["geometry"][key], first["geometry"][key]):
                raise ValueError("chain geometries/common coordinate bases differ")
    times, indices = _align(chains, "scalar", postburn=True)
    if recent_iterations is not None and len(times):
        keep = times > times[-1] - recent_iterations
        times, indices = times[keep], [idx[keep] for idx in indices]
    names = first["manifest"]["scalar_names"]
    scalars = {}
    for col, name in enumerate(names):
        values = np.stack([c["scalar"]["values"][ids, col] for c, ids in zip(chains, indices)]) if indices else np.empty((len(chains), 0))
        scalars[name] = scalar_diagnostic(values)
    noise_rhat = [row["rhat"] for name, row in scalars.items() if name.startswith("sigma2[") and "rhat" in row]
    scalar_pass = all(row["status"] == "PASS" for row in scalars.values())
    integrity_pass = all(c["manifest"]["integrity"]["failures"] == 0 for c in chains)
    probs = [c["summary"]["edge_prob"] for c in chains]
    if truth is not None:
        triangle_edge_recovery(probs[0], truth)  # Validate truth shape/values before masks.
    T, p, _ = probs[0].shape
    mask = np.broadcast_to(~np.eye(p, dtype=bool), (T, p, p))
    skeleton_mask = np.broadcast_to(np.triu(np.ones((p, p), dtype=bool), 1), (T, p, p))
    pairwise = {}
    compelled = np.stack([dag_to_cpdag(x)[0] for x in truth]).astype(bool) if truth is not None else None
    for a, b in combinations(range(len(chains)), 2):
        pairwise[f"{a}:{b}"] = {"directed": compare_pip(probs[a], probs[b], mask),
            "skeleton": compare_pip(probs[a] + probs[a].swapaxes(1, 2), probs[b] + probs[b].swapaxes(1, 2), skeleton_mask),
            "coefficient_signed_mean_max_difference": float(np.max(np.abs(chains[a]["summary"]["signed_candidate_mean"] - chains[b]["summary"]["signed_candidate_mean"]))),
            "coefficient_rms_mean_max_difference": float(np.max(np.abs(chains[a]["summary"]["rms_candidate_mean"] - chains[b]["summary"]["rms_candidate_mean"])))}
        if compelled is not None and np.any(compelled):
            pairwise[f"{a}:{b}"]["truth_compelled_directions"] = compare_pip(probs[a], probs[b], compelled)
    spread = np.ptp(np.stack(probs), axis=0)
    candidates = np.argwhere(mask)
    if child is not None:
        if not 0 <= child < p:
            raise ValueError("--child uses zero-based variable index")
        candidates = candidates[candidates[:, 1] == child]
    ranked = sorted(candidates.tolist(), key=lambda x: spread[tuple(x)], reverse=True)[:top]
    targets = [(j, k, t) for t, j, k in ranked]
    direction_rows = []
    gamma_times, gamma_indices = _align(chains, "gamma")
    for target in targets:
        direction = {"target_child_parent_triangle": target,
                     "selection": "post-hoc largest chain PIP differences; diagnostic only",
                     "chains": [direction_noise_report(c, target) for c in chains]}
        if gamma_indices:
            values = np.stack([(_direction_trace(c, *target)[idx] == 1).astype(float) for c, idx in zip(chains, gamma_indices)])
            direction["saved_gamma_indicator_diagnostic"] = scalar_diagnostic(values)
            direction["saved_gamma_aligned_draws_per_chain"] = len(gamma_times)
        direction_rows.append(direction)
    projection_times, projection_indices = _align(chains, "projection")
    coefficient_diag = {}
    same_targets = all(c["manifest"]["targets"].get("projections", []) == first["manifest"]["targets"].get("projections", []) for c in chains)
    if projection_indices and same_targets:
        for col in range(first["projection"]["values"].shape[1]):
            coefficient_diag[str(col)] = scalar_diagnostic(np.stack([c["projection"]["values"][idx, col] for c, idx in zip(chains, projection_indices)]))
    chain_rows = []
    for c, prob in zip(chains, probs):
        b = c["blocks"]
        block_counts = list(b.get("count", [])) + [c["summary"]["partial_block_count"]]
        block_n = list(b.get("n", [])) + [int(c["summary"]["partial_block_n"])]
        block_means = [x / n for x, n in zip(block_counts, block_n) if n]
        row = {"directory": str(c["directory"]), "seed": c["manifest"]["seed"], "initialization": c["manifest"]["initialization"],
               "retained_summary_draws": int(c["summary"]["kept"]),
               "block_counts": block_n, "block_first_last_pip_max_difference": float(np.max(np.abs(block_means[0] - block_means[-1]))),
               "resources": c["manifest"]["resources"], "integrity": c["manifest"]["integrity"]}
        if truth is not None:
            row["triangle_recovery"] = triangle_edge_recovery(prob, truth)
        if motifs is not None:
            row["motif_recovery"] = evaluate_reversal_recovery(prob, motifs)
        chain_rows.append(row)
    counts = np.asarray([int(c["summary"]["kept"]) for c in chains])
    result = {
        "status": "REVIEW" if scalar_pass and integrity_pass else "FAIL",
        "scalar_gate_pass": scalar_pass, "integrity_pass": integrity_pass,
        "graph_and_coefficient_review_required": True, "arviz_version": az.__version__,
        "thresholds": {"rhat_strictly_below": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400},
        "scalar_aligned_draws_per_chain": len(times), "recent_iterations": recent_iterations,
        "scalar_iteration_range": [int(times[0]), int(times[-1])] if len(times) else None,
        "max_noise_rhat": max(noise_rhat) if noise_rhat else None,
        "noise_diagnostics_available": len(noise_rhat), "noise_parameters": p,
        "scalars": scalars, "chains": chain_rows, "pairwise": pairwise,
        "coefficient_projection_diagnostics": coefficient_diag,
        "projection_status": "AVAILABLE" if coefficient_diag else "UNAVAILABLE: no matching selected projections",
        "directions_and_noise": direction_rows,
        "limitations": ["Scalar gates cannot establish graph or full high-dimensional convergence.",
                        "Gamma MCSE/ESS and observed transitions concern the saved gamma resolution, not the summary-thin estimator.",
                        "Direction/noise associations are descriptive, not proof of a causal mixing mechanism.",
                        "A thresholded PIP graph need not be a DAG; compelled masks here are defined by supplied truth.",
                        "Pooled recovery is descriptive; chains retain their identity and failures remain reported."],
    }
    if truth is not None:
        result["pooled_triangle_descriptive_only"] = triangle_edge_recovery(np.average(probs, axis=0, weights=counts), truth)
    return _clean(result)


def report_markdown(report):
    """Render a short Markdown summary of a diagnostics report; full detail stays in the JSON."""
    rows = ["# SpaDAG diagnostics", "", f"Status: **{report['status']}**. Graph/coefficient review remains required.", "",
            f"Scalar draws per chain: {report['scalar_aligned_draws_per_chain']}; max noise R-hat: {report['max_noise_rhat']}.", "",
            "| Chain | Start | TPR | FDR | F1 |", "| --- | --- | ---: | ---: | ---: |"]
    for c in report["chains"]:
        metric = c.get("triangle_recovery", {})
        values = [f"{metric[key]:.4f}" if key in metric else "not supplied" for key in ("tpr", "fdr", "f1")]
        rows.append(f"| {c['seed']} | {c['initialization']} | {' | '.join(values)} |")
    failures = [name for name, row in report["scalars"].items() if row["status"] != "PASS"]
    rows += ["", "Scalar failures/unavailable: " + (", ".join(failures) if failures else "none"), "",
             "Coefficient projections: " + report["projection_status"], "",
             "Triangle recovery and coefficient size summarize the fit; noise, direction states, PIP disagreements and convergence gates are diagnostics.", "",
             "See the adjacent JSON for all chain-specific and pairwise results, selected direction/noise associations, resource observations and limitations.", ""]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chains", nargs="+", type=Path)
    parser.add_argument("--truth", type=Path, help="explicit triangle-edge truth .npy; never read during fitting")
    parser.add_argument("--motifs", type=Path, help="JSON list of reversal motifs")
    parser.add_argument("--child", type=int)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--recent-iterations", type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.top < 0 or (args.recent_iterations is not None and args.recent_iterations <= 0):
        parser.error("--top must be nonnegative; --recent-iterations must be positive")
    report = diagnose(args.chains, truth=np.load(args.truth, allow_pickle=False) if args.truth else None,
                      motifs=json.loads(args.motifs.read_text(encoding="utf-8")) if args.motifs else None,
                      top=args.top, child=args.child, recent_iterations=args.recent_iterations)
    atomic_json(args.out, report)
    args.out.with_suffix(".md").write_text(report_markdown(report), encoding="utf-8")
    print(f"{report['status']}: {args.out}")


if __name__ == "__main__":
    main()
