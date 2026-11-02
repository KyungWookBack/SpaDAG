"""Regional-reversal graph fields and coefficients for the manuscript simulations.

A given ``seed`` reproduces the graphs and coefficients of the manuscript's
simulation cohort exactly, so the random-number call order must not change.

Design:

* Variable labels are permuted at random.  The left ordering is the permuted
  order; the right ordering swaps positions ``(2,3), (4,5), (6,7), (8,9)`` in
  every complete ten-variable block and ``(1,2), (3,4)`` in a trailing
  five-variable block.  Dimensions must be multiples of five, so ``p = 25, 50,
  75`` carry ``10, 20, 30`` reversing pairs.
* Every block places unshielded-collider motifs ``a -> b <- c`` on a
  left-hand core (four vertices, ``x <= 0.2``) and ``b -> a <- c`` on the
  disjoint right-hand core (``x >= 0.8``) of a prescribed mesh row.
* Background directions compatible with both orderings are added on
  connected vertex patches until the area-weighted mean number of edges per
  triangle reaches ``target_mean_edges`` (``0.65 p`` in the manuscript),
  subject to a coverage window, a maximum local indegree of two, no opposite
  support, and triangle-wise acyclicity.  No recovery-based selection or
  reseeding takes place.
* Coefficients share one sign per directed pair and are independent
  ``Uniform(0.5, 1.5)`` draws at active vertices; inactive entries are zero.

The generator requires the regular mesh with 36 vertices and 50 triangles
(``build_regular_mesh(5)``) used by every manuscript simulation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from spadag.acyclic import creates_cycle, is_dag, triangle_edge_tensor

REFERENCE_H = 36
REFERENCE_T = 50


@dataclass(frozen=True)
class PartitionConfig:
    """Background-patch settings of the reversal graph generator."""

    p: int = 10
    target_mean_edges: float = 6.5
    max_indegree: int = 2
    max_attempts: int = 20000
    coverage_min: float = .08
    coverage_max: float = .36
    patch_sizes: tuple = (2, 3, 4)
    patch_probabilities: tuple = (.4, .4, .2)
    compactness_scale: float = .04


def cohort_config(p: int) -> PartitionConfig:
    """Main-experiment configuration: patch sizes 2-4, density ``0.65 p``."""
    return PartitionConfig(p=p, target_mean_edges=.65 * p)


def mesh_geometry(points, triangles):
    """Vertex neighbours, normalized triangle areas, and edge-adjacent pairs."""
    neighbors = [set() for _ in points]
    for tri in triangles:
        for v in tri:
            neighbors[v].update(int(u) for u in tri if u != v)
    xy = points[triangles]
    areas = np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                   - (xy[:, 1, 1] - xy[:, 0, 1]) * (xy[:, 2, 0] - xy[:, 0, 0])) / 2
    areas /= areas.sum()
    adjacent = [(a, b) for a, b in combinations(range(len(triangles)), 2)
                if len(set(triangles[a]) & set(triangles[b])) == 2]
    return neighbors, areas, adjacent


def draw_partitioned_graph(points, triangles, seed, config=PartitionConfig()):
    """Draw nodal indicators ``gamma[child, parent, vertex]`` and triangle edges.

    Returns ``(gamma, edge, report)`` where ``edge[t, child, parent]`` is the
    nodal-OR triangle graph and ``report`` records the permutation, orderings,
    motifs, accepted patches, rejection counts, and summary statistics.
    """
    c = config
    if c.p < 10 or c.p % 5 or c.max_indegree != 2:
        raise ValueError("Requires p >= 10, multiple of 5, and max indegree 2")
    if len(points) != REFERENCE_H or len(triangles) != REFERENCE_T:
        raise ValueError("requires the reference 36-vertex / 50-triangle regular mesh")
    neighbors, areas, adjacent = mesh_geometry(points, triangles)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(c.p)  # canonical index -> observed variable label
    left_order = permutation[np.arange(c.p)]
    blocks = [10 * t + np.array([0, 1, 3, 2, 5, 4, 7, 6, 9, 8]) for t in range(c.p // 10)]
    template = [(10 * t + cc, 10 * t + aa, 10 * t + bb, row) for t in range(c.p // 10)
                for cc, aa, bb, row in ((0, 2, 3, 0), (0, 4, 5, 1), (1, 6, 7, 3), (1, 8, 9, 4))]
    if c.p % 10:
        offset = 10 * (c.p // 10)
        blocks.append(offset + np.array([0, 2, 1, 4, 3]))
        template.extend([(offset, offset + 1, offset + 2, 0),
                         (offset, offset + 3, offset + 4, 1)])
    right_order = permutation[np.concatenate(blocks)]
    rank_l, rank_r = np.argsort(left_order), np.argsort(right_order)
    gamma = np.zeros((c.p, c.p, len(points)), dtype=np.int8)
    motifs, reserved, accepted = [], set(), []
    for i, (cc, aa, bb, row) in enumerate(template, 1):
        anchor, a, b = map(int, permutation[[cc, aa, bb]])
        y = (points[:, 1] >= row / 5 - 1e-12) & (points[:, 1] <= (row + 1) / 5 + 1e-12)
        left = y & (points[:, 0] <= .2 + 1e-12)
        right = y & (points[:, 0] >= .8 - 1e-12)
        assert left.sum() == right.sum() == 4
        tl, tr = left[triangles].any(1), right[triangles].any(1)
        assert not np.any(tl & tr)
        for parent, child, mask in ((a, b, left), (anchor, b, left),
                                    (b, a, right), (anchor, a, right)):
            gamma[child, parent, mask] = 1
        reserved.update(tuple(sorted(pair)) for pair in ((a, b), (anchor, a), (anchor, b)))
        motifs.append(dict(motif=i, a=a, b=b, c=anchor,
                           left_triangles=np.flatnonzero(tl).tolist(),
                           right_triangles=np.flatnonzero(tr).tolist(),
                           left_vertices=np.flatnonzero(left).tolist(),
                           right_vertices=np.flatnonzero(right).tolist()))
    edge = triangle_edge_tensor(gamma, triangles).astype(np.int8)
    base_mean = float(np.einsum("t,tjk->", areas, edge))
    allowed = [(parent, child) for parent in range(c.p) for child in range(c.p)
               if rank_l[parent] < rank_l[child] and rank_r[parent] < rank_r[child]
               and tuple(sorted((parent, child))) not in reserved]
    assert len(allowed) > 0
    total, attempts = base_mean, 0
    rejects = dict.fromkeys(("pair_used", "coverage", "opposite_overlap", "indegree", "cycle"), 0)
    while total < c.target_mean_edges and attempts < c.max_attempts:
        attempts += 1
        parent, child = allowed[int(rng.integers(len(allowed)))]
        if gamma[child, parent].any():
            rejects["pair_used"] += 1
            continue
        patch = {int(rng.integers(len(points)))}
        size = int(rng.choice(c.patch_sizes, p=c.patch_probabilities))
        while len(patch) < size:
            frontier = sorted(set().union(*(neighbors[v] for v in patch)) - patch)
            center = points[sorted(patch)].mean(axis=0)
            dist = np.sum((points[frontier] - center) ** 2, axis=1)
            weights = np.exp(-(dist - dist.min()) / c.compactness_scale)
            patch.add(int(rng.choice(frontier, p=weights / weights.sum())))
        mask = np.isin(np.arange(len(points)), sorted(patch))
        affected = mask[triangles].any(axis=1)
        coverage = float(areas[affected].sum())
        if not c.coverage_min - 1e-12 <= coverage <= c.coverage_max + 1e-12:
            rejects["coverage"] += 1
            continue
        if edge[affected, parent, child].any():
            rejects["opposite_overlap"] += 1
            continue
        if np.any(edge[affected, child, :].sum(axis=1) >= c.max_indegree):
            rejects["indegree"] += 1
            continue
        if any(creates_cycle(edge[t], child, parent) for t in np.flatnonzero(affected)):
            rejects["cycle"] += 1
            continue
        gamma[child, parent, mask] = 1
        edge[affected, child, parent] = 1
        total += coverage
        accepted.append(dict(attempt=attempts, parent=parent, child=child,
                             vertices=sorted(patch), triangles=np.flatnonzero(affected).tolist(),
                             coverage=coverage))
    assert np.array_equal(edge, triangle_edge_tensor(gamma, triangles))
    assert all(is_dag(g) for g in edge)
    assert int(edge.sum(axis=2).max()) <= 2
    right_triangles = (points[:, 0] >= .8 - 1e-12)[triangles].any(axis=1)
    for t, graph in enumerate(edge):
        rank = rank_r if right_triangles[t] else rank_l
        assert all(rank[parent] < rank[child] for child, parent in np.argwhere(graph))
    for motif in motifs:
        a, b, anchor = motif["a"], motif["b"], motif["c"]
        tl, tr = motif["left_triangles"], motif["right_triangles"]
        assert np.all(edge[tl, b, a] & edge[tl, b, anchor])
        assert not np.any(edge[tl, a, anchor] | edge[tl, anchor, a])
        assert np.all(edge[tr, a, b] & edge[tr, a, anchor])
        assert not np.any(edge[tr, b, anchor] | edge[tr, anchor, b])
    union = edge.any(axis=0)
    reversals = [(a, b) for a, b in combinations(range(c.p), 2) if union[a, b] and union[b, a]]
    assert len(reversals) == 2 * (c.p // 5) and not is_dag(union)
    assert not np.any(edge & edge.transpose(0, 2, 1))
    coverage = np.einsum("t,tjk->jk", areas, edge)[union]
    assert coverage.max() <= c.coverage_max + 1e-12
    difference = np.array([np.count_nonzero(edge[a] != edge[b]) for a, b in adjacent])
    counts = gamma[:, :, triangles].sum(axis=-1).transpose(2, 0, 1)[edge.astype(bool)]
    report = dict(
        config=asdict(c), graph_seed=int(seed), attempts=attempts, rejections=rejects,
        label_permutation=permutation.tolist(), left_order=left_order.tolist(),
        right_order=right_order.tolist(), reversal_motifs=motifs,
        target_reached=bool(total >= c.target_mean_edges),
        stop_reason="soft_density_target" if total >= c.target_mean_edges else "attempt_budget",
        base_motif_mean_local_edges=base_mean, accepted_background_patches=accepted,
        background_candidates=[list(x) for x in allowed],
        realized_reversal_pairs=[list(x) for x in reversals],
        mean_local_edges=float(np.einsum("t,tjk->", areas, edge)),
        local_edge_range=[int(edge.sum((1, 2)).min()), int(edge.sum((1, 2)).max())],
        maximum_local_indegree=int(edge.sum(axis=2).max()),
        active_directed_pairs=int(union.sum()), global_edges=int(np.isclose(coverage, 1).sum()),
        coverage_mean=float(coverage.mean()), coverage_range=[float(coverage.min()), float(coverage.max())],
        gamma_density=float(gamma.sum() / (c.p * (c.p - 1) * len(points))),
        adjacent_directed_difference_mean=float(difference.mean()),
        adjacent_identical_pairs=int((difference == 0).sum()),
        unique_graphs=int(len(np.unique(edge.reshape(len(edge), -1), axis=0))),
        active_vertex_count_distribution={str(k): int((counts == k).sum()) for k in (1, 2, 3)},
        common_order_exists=False, all_local_dags=True, colliders_unshielded=True,
    )
    return gamma, edge, report


def draw_coefficients(points, triangles, gamma, edge, seed):
    """Pair-shared signs and independent active-vertex ``Uniform(0.5, 1.5)``."""
    rng = np.random.default_rng(seed)
    b = np.zeros(gamma.shape, dtype=float)
    for j, k in np.argwhere(gamma.any(axis=2)):
        active = gamma[j, k].astype(bool)
        sign = float(rng.choice([-1, 1]))
        b[j, k, active] = sign * rng.uniform(.5, 1.5, int(active.sum()))
    assert np.all(b[gamma == 0] == 0)
    assert np.all((abs(b[gamma != 0]) >= .5) & (abs(b[gamma != 0]) <= 1.5))
    return b


def reversal_motifs_for_evaluation(motifs) -> list[dict[str, object]]:
    """Express core motifs in the ``spadag-evaluate`` ``reversal_motifs`` schema.

    On the left core the reversing pair is oriented ``a -> b``; on the right
    core it is ``b -> a``.  ``parent``/``child`` follow the sampler's
    ``gamma[child, parent]`` convention.
    """
    return [
        {
            "name": f"R{motif['motif']}",
            "nodes": {"a": int(motif["a"]), "b": int(motif["b"]), "c": int(motif["c"])},
            "left": {"parent": int(motif["a"]), "child": int(motif["b"]),
                     "triangles": [int(t) for t in motif["left_triangles"]]},
            "right": {"parent": int(motif["b"]), "child": int(motif["a"]),
                      "triangles": [int(t) for t in motif["right_triangles"]]},
        }
        for motif in motifs
    ]
