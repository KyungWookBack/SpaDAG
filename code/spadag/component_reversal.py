"""Component-reversal Metropolis-Hastings move.

Starting from an active nodal slot, C is the maximal connected set of mesh
nodes on which the indicator of k -> j is active; the move proposes to replace
k -> j by j -> k on C.  The
acceptance ratio integrates out the H nodal coefficients of both (j, k) and
(k, j), conditional on all other coefficients, and both blocks are then redrawn
from their conditionals whether or not the move is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from spadag.acyclic import is_dag, node_triangles
from spadag.mcmc_steps import alpha_at_obs, parent_pred, spike_slab_variances


@dataclass
class Proposal:
    child: int
    parent: int
    component: np.ndarray
    affected_triangles: np.ndarray
    active_slots: int
    reason: str = "valid"

    @property
    def probability(self):
        return len(self.component) / self.active_slots


class MeshComponents:
    """Mesh node adjacency used to find components by depth-first search."""

    def __init__(self, data):
        self.triangles = data.triangles
        self.incident = node_triangles(data.triangles, data.H)
        neighbors = [set() for _ in range(data.H)]
        for triangle in data.triangles:
            for h in triangle:
                neighbors[h].update(int(v) for v in triangle if v != h)
        self.neighbors = [np.array(sorted(v), dtype=int) for v in neighbors]
        self.candidates = set(data.pairs)

    def propose(self, gamma, slot, active_slots=None):
        j, k, h0 = map(int, slot)
        if gamma[j, k, h0] != 1 or j == k:
            raise ValueError("seed must be an active candidate slot")
        seen, stack = {h0}, [h0]
        while stack:
            for v in self.neighbors[stack.pop()]:
                v = int(v)
                if gamma[j, k, v] and v not in seen:
                    seen.add(v)
                    stack.append(v)
        C = np.array(sorted(seen), dtype=int)
        affected = np.unique(np.concatenate([self.incident[v] for v in C]))
        proposal = Proposal(j, k, C, affected,
                            int(gamma.sum()) if active_slots is None else active_slots)
        if (k, j) not in self.candidates:
            proposal.reason = "reverse_not_candidate"
            return proposal
        boundary = set(C)
        for h in C:
            boundary.update(self.neighbors[h])
        if np.any(gamma[k, j, list(boundary)]):
            proposal.reason = "opposite_merge"
            return proposal
        # Only triangles touching C can change, so only they are checked.
        for t in affected:
            vertices = self.triangles[t]
            adjacency = gamma[:, :, vertices].max(axis=2).astype(bool)
            adjacency[j, k] = any(gamma[j, k, v] and v not in seen for v in vertices)
            adjacency[k, j] = True
            if not is_dag(adjacency):
                proposal.reason = "dag"
                break
        return proposal


def apply_reversal(gamma, proposal):
    if proposal.reason != "valid":
        raise ValueError("cannot apply an inadmissible reversal")
    j, k, C = proposal.child, proposal.parent, proposal.component
    gamma[j, k, C] = 0
    gamma[k, j, C] = 1


def gaussian_block(gram, rhs, variances):
    """Log marginal likelihood of one coefficient block, Cholesky factor and mean.

    The returned score omits -n/2 log(2 pi sigma2) - r'r/(2 sigma2), which does
    not depend on gamma.  Both log-determinant terms depend on gamma through the
    spike and slab prior variances and must be kept.
    """
    precision = gram + np.diag(1.0 / variances)
    factor, _ = cho_factor(precision, lower=True, check_finite=False)
    mean = cho_solve((factor, True), rhs, check_finite=False)
    score = -0.5 * np.log(variances).sum() - np.log(np.diag(factor)).sum()
    score += 0.5 * rhs @ mean
    return float(score), factor, mean


def score_pair(data, state, config, proposal):
    """Log likelihood and log prior differences of a reversal for the two child regressions.

    All coefficients other than b[j, k] and b[k, j] are held fixed.
    """
    j, k, C = proposal.child, proposal.parent, proposal.component
    current, proposed = [], []
    delta_prior = 0.0
    for child, parent, new_bit in ((j, k, 0), (k, j, 1)):
        residual = (data.X[:, child] - alpha_at_obs(data, state, child)
                    - parent_pred(data, state, child, exclude_k=parent))
        gram = data.ZtZ[parent] / state.sigma2[child]
        rhs = data.Zk[parent].T @ residual / state.sigma2[child]
        old_gamma = state.gamma[child, parent]
        new_gamma = old_gamma.copy()
        new_gamma[C] = new_bit
        current.append(gaussian_block(gram, rhs, spike_slab_variances(old_gamma, state.sigma_b2, config)))
        proposed.append(gaussian_block(gram, rhs, spike_slab_variances(new_gamma, state.sigma_b2, config)))
        # Same clipping as the single-site indicator update.
        pi = np.clip(state.pi[child, parent, C], 1e-12, 1 - 1e-12)
        delta_prior += float(np.sum((new_gamma[C] - old_gamma[C]) * (np.log(pi) - np.log1p(-pi))))
    delta_likelihood = sum(x[0] - y[0] for x, y in zip(proposed, current))
    return float(delta_likelihood), float(delta_prior), current, proposed


def _draw(block, rng):
    _, factor, mean = block
    return mean + solve_triangular(factor, rng.standard_normal(len(mean)),
                                   lower=True, trans="T", check_finite=False)


def attempt_component_reversal(data, state, config, rng, mesh):
    """Run one component-reversal attempt.

    The seed is drawn uniformly from the active nodal indicators A, so a
    component C is proposed with probability |C|/|A| and its reverse with the
    same probability; components are never enumerated.  Selection depends only
    on gamma, not on the integrated coefficients.  If the proposal is
    inadmissible, both coefficient blocks are still redrawn under the unchanged
    gamma, which keeps the joint target invariant.
    """
    active = np.argwhere(state.gamma == 1)
    if len(active) == 0:
        return dict(reason="empty", accepted=False, size=0, coefficient_blocks=0)
    slot = active[rng.integers(len(active))]
    proposal = mesh.propose(state.gamma, slot, len(active))
    j, k = proposal.child, proposal.parent
    result = dict(reason=proposal.reason, accepted=False, size=len(proposal.component),
                  child=j, parent=k, affected_triangles=len(proposal.affected_triangles),
                  proposal_probability=proposal.probability, coefficient_blocks=0)
    if proposal.reason == "reverse_not_candidate":
        return result
    if proposal.reason != "valid":
        blocks = []
        for child, parent in ((j, k), (k, j)):
            residual = data.X[:, child] - alpha_at_obs(data, state, child) - parent_pred(data, state, child, exclude_k=parent)
            blocks.append(gaussian_block(data.ZtZ[parent] / state.sigma2[child],
                          data.Zk[parent].T @ residual / state.sigma2[child],
                          spike_slab_variances(state.gamma[child, parent], state.sigma_b2, config)))
    else:
        dl, dp, old, new = score_pair(data, state, config, proposal)
        log_ratio = dl + dp
        if not np.isfinite(log_ratio):
            raise FloatingPointError("nonfinite component-reversal MH score")
        accepted = np.log(max(rng.random(), np.finfo(float).tiny)) < min(0.0, log_ratio)
        if accepted:
            apply_reversal(state.gamma, proposal)
        blocks = new if accepted else old
        result.update(reason="accepted" if accepted else "mh", accepted=bool(accepted),
                      delta_likelihood=dl, delta_prior=dp, log_ratio=log_ratio)
    # The two blocks have distinct children and are conditionally independent.
    for (child, parent), block in zip(((j, k), (k, j)), blocks):
        state.b[child, parent] = _draw(block, rng)
    result["coefficient_blocks"] = 2
    # b changed outside the regular sweep, so the cached RSS and log likelihood are stale.
    state.rss = None
    state.log_likelihood = None
    return result
