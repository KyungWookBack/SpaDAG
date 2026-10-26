"""Independent graph, dense marginal and detailed-balance checks."""
from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from spadag.acyclic import all_triangle_dags, triangle_edge_tensor
from spadag.mcmc_steps import SamplerConfig, spike_slab_variances
from spadag.component_reversal import (MeshComponents, apply_reversal,
    attempt_component_reversal, gaussian_block, score_pair, _draw)


def toy(p=3, H=5):
    rng = np.random.default_rng(7081)
    n = 17
    phi = rng.dirichlet(np.ones(H), n)
    X = rng.normal(size=(n, p))
    Z = np.array([phi * X[:, k, None] for k in range(p)])
    data = SimpleNamespace(p=p, H=H, n=n, X=X, Phi=phi,
        triangles=np.array([[0, 1, 2], [2, 3, 4]]) if H == 5 else np.array([[0, 1, 2]]),
        pairs=[(j,k) for j in range(p) for k in range(p) if j != k],
        parents=np.ones((p,p), int) - np.eye(p, dtype=int), Zk=Z,
        ZtZ=np.array([z.T @ z for z in Z]))
    state = SimpleNamespace(gamma=np.zeros((p,p,H), int),
        b=rng.normal(size=(p,p,H))*.15, pi=rng.uniform(.1,.9,(p,p,H)),
        sigma_b2=1.7, sigma2=rng.uniform(.7,1.8,p), a0=rng.normal(size=p)*.1,
        a_spatial=rng.normal(size=(p,H))*.1)
    return data, state, SamplerConfig()


def test_component_involution_slot_count_and_symmetric_probability():
    data, st, _ = toy()
    st.gamma[1,0,[0,1]] = 1
    st.gamma[0,1,4] = 1  # Separate opposite component is allowed.
    st.gamma[2,0,[0,1,2]] = 1
    mesh = MeshComponents(data)
    original = st.gamma.copy()
    p = mesh.propose(st.gamma,(1,0,0))
    assert p.reason == "valid"
    np.testing.assert_array_equal(p.component, [0,1])
    np.testing.assert_array_equal(p.affected_triangles, [0])
    assert p.probability == 2/6
    for h in p.component:
        np.testing.assert_array_equal(mesh.propose(st.gamma,(1,0,h)).component,p.component)
    apply_reversal(st.gamma,p)
    assert st.gamma.sum() == original.sum()
    assert all_triangle_dags(triangle_edge_tensor(st.gamma,data.triangles))
    reverse = mesh.propose(st.gamma,(0,1,0))
    assert reverse.reason == "valid" and reverse.probability == p.probability
    apply_reversal(st.gamma,reverse)
    np.testing.assert_array_equal(st.gamma,original)


def test_cycle_and_merge_rejection_no_mutation():
    data, st, _ = toy()
    st.gamma[1,0,0] = st.gamma[2,0,0] = st.gamma[1,2,0] = 1
    original = st.gamma.copy()
    assert MeshComponents(data).propose(st.gamma,(1,0,0)).reason == "dag"
    np.testing.assert_array_equal(st.gamma,original)
    # Defensive merge rejection: adjacent opposite supports already violate OR.
    st.gamma.fill(0)
    st.gamma[1,0,0] = st.gamma[0,1,1] = 1
    assert MeshComponents(data).propose(st.gamma,(1,0,0)).reason == "opposite_merge"


@pytest.mark.parametrize("mode", ["decoupled", "coupled"])
def test_collapsed_two_child_score_matches_dense_reference(mode):
    data, st, cfg = toy()
    cfg.spike_mode = mode
    st.gamma[1,0,[0,1]] = 1
    pr = MeshComponents(data).propose(st.gamma,(1,0,0))
    dl, dp, old, new = score_pair(data,st,cfg,pr)
    other = deepcopy(st)
    apply_reversal(other.gamma,pr)
    scores = []
    priors = []
    for s in (st,other):
        total = prior = 0.0
        for child,parent in ((1,0),(0,1)):
            r = data.X[:,child] - s.a0[child] - data.Phi @ s.a_spatial[child]
            for k in range(data.p):
                if k not in (child,parent):
                    r -= data.Zk[k] @ s.b[child,k]
            v = spike_slab_variances(s.gamma[child,parent],s.sigma_b2,cfg)
            cov = s.sigma2[child]*np.eye(data.n) + (data.Zk[parent]*v) @ data.Zk[parent].T
            total += multivariate_normal.logpdf(r, cov=cov)
            pi = np.clip(s.pi[child,parent],1e-12,1-1e-12)
            prior += np.sum(s.gamma[child,parent]*np.log(pi)+(1-s.gamma[child,parent])*np.log1p(-pi))
        scores.append(total)
        priors.append(prior)
    assert dl == pytest.approx(scores[1]-scores[0],abs=2e-12)
    assert dp == pytest.approx(priors[1]-priors[0],abs=2e-12)
    rdl, rdp, _, _ = score_pair(data,other,cfg,MeshComponents(data).propose(other.gamma,(0,1,0)))
    assert dl+dp == pytest.approx(-rdl-rdp,abs=2e-12)


def test_exact_finite_state_detailed_balance():
    data, st, cfg = toy(p=2,H=3)
    mesh = MeshComponents(data)
    states = []
    for bits in product((0,1),repeat=6):
        g = np.zeros_like(st.gamma)
        g[0,1],g[1,0] = bits[:3],bits[3:]
        if all_triangle_dags(triangle_edge_tensor(g,data.triangles)):
            states.append(g)
    lookup = {g.tobytes():i for i,g in enumerate(states)}
    Q = np.zeros((len(states),len(states)))
    log_target = []
    for i,g in enumerate(states):
        st.gamma = g
        val = 0.
        for j,k in data.pairs:
            r = data.X[:,j]-st.a0[j]-data.Phi@st.a_spatial[j]
            v = spike_slab_variances(g[j,k],st.sigma_b2,cfg)
            val += gaussian_block(data.ZtZ[k]/st.sigma2[j],data.Zk[k].T@r/st.sigma2[j],v)[0]
            val += np.sum(g[j,k]*np.log(st.pi[j,k])+(1-g[j,k])*np.log1p(-st.pi[j,k]))
        log_target.append(val)
        active = np.argwhere(g)
        for slot in active:
            pr = mesh.propose(g,slot)
            assert pr.reason == "valid"
            proposal = g.copy()
            apply_reversal(proposal,pr)
            Q[i,lookup[proposal.tobytes()]] += 1/len(active)
    np.testing.assert_allclose(Q,Q.T,atol=1e-15)
    log_target=np.array(log_target)
    target=np.exp(log_target-logsumexp(log_target))
    K=Q*np.exp(np.minimum(0,log_target[None,:]-log_target[:,None]))
    K[np.diag_indices(len(states))]=1-K.sum(1)
    np.testing.assert_allclose(target[:,None]*K,(target[:,None]*K).T,atol=1e-15)
    np.testing.assert_allclose(target@K,target,atol=1e-15)


def test_resampling_after_accept_and_reject_and_gaussian_moments():
    data, st, cfg = toy()
    st.gamma[1,0,[0,1]] = 1
    st.pi.fill(.5)
    # Extreme old-direction field guarantees numerical rejection for this seed.
    st.pi[1,0] = 1-1e-12
    st.pi[0,1] = 1e-12
    before=deepcopy(st)
    result=attempt_component_reversal(data,st,cfg,np.random.default_rng(1),MeshComponents(data))
    assert result['reason'] == 'mh' and result['coefficient_blocks'] == 2
    np.testing.assert_array_equal(st.gamma,before.gamma)
    assert not np.array_equal(st.b[1,0],before.b[1,0])
    assert not np.array_equal(st.b[0,1],before.b[0,1])
    np.testing.assert_array_equal(st.b[2],before.b[2])
    st.pi[1,0],st.pi[0,1]=1e-12,1-1e-12
    result=attempt_component_reversal(data,st,cfg,np.random.default_rng(1),MeshComponents(data))
    assert result['accepted'] and result['coefficient_blocks'] == 2
    assert all_triangle_dags(triangle_edge_tensor(st.gamma,data.triangles))
    block=gaussian_block(np.array([[2.,.7],[.7,3.]]),np.array([1.,2.]),np.array([.5,1.7]))
    rng=np.random.default_rng(951)
    draws=np.array([_draw(block,rng) for _ in range(30000)])
    precision=np.array([[4.,.7],[.7,3.+1/1.7]])
    np.testing.assert_allclose(draws.mean(0),np.linalg.solve(precision,[1.,2.]),atol=.015)
    np.testing.assert_allclose(np.cov(draws.T),np.linalg.inv(precision),atol=.015)


def test_empty_and_missing_reverse_candidate():
    data,st,cfg=toy()
    mesh=MeshComponents(data)
    assert attempt_component_reversal(data,st,cfg,np.random.default_rng(1),mesh)['reason']=='empty'
    st.gamma[1,0,0]=1
    data.pairs.remove((0,1))
    assert attempt_component_reversal(data,st,cfg,np.random.default_rng(1),MeshComponents(data))['reason']=='reverse_not_candidate'
