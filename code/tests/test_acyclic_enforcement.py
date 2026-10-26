"""Focused checks for full-directed candidate DAG enforcement."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from spadag.acyclic import all_triangle_dags, triangle_edge_tensor
from spadag.mcmc_data import InferenceData
from spadag.mcmc_steps import SamplerConfig, init_state, update_gamma


def _minimal_data() -> InferenceData:
    n, p, H, L = 4, 3, 3, 1
    Phi = np.eye(H)[[0, 1, 2, 0]]
    X = np.ones((n, p))
    triangles = np.array([[0, 1, 2]], dtype=int)
    parents = np.ones((p, p), dtype=int)
    np.fill_diagonal(parents, 0)
    pairs = [(j, k) for j in range(p) for k in np.where(parents[j] == 1)[0]]
    return InferenceData(
        n=n,
        p=p,
        H=H,
        L=L,
        X=X,
        Phi=Phi,
        mesh_points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        triangles=triangles,
        A_kl=np.ones((H, L)),
        pairs=pairs,
        parents=parents,
        Zk=np.ones((p, n, H)),
        ZtZ=np.stack([np.eye(H) for _ in range(p)]),
        U=np.ones((n, H + 1)),
        UtU=np.eye(H + 1),
    )


def test_cycle_closing_flip_is_rejected() -> None:
    data = _minimal_data()
    cfg = SamplerConfig()
    rng = np.random.default_rng(1)
    st = init_state(replace(data, pairs=[]), cfg, rng)

    # adj[j, k] means k -> j. These two edges create path 0 -> 1 -> 2.
    st.gamma[1, 0, :] = 1
    st.gamma[2, 1, :] = 1
    assert all_triangle_dags(triangle_edge_tensor(st.gamma, data.triangles))

    # Force only candidate 2 -> 0, which would close 0 -> 1 -> 2 -> 0.
    data = replace(data, pairs=[(0, 2)], parents=np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]))
    st.b[0, 2, :] = 10.0
    st.pi[0, 2, :] = 1.0 - 1e-12
    update_gamma(data, st, cfg, rng)

    assert np.all(st.gamma[0, 2, :] == 0)
    assert all_triangle_dags(triangle_edge_tensor(st.gamma, data.triangles))


def test_random_initialization_is_acyclic() -> None:
    data = _minimal_data()
    cfg = SamplerConfig()
    for seed in range(20):
        st = init_state(
            data,
            cfg,
            np.random.default_rng(seed),
            gamma_init_prob=0.9,
        )
        assert all_triangle_dags(
            triangle_edge_tensor(st.gamma, data.triangles)
        )


if __name__ == "__main__":
    test_cycle_closing_flip_is_rejected()
    test_random_initialization_is_acyclic()
    print("acyclic enforcement checks passed")
