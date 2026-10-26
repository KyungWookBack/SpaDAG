"""Triangle graphs and acyclicity checks.

Edge k -> j is active on triangle K iff some vertex h of K has nodal indicator
gamma[j, k, h] = 1 (OR rule), and the structure prior requires every triangle
graph G_K to be acyclic.  Throughout the package ``adj[j, k] == 1`` denotes the
directed edge k -> j (parent k of child j).
"""
from __future__ import annotations

from typing import List

import numpy as np


def node_triangles(triangles: np.ndarray, H: int) -> List[np.ndarray]:
    """For each node h in 0..H-1, indices of triangles whose vertices include h."""
    tris = np.asarray(triangles, dtype=int)
    out: List[List[int]] = [[] for _ in range(H)]
    for Kidx, verts in enumerate(tris):
        for h in verts:
            out[int(h)].append(Kidx)
    return [np.asarray(v, dtype=int) for v in out]


def triangle_edge_tensor(gamma: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """(T, p, p) int8 edge tensor; edge[K,j,k]=1 iff exists h in K with gamma[j,k,h]=1."""
    tris = np.asarray(triangles, dtype=int)
    p = gamma.shape[0]
    T = tris.shape[0]
    edge = np.zeros((T, p, p), dtype=np.int8)
    for Kidx, verts in enumerate(tris):
        edge[Kidx] = (gamma[:, :, verts].max(axis=2) > 0).astype(np.int8)
    return edge


def is_dag(adj: np.ndarray) -> bool:
    """Acyclicity via Kahn's algorithm. adj[j,k]=1 means edge k -> j."""
    A = np.asarray(adj, dtype=int)
    p = A.shape[0]
    indeg = A.sum(axis=1).astype(int)        # in-degree of j = number of parents
    queue = [j for j in range(p) if indeg[j] == 0]
    seen = 0
    while queue:
        node = queue.pop()
        seen += 1
        children = np.where(A[:, node] == 1)[0]   # node -> child
        for c in children:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(int(c))
    return seen == p


def all_triangle_dags(edge_tensor: np.ndarray) -> bool:
    """True iff every triangle-level graph G_K is acyclic."""
    return all(is_dag(edge_tensor[K]) for K in range(edge_tensor.shape[0]))


def creates_cycle(adj: np.ndarray, j: int, k: int) -> bool:
    """Would adding edge k -> j create a directed cycle in graph `adj`?

    A cycle appears iff there is already a directed path j ~> k (then k -> j closes
    it). Detect by reachability from j. `adj[a,b]=1` means edge b -> a.
    """
    if adj[j, k] == 1:
        return False                          # edge already present: no change
    A = np.asarray(adj, dtype=int)
    p = A.shape[0]
    stack = [int(j)]
    visited = np.zeros(p, dtype=bool)
    visited[j] = True
    while stack:
        u = stack.pop()
        if u == k:
            return True
        for w in np.where(A[:, u] == 1)[0]:   # u -> w
            if not visited[w]:
                visited[w] = True
                stack.append(int(w))
    return False
