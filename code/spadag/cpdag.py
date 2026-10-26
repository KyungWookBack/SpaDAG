"""Exact DAG-to-CPDAG conversion for fully observed graphs.

SpaDAG stores adjacency matrices as ``adj[child, parent] == 1``.  The
returned directed array follows the same convention; the undirected array is
symmetric.  Conversion starts from the skeleton, orients every unshielded
collider, and closes the PDAG under Meek's four orientation rules.
"""
from __future__ import annotations

import numpy as np

from .acyclic import is_dag


def _adjacent(
    directed: np.ndarray,
    undirected: np.ndarray,
    first: int,
    second: int,
) -> bool:
    return bool(
        undirected[first, second]
        or directed[first, second]
        or directed[second, first]
    )


def _orient(
    directed: np.ndarray,
    undirected: np.ndarray,
    parent: int,
    child: int,
) -> None:
    """Orient an undirected edge in the working parent-to-child convention."""
    if not undirected[parent, child]:
        return
    undirected[parent, child] = False
    undirected[child, parent] = False
    directed[parent, child] = True


def _orientation_rule(
    directed: np.ndarray,
    undirected: np.ndarray,
    parent: int,
    child: int,
) -> bool:
    """Return whether a Meek rule orients ``parent - child`` forward."""
    p = directed.shape[0]

    # R1: source -> parent - child, with source and child nonadjacent.
    for source in range(p):
        if directed[source, parent] and not _adjacent(
            directed, undirected, source, child
        ):
            return True

    # R2: parent -> middle -> child.
    for middle in range(p):
        if directed[parent, middle] and directed[middle, child]:
            return True

    # R3: parent - first -> child and parent - second -> child, where first
    # and second are nonadjacent.
    candidates = [
        node
        for node in range(p)
        if undirected[parent, node] and directed[node, child]
    ]
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            if not _adjacent(directed, undirected, first, second):
                return True

    # R4: parent - near -> far -> child, parent - far, and near is
    # nonadjacent to child.
    for near in range(p):
        if not undirected[parent, near]:
            continue
        for far in range(p):
            if (
                undirected[parent, far]
                and directed[near, far]
                and directed[far, child]
                and not _adjacent(directed, undirected, near, child)
            ):
                return True
    return False


def dag_to_cpdag(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return compelled directions and reversible edges of a DAG."""
    adjacency = np.asarray(adjacency, dtype=bool)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if np.any(np.diag(adjacency)):
        raise ValueError("self edges are not allowed")
    if not is_dag(adjacency):
        raise ValueError("adjacency must represent a DAG")

    # The working convention in this routine is directed[parent, child].
    dag = adjacency.T.copy()
    skeleton = dag | dag.T
    directed = np.zeros_like(dag)
    undirected = skeleton.copy()
    p = dag.shape[0]

    for child in range(p):
        parents = np.flatnonzero(dag[:, child])
        for first_index, first_value in enumerate(parents):
            first = int(first_value)
            for second_value in parents[first_index + 1 :]:
                second = int(second_value)
                if not skeleton[first, second]:
                    _orient(directed, undirected, first, child)
                    _orient(directed, undirected, second, child)

    changed = True
    while changed:
        changed = False
        for first in range(p):
            for second in range(first + 1, p):
                if not undirected[first, second]:
                    continue
                first_to_second = _orientation_rule(
                    directed, undirected, first, second
                )
                second_to_first = _orientation_rule(
                    directed, undirected, second, first
                )
                if first_to_second and second_to_first:
                    raise RuntimeError("conflicting Meek orientations")
                if first_to_second:
                    _orient(directed, undirected, first, second)
                    changed = True
                elif second_to_first:
                    _orient(directed, undirected, second, first)
                    changed = True

    if np.any(directed & ~dag):
        raise RuntimeError("CPDAG orientation disagrees with input DAG")
    if not np.array_equal(directed | directed.T | undirected, skeleton):
        raise RuntimeError("CPDAG skeleton changed during orientation")
    return directed.T.astype(np.int8), undirected.astype(np.int8)
