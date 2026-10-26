"""Focused checks for exact DAG-to-CPDAG conversion."""
from __future__ import annotations

import itertools

import numpy as np

from spadag.acyclic import is_dag
from spadag.cpdag import dag_to_cpdag


def _signature(adjacency: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    skeleton: list[int] = []
    colliders: list[int] = []
    p = len(adjacency)
    for first in range(p):
        for second in range(first + 1, p):
            skeleton.append(int(adjacency[first, second] or adjacency[second, first]))
    for child in range(p):
        parents = np.flatnonzero(adjacency[child])
        for first_index, first in enumerate(parents):
            for second in parents[first_index + 1 :]:
                if not (
                    adjacency[first, second] or adjacency[second, first]
                ):
                    colliders.extend((int(first), child, int(second)))
    return tuple(skeleton), tuple(colliders)


def _all_dags(p: int) -> list[np.ndarray]:
    pairs = [(first, second) for first in range(p) for second in range(first + 1, p)]
    dags: list[np.ndarray] = []
    for states in itertools.product((0, 1, 2), repeat=len(pairs)):
        adjacency = np.zeros((p, p), dtype=np.int8)
        for state, (first, second) in zip(states, pairs):
            if state == 1:
                adjacency[second, first] = 1
            elif state == 2:
                adjacency[first, second] = 1
        if is_dag(adjacency):
            dags.append(adjacency)
    return dags


def test_p4_cpdag_matches_exact_markov_equivalence_classes() -> None:
    classes: dict[tuple[tuple[int, ...], tuple[int, ...]], list[np.ndarray]] = {}
    for adjacency in _all_dags(4):
        classes.setdefault(_signature(adjacency), []).append(adjacency)

    for members in classes.values():
        directed, undirected = dag_to_cpdag(members[0])
        compelled = np.logical_and.reduce(members)
        skeleton = (members[0] | members[0].T).astype(bool)
        expected_undirected = skeleton & ~(compelled | compelled.T)
        assert np.array_equal(directed.astype(bool), compelled)
        assert np.array_equal(undirected.astype(bool), expected_undirected)
