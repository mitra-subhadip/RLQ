"""Deterministic and stochastic initial-placement baselines."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap

from .environment import PlacementEnvironment
from .problem import PlacementProblem


def identity_mapping(problem: PlacementProblem) -> tuple[int, ...]:
    allowed = np.flatnonzero(problem.allowed_physical_mask).tolist()
    mapping = [-1] * problem.num_logical_qubits
    unused = set(allowed)
    for logical in range(problem.num_logical_qubits):
        if logical in unused:
            mapping[logical] = logical
            unused.remove(logical)
    remaining = iter(sorted(unused))
    for logical, physical in enumerate(mapping):
        if physical < 0:
            mapping[logical] = next(remaining)
    return tuple(mapping)


def random_mapping(
    problem: PlacementProblem, *, seed: int | None = None
) -> tuple[int, ...]:
    generator = np.random.default_rng(seed)
    allowed = np.flatnonzero(problem.allowed_physical_mask)
    selected = generator.choice(
        allowed, size=problem.num_logical_qubits, replace=False
    )
    return tuple(int(value) for value in selected)


def degree_centrality_mapping(problem: PlacementProblem) -> tuple[int, ...]:
    hardware_score = (
        problem.hardware.static_node_features[:, 0]
        + problem.hardware.static_node_features[:, 1]
        - problem.hardware.static_node_features[:, 5]
    )
    candidates = np.flatnonzero(problem.allowed_physical_mask)
    physical_order = candidates[np.argsort(hardware_score[candidates])[::-1]]
    mapping = [-1] * problem.num_logical_qubits
    for logical, physical in zip(
        problem.placement_order, physical_order, strict=False
    ):
        mapping[logical] = int(physical)
    return tuple(mapping)


def greedy_mapping(problem: PlacementProblem) -> tuple[int, ...]:
    environment = PlacementEnvironment(problem)
    state = environment.reset()
    while not state.done:
        logical = environment.current_logical_qubit(state)
        candidates = np.flatnonzero(environment.valid_action_mask(state))
        action = min(
            candidates,
            key=lambda physical: environment.incremental_score(
                state, logical, int(physical)
            ).combined,
        )
        state, _, _, _ = environment.step(int(action))
    return state.logical_to_physical


def sabre_mapping(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    *,
    heuristic: str = "decay",
    seed: int = 0,
    max_iterations: int = 3,
) -> tuple[int, ...]:
    """Return SABRE's initial layout using basic, lookahead, or decay routing."""
    if heuristic not in {"basic", "lookahead", "decay"}:
        raise ValueError("SABRE heuristic must be basic, lookahead, or decay.")
    undirected_edges = problem.hardware.edges.tolist()
    coupling = CouplingMap(
        undirected_edges + [[right, left] for left, right in undirected_edges]
    )
    routing = SabreSwap(
        coupling,
        heuristic=heuristic,
        seed=seed,
        trials=1,
    )
    layout_pass = SabreLayout(
        coupling,
        routing_pass=routing,
        seed=seed,
        max_iterations=max_iterations,
    )
    manager = PassManager(layout_pass)
    manager.run(circuit)
    layout = manager.property_set["layout"]
    return tuple(int(layout[qubit]) for qubit in circuit.qubits)
