"""Proxy and routed metrics for placement comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log1p
from time import perf_counter

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from .environment import PlacementEnvironment
from .problem import PlacementProblem


@dataclass(frozen=True)
class ProxyMetrics:
    distance_score: float
    calibration_score: float
    combined_score: float
    scoring_runtime_seconds: float


@dataclass(frozen=True)
class RoutedMetrics:
    swap_count: int
    two_qubit_gate_count: int
    depth: int
    estimated_log_infidelity: float
    normalized_log_infidelity: float
    estimated_success_probability: float
    runtime_seconds: float


def evaluate_mapping(
    problem: PlacementProblem,
    mapping: tuple[int, ...] | list[int],
    *,
    started_at: float | None = None,
) -> ProxyMetrics:
    start = started_at if started_at is not None else perf_counter()
    score = PlacementEnvironment(problem).score_mapping(mapping)
    return ProxyMetrics(
        score.distance,
        score.calibration,
        score.combined,
        perf_counter() - start,
    )


def evaluate_routed_circuit(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    mapping: tuple[int, ...] | list[int],
    *,
    seed: int = 0,
    optimization_level: int = 1,
) -> RoutedMetrics:
    """Route with Qiskit SABRE and collect hardware-aware diagnostics."""
    start = perf_counter()
    undirected_edges = problem.hardware.edges.tolist()
    coupling = CouplingMap(
        undirected_edges + [[right, left] for left, right in undirected_edges]
    )
    routed = transpile(
        circuit,
        coupling_map=coupling,
        initial_layout=list(mapping),
        routing_method="sabre",
        seed_transpiler=seed,
        optimization_level=optimization_level,
    )
    elapsed = perf_counter() - start
    two_qubit_count = 0
    log_infidelity = 0.0
    error_by_edge = {
        tuple(sorted((int(left), int(right)))): float(error)
        for (left, right), error in zip(
            problem.hardware.edges,
            problem.hardware.cz_errors,
            strict=True,
        )
    }
    for instruction in routed.data:
        if len(instruction.qubits) != 2:
            continue
        two_qubit_count += 1
        physical = tuple(
            sorted(routed.find_bit(qubit).index for qubit in instruction.qubits)
        )
        error = error_by_edge.get(physical)
        if error is not None:
            multiplier = 3 if instruction.operation.name == "swap" else 1
            log_infidelity += multiplier * -log1p(-error)
    return RoutedMetrics(
        routed.count_ops().get("swap", 0),
        two_qubit_count,
        routed.depth(),
        log_infidelity,
        log_infidelity / max(problem.original_two_qubit_gate_count, 1),
        exp(-log_infidelity),
        elapsed,
    )


def evaluate_problem_mapping(
    problem: PlacementProblem,
    mapping: tuple[int, ...] | list[int],
    *,
    seed: int = 0,
    optimization_level: int = 1,
) -> RoutedMetrics:
    """Rebuild and route the problem's compact two-qubit interaction stream."""
    circuit = QuantumCircuit(problem.num_logical_qubits)
    for left, right in problem.routing_pairs:
        circuit.cz(int(left), int(right))
    return evaluate_routed_circuit(
        circuit,
        problem,
        mapping,
        seed=seed,
        optimization_level=optimization_level,
    )
