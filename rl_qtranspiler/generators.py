"""Circuit-family generators for curriculum training."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, qasm2, qasm3


SUPPORTED_FAMILIES = (
    "random",
    "linear",
    "star",
    "clustered",
    "qaoa",
    "qft_like",
)


def generate_layered_random_circuit(
    num_qubits: int,
    depth: int,
    *,
    two_qubit_density: float = 0.6,
    seed: int | None = None,
) -> QuantumCircuit:
    """Generate an exact-depth random circuit using RZ, SX, X, and CZ gates.

    Every qubit participates in exactly one operation per layer. Consequently,
    all operations within a layer are disjoint and the resulting circuit depth
    equals ``depth`` exactly.
    """
    if num_qubits < 2:
        raise ValueError("num_qubits must be at least two.")
    if depth < 1:
        raise ValueError("depth must be positive.")
    if not 0.0 <= two_qubit_density <= 1.0:
        raise ValueError("two_qubit_density must be in [0, 1].")

    generator = np.random.default_rng(seed)
    circuit = QuantumCircuit(
        num_qubits,
        name=f"random_q{num_qubits}_d{depth}",
    )
    pair_count = min(
        num_qubits // 2,
        int(round(two_qubit_density * num_qubits / 2)),
    )
    previous_pairs: set[tuple[int, int]] = set()

    for _ in range(depth):
        # Avoid immediately repeating a CZ pair, which would be vulnerable to
        # trivial cancellation during a later peephole-optimization pass.
        for _attempt in range(16):
            permutation = generator.permutation(num_qubits)
            pairs = [
                tuple(
                    sorted(
                        (
                            int(permutation[2 * index]),
                            int(permutation[2 * index + 1]),
                        )
                    )
                )
                for index in range(pair_count)
            ]
            if previous_pairs.isdisjoint(pairs):
                break

        paired_qubits = {qubit for pair in pairs for qubit in pair}
        for left, right in pairs:
            circuit.cz(left, right)

        for qubit in range(num_qubits):
            if qubit in paired_qubits:
                continue
            choice = float(generator.random())
            if choice < 0.6:
                angle = float(generator.uniform(-np.pi, np.pi))
                circuit.rz(angle, qubit)
            elif choice < 0.85:
                circuit.sx(qubit)
            else:
                circuit.x(qubit)

        previous_pairs = set(pairs)

    return circuit


def generate_circuit(
    family: str,
    num_qubits: int,
    *,
    layers: int = 4,
    seed: int | None = None,
) -> QuantumCircuit:
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unknown circuit family {family!r}.")
    if not 1 <= num_qubits <= 30:
        raise ValueError("Generated circuits must contain 1-30 qubits.")
    generator = np.random.default_rng(seed)
    circuit = QuantumCircuit(num_qubits, name=f"{family}_{num_qubits}")

    for layer in range(layers):
        if family == "linear":
            pairs = [
                (left, left + 1)
                for left in range(layer % 2, num_qubits - 1, 2)
            ]
        elif family == "star":
            center = layer % num_qubits
            pairs = [
                (center, target)
                for target in range(num_qubits)
                if target != center
            ]
        elif family == "clustered":
            cluster_size = max(2, int(np.sqrt(num_qubits)))
            pairs = []
            for start in range(0, num_qubits, cluster_size):
                cluster = list(
                    range(start, min(start + cluster_size, num_qubits))
                )
                pairs.extend(zip(cluster[:-1], cluster[1:]))
            if num_qubits > cluster_size:
                pairs.append((cluster_size - 1, cluster_size))
        elif family == "qft_like":
            pairs = [
                (left, right)
                for left in range(num_qubits)
                for right in range(left + 1, num_qubits)
                if (right - left + layer) % max(layers, 1) == 0
            ]
        elif family == "qaoa":
            graph = nx.gnp_random_graph(
                num_qubits,
                min(4 / max(num_qubits - 1, 1), 1.0),
                seed=int(generator.integers(0, 2**31)),
            )
            pairs = list(graph.edges)
        else:
            permutation = generator.permutation(num_qubits)
            pairs = [
                (int(permutation[index]), int(permutation[index + 1]))
                for index in range(0, num_qubits - 1, 2)
            ]

        for left, right in pairs:
            circuit.cz(left, right)
        for qubit in range(num_qubits):
            circuit.rz(float(generator.uniform(-np.pi, np.pi)), qubit)
    return circuit


def load_qasm_benchmarks(directory: str | Path) -> list[QuantumCircuit]:
    circuits: list[QuantumCircuit] = []
    for path in sorted(Path(directory).glob("*.qasm")):
        text = path.read_text(encoding="utf-8")
        circuit = qasm3.loads(text) if "OPENQASM 3" in text else qasm2.loads(text)
        circuit.name = path.stem
        circuits.append(circuit)
    return circuits
