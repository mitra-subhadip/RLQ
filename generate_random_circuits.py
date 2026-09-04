#!/usr/bin/env python3
"""Generate large, exact-depth random circuits for placement benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from qiskit import qasm2, qasm3

from rl_qtranspiler.generators import generate_layered_random_circuit


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--min-qubits", type=int, default=50)
    parser.add_argument("--max-qubits", type=int, default=60)
    parser.add_argument("--min-depth", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=100)
    parser.add_argument("--two-qubit-density", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("random_circuits"))
    parser.add_argument("--format", choices=("qasm2", "qasm3"), default="qasm3")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.count < 1:
        raise ValueError("count must be positive.")
    if arguments.min_qubits < 2 or arguments.min_qubits > arguments.max_qubits:
        raise ValueError("Require 2 <= min-qubits <= max-qubits.")
    if arguments.min_depth < 1 or arguments.min_depth > arguments.max_depth:
        raise ValueError("Require 1 <= min-depth <= max-depth.")
    if not 0.0 <= arguments.two_qubit_density <= 1.0:
        raise ValueError("two-qubit-density must be in [0, 1].")


def _print_results(records: list[dict[str, object]]) -> None:
    columns = (
        ("File", "file"),
        ("Qubits", "num_qubits"),
        ("Depth", "actual_depth"),
        ("CZ", "cz"),
        ("RZ", "rz"),
        ("SX", "sx"),
        ("X", "x"),
        ("Seed", "seed"),
    )
    widths = {
        key: max(len(heading), *(len(str(record[key])) for record in records))
        for heading, key in columns
    }
    header = "  ".join(
        heading.ljust(widths[key]) for heading, key in columns
    )
    separator = "  ".join("-" * widths[key] for _, key in columns)
    print(header)
    print(separator)
    for record in records:
        print(
            "  ".join(
                str(record[key]).ljust(widths[key]) for _, key in columns
            )
        )


def main() -> None:
    arguments = parse_arguments()
    _validate_arguments(arguments)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(arguments.seed)
    records: list[dict[str, object]] = []

    for index in range(arguments.count):
        num_qubits = int(
            generator.integers(arguments.min_qubits, arguments.max_qubits + 1)
        )
        target_depth = int(
            generator.integers(arguments.min_depth, arguments.max_depth + 1)
        )
        circuit_seed = int(generator.integers(0, 2**32, dtype=np.uint64))
        circuit = generate_layered_random_circuit(
            num_qubits,
            target_depth,
            two_qubit_density=arguments.two_qubit_density,
            seed=circuit_seed,
        )
        stem = f"random_{index:03d}_q{num_qubits}_d{target_depth}"
        path = arguments.output_dir / f"{stem}.qasm"
        text = qasm3.dumps(circuit) if arguments.format == "qasm3" else qasm2.dumps(circuit)
        path.write_text(text, encoding="utf-8")

        operation_counts = {
            name: int(count) for name, count in circuit.count_ops().items()
        }
        records.append(
            {
            "file": path.name,
            "seed": circuit_seed,
            "num_qubits": circuit.num_qubits,
            "actual_depth": circuit.depth(),
                "cz": operation_counts.get("cz", 0),
                "rz": operation_counts.get("rz", 0),
                "sx": operation_counts.get("sx", 0),
                "x": operation_counts.get("x", 0),
            }
        )

    _print_results(records)
    print(f"\nOutput directory: {arguments.output_dir}")


if __name__ == "__main__":
    main()
